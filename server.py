"""
server.py — nv-segment-ctmr-mcp
=================================
FastMCP server exposing NV-Segment-CTMR (VISTA3D) 3D medical image
segmentation as MCP tools, deployed via Prefect Horizon.

Unlike the 2D grounding servers in this family, no pixel data passes through
this container. A CT/MR volume is 50-200 MB; base64 in a tool argument is not
a viable transport. This server brokers *URIs* — it validates them, resolves
anatomical structure names to class indices, dispatches to the Modal endpoint,
and returns agent-readable statistics plus a download URL for the mask.

Required environment variables:
    MODAL_API_URL       Base URL of the Modal ASGI app,
                        e.g. https://mathgcloud--nv-segment-ctmr-api.modal.run

Optional environment variables:
    DEFAULT_WAIT_S      Seconds segment_structures polls before handing back
                        a call_id (default 240)
    FASTMCP_DOCKET_URL  rediss://<host>:<port>  Redis for background tasks

Tools:
    list_anatomical_structures(query, limit)      -> searchable class vocabulary
    check_image_uri(image_uri)                    -> reachability / size preflight
    segment_structures(image_uri, structures, …)  -> named organs, submit + wait
    segment_everything(image_uri, modality, …)    -> whole-body, always async
    get_segmentation_status(call_id, …)           -> poll a running job
    health()                                      -> liveness check

Research use only. NV-Segment-CTMR is released under the NVIDIA OneWay
Non-Commercial License and is not a cleared diagnostic device. Outputs are
unvalidated for clinical use and must not drive patient care decisions.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Literal

import requests
from fastmcp import FastMCP

logging.basicConfig(format="[%(levelname)s]: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

MODAL_API_URL = os.environ.get("MODAL_API_URL", "").rstrip("/")

if not MODAL_API_URL:
    logger.warning("MODAL_API_URL is not set — every tool call will fail.")

DEFAULT_WAIT_S = int(os.environ.get("DEFAULT_WAIT_S", "240"))

# Whole-body "everything" mode is ~117 class prompts and runs for minutes.
# Anything at or above this many prompts is forced onto the async path.
ASYNC_PROMPT_THRESHOLD = 24

# Cap the structure table returned to an agent. A 117-class result is a lot of
# tokens to spend on organs with 0.2 mL of predicted volume.
DEFAULT_MAX_STRUCTURES = 40

_VOCAB_CACHE: dict = {}
_VOCAB_FETCHED_AT: float = 0.0
_VOCAB_TTL_S = 3600


# ---------------------------------------------------------------------------
# Modal client
# ---------------------------------------------------------------------------


def _modal_get(path: str, timeout: int = 60) -> dict:
    if not MODAL_API_URL:
        raise RuntimeError("MODAL_API_URL is not set.")
    resp = requests.get(f"{MODAL_API_URL}{path}", timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _modal_post(path: str, payload: dict, timeout: int = 120) -> dict:
    if not MODAL_API_URL:
        raise RuntimeError("MODAL_API_URL is not set.")
    logger.info(f"POST {MODAL_API_URL}{path}")
    resp = requests.post(f"{MODAL_API_URL}{path}", json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _vocabulary() -> dict:
    """Fetch and memoize the class vocabulary. Served from Modal's CPU tier."""
    global _VOCAB_FETCHED_AT
    now = time.time()
    if _VOCAB_CACHE and (now - _VOCAB_FETCHED_AT) < _VOCAB_TTL_S:
        return _VOCAB_CACHE

    data = _modal_get("/labels")
    index_to_name = {int(k): v for k, v in data["index_to_name"].items()}
    _VOCAB_CACHE.clear()
    _VOCAB_CACHE.update(
        {
            "index_to_name": index_to_name,
            "name_to_index": {v.strip().lower(): k for k, v in index_to_name.items()},
            "unsupported_indices": set(data.get("unsupported_indices", [])),
            "everything_sets": data.get("everything_sets", {}),
            "n_classes": data.get("n_classes", len(index_to_name)),
        }
    )
    _VOCAB_FETCHED_AT = now
    logger.info(f"Vocabulary cached: {_VOCAB_CACHE['n_classes']} classes")
    return _VOCAB_CACHE


# ---------------------------------------------------------------------------
# Local validation — runs on Horizon, no GPU spin-up
# ---------------------------------------------------------------------------


def _resolve_structures(structures: list[str]) -> tuple[list[int], list[str]]:
    """
    Map structure names to class indices.

    Returns (indices, unresolved). Matching is case-insensitive exact first,
    then substring, so "left kidney" and "kidney_left" both land somewhere
    sensible without the agent having to memorize NVIDIA's naming.
    """
    vocab = _vocabulary()
    name_to_index = vocab["name_to_index"]

    indices: list[int] = []
    unresolved: list[str] = []

    for raw in structures:
        key = str(raw).strip().lower()
        if key in name_to_index:
            indices.append(name_to_index[key])
            continue
        normalized = key.replace("_", " ").replace("-", " ")
        hits = [
            idx
            for name, idx in name_to_index.items()
            if normalized == name.replace("_", " ").replace("-", " ")
        ]
        if not hits:
            hits = [
                idx
                for name, idx in name_to_index.items()
                if normalized in name.replace("_", " ").replace("-", " ")
            ]
        if len(hits) == 1:
            indices.append(hits[0])
        else:
            unresolved.append(str(raw))

    # De-duplicate, preserving order (VISTA3D resolves overlaps by prompt order).
    seen, ordered = set(), []
    for i in indices:
        if i not in seen:
            seen.add(i)
            ordered.append(i)
    return ordered, unresolved


def _validate_indices(indices: list[int]) -> list[str]:
    """Return human-readable problems with a prompt list, or an empty list."""
    vocab = _vocabulary()
    problems = []
    out_of_range = [i for i in indices if not (0 < i < 512)]
    if out_of_range:
        problems.append(f"indices outside (0, 512): {out_of_range}")
    unsupported = sorted(set(indices) & vocab["unsupported_indices"])
    if unsupported:
        names = [vocab["index_to_name"].get(i, "?") for i in unsupported]
        problems.append(
            f"indices unsupported by this checkpoint: "
            f"{list(zip(unsupported, names))}"
        )
    unknown = [i for i in indices if i not in vocab["index_to_name"]]
    if unknown:
        problems.append(f"indices not present in metadata.json: {unknown}")
    return problems


def _validate_presigned(url: str, field: str) -> str | None:
    """
    Reject anything that isn't a presigned https URL.

    Raw s3:// URIs would require credentials somewhere in the pipeline; the
    whole point of the presigned design is that neither this container nor
    the Modal container holds any. Failing here gives a clear message instead
    of a 403 six minutes into a GPU job.
    """
    if not url or not isinstance(url, str):
        return f"{field} is required."
    if url.startswith("s3://"):
        return (
            f"{field} must be a presigned https URL, not a raw s3:// URI. "
            "Nothing in this pipeline holds AWS credentials by design — "
            "generate a presigned URL scoped to the single object."
        )
    if not url.startswith("https://"):
        return f"{field} must be an https:// URL."
    if "X-Amz-Signature" not in url and "Signature=" not in url:
        return (
            f"{field} does not look presigned (no signature in the query "
            "string). An unsigned bucket URL will 403."
        )
    return None


def _absolute_download_url(result: dict) -> str | None:
    path = result.get("mask_download_path")
    return f"{MODAL_API_URL}{path}" if path else None


def _summarize(
    result: dict,
    max_structures: int = DEFAULT_MAX_STRUCTURES,
    include_bboxes: bool = False,
) -> dict:
    """Trim a Modal result to something worth putting in an agent's context."""
    if not result.get("success"):
        return result

    structures = result.get("structures", {}) or {}
    ranked = sorted(
        structures.items(),
        key=lambda kv: kv[1].get("volume_ml", 0.0),
        reverse=True,
    )
    kept = ranked[:max_structures]

    table = {}
    for name, s in kept:
        entry = {
            "label_index": s["label_index"],
            "volume_ml": s["volume_ml"],
            "voxels": s["voxels"],
        }
        if include_bboxes:
            entry["bbox_voxel"] = s.get("bbox_voxel")
        table[name] = entry

    summary = {
        "success": True,
        "job_id": result.get("job_id"),
        "study_id": result.get("study_id"),
        "mask_download_url": _absolute_download_url(result),
        "mask_uploaded_to_presigned_url": bool(result.get("mask_uploaded")),
        "mask_size_mb": round((result.get("mask_bytes") or 0) / 1024**2, 2),
        "volume_shape": result.get("shape"),
        "voxel_volume_mm3": result.get("voxel_volume_mm3"),
        "prompt_source": result.get("prompt_source"),
        "n_structures_requested": result.get("n_prompts"),
        "n_structures_found": len(structures),
        "structures": table,
        "timing_s": result.get("timing_s"),
    }

    if len(ranked) > len(kept):
        summary["structures_truncated"] = {
            "shown": len(kept),
            "total": len(ranked),
            "note": "Ranked by volume. Raise max_structures to see the rest.",
        }

    absent = result.get("labels_absent") or []
    if absent:
        vocab = _vocabulary()
        summary["structures_not_detected"] = [
            vocab["index_to_name"].get(i, f"class_{i}") for i in absent[:30]
        ]

    summary["disclaimer"] = (
        "Research output from an unvalidated model. Not for clinical use."
    )
    return summary


def _poll_until(call_id: str, wait_s: int) -> dict:
    """Poll the Modal job endpoint with backoff. Returns the last status seen."""
    deadline = time.time() + max(0, wait_s)
    delay = 3.0
    status = {"call_id": call_id, "status": "pending"}
    while time.time() < deadline:
        status = _modal_get(f"/jobs/{call_id}", timeout=30)
        if status.get("status") != "pending":
            return status
        time.sleep(min(delay, max(1.0, deadline - time.time())))
        delay = min(delay * 1.4, 20.0)
    return status


def _pending_response(call_id: str, note: str) -> str:
    return json.dumps(
        {
            "success": True,
            "status": "pending",
            "call_id": call_id,
            "note": note,
            "next": "Call get_segmentation_status with this call_id.",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    )


# ---------------------------------------------------------------------------
# FastMCP app
# ---------------------------------------------------------------------------

mcp = FastMCP("nv-segment-ctmr-mcp")


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
async def list_anatomical_structures(
    query: str | None = None,
    limit: int = 50,
) -> str:
    """
    Search the anatomical structure vocabulary this model can segment.

    NV-Segment-CTMR supports 345+ named classes spanning abdominal organs,
    cardiovascular and respiratory structures, the full vertebral column,
    ribs, long bones, detailed brain parcellation, and some pathology
    (lung/hepatic/pancreatic tumors, bone lesions, kidney cysts).

    Call this before segment_structures if you are unsure whether a structure
    is available or what NVIDIA calls it. Cheap — served from cache.

    Args:
        query:  Optional case-insensitive substring, e.g. "kidney", "vertebrae",
                "lung". Omit to get the size of each modality's everything-set.
        limit:  Maximum matches to return (default 50).

    Returns:
        JSON mapping structure names to class indices.
    """
    try:
        vocab = _vocabulary()
        index_to_name = vocab["index_to_name"]

        if query:
            q = str(query).strip().lower()
            matches = {
                name: idx
                for idx, name in sorted(index_to_name.items())
                if q in name.lower()
            }
            truncated = len(matches) > limit
            matches = dict(list(matches.items())[:limit])
            return json.dumps(
                {
                    "success": True,
                    "query": query,
                    "n_matches": len(matches),
                    "truncated": truncated,
                    "structures": matches,
                }
            )

        everything = vocab["everything_sets"]
        return json.dumps(
            {
                "success": True,
                "n_classes": vocab["n_classes"],
                "everything_set_sizes": {
                    k: len(v) for k, v in everything.items()
                },
                "sample": dict(list(sorted(index_to_name.items()))[:limit]),
                "note": "Pass a query to search, e.g. query='rib' or query='brain'.",
            }
        )
    except Exception as e:
        logger.error(f"list_anatomical_structures failed: {e}", exc_info=True)
        return json.dumps({"success": False, "error": str(e)})


@mcp.tool()
async def check_image_uri(image_uri: str) -> str:
    """
    Preflight a NIfTI URI before committing GPU time to it.

    Issues a HEAD request to confirm the URI resolves and reports the declared
    size and content type. Catches expired presigned links and typo'd paths
    without cold-starting the inference container.

    Args:
        image_uri:  Presigned https:// GET URL for a .nii or .nii.gz object.

    Returns:
        JSON with reachability, size, and a rough runtime expectation.
    """
    try:
        if image_uri.startswith("s3://"):
            return json.dumps(
                {
                    "success": False,
                    "reason": (
                        "Raw s3:// URIs are not supported anywhere in this "
                        "pipeline — no component holds AWS credentials. "
                        "Generate a presigned GET URL scoped to the single "
                        "object and pass that instead."
                    ),
                }
            )

        if not image_uri.startswith("https://"):
            return json.dumps(
                {
                    "success": False,
                    "reason": "image_uri must be an https:// URL.",
                }
            )

        resp = requests.head(image_uri, timeout=30, allow_redirects=True)
        size = int(resp.headers.get("Content-Length") or 0)
        looks_nifti = any(
            image_uri.lower().split("?")[0].endswith(ext)
            for ext in (".nii", ".nii.gz")
        )

        return json.dumps(
            {
                "success": resp.status_code < 400,
                "status_code": resp.status_code,
                "size_mb": round(size / 1024**2, 2) if size else None,
                "content_type": resp.headers.get("Content-Type"),
                "filename_looks_like_nifti": looks_nifti,
                "warning": (
                    None
                    if looks_nifti
                    else "URI does not end in .nii/.nii.gz — confirm the format."
                ),
            }
        )
    except Exception as e:
        return json.dumps({"success": False, "error": str(e), "image_uri": image_uri})


@mcp.tool()
async def segment_structures(
    image_uri: str,
    study_id: str,
    structures: list[str] | None = None,
    label_indices: list[int] | None = None,
    wait_s: int = DEFAULT_WAIT_S,
    max_structures: int = DEFAULT_MAX_STRUCTURES,
    include_bboxes: bool = False,
    output_put_url: str | None = None,
) -> str:
    """
    Segment specific named anatomical structures in a 3D CT or MR volume.

    Use this when you know which organs you want — "liver and spleen",
    "L1 through L5", "both kidneys". For a whole-body pass use
    segment_everything instead.

    The mask is written server-side and returned as a download URL, never
    inlined. What comes back here is a table of per-structure volumes in mL,
    which is what you should reason over.

    Submits and polls for up to wait_s. If the job is still running when that
    elapses, returns a call_id — pass it to get_segmentation_status.

    Args:
        image_uri:       Presigned https:// GET URL for a single .nii/.nii.gz
                         object. Raw s3:// URIs are not accepted — nothing in
                         this pipeline holds AWS credentials.
        study_id:        Identifier for this study; used in output paths and logs.
        structures:      Structure names, e.g. ["liver", "spleen", "aorta"].
                         Resolved against the model vocabulary; call
                         list_anatomical_structures if unsure of naming.
        label_indices:   Class indices, as an alternative to structures.
        wait_s:          Seconds to poll before handing back a call_id (default 240).
        max_structures:  Cap on the returned structure table (default 40).
        include_bboxes:  Include per-structure voxel bounding boxes (default False).
        output_put_url:  Optional presigned https:// PUT URL for a single output
                         object. The mask is uploaded there in addition to the
                         Modal volume.

    Returns:
        JSON with per-structure volumes, a mask download URL, and timings —
        or a call_id if still running.
    """
    try:
        if not structures and not label_indices:
            return json.dumps(
                {
                    "success": False,
                    "reason": "Provide either structures (names) or label_indices.",
                }
            )

        uri_problem = _validate_presigned(image_uri, "image_uri")
        if uri_problem:
            return json.dumps({"success": False, "reason": uri_problem})
        if output_put_url:
            put_problem = _validate_presigned(output_put_url, "output_put_url")
            if put_problem:
                return json.dumps({"success": False, "reason": put_problem})

        if label_indices:
            indices = [int(i) for i in label_indices]
            unresolved: list[str] = []
        else:
            indices, unresolved = _resolve_structures(structures or [])

        if unresolved:
            return json.dumps(
                {
                    "success": False,
                    "reason": f"Could not resolve structure name(s): {unresolved}",
                    "hint": "Call list_anatomical_structures with a query to find "
                    "the exact naming, or pass label_indices directly.",
                    "resolved": indices,
                }
            )

        problems = _validate_indices(indices)
        if problems:
            return json.dumps(
                {"success": False, "reason": "; ".join(problems), "indices": indices}
            )

        payload = {
            "image_uri": image_uri,
            "study_id": study_id,
            "label_prompt": indices,
            "compute_stats": True,
        }
        if output_put_url:
            payload["output_put_url"] = output_put_url

        # Peak VRAM scales with simultaneous prompt count, not model size.
        if len(indices) > ASYNC_PROMPT_THRESHOLD:
            payload["label_chunk_size"] = 16

        submitted = _modal_post("/jobs", payload)
        call_id = submitted["call_id"]
        logger.info(
            f"segment_structures: {study_id}  n_prompts={len(indices)}  "
            f"call_id={call_id}"
        )

        status = _poll_until(call_id, wait_s)
        if status.get("status") == "pending":
            return _pending_response(
                call_id,
                f"Still segmenting after {wait_s}s ({len(indices)} structures).",
            )
        if status.get("status") != "done":
            return json.dumps(
                {"success": False, "call_id": call_id, "status": status}
            )

        out = _summarize(status["result"], max_structures, include_bboxes)
        out["call_id"] = call_id
        return json.dumps(out)

    except Exception as e:
        logger.error(f"segment_structures failed: {e}", exc_info=True)
        return json.dumps({"success": False, "error": str(e), "study_id": study_id})


@mcp.tool()
async def segment_everything(
    image_uri: str,
    study_id: str,
    modality: Literal["CT_BODY", "MRI_BODY", "MRI_BRAIN"],
    brain_preprocessed: bool = False,
    label_chunk_size: int = 16,
    output_put_url: str | None = None,
) -> str:
    """
    Run whole-body (or whole-brain) automatic segmentation over a volume.

    Uses the model's predefined class set for the given modality — roughly 117
    classes for CT_BODY, 50 for MRI_BODY, 132 for MRI_BRAIN. This takes several
    minutes, so it always returns a call_id immediately; poll with
    get_segmentation_status.

    MRI_BRAIN requires a standard T1 that has already been skull-stripped and
    intensity-normalized. Running it on a raw T1 produces confident nonsense,
    so brain_preprocessed must be set True to acknowledge this.

    Args:
        image_uri:          Presigned https:// GET URL for a single .nii/.nii.gz.
        study_id:           Identifier for this study.
        modality:           CT_BODY, MRI_BODY, or MRI_BRAIN.
        brain_preprocessed: Required True for MRI_BRAIN — confirms the volume is
                            skull-stripped and normalized.
        label_chunk_size:   Prompts per forward pass; lower to cut peak VRAM
                            (default 16, 0 disables chunking).
        output_put_url:     Optional presigned https:// PUT URL for the mask.

    Returns:
        JSON with a call_id to poll.
    """
    try:
        uri_problem = _validate_presigned(image_uri, "image_uri")
        if uri_problem:
            return json.dumps({"success": False, "reason": uri_problem})
        if output_put_url:
            put_problem = _validate_presigned(output_put_url, "output_put_url")
            if put_problem:
                return json.dumps({"success": False, "reason": put_problem})

        if modality == "MRI_BRAIN" and not brain_preprocessed:
            return json.dumps(
                {
                    "success": False,
                    "reason": (
                        "MRI_BRAIN only supports standard T1 images that have "
                        "been skull-stripped and intensity-normalized. Preprocess "
                        "the volume first, then set brain_preprocessed=True."
                    ),
                    "reference": "https://github.com/junyuchen245/MIR/tree/main/"
                    "tutorials/brain_MRI_preprocessing",
                }
            )

        payload = {
            "image_uri": image_uri,
            "study_id": study_id,
            "modality": modality,
            "label_chunk_size": int(label_chunk_size),
            "compute_stats": True,
        }
        if output_put_url:
            payload["output_put_url"] = output_put_url

        submitted = _modal_post("/jobs", payload)
        call_id = submitted["call_id"]
        n_expected = len(_vocabulary()["everything_sets"].get(modality, []))
        logger.info(f"segment_everything: {study_id}  {modality}  call_id={call_id}")

        return _pending_response(
            call_id,
            f"{modality} everything-mode submitted (~{n_expected} classes). "
            "Expect several minutes.",
        )

    except Exception as e:
        logger.error(f"segment_everything failed: {e}", exc_info=True)
        return json.dumps({"success": False, "error": str(e), "study_id": study_id})


@mcp.tool()
async def get_segmentation_status(
    call_id: str,
    wait_s: int = 0,
    max_structures: int = DEFAULT_MAX_STRUCTURES,
    include_bboxes: bool = False,
) -> str:
    """
    Check on a segmentation job submitted by segment_structures or
    segment_everything.

    Args:
        call_id:         Job identifier returned at submission.
        wait_s:          Optionally block up to this many seconds waiting for
                         completion (default 0 = check once and return).
        max_structures:  Cap on the returned structure table (default 40).
        include_bboxes:  Include per-structure voxel bounding boxes.

    Returns:
        JSON with status pending / done / error / expired. When done, includes
        per-structure volumes and the mask download URL. Results are retained
        for 7 days.
    """
    try:
        status = (
            _poll_until(call_id, wait_s)
            if wait_s > 0
            else _modal_get(f"/jobs/{call_id}", timeout=30)
        )

        state = status.get("status")
        if state == "pending":
            return json.dumps(
                {"success": True, "status": "pending", "call_id": call_id}
            )
        if state == "expired":
            return json.dumps(
                {
                    "success": False,
                    "status": "expired",
                    "call_id": call_id,
                    "reason": "Results are retained for 7 days. Resubmit the job.",
                }
            )
        if state != "done":
            return json.dumps(
                {"success": False, "call_id": call_id, "status": status}
            )

        out = _summarize(status["result"], max_structures, include_bboxes)
        out["call_id"] = call_id
        out["status"] = "done"
        return json.dumps(out)

    except Exception as e:
        logger.error(f"get_segmentation_status failed: {e}", exc_info=True)
        return json.dumps({"success": False, "error": str(e), "call_id": call_id})


@mcp.tool()
async def health() -> str:
    """Liveness probe. Reports Modal endpoint configuration and reachability."""
    info = {
        "status": "ok",
        "service": "nv-segment-ctmr-mcp",
        "modal": {
            "api_url": MODAL_API_URL or "(not set)",
            "configured": bool(MODAL_API_URL),
        },
        "defaults": {
            "wait_s": DEFAULT_WAIT_S,
            "max_structures": DEFAULT_MAX_STRUCTURES,
            "async_prompt_threshold": ASYNC_PROMPT_THRESHOLD,
        },
    }
    if MODAL_API_URL:
        try:
            info["modal"]["upstream"] = _modal_get("/health", timeout=15)
            info["modal"]["reachable"] = True
        except Exception as e:
            info["status"] = "degraded"
            info["modal"]["reachable"] = False
            info["modal"]["error"] = str(e)
    return json.dumps(info)


if __name__ == "__main__":
    mcp.run(
        stateless_http=True,
        json_response=True,
        # No image payloads cross this boundary — only URIs and JSON results.
        max_request_body_size=2 * 1024 * 1024,
    )
