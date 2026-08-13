# mcp-nv-segment-ctmr

A [FastMCP](https://github.com/jlowin/fastmcp) server exposing
**NV-Segment-CTMR** (NVIDIA's VISTA3D-based 3D medical image segmentation
foundation model) as MCP tools, deployed via Prefect Horizon. GPU inference is
dispatched to a **Modal** serverless L40S endpoint, so the container needs no
GPU, no MONAI, and no model weights.

**Research use only.** NV-Segment-CTMR is released under the NVIDIA OneWay
Non-Commercial License. It is not a cleared diagnostic device and its outputs
must not drive patient care decisions.

## Why this one is different

The other servers in this family (locate-anything, fundus-agent, MIRAGE) pass
images as base64 in a tool argument. That does not work here:

| | 2D fundus / CXR | 3D CT / MR |
|---|---|---|
| Typical payload | 0.3–3 MB | 50–200 MB (`.nii.gz`, already compressed) |
| Runtime | 2–20 s | 1–10 min (sliding window over 117 class prompts) |
| Output | a few boxes | a mask the same size as the input |

So the contract inverts. **Volumes are passed by URI**, the mask is written
server-side and returned as a download URL, and long jobs use submit/poll
rather than a blocking request. What crosses the MCP boundary is structure
names in and volumes in mL out — things an agent can actually reason over.

## Architecture

```
MCP client ──► Horizon (this container, CPU)
                 │  name→index resolution, URI preflight, job brokering
                 ▼
              Modal ASGI app (CPU)  ──►  SegmentCTMR class (L40S)
                 │  /jobs /labels /files      VISTA3D sliding-window inference
                 ▼
              Modal Volume  ──►  mask .nii.gz download URL
```

The Modal side lives in [`modal_nv_segment_ctmr.py`](modal_nv_segment_ctmr.py);
see [`download_nv_segment_ctmr.py`](download_nv_segment_ctmr.py) for fetching
the HF repo and the volume setup steps.

## Tools

| Tool | Description | Returns |
|------|-------------|---------|
| `list_anatomical_structures` | Search the 345+ class vocabulary | Name → index map |
| `check_image_uri` | HEAD preflight on a NIfTI URI | Reachability, size |
| `segment_structures` | Segment named organs; submits and polls | Volumes + mask URL |
| `segment_everything` | Whole-body/brain pass; always async | `call_id` |
| `get_segmentation_status` | Poll a running job | Status or full result |
| `health` | Liveness probe, includes upstream check | Status |

### Typical agent flow

```
list_anatomical_structures(query="kidney")
  → {"kidney_right": 14, "kidney_left": 5, "kidney_cyst_left": 116, …}

check_image_uri("https://…/abdomen_ct.nii.gz")
  → {"success": true, "size_mb": 62.4, "filename_looks_like_nifti": true}

segment_structures(
    image_uri="https://…/abdomen_ct.nii.gz",
    study_id="case_0042",
    structures=["liver", "spleen", "kidney_left", "kidney_right"])
  → {"structures": {"liver": {"volume_ml": 1487.2, …}, …},
     "mask_download_url": "https://…/files/case_0042-17.../case_0042_seg.nii.gz"}
```

`segment_everything` returns a `call_id` immediately; feed it to
`get_segmentation_status` (optionally with `wait_s`) until it reports `done`.

## Design notes

**Name resolution happens here, not on the GPU.** The class vocabulary is
fetched once from Modal's CPU tier and memoized for an hour. Matching is
case-insensitive exact, then normalized, then unique-substring — so `"left
kidney"`, `"kidney_left"`, and `"Kidney Left"` all resolve. Ambiguous or
unknown names come back as a validation error with a pointer to
`list_anatomical_structures` rather than a MONAI stack trace.

**Prompt validation is local.** Indices are range-checked and filtered against
the checkpoint's unsupported set (16, 129–131, 133, 137–145, 162) before
anything is dispatched, so a bad request never costs GPU time.

**Results are trimmed.** A 117-class result is a lot of tokens to spend on
organs with 0.2 mL of predicted volume. Structures are ranked by volume and
capped at `max_structures` (default 40); bounding boxes are opt-in.

**MRI_BRAIN has a guard.** The checkpoint only supports skull-stripped,
intensity-normalized T1. Running it on a raw T1 produces confident nonsense, so
`brain_preprocessed=True` is required as an explicit acknowledgment.

## Configuration

| Variable | Required | Description |
|----------|----------|-------------|
| `MODAL_API_URL` | yes | Modal ASGI base URL, e.g. `https://mathgcloud--nv-segment-ctmr-api.modal.run` |
| `DEFAULT_WAIT_S` | no | Seconds `segment_structures` polls before returning a `call_id` (default `240`) |
| `FASTMCP_DOCKET_URL` | no | Redis URL (`rediss://…`) for background tasks |

## Connecting

```json
{
  "mcpServers": {
    "nv-segment-ctmr": {
      "url": "https://nv-segment-ctmr.fastmcp.app/mcp"
    }
  }
}
```

## Running

```bash
# Docker
docker build -t nv-segment-ctmr-mcp .
docker run -p 8080:8080 \
  -e MODAL_API_URL=https://<your-modal-app>.modal.run \
  nv-segment-ctmr-mcp

# Local
pip install -r requirements.txt
export MODAL_API_URL=https://<your-modal-app>.modal.run
python server.py
```

Multi-stage `python:3.11-slim` build, runs as a non-root user, exposes port
8080. Serves over stateless HTTP with JSON responses.

## License

Apache License 2.0 for this code — see [LICENSE](LICENSE). The underlying model
is NVIDIA OneWay Non-Commercial.
