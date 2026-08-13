# nv-segment-ctmr-mcp — Horizon container
# CPU only, no model weights. GPU inference lives on Modal.

FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


FROM python:3.11-slim

RUN useradd --create-home --shell /bin/bash mcp
COPY --from=builder /install /usr/local
WORKDIR /app
COPY server.py .
RUN chown -R mcp:mcp /app
USER mcp

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080

EXPOSE 8080

CMD ["python", "server.py"]
