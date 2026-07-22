# ── TurboParse Dockerfile ──────────────────────────────────────────────
# Cloud Run deployment:
#   1. Upload model/ to a GCS bucket
#   2. Mount the bucket at /mnt/model via Cloud Storage FUSE
#   3. Set MODEL_PATH=/mnt/model/pdf_tokens_type.model (or use default)
FROM python:3.13-slim

# System dependencies for OpenCV, lightgbm, pypdfium2
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Install Python deps ──────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy source only (model is mounted at runtime) ───────────────
COPY src/ ./src/

# ── Model mount point for Cloud Storage FUSE ─────────────────────
# Upload your model/ directory to a GCS bucket, then mount it here.
RUN mkdir -p /mnt/model

# ── Config ───────────────────────────────────────────────────────
ENV MODEL_PATH=/mnt/model/pdf_tokens_type.model
ENV PYTHONUNBUFFERED=1

EXPOSE 8080

CMD sh -c "python -m uvicorn api:app --app-dir src --host 0.0.0.0 --port $PORT"
