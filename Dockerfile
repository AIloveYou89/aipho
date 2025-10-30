FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    CT2_USE_CUDA=1 \
    CT2_CUDA_ENABLE_SDP_ATTENTION=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip ffmpeg curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN python3 -m pip install --upgrade pip && pip3 install --no-cache-dir -r requirements.txt

COPY st_handler.py .

# Dùng model CT2 có sẵn (không convert ở bước build)
ENV MODEL_ID="kiendt/PhoWhisper-large-ct2" \
    MODEL_DIR="/models" \
    OUT_DIR="/runpod-volume/out" \
    DEVICE="cuda" \
    COMPUTE_TYPE="float16" \
    VAD_FILTER="1" \
    LANG="vi" \
    MAX_CHUNK_LEN="30" \
    SRT="1" \
    VTT="0"

CMD ["python3", "-u", "st_handler.py"]
