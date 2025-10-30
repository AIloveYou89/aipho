FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    CT2_USE_CUDA=1 \
    CT2_CUDA_ENABLE_SDP_ATTENTION=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip ffmpeg curl ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN python3 -m pip install --upgrade pip && pip3 install -r requirements.txt

# Convert PhoWhisper model to CTranslate2 format
# Chọn model size: tiny, base, small, medium, large
ARG MODEL_SIZE=large
ENV MODEL_SIZE=${MODEL_SIZE}

RUN echo "Converting vinai/PhoWhisper-${MODEL_SIZE} to CTranslate2..." && \
    ct2-transformers-converter \
      --model vinai/PhoWhisper-${MODEL_SIZE} \
      --output_dir /models/PhoWhisper-${MODEL_SIZE}-ct2 \
      --quantization float16 \
      --force

# Copy handler
COPY st_handler.py .

# Environment variables
ENV MODEL_DIR="/models/PhoWhisper-large-ct2" \
    OUT_DIR="/runpod-volume/out" \
    DEVICE="cuda" \
    COMPUTE_TYPE="float16" \
    VAD_FILTER="1" \
    LANG="vi" \
    MAX_CHUNK_LEN="30" \
    SRT="1" \
    VTT="0"

CMD ["python3", "-u", "st_handler.py"]
