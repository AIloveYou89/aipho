FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    CT2_USE_CUDA=1 \
    CT2_CUDA_ENABLE_SDP_ATTENTION=1 \
    CUDA_HOME=/usr/local/cuda \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip python3-dev \
      ffmpeg curl ca-certificates git \
      libcudnn8 libcudnn8-dev \
      build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN python3 -m pip install --upgrade pip && \
    pip3 install -r requirements.txt

# Convert PhoWhisper model to CTranslate2 format
ARG MODEL_SIZE=large
ENV MODEL_SIZE=${MODEL_SIZE}

RUN echo "Converting vinai/PhoWhisper-${MODEL_SIZE} to CTranslate2..." && \
    ct2-transformers-converter \
      --model vinai/PhoWhisper-${MODEL_SIZE} \
      --output_dir /models/PhoWhisper-${MODEL_SIZE}-ct2 \
      --quantization float16 \
      --force && \
    echo "Model files:" && \
    ls -lah /models/PhoWhisper-${MODEL_SIZE}-ct2/

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
