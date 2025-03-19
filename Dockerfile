# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3 as default
RUN ln -s /usr/bin/python3 /usr/bin/python

# Install Python dependencies
RUN pip3 install --no-cache-dir --upgrade pip

# Install runpod module (required for RunPod serverless)
RUN pip3 install --no-cache-dir runpod

# Clone and install faster-whisper
RUN git clone https://github.com/Infinity242/faster-whisper.git /tmp/faster-whisper \
    && cd /tmp/faster-whisper \
    && pip3 install --no-cache-dir . \
    && rm -rf /tmp/faster-whisper

# Install PyTorch with CUDA support
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu122

# Copy handler script (you’ll need to create this)
COPY rp_handler.py .

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    WHISPER_MODEL=large-v3 \
    DEVICE=cuda

# Run the RunPod serverless handler
CMD ["python", "rp_handler.py"]
