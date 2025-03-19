# Use NVIDIA CUDA base image with runtime support
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    ffmpeg \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install cuDNN 9.x manually
RUN wget https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-9.1.0.70_cuda12-archive.tar.xz \
    && tar -xvf cudnn-linux-x86_64-9.1.0.70_cuda12-archive.tar.xz \
    && cp cudnn-linux-x86_64-9.1.0.70_cuda12-archive/lib/* /usr/local/cuda/lib64/ \
    && cp cudnn-linux-x86_64-9.1.0.70_cuda12-archive/include/* /usr/local/cuda/include/ \
    && rm -rf cudnn-linux-x86_64-9.1.0.70_cuda12-archive.tar.xz cudnn-linux-x86_64-9.1.0.70_cuda12-archive \
    && ldconfig

# Set Python 3 as default
RUN ln -s /usr/bin/python3 /usr/bin/python

# Install Python dependencies
RUN pip3 install --no-cache-dir --upgrade pip

# Install runpod module
RUN pip3 install --no-cache-dir runpod

# Clone and install faster-whisper
RUN git clone https://github.com/Infinity242/faster-whisper.git /tmp/faster-whisper \
    && cd /tmp/faster-whisper \
    && pip3 install --no-cache-dir . \
    && rm -rf /tmp/faster-whisper

# Install PyTorch with CUDA 12.1 support (cuDNN included in runtime image)
RUN pip3 install --no-cache-dir \
    torch==2.4.1+cu121 \
    torchvision==0.19.1+cu121 \
    torchaudio==2.4.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Copy handler script
COPY rp_handler.py .

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    WHISPER_MODEL=large-v3 \
    DEVICE=cuda

# Run the RunPod serverless handler
CMD ["python", "rp_handler.py"]
