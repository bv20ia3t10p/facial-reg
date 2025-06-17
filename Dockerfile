# Use PyTorch CUDA base image
ARG BASE_IMAGE=pytorch/pytorch:2.7.1-cuda11.8-cudnn9-runtime
FROM ${BASE_IMAGE}

# Set environment variables
ENV CUDA_VISIBLE_DEVICES=0
ENV TORCH_USE_CUDA_DSA=1
ENV CUDA_LAUNCH_BLOCKING=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    sqlite3 \
    iputils-ping \
    && rm -rf /var/lib/apt/lists/*

# Copy application files
COPY . /app
WORKDIR /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set default command
CMD ["python", "api/src/improved_privacy_training.py"] 