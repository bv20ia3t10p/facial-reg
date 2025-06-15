# Multi-stage build for minimal image size
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 as builder

# Install Python and build dependencies
# Split apt-get update and install to better use cache
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    gcc \
    g++ \
    pkg-config \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install pip and wheel first to optimize caching
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip wheel setuptools

# Split requirements into core and optional for better caching
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm -rf /root/.cache/pip/*

# Production stage
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Install Python and runtime dependencies
# Split apt-get update and install to better use cache
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory and create required directories early
WORKDIR /app
RUN mkdir -p /data/cache /data/models /data/logs

# Set environment variables
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Setup non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser && \
    chown -R appuser:appuser /app /data

# Copy application code (do this late to maximize cache usage)
COPY api/ /app/

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3.11 -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Expose port
EXPOSE 8000

# Start application
CMD ["python3.11", "main.py"] 