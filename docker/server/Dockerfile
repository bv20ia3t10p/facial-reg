FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libopenblas-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY src/ /app/src/
COPY run.py /app/
COPY config.py /app/

# Create directories
RUN mkdir -p /app/data /app/models /app/shared

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV ROLE=server

# Set the entrypoint
ENTRYPOINT ["python"]

# Default command - use the Docker-specific server script
CMD ["/app/src/federated_server_docker.py"] 