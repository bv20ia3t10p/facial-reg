# Use PyTorch CUDA base image
FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-runtime


# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir timm

# Copy the rest of the application
COPY . .

# Expose port
EXPOSE 8080

# Run the API server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"] 