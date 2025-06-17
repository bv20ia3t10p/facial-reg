# Emotion Recognition API

A standalone FastAPI service that performs emotion recognition using the FERPlus EMA model.

## Setup

1. Make sure the FERPlus EMA checkpoint file is in the root directory:
```bash
# Verify the model file exists
ls ferplus_ema.pth
```

2. Build and run the Docker container:
```bash
docker-compose up --build
```

The API will be available at `http://localhost:8080`

## API Endpoints

### POST /predict
Upload an image to get emotion predictions.

Example using curl:
```bash
curl -X POST -F "file=@/path/to/image.jpg" http://localhost:8080/predict
```

Response format:
```json
[
  {
    "emotion": "happiness",
    "probability": 0.85
  },
  {
    "emotion": "neutral",
    "probability": 0.10
  },
  ...
]
```

### GET /health
Health check endpoint.

Example:
```bash
curl http://localhost:8080/health
```

Response:
```json
{
  "status": "healthy"
}
```

## Supported Emotions
- neutral
- happiness
- surprise
- sadness
- anger
- disgust
- fear
- contempt

## Requirements
- NVIDIA GPU with CUDA support
- Docker with NVIDIA Container Toolkit
- FERPlus EMA checkpoint file (ferplus_ema.pth) 