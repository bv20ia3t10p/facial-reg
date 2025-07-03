# BioEmo API

Privacy-Preserving Biometric Authentication API with Federated Learning, Homomorphic Encryption, and Differential Privacy.

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

2. Install the package in development mode:
```bash
pip install -e .
```

## Running the API

1. Make sure you're in the `api` directory:
```bash
cd api
```

2. Run the API server:
```bash
uvicorn src.app:app --host 0.0.0.0 --port 8080 --reload
```

Or use the Python entry point:
```bash
python -m src.app
```

## Development

The project structure is organized as follows:

```
api/
├── src/                    # Main package directory
│   ├── __init__.py        # Package initialization
│   ├── app.py             # FastAPI application
│   ├── client_api.py      # Client API endpoints
│   ├── db/                # Database models and utilities
│   ├── models/            # ML models
│   ├── privacy/           # Privacy-related components
│   ├── routes/            # API routes
│   ├── services/          # Service layer
│   └── utils/             # Utility functions
├── setup.py               # Package setup
├── requirements.txt       # Dependencies
└── README.md             # This file
```

## Environment Variables

The following environment variables can be set:

- `DATABASE_URL`: SQLite database URL (default: `sqlite:///data/biometric.db`)
- `MAX_DP_EPSILON`: Maximum differential privacy epsilon (default: `100.0`)
- `DP_DELTA`: Differential privacy delta (default: `1e-5`)
- `DP_MAX_GRAD_NORM`: Maximum gradient norm for DP-SGD (default: `5.0`)
- `DP_NOISE_MULTIPLIER`: Noise multiplier for DP-SGD (default: `0.2`) 