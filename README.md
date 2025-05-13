# Privacy-Preserving Facial Recognition System

This is a refactored implementation of a privacy-preserving facial recognition system. The codebase has been completely restructured following SOLID principles and best practices for maintainability and extensibility.

## Features

- 🔍 **Face Detection & Recognition**: Detect and recognize faces using modern deep learning techniques
- 🔒 **Privacy Preservation**: Options for differential privacy and homomorphic encryption
- 🌐 **Federated Learning**: Train models across multiple clients without sharing raw data
- 🔧 **Modular Architecture**: Follows SOLID principles with clear separation of concerns
- 📊 **Benchmarking**: Tools to evaluate performance and privacy tradeoffs

## Directory Structure

```
├─ src/                     # Source code
│   ├─ config/              # Configuration settings
│   ├─ interfaces/          # Abstract interfaces
│   ├─ models/              # Model implementations
│   ├─ data/                # Data handling
│   ├─ services/            # Business logic
│   ├─ repositories/        # Data access
│   ├─ controllers/         # Control logic
│   ├─ utils/               # Utility functions
│   ├─ training/            # Training modules
│   │   ├─ standard/        # Supervised learning
│   │   ├─ federated/       # Federated learning
│   │   └─ utils/           # Training utilities
│   └─ scripts/             # Command-line scripts
│       ├─ data/            # Data processing scripts
│       ├─ training/        # Training scripts
│       └─ deployment/      # Deployment scripts
├─ data/                    # Data directory
├─ models/                  # Saved models
├─ docker/                  # Docker configurations
├─ run.py                   # Main entry point
├─ run.sh                   # Unix shell script
└─ run.bat                  # Windows batch script
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/facial-reg.git
cd facial-reg
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

The system provides a unified command-line interface through the `run.py` script. You can also use the convenient shell scripts `run.sh` (Unix) or `run.bat` (Windows).

### Data Extraction

Extract faces from the CASIA WebFace dataset:

```bash
python run.py extract --rec_file path/to/train.rec --idx_file path/to/train.idx --lst_file path/to/train.lst --output_dir data/extracted
```

### Dataset Preparation

Prepare and partition the dataset for training:

```bash
python run.py prepare --data_dir data/extracted --output_dir data/partitioned --validate --clean --partition
```

### Training

Train a standard supervised model:

```bash
python run.py train standard --data_dir data/partitioned/server --model_dir models
```

Train with differential privacy:

```bash
python run.py train standard --data_dir data/partitioned/server --model_dir models --use_dp
```

### Federated Learning

Prepare for federated learning:

```bash
python run.py train federated-prep --dataset_dir data/extracted --output_dir data/partitioned --add_unseen_classes
```

Deploy federated learning system:

```bash
python run.py deploy --model_dir models --data_dir data/partitioned --num_rounds 10 --allow_new_classes
```

Run the entire federated learning workflow:

```bash
python src/scripts/run_federated.py --dataset_dir data/extracted --partitioned_dir data/partitioned --num_rounds 10
```

## Docker Support

The system includes Docker support for federated learning:

```bash
# Build and start containers
docker-compose up -d

# View logs
docker-compose logs -f

# Stop containers
docker-compose down
```

## Privacy Features

### Differential Privacy

Differential privacy is implemented for both standard and federated training. It provides mathematical guarantees about the privacy of training data.

### Federated Learning

Federated learning enables training models across multiple clients without sharing raw data. The system supports:

- Server-client architecture
- Secure aggregation of model updates
- Handling of non-IID data with new/unseen classes
- Optional differential privacy during training

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 