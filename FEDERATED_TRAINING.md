# Two-Phase Facial Recognition with Federated Learning

This guide describes how to use the two-script approach for training and deploying the privacy-preserving facial recognition system using federated learning.

## Overview

The system is set up in two phases:

1. **Training Phase**: Train and save initial models for the server and clients
2. **Deployment Phase**: Deploy the trained models in Docker containers and run federated learning

## Prerequisites

- Docker and Docker Compose installed
- CASIA-WebFace dataset extracted to a directory
- Python 3.9+ with required packages installed (`pip install -r requirements.txt`)

## Phase 1: Train and Save Models

The first script, `train_and_save_models.py`, handles:

1. Partitioning the dataset between server (90%) and clients (5% each)
2. Adding unseen classes to clients if requested
3. Training initial models for the server and both clients
4. Evaluating model performance
5. Saving models and metadata for deployment

### Usage

```bash
python train_and_save_models.py --dataset_dir /path/to/extracted/dataset --output_dir data/partitioned [OPTIONS]
```

### Options

- `--dataset_dir`: Directory containing the extracted CASIA-WebFace dataset (required)
- `--output_dir`: Directory to store the partitioned dataset (default: data/partitioned)
- `--use_dp`: Enable differential privacy during training
- `--initial_epochs`: Number of epochs for initial model training (default: 5)
- `--add_unseen_classes`: Add unseen classes to clients
- `--num_unseen_classes`: Number of unseen classes to add per client (default: 10)

### Example

```bash
python train_and_save_models.py --dataset_dir data/extracted --output_dir data/partitioned --initial_epochs 3 --add_unseen_classes --num_unseen_classes 5
```

## Phase 2: Deploy Federated Learning in Docker

The second script, `deploy_federated_learning.py`, handles:

1. Setting up the environment for federated learning
2. Copying trained models to the shared volume
3. Creating configuration files for Docker
4. Starting the Docker containers

### Usage

```bash
python deploy_federated_learning.py [OPTIONS]
```

### Options

- `--model_dir`: Directory containing the saved models (default: models)
- `--data_dir`: Directory containing the partitioned dataset (default: data/partitioned)
- `--num_rounds`: Number of federated learning rounds (default: 10)
- `--use_dp`: Enable differential privacy during federated training
- `--allow_new_classes`: Allow clients to handle new classes
- `--setup_only`: Only set up environment without starting Docker
- `--clean`: Clean up Docker containers and shared volumes before starting

### Example

```bash
python deploy_federated_learning.py --num_rounds 5 --allow_new_classes --clean
```

## Complete Workflow Example

```bash
# Step 1: Extract the CASIA-WebFace dataset
python run.py extract --rec_file path/to/train.rec --idx_file path/to/train.idx --lst_file path/to/train.lst --output_dir data/extracted

# Step 2: Train and save initial models
python train_and_save_models.py --dataset_dir data/extracted --add_unseen_classes

# Step 3: Deploy federated learning in Docker
python deploy_federated_learning.py --num_rounds 10 --allow_new_classes
```

## Additional Information

### Docker Deployment Architecture

The Docker deployment consists of:
- 1 server container (`fl_server`)
- 2 client containers (`fl_client1` and `fl_client2`)
- Shared volumes for data exchange and model storage

### Model Training Process

1. In Phase 1, each model (server, client1, client2) is trained independently on its portion of the data
2. In Phase 2, the models are loaded in Docker containers and continue training via federated learning
3. The server coordinates the federated learning process, aggregating model updates from clients

### Non-Existing Classes Support

When using the `--allow_new_classes` option, clients can handle classes not present in the global model without affecting the server's model structure. This is achieved through:

1. Class mapping between local and global indices
2. Dynamic model adaptation for handling new classes
3. Weight projection when sending updates to the server

### Monitoring Training Progress

During Phase 2, you can monitor the federated learning process by:

1. Viewing Docker logs: `docker-compose logs -f`
2. Accessing the server's status endpoint: `curl http://localhost:8080/status`
3. Checking saved models in the `shared/models` directory 