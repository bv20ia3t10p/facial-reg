# Containerized Federated Learning for Facial Recognition

This document provides instructions for running the federated learning system using Docker containers. The system consists of a central server and multiple client containers that train a shared facial recognition model while maintaining data privacy.

## Prerequisites

- Docker and Docker Compose
- The CASIA-WebFace dataset extracted to the `data/extracted` directory

## Setup

1. Extract the CASIA-WebFace dataset if you haven't already:

```bash
python run.py extract --rec_file path/to/train.rec --idx_file path/to/train.idx --lst_file path/to/train.lst --output_dir data/extracted
```

2. Build and start the Docker containers:

```bash
docker-compose up --build
```

This will start:
- 1 federated server (`fl_server`)
- 2 federated clients (`fl_client1` and `fl_client2`)

## Key Features

### Non-existing Classes Support

The client implementation handles non-existing classes that aren't present in the global model:

1. **Class Mapping**: Clients create a mapping between local and global class indices
2. **Model Architecture Adaptation**: Local models automatically expand to handle new classes
3. **Weight Projection**: When sending updates to the server, clients project their weights to include only known global classes

This enables clients to recognize new faces (classes) locally without affecting the global model.

### Testing with New Classes

To test with new classes on a specific client:

```bash
docker-compose exec fl_client1 python /app/src/federated_client.py --client_id 1 --server_host fl_server --server_port 8080 --allow_new_classes --add_new_classes --num_new_classes 5
```

### Monitoring the Training

You can view the status of the training by accessing the server's status endpoint:

```bash
curl http://localhost:8080/status
```

## Customization

You can modify the Docker Compose file to:
- Add more clients
- Change the number of training rounds
- Enable differential privacy
- Adjust how many new classes to test with

## Running Independently

To run components separately without Docker:

### Start the server:

```bash
python src/federated_server.py --server_port 8080 --num_rounds 10
```

### Start client 1:

```bash
python src/federated_client.py --client_id 1 --server_host localhost --server_port 8080 --data_dir data/extracted/client_1 --allow_new_classes
```

### Start client 2:

```bash
python src/federated_client.py --client_id 2 --server_host localhost --server_port 8080 --data_dir data/extracted/client_2 --allow_new_classes
```

## Results

The training results are saved to:

- Models: `models/federated_model_*.h5`
- Metrics: `models/federated_metrics.json` and `models/federated_metrics.png` 