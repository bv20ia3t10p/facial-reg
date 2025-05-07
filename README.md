# Privacy-Preserving Facial Recognition System

This project implements a privacy-preserving facial recognition model using the CASIA-WebFace dataset with the following security and privacy features:

- **Differential Privacy**: Protects against inference attacks by adding noise to the training process
- **Homomorphic Encryption**: Allows computations on encrypted data without decryption
- **Federated Learning**: Enables training across multiple decentralized devices without sharing raw data

## Components

- Data preprocessing and loading utilities for MXNet RecordIO format
- Privacy-preserving model training with differential privacy
- Federated learning across multiple clients
- Secure inference using homomorphic encryption
- Evaluation and testing scripts
- Web application for demonstration

## Setup

1. Install dependencies:
```
pip install -r requirements.txt
```

2. Extract the CASIA-WebFace dataset from RecordIO format:
```
python run.py extract --rec_file path/to/train.rec --idx_file path/to/train.idx --lst_file path/to/train.lst --output_dir data/extracted
```

3. Configure dataset path in `config.py`

4. Choose your training approach:

   a. Centralized training with differential privacy:
   ```
   python run.py train --dataset_path data/extracted --use_dp
   ```

   b. Federated learning with differential privacy:
   ```
   python run.py federated --data_dir data/extracted --num_clients 5 --num_rounds 10 --use_dp
   ```

5. For secure inference:
```
python run.py inference --query_image path/to/image.jpg
```

6. To run the web application:
```
python run.py webapp
```

## Security and Privacy Features

- **Differential Privacy**: Implemented using TensorFlow Privacy to limit information leakage during training
- **Homomorphic Encryption**: Using TenSEAL and Pyfhel libraries for encrypted inference 
- **Federated Learning**: Using TensorFlow Federated to train models across multiple clients without sharing raw data

## Dataset Extraction

The CASIA-WebFace dataset often comes in MXNet RecordIO format, which includes:
- `train.rec`: Main data file containing the images
- `train.idx`: Index file for quicker access to records
- `train.lst`: Listing file with metadata

Our data extraction utility extracts these files into a directory structure suitable for training:
```
python run.py extract --rec_file path/to/train.rec --idx_file path/to/train.idx --lst_file path/to/train.lst
```

## Federated Learning

Federated learning enables training across multiple devices or clients without sharing raw data:

1. The dataset is partitioned among simulated clients
2. Each client trains on their local data
3. Model updates are aggregated on a central server
4. The process repeats for multiple rounds

To run federated learning:
```
python run.py federated --data_dir data/extracted --num_clients 5 --num_rounds 10
```

You can also combine federated learning with differential privacy:
```
python run.py federated --data_dir data/extracted --num_clients 5 --num_rounds 10 --use_dp
``` 