# Facial Recognition System with SE-ECA ResNet50

A complete federated learning system for facial recognition using the CASIA-WebFace dataset, featuring advanced attention-based deep learning models.

## Features

- **Advanced Model Architecture**: ResNet50 with Squeeze-and-Excitation (SE) and Efficient Channel Attention (ECA) blocks
- **Data Processing Pipeline**: Full extraction from MXNet RecordIO format, class-based partitioning, and training
- **Federated Learning Support**: Equal 1:1:1 split with 200 classes per partition for server, client1, and client2
- **Dataset Validation**: Automatic validation of dataset integrity and visual inspection tools
- **Attention Visualization**: Tools to visualize model attention for explainability

## Quick Start

### Full Pipeline

Run the complete pipeline from extraction to training:

```bash
./extract_and_train.sh
```

### Skip Specific Steps

Use flags to skip specific steps:

```bash
# Skip extraction if images are already extracted
./extract_and_train.sh --skip-extraction

# Skip partitioning if already partitioned
./extract_and_train.sh --skip-partitioning

# Skip training if only preparing data
./extract_and_train.sh --skip-training
```

### Train Individual Models

Train specific models separately:

```bash
# Train server model
python train_server_model.py --img-size 224 --batch-size 32 --debug

# Train client models
python train_client_model.py --client-id client1 --img-size 224 --batch-size 32 --debug
python train_client_model.py --client-id client2 --img-size 224 --batch-size 32 --debug
```

### Visualize Attention Maps

After training, visualize how the model attends to facial features:

```bash
python visualize_attention.py --image-path path/to/face_image.jpg
```

## Model Architecture

Our system uses a ResNet50 backbone enhanced with two attention mechanisms:

1. **Squeeze-and-Excitation (SE) Blocks**: Model channel relationships to emphasize important feature channels
2. **Efficient Channel Attention (ECA)**: Lightweight attention mechanism that captures local cross-channel interactions

These attention mechanisms significantly improve the model's ability to focus on discriminative facial features, leading to higher accuracy.

## Dataset Structure

The pipeline processes the CASIA-WebFace dataset with the following structure:

- **Extraction**: `/data/extracted/casia-images/`
- **Partitioning**: 
  - Server: `/data/partitioned/server/` (200 classes)
  - Client1: `/data/partitioned/client1/` (200 classes)
  - Client2: `/data/partitioned/client2/` (200 classes)

## Debugging

Use the `--debug` flag to validate dataset integrity, remove empty classes, and generate sample visualizations.

## Performance

The enhanced ResNet50 SE-ECA model can achieve >90% accuracy on facial recognition tasks, significantly outperforming baseline models like MobileNetV2.

## Requirements

- Python 3.8+
- TensorFlow 2.x
- OpenCV
- NumPy
- Matplotlib
- PIL 