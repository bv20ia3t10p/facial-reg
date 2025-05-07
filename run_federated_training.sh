#!/bin/bash
# Script to run the entire workflow from data extraction to federated learning

# Set default values
DATASET_DIR="data/extracted"
PARTITIONED_DIR="data/partitioned"
NUM_ROUNDS=10
USE_DP=false
ALLOW_NEW_CLASSES=true
NUM_UNSEEN_CLASSES=5
CLEAN_DOCKER=true

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --dataset_dir)
      DATASET_DIR="$2"
      shift 2
      ;;
    --partitioned_dir)
      PARTITIONED_DIR="$2"
      shift 2
      ;;
    --num_rounds)
      NUM_ROUNDS="$2"
      shift 2
      ;;
    --use_dp)
      USE_DP=true
      shift
      ;;
    --no_allow_new_classes)
      ALLOW_NEW_CLASSES=false
      shift
      ;;
    --num_unseen_classes)
      NUM_UNSEEN_CLASSES="$2"
      shift 2
      ;;
    --no_clean_docker)
      CLEAN_DOCKER=false
      shift
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --dataset_dir DIR        Path to extracted dataset (default: data/extracted)"
      echo "  --partitioned_dir DIR    Path for partitioned data (default: data/partitioned)"
      echo "  --num_rounds N           Number of federated learning rounds (default: 10)"
      echo "  --use_dp                 Enable differential privacy"
      echo "  --no_allow_new_classes   Disable handling of new classes by clients"
      echo "  --num_unseen_classes N   Number of unseen classes per client (default: 5)"
      echo "  --no_clean_docker        Don't clean Docker environment before starting"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

echo "=========== Privacy-Preserving Facial Recognition Federated Learning ==========="
echo "Dataset directory: $DATASET_DIR"
echo "Partitioned data directory: $PARTITIONED_DIR"
echo "Number of rounds: $NUM_ROUNDS"
echo "Use differential privacy: $USE_DP"
echo "Allow new classes: $ALLOW_NEW_CLASSES"
echo "Number of unseen classes: $NUM_UNSEEN_CLASSES"
echo "Clean Docker environment: $CLEAN_DOCKER"
echo "=============================================================================="

# Check if dataset exists
if [ ! -d "$DATASET_DIR" ]; then
  echo "Error: Dataset directory $DATASET_DIR does not exist."
  echo "Please extract the CASIA-WebFace dataset first using:"
  echo "python run.py extract --rec_file path/to/train.rec --idx_file path/to/train.idx --lst_file path/to/train.lst --output_dir $DATASET_DIR"
  exit 1
fi

# Step 1: Train and save initial models
echo ""
echo "==== Step 1: Training initial models ===="
DP_FLAG=""
if [ "$USE_DP" = true ]; then
  DP_FLAG="--use_dp"
fi

CLASS_FLAG=""
if [ "$ALLOW_NEW_CLASSES" = true ]; then
  CLASS_FLAG="--add_unseen_classes --num_unseen_classes $NUM_UNSEEN_CLASSES"
fi

TRAIN_CMD="python train_and_save_models.py --dataset_dir $DATASET_DIR --output_dir $PARTITIONED_DIR $DP_FLAG $CLASS_FLAG"
echo "Running: $TRAIN_CMD"
eval $TRAIN_CMD

if [ $? -ne 0 ]; then
  echo "Error: Initial model training failed."
  exit 1
fi

# Step 2: Deploy federated learning in Docker
echo ""
echo "==== Step 2: Deploying federated learning in Docker ===="
DP_FLAG=""
if [ "$USE_DP" = true ]; then
  DP_FLAG="--use_dp"
fi

CLASS_FLAG=""
if [ "$ALLOW_NEW_CLASSES" = true ]; then
  CLASS_FLAG="--allow_new_classes"
fi

CLEAN_FLAG=""
if [ "$CLEAN_DOCKER" = true ]; then
  CLEAN_FLAG="--clean"
fi

DEPLOY_CMD="python deploy_federated_learning.py --model_dir models --data_dir $PARTITIONED_DIR --num_rounds $NUM_ROUNDS $DP_FLAG $CLASS_FLAG $CLEAN_FLAG"
echo "Running: $DEPLOY_CMD"
eval $DEPLOY_CMD

if [ $? -ne 0 ]; then
  echo "Error: Deployment of federated learning failed."
  exit 1
fi

echo ""
echo "Federated learning deployment complete!"
echo "To view logs: docker-compose logs -f"
echo "To stop containers: docker-compose down" 