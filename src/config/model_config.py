"""
Configuration settings for models in the facial recognition system.
"""

# Model paths
MODEL_SAVE_PATH = 'models'
SERVER_MODEL_PATH = 'models/server_model.h5'
CLIENT1_MODEL_PATH = 'models/client1_model.h5'
CLIENT2_MODEL_PATH = 'models/client2_model.h5'

# Model parameters
EMBEDDING_SIZE = 512
INPUT_SHAPE = (112, 112, 3)
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# Differential privacy parameters
DP_L2_NORM_CLIP = 1.0
DP_NOISE_MULTIPLIER = 0.1  # Lower values = less privacy but better accuracy
DP_MICROBATCHES = 1

# Federated learning parameters
FEDERATED_ROUNDS = 10
FEDERATED_CLIENT_EPOCHS = 5
FEDERATED_SERVER_PORT = 8080 