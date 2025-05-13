"""
Configuration settings for the privacy-preserving facial recognition system.
"""

# Dataset Configuration
DATASET_PATH = "path/to/casia-webface"  # Update with actual dataset path
IMG_SIZE = (112, 112)  # Standard size for face recognition
BATCH_SIZE = 64
VALIDATION_SPLIT = 0.2

# Model Configuration
EMBEDDING_SIZE = 512  # Face embedding dimension
MODEL_SAVE_PATH = "models/face_recognition_model"

# Differential Privacy Parameters
DP_NOISE_MULTIPLIER = 1.1
DP_L2_NORM_CLIP = 1.0
DP_MICROBATCHES = 16
DP_DELTA = 1e-5  # Privacy leakage parameter

# Homomorphic Encryption Parameters
HE_POLYNOMIAL_MODULUS = 8192
HE_COEFFICIENT_MODULUS = [40, 40, 40, 40, 40]
HE_SCALE = 2**40
HE_SECURITY_LEVEL = 128

# Training Parameters
EPOCHS = 20
LEARNING_RATE = 0.001
LR_DECAY_STEPS = 1000
LR_DECAY_RATE = 0.9 