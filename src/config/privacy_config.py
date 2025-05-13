"""
Privacy-related configuration settings for the privacy-preserving facial recognition system.
"""

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