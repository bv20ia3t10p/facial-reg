"""
Homomorphic encryption implementation for secure facial recognition inference.

This module provides utilities for encrypting face embeddings and performing 
secure similarity computations using homomorphic encryption.
"""

import numpy as np
import tenseal as ts
from pyfhel import Pyfhel, PyCtxt
import pickle
import os
import time

from config import (
    HE_POLYNOMIAL_MODULUS, 
    HE_COEFFICIENT_MODULUS,
    HE_SCALE,
    HE_SECURITY_LEVEL,
    EMBEDDING_SIZE
)

class HomomorphicEncryptionManager:
    """Manages homomorphic encryption operations for facial recognition."""
    
    def __init__(self, use_tenseal=True):
        """
        Initialize the homomorphic encryption manager.
        
        Args:
            use_tenseal: Whether to use TenSEAL (True) or Pyfhel (False)
        """
        self.use_tenseal = use_tenseal
        
        if use_tenseal:
            # Initialize TenSEAL context
            self.context = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=HE_POLYNOMIAL_MODULUS,
                coeff_mod_bit_sizes=HE_COEFFICIENT_MODULUS,
                encryption_type=ts.ENCRYPTION_TYPE.SYMMETRIC
            )
            self.context.global_scale = 2**HE_SCALE
            self.context.generate_galois_keys()
        else:
            # Initialize Pyfhel context
            self.context = Pyfhel()
            self.context.contextGen(
                scheme='CKKS',
                n=HE_POLYNOMIAL_MODULUS,
                scale=2**HE_SCALE,
                qi_sizes=HE_COEFFICIENT_MODULUS
            )
            self.context.keyGen()
            self.context.relinKeyGen()
            self.context.rotateKeyGen()
    
    def encrypt_embedding(self, embedding):
        """
        Encrypt a face embedding vector using homomorphic encryption.
        
        Args:
            embedding: Face embedding vector (numpy array)
            
        Returns:
            Encrypted embedding
        """
        if self.use_tenseal:
            return ts.ckks_vector(self.context, embedding)
        else:
            return self.context.encrypt(embedding)
    
    def decrypt_embedding(self, encrypted_embedding):
        """
        Decrypt an encrypted face embedding.
        
        Args:
            encrypted_embedding: Encrypted face embedding
            
        Returns:
            Decrypted embedding as numpy array
        """
        if self.use_tenseal:
            return encrypted_embedding.decrypt()
        else:
            return self.context.decrypt(encrypted_embedding)
    
    def compute_encrypted_similarity(self, encrypted_query, reference_embedding):
        """
        Compute similarity between encrypted query and plain reference embedding.
        
        For secure matching, we implement the cosine similarity calculation
        in the encrypted domain:
        cos(a, b) = dot(a, b) / (||a|| * ||b||)
        
        Args:
            encrypted_query: Encrypted query embedding
            reference_embedding: Reference embedding (not encrypted)
            
        Returns:
            Encrypted similarity score
        """
        if self.use_tenseal:
            # Normalize reference embedding
            ref_norm = np.linalg.norm(reference_embedding)
            normalized_ref = reference_embedding / ref_norm
            
            # Compute dot product in encrypted domain
            # Note: TenSEAL allows multiplication between encrypted and plain vectors
            encrypted_dot = encrypted_query.dot(normalized_ref)
            
            return encrypted_dot
        else:
            # Normalize reference embedding
            ref_norm = np.linalg.norm(reference_embedding)
            normalized_ref = reference_embedding / ref_norm
            
            # With Pyfhel, we need a different approach
            # Convert reference to an encrypted vector
            encrypted_ref = self.context.encrypt(normalized_ref)
            
            # Compute inner product
            result = self.context.scalar_prod(encrypted_query, encrypted_ref)
            
            return result
    
    def save_context(self, filepath):
        """
        Save the HE context to a file.
        
        Args:
            filepath: Path to save the context
        """
        if self.use_tenseal:
            with open(filepath, 'wb') as f:
                pickle.dump(self.context, f)
        else:
            self.context.save_context(filepath)
    
    def load_context(self, filepath):
        """
        Load the HE context from a file.
        
        Args:
            filepath: Path to load the context from
        """
        if self.use_tenseal:
            with open(filepath, 'rb') as f:
                self.context = pickle.load(f)
        else:
            self.context.load_context(filepath)

class SecureFaceDatabase:
    """
    Secure face database using homomorphic encryption.
    Stores reference embeddings and provides secure matching capabilities.
    """
    
    def __init__(self, encryption_manager=None):
        """
        Initialize a secure face database.
        
        Args:
            encryption_manager: HomomorphicEncryptionManager instance
        """
        if encryption_manager is None:
            self.encryption_manager = HomomorphicEncryptionManager()
        else:
            self.encryption_manager = encryption_manager
        
        self.database = {}  # identity -> embedding
    
    def add_identity(self, identity_id, embedding, encrypt=False):
        """
        Add an identity to the secure database.
        
        Args:
            identity_id: Unique identifier for the identity
            embedding: Face embedding vector
            encrypt: Whether to encrypt the stored embedding
        """
        if encrypt:
            encrypted_embedding = self.encryption_manager.encrypt_embedding(embedding)
            self.database[identity_id] = encrypted_embedding
        else:
            self.database[identity_id] = embedding
    
    def match_encrypted_query(self, encrypted_query, threshold=0.7):
        """
        Match an encrypted query against the database.
        
        Args:
            encrypted_query: Encrypted query embedding
            threshold: Similarity threshold for matching
            
        Returns:
            List of (identity_id, similarity) pairs for matches above threshold
        """
        results = []
        
        for identity_id, reference_embedding in self.database.items():
            # For simplicity, we assume reference_embedding is not encrypted
            # In a real system, you might have different encryption schemes
            encrypted_similarity = self.encryption_manager.compute_encrypted_similarity(
                encrypted_query, reference_embedding
            )
            
            # Decrypt the similarity score
            similarity = self.encryption_manager.decrypt_embedding(encrypted_similarity)
            
            if isinstance(similarity, np.ndarray):
                similarity = similarity[0]  # Sometimes returned as array
            
            if similarity > threshold:
                results.append((identity_id, similarity))
        
        # Sort by similarity (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
    
    def save_database(self, filepath):
        """
        Save the secure database to a file.
        
        Args:
            filepath: Path to save the database
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self.database, f)
    
    def load_database(self, filepath):
        """
        Load the secure database from a file.
        
        Args:
            filepath: Path to load the database from
        """
        with open(filepath, 'rb') as f:
            self.database = pickle.load(f)

def benchmark_encryption(embedding_size=EMBEDDING_SIZE, trials=10):
    """
    Benchmark the performance of homomorphic encryption.
    
    Args:
        embedding_size: Size of the face embedding vector
        trials: Number of trials to run
        
    Returns:
        Dictionary with benchmark results
    """
    results = {
        'tenseal': {
            'encryption_time': 0.0,
            'decryption_time': 0.0,
            'similarity_time': 0.0
        },
        'pyfhel': {
            'encryption_time': 0.0,
            'decryption_time': 0.0,
            'similarity_time': 0.0
        }
    }
    
    # Generate random embeddings
    query_embedding = np.random.random(embedding_size)
    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    
    reference_embedding = np.random.random(embedding_size)
    reference_embedding = reference_embedding / np.linalg.norm(reference_embedding)
    
    # Benchmark TenSEAL
    print("Benchmarking TenSEAL...")
    tenseal_manager = HomomorphicEncryptionManager(use_tenseal=True)
    
    # Encryption
    start_time = time.time()
    for _ in range(trials):
        encrypted_query = tenseal_manager.encrypt_embedding(query_embedding)
    encryption_time = (time.time() - start_time) / trials
    results['tenseal']['encryption_time'] = encryption_time
    
    # Similarity computation
    start_time = time.time()
    for _ in range(trials):
        encrypted_similarity = tenseal_manager.compute_encrypted_similarity(
            encrypted_query, reference_embedding
        )
    similarity_time = (time.time() - start_time) / trials
    results['tenseal']['similarity_time'] = similarity_time
    
    # Decryption
    start_time = time.time()
    for _ in range(trials):
        decrypted_similarity = tenseal_manager.decrypt_embedding(encrypted_similarity)
    decryption_time = (time.time() - start_time) / trials
    results['tenseal']['decryption_time'] = decryption_time
    
    # Benchmark Pyfhel
    print("Benchmarking Pyfhel...")
    pyfhel_manager = HomomorphicEncryptionManager(use_tenseal=False)
    
    # Encryption
    start_time = time.time()
    for _ in range(trials):
        encrypted_query = pyfhel_manager.encrypt_embedding(query_embedding)
    encryption_time = (time.time() - start_time) / trials
    results['pyfhel']['encryption_time'] = encryption_time
    
    # Similarity computation
    start_time = time.time()
    for _ in range(trials):
        encrypted_similarity = pyfhel_manager.compute_encrypted_similarity(
            encrypted_query, reference_embedding
        )
    similarity_time = (time.time() - start_time) / trials
    results['pyfhel']['similarity_time'] = similarity_time
    
    # Decryption
    start_time = time.time()
    for _ in range(trials):
        decrypted_similarity = pyfhel_manager.decrypt_embedding(encrypted_similarity)
    decryption_time = (time.time() - start_time) / trials
    results['pyfhel']['decryption_time'] = decryption_time
    
    # Print results
    print("\nBenchmark Results:")
    print("=================")
    print(f"Embedding size: {embedding_size}, Trials: {trials}")
    
    print("\nTenSEAL:")
    print(f"  Encryption time:  {results['tenseal']['encryption_time']*1000:.2f} ms")
    print(f"  Similarity time:  {results['tenseal']['similarity_time']*1000:.2f} ms")
    print(f"  Decryption time:  {results['tenseal']['decryption_time']*1000:.2f} ms")
    
    print("\nPyfhel:")
    print(f"  Encryption time:  {results['pyfhel']['encryption_time']*1000:.2f} ms")
    print(f"  Similarity time:  {results['pyfhel']['similarity_time']*1000:.2f} ms")
    print(f"  Decryption time:  {results['pyfhel']['decryption_time']*1000:.2f} ms")
    
    return results

if __name__ == "__main__":
    # Run benchmarks if script is executed directly
    benchmark_encryption() 