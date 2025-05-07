"""
Secure inference using homomorphic encryption for facial recognition.
"""

import os
import numpy as np
import tensorflow as tf
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2

from config import MODEL_SAVE_PATH, EMBEDDING_SIZE
from data_utils import preprocess_image, create_dataset_index
from model import get_embedding_model
from homomorphic_encryption import HomomorphicEncryptionManager, SecureFaceDatabase

def load_face_model(model_path=None):
    """
    Load a trained facial recognition model.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Loaded model for extracting face embeddings
    """
    if model_path is None:
        model_path = os.path.join(MODEL_SAVE_PATH, 'face_recognition_model_best.h5')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load the full model
    full_model = tf.keras.models.load_model(model_path)
    
    # Get the embedding part
    embedding_model = get_embedding_model(full_model)
    
    return embedding_model

def extract_embedding(model, image_path):
    """
    Extract face embedding from an image.
    
    Args:
        model: Face embedding model
        image_path: Path to the image file
        
    Returns:
        Face embedding vector
    """
    # Preprocess image
    img = preprocess_image(image_path)
    if img is None:
        return None
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    # Extract embedding
    embedding = model.predict(img)[0]
    
    # Normalize embedding
    embedding = embedding / np.linalg.norm(embedding)
    
    return embedding

def build_secure_database(model, dataset_path, num_identities=100, images_per_identity=3):
    """
    Build a secure face database from a dataset.
    
    Args:
        model: Face embedding model
        dataset_path: Path to the face dataset
        num_identities: Number of identities to include
        images_per_identity: Number of images per identity
        
    Returns:
        SecureFaceDatabase instance
    """
    print(f"Building secure database with {num_identities} identities...")
    
    # Create dataset index
    dataset_index = create_dataset_index(dataset_path)
    
    # Group by identity
    identities = dataset_index.groupby('person_id')
    
    # Create encryption manager
    he_manager = HomomorphicEncryptionManager()
    
    # Create secure database
    secure_db = SecureFaceDatabase(he_manager)
    
    # Counter for processed identities
    processed = 0
    
    # Process identities
    for person_id, group in tqdm(identities):
        if processed >= num_identities:
            break
        
        # Get images for this identity
        image_paths = group['image_path'].values[:images_per_identity]
        
        # Skip if not enough images
        if len(image_paths) < images_per_identity:
            continue
        
        # Extract embeddings for each image
        embeddings = []
        for img_path in image_paths:
            emb = extract_embedding(model, img_path)
            if emb is not None:
                embeddings.append(emb)
        
        # Skip if not enough valid embeddings
        if len(embeddings) < images_per_identity:
            continue
        
        # Average the embeddings to get a single reference
        avg_embedding = np.mean(embeddings, axis=0)
        avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
        
        # Add to database
        secure_db.add_identity(person_id, avg_embedding, encrypt=False)
        
        processed += 1
    
    print(f"Secure database built with {len(secure_db.database)} identities")
    return secure_db

def secure_identification(query_image_path, model, secure_db, threshold=0.7, visualize=False):
    """
    Perform secure identification using homomorphic encryption.
    
    Args:
        query_image_path: Path to the query image
        model: Face embedding model
        secure_db: Secure face database
        threshold: Matching threshold
        visualize: Whether to visualize the results
        
    Returns:
        List of matching identities with scores
    """
    # Extract query embedding
    query_embedding = extract_embedding(model, query_image_path)
    if query_embedding is None:
        print(f"Failed to extract embedding from query image: {query_image_path}")
        return []
    
    # Encrypt query embedding
    encrypted_query = secure_db.encryption_manager.encrypt_embedding(query_embedding)
    
    # Match against database
    matches = secure_db.match_encrypted_query(encrypted_query, threshold)
    
    # Print results
    print(f"Found {len(matches)} matches:")
    for identity_id, score in matches:
        print(f"  Identity: {identity_id}, Score: {score:.4f}")
    
    # Visualize if requested
    if visualize and matches:
        plt.figure(figsize=(10, 5))
        
        # Show query image
        plt.subplot(1, 2, 1)
        query_img = cv2.imread(query_image_path)
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
        plt.imshow(query_img)
        plt.title("Query Image")
        plt.axis('off')
        
        # Show top match identity
        top_match_id, top_score = matches[0]
        plt.subplot(1, 2, 2)
        plt.text(0.5, 0.5, f"Matched ID: {top_match_id}\nScore: {top_score:.4f}",
                 ha='center', va='center', fontsize=12)
        plt.title("Match Result")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig("match_result.png")
        plt.show()
    
    return matches

def main(args):
    """Main function for secure inference."""
    # Load model
    print("Loading face embedding model...")
    model = load_face_model(args.model_path)
    
    # Create or load secure database
    if args.database_path and os.path.exists(args.database_path):
        print(f"Loading secure database from {args.database_path}...")
        he_manager = HomomorphicEncryptionManager()
        secure_db = SecureFaceDatabase(he_manager)
        secure_db.load_database(args.database_path)
    else:
        print("Building secure database...")
        secure_db = build_secure_database(
            model, 
            args.dataset_path, 
            num_identities=args.num_identities, 
            images_per_identity=args.images_per_identity
        )
        
        # Save database if path provided
        if args.database_path:
            os.makedirs(os.path.dirname(args.database_path), exist_ok=True)
            secure_db.save_database(args.database_path)
            print(f"Secure database saved to {args.database_path}")
    
    # Perform secure identification if query image provided
    if args.query_image:
        secure_identification(
            args.query_image, 
            model, 
            secure_db, 
            threshold=args.threshold,
            visualize=args.visualize
        )

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Secure facial recognition using homomorphic encryption")
    
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to the trained model')
    parser.add_argument('--dataset_path', type=str, default=None,
                        help='Path to the face dataset')
    parser.add_argument('--database_path', type=str, default=None,
                        help='Path to save/load the secure database')
    parser.add_argument('--query_image', type=str, default=None,
                        help='Path to the query image for identification')
    parser.add_argument('--num_identities', type=int, default=100,
                        help='Number of identities to include in the database')
    parser.add_argument('--images_per_identity', type=int, default=3,
                        help='Number of images per identity to use')
    parser.add_argument('--threshold', type=float, default=0.7,
                        help='Matching threshold')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize the results')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(args) 