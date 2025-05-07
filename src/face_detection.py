"""
Face detection utilities for identifying and cropping faces from images.
"""

import cv2
import numpy as np
import os
from pathlib import Path
import tensorflow as tf
from PIL import Image
import io

# Load OpenCV's pre-trained face detector
FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
FACE_CASCADE = cv2.CascadeClassifier(FACE_CASCADE_PATH)

# Constants for face preprocessing
TARGET_SIZE = (112, 112)  # Standard size for face recognition models

def detect_faces(image, min_size=(30, 30)):
    """
    Detect faces in an image using OpenCV's Haar Cascade classifier.
    
    Args:
        image: Input image (numpy array in BGR format for OpenCV)
        min_size: Minimum size for detected faces
        
    Returns:
        List of (x, y, w, h) tuples for detected faces
    """
    # Convert to grayscale for face detection
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Detect faces
    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=min_size
    )
    
    return faces

def crop_face(image, face_rect=None, add_margin=0.1):
    """
    Crop a face from an image, either using a provided rectangle or by detecting the largest face.
    
    Args:
        image: Input image (numpy array)
        face_rect: Optional (x, y, w, h) tuple for the face. If None, detect the largest face.
        add_margin: Fraction to add as margin around the face
        
    Returns:
        Cropped and preprocessed face image, or None if no face detected
    """
    # If no face rectangle provided, detect the largest face
    if face_rect is None:
        faces = detect_faces(image)
        if len(faces) == 0:
            return None
        
        # Select the largest face by area
        face_rect = max(faces, key=lambda rect: rect[2] * rect[3])
    
    # Unpack the face rectangle
    x, y, w, h = face_rect
    
    # Add margin
    margin_x = int(w * add_margin)
    margin_y = int(h * add_margin)
    
    # Calculate new boundaries with margins
    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(image.shape[1], x + w + margin_x)
    y2 = min(image.shape[0], y + h + margin_y)
    
    # Crop the face
    face_img = image[y1:y2, x1:x2]
    
    # Resize to standard size
    face_img = cv2.resize(face_img, TARGET_SIZE)
    
    return face_img

def preprocess_face(face_img):
    """
    Preprocess a face image for input to the recognition model.
    
    Args:
        face_img: Cropped face image
        
    Returns:
        Preprocessed face image ready for model input
    """
    # Convert to RGB if needed (OpenCV uses BGR)
    if len(face_img.shape) == 3 and face_img.shape[2] == 3:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    
    # Normalize pixel values to [0, 1]
    face_img = face_img.astype(np.float32) / 255.0
    
    # Add batch dimension if needed
    if len(face_img.shape) == 3:
        face_img = np.expand_dims(face_img, axis=0)
    
    return face_img

def process_image_bytes(image_bytes):
    """
    Process an image from bytes, detect the largest face, and preprocess it.
    
    Args:
        image_bytes: Input image as bytes
        
    Returns:
        Preprocessed face image, original image, and face rectangle
    """
    # Convert bytes to numpy array
    np_array = np.frombuffer(image_bytes, np.uint8)
    
    # Decode image
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image")
    
    # Detect faces
    faces = detect_faces(img)
    if len(faces) == 0:
        raise ValueError("No face detected in the image")
    
    # Select the largest face
    face_rect = max(faces, key=lambda rect: rect[2] * rect[3])
    
    # Crop and preprocess the face
    face_img = crop_face(img, face_rect)
    if face_img is None:
        raise ValueError("Failed to crop face")
    
    # Preprocess for model input
    processed_face = preprocess_face(face_img)
    
    return processed_face, img, face_rect

def process_image_file(image_path):
    """
    Process an image file, detect the largest face, and preprocess it.
    
    Args:
        image_path: Path to the input image file
        
    Returns:
        Preprocessed face image, original image, and face rectangle
    """
    # Read image file
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    
    return process_image_bytes(image_bytes)

def visualize_face_detection(image, face_rect, identity=None):
    """
    Visualize face detection results on an image.
    
    Args:
        image: Original image
        face_rect: (x, y, w, h) tuple for the detected face
        identity: Optional identity label to display
        
    Returns:
        Image with face detection visualization
    """
    # Make a copy of the image
    viz_img = image.copy()
    
    # Unpack face rectangle
    x, y, w, h = face_rect
    
    # Draw rectangle around face
    cv2.rectangle(viz_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Add identity text if provided
    if identity:
        cv2.putText(viz_img, identity, (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return viz_img

def face_image_to_bytes(face_img, format='JPEG'):
    """
    Convert a face image to bytes.
    
    Args:
        face_img: Face image as numpy array
        format: Image format (JPEG, PNG, etc.)
        
    Returns:
        Image bytes
    """
    # Convert to uint8 if needed
    if face_img.dtype == np.float32 or face_img.dtype == np.float64:
        face_img = (face_img * 255).astype(np.uint8)
    
    # Convert to RGB if needed
    if len(face_img.shape) == 3 and face_img.shape[2] == 3:
        pil_img = Image.fromarray(face_img)
    else:
        pil_img = Image.fromarray(face_img, 'L')  # Grayscale
    
    # Save to bytes
    buffer = io.BytesIO()
    pil_img.save(buffer, format=format)
    
    return buffer.getvalue()

if __name__ == "__main__":
    # Test face detection
    import argparse
    
    parser = argparse.ArgumentParser(description="Test face detection")
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--output', type=str, default='detected_face.jpg', help='Path to output image')
    
    args = parser.parse_args()
    
    try:
        # Process image
        processed_face, original_img, face_rect = process_image_file(args.image)
        
        # Visualize detection
        viz_img = visualize_face_detection(original_img, face_rect)
        
        # Save results
        cv2.imwrite(args.output, viz_img)
        
        print(f"Detected face saved to {args.output}")
        
        # Save cropped face
        face_output = os.path.splitext(args.output)[0] + "_face.jpg"
        cv2.imwrite(face_output, (processed_face[0] * 255).astype(np.uint8))
        
        print(f"Cropped face saved to {face_output}")
        
    except Exception as e:
        print(f"Error: {str(e)}") 