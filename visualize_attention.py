import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import cv2
import argparse
from tensorflow.keras.applications.resnet50 import preprocess_input

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize attention maps for facial recognition model')
    parser.add_argument('--model-path', type=str, default='models/server_model.h5',
                        help='Path to the trained model')
    parser.add_argument('--image-path', type=str, required=True,
                        help='Path to the facial image to analyze')
    parser.add_argument('--output-dir', type=str, default='attention_maps',
                        help='Directory to save attention visualizations')
    return parser.parse_args()

def get_attention_model(model_path):
    """Load the trained model and create variants to extract attention maps"""
    # Load the full model
    model = load_model(model_path, compile=False)
    
    # Identify layers that could contain attention outputs
    attention_layers = []
    for i, layer in enumerate(model.layers):
        # Look for multiply layers which are typically used in attention mechanisms
        if 'multiply' in layer.name:
            attention_layers.append(layer.name)
    
    print(f"Found {len(attention_layers)} potential attention layers")
    
    # Create models to extract intermediate outputs
    attention_models = []
    for layer_name in attention_layers:
        # Find the layer
        for i, layer in enumerate(model.layers):
            if layer.name == layer_name:
                # Create a model that outputs this layer's output
                attention_model = tf.keras.Model(
                    inputs=model.input,
                    outputs=model.layers[i].output
                )
                attention_models.append((layer_name, attention_model))
                break
                
    return model, attention_models

def preprocess_image(img_path, target_size=(224, 224)):
    """Load and preprocess an image for model input"""
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img, img_array

def visualize_attention(img, attention_outputs, layer_names, output_dir):
    """Visualize attention maps by overlaying them on the original image"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert original image to numpy array for processing
    original_img = np.array(img)
    
    # Create a composite visualization
    n_layers = len(attention_outputs)
    plt.figure(figsize=(20, 10))
    
    # Plot original image
    plt.subplot(1, n_layers + 1, 1)
    plt.imshow(original_img)
    plt.title('Original Image')
    plt.axis('off')
    
    # For each attention layer
    for i, (layer_name, attention) in enumerate(zip(layer_names, attention_outputs)):
        # Get attention map
        if len(attention.shape) == 4:
            # Channel-wise attention (take the mean across channels)
            attention_map = np.mean(attention[0], axis=-1)
        else:
            # Spatial attention
            attention_map = attention[0, :, :, 0]
            
        # Normalize attention map
        attention_map = (attention_map - np.min(attention_map)) / (np.max(attention_map) - np.min(attention_map) + 1e-8)
        
        # Resize to match original image size
        attention_map_resized = cv2.resize(attention_map, (original_img.shape[1], original_img.shape[0]))
        
        # Create heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * attention_map_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Superimpose heatmap on original image
        superimposed = heatmap * 0.6 + original_img * 0.4
        superimposed = np.clip(superimposed, 0, 255).astype(np.uint8)
        
        # Save individual heatmap
        plt.subplot(1, n_layers + 1, i + 2)
        plt.imshow(superimposed)
        plt.title(f'Attention: {layer_name}')
        plt.axis('off')
        
        # Save the superimposed image
        cv2.imwrite(os.path.join(output_dir, f"attention_{layer_name}.jpg"), 
                   cv2.cvtColor(superimposed, cv2.COLOR_RGB2BGR))
    
    # Save the combined figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "combined_attention.jpg"))
    plt.close()
    
    print(f"Attention visualizations saved to {output_dir}")

def main():
    args = parse_args()
    
    # Load the model and create attention extraction models
    print(f"Loading model from {args.model_path}")
    model, attention_models = get_attention_model(args.model_path)
    
    if not attention_models:
        print("No attention layers found in the model.")
        return
    
    # Load and preprocess the image
    print(f"Processing image: {args.image_path}")
    img, img_array = preprocess_image(args.image_path)
    
    # Extract attention maps
    attention_outputs = []
    layer_names = []
    for layer_name, attn_model in attention_models:
        attention_output = attn_model.predict(img_array)
        attention_outputs.append(attention_output)
        layer_names.append(layer_name)
    
    # Visualize attention maps
    output_dir = os.path.join(args.output_dir, os.path.basename(args.image_path).split('.')[0])
    visualize_attention(img, attention_outputs, layer_names, output_dir)
    
    # Also run the full model for classification
    predictions = model.predict(img_array)[0]
    top_idx = np.argsort(predictions)[-5:][::-1]  # Get top 5 classes
    
    print("\nTop 5 predictions:")
    for i, idx in enumerate(top_idx):
        print(f"{i+1}. Class {idx}: {predictions[idx]:.4f}")

if __name__ == "__main__":
    main() 