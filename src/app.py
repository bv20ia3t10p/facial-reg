"""
Web application for demonstrating the privacy-preserving facial recognition system.
"""

import os
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from flask import Flask, request, render_template, jsonify, send_file
import tempfile
import base64
from io import BytesIO
import time
import argparse

from config import MODEL_SAVE_PATH
from model import get_embedding_model
from data_utils import preprocess_image
from homomorphic_encryption import HomomorphicEncryptionManager, SecureFaceDatabase
from secure_inference import load_face_model, extract_embedding

app = Flask(__name__)

# Global variables
model = None
secure_db = None
temp_dir = tempfile.mkdtemp()

def init_app(model_path=None, database_path=None):
    """
    Initialize the application.
    
    Args:
        model_path: Path to the model file
        database_path: Path to the secure database
    """
    global model, secure_db
    
    print("Initializing the application...")
    
    # Load model
    model = load_face_model(model_path)
    
    # Load or create secure database
    if database_path and os.path.exists(database_path):
        print(f"Loading secure database from {database_path}...")
        he_manager = HomomorphicEncryptionManager()
        secure_db = SecureFaceDatabase(he_manager)
        secure_db.load_database(database_path)
        print(f"Loaded secure database with {len(secure_db.database)} identities")
    else:
        print("No database provided. Starting with empty database.")
        he_manager = HomomorphicEncryptionManager()
        secure_db = SecureFaceDatabase(he_manager)

@app.route('/')
def index():
    """Render the home page."""
    return render_template('index.html')

@app.route('/identify', methods=['POST'])
def identify():
    """
    Process uploaded image for identification.
    
    Returns:
        JSON response with identification results
    """
    global model, secure_db
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    # Save uploaded file
    img_path = os.path.join(temp_dir, 'query_image.jpg')
    file.save(img_path)
    
    # Start timer
    start_time = time.time()
    
    # Extract query embedding
    query_embedding = extract_embedding(model, img_path)
    if query_embedding is None:
        return jsonify({'error': 'Failed to extract embedding from the image'})
    
    embedding_time = time.time() - start_time
    
    # If database is empty, return only the embedding time
    if len(secure_db.database) == 0:
        return jsonify({
            'result': 'No identities in the database',
            'embedding_time': f"{embedding_time:.4f}s",
            'matches': []
        })
    
    # Encrypt query embedding
    encrypt_start = time.time()
    encrypted_query = secure_db.encryption_manager.encrypt_embedding(query_embedding)
    encryption_time = time.time() - encrypt_start
    
    # Match against database
    match_start = time.time()
    matches = secure_db.match_encrypted_query(encrypted_query, threshold=0.7)
    matching_time = time.time() - match_start
    
    # Format results
    total_time = time.time() - start_time
    match_results = [{
        'identity': identity_id,
        'score': float(score)
    } for identity_id, score in matches]
    
    return jsonify({
        'result': 'success',
        'embedding_time': f"{embedding_time:.4f}s",
        'encryption_time': f"{encryption_time:.4f}s",
        'matching_time': f"{matching_time:.4f}s",
        'total_time': f"{total_time:.4f}s",
        'matches': match_results
    })

@app.route('/add_identity', methods=['POST'])
def add_identity():
    """
    Add a new identity to the database.
    
    Returns:
        JSON response indicating success or failure
    """
    global model, secure_db
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    identity_id = request.form.get('identity_id', f"person_{len(secure_db.database) + 1}")
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    # Save uploaded file
    img_path = os.path.join(temp_dir, f"{identity_id}.jpg")
    file.save(img_path)
    
    # Extract embedding
    embedding = extract_embedding(model, img_path)
    if embedding is None:
        return jsonify({'error': 'Failed to extract embedding from the image'})
    
    # Add to database
    secure_db.add_identity(identity_id, embedding, encrypt=False)
    
    return jsonify({
        'result': 'success',
        'identity_id': identity_id,
        'database_size': len(secure_db.database)
    })

@app.route('/database_info', methods=['GET'])
def database_info():
    """
    Get information about the database.
    
    Returns:
        JSON response with database information
    """
    global secure_db
    
    return jsonify({
        'size': len(secure_db.database),
        'identities': list(secure_db.database.keys())
    })

@app.route('/visualize', methods=['POST'])
def visualize():
    """
    Visualize identification results.
    
    Returns:
        Image file with visualization
    """
    global model, secure_db
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    # Save uploaded file
    img_path = os.path.join(temp_dir, 'query_image_vis.jpg')
    file.save(img_path)
    
    # Extract query embedding
    query_embedding = extract_embedding(model, img_path)
    if query_embedding is None:
        return jsonify({'error': 'Failed to extract embedding from the image'})
    
    # If database is empty, return only the image
    if len(secure_db.database) == 0:
        query_img = cv2.imread(img_path)
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(6, 6))
        plt.imshow(query_img)
        plt.title("Query Image (No matches - Empty database)")
        plt.axis('off')
        
        # Save visualization
        vis_path = os.path.join(temp_dir, 'visualization.png')
        plt.savefig(vis_path)
        plt.close()
        
        return send_file(vis_path, mimetype='image/png')
    
    # Encrypt query embedding
    encrypted_query = secure_db.encryption_manager.encrypt_embedding(query_embedding)
    
    # Match against database
    matches = secure_db.match_encrypted_query(encrypted_query, threshold=0.7)
    
    # Create visualization
    query_img = cv2.imread(img_path)
    query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(12, 6))
    
    # Show query image
    plt.subplot(1, 2, 1)
    plt.imshow(query_img)
    plt.title("Query Image")
    plt.axis('off')
    
    # Show matches or no match message
    plt.subplot(1, 2, 2)
    if matches:
        match_text = "Top Matches:\n"
        for i, (identity_id, score) in enumerate(matches[:3]):
            match_text += f"{i+1}. ID: {identity_id}, Score: {score:.4f}\n"
        
        plt.text(0.5, 0.5, match_text, ha='center', va='center', fontsize=12)
    else:
        plt.text(0.5, 0.5, "No matches found", ha='center', va='center', fontsize=14)
    
    plt.title("Secure Matching Results")
    plt.axis('off')
    
    # Save visualization
    vis_path = os.path.join(temp_dir, 'visualization.png')
    plt.savefig(vis_path)
    plt.close()
    
    return send_file(vis_path, mimetype='image/png')

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Web application for privacy-preserving facial recognition")
    
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to the trained model')
    parser.add_argument('--database_path', type=str, default=None,
                        help='Path to the secure database')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Host to run the server on')
    parser.add_argument('--port', type=int, default=5000,
                        help='Port to run the server on')
    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    # Initialize the application
    init_app(args.model_path, args.database_path)
    
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Create index.html template
    with open('templates/index.html', 'w') as f:
        f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>Privacy-Preserving Facial Recognition</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .spinner {
            display: none;
            margin: 20px auto;
        }
        #result-container {
            margin-top: 20px;
            display: none;
        }
        #visualization {
            max-width: 100%;
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="row">
            <div class="col-md-12 text-center">
                <h1>Privacy-Preserving Facial Recognition</h1>
                <p class="lead">Secure identification with differential privacy and homomorphic encryption</p>
            </div>
        </div>
        
        <div class="row mt-4">
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h4>Add Identity to Database</h4>
                    </div>
                    <div class="card-body">
                        <form id="add-form" enctype="multipart/form-data">
                            <div class="mb-3">
                                <label for="identity-id" class="form-label">Identity ID</label>
                                <input type="text" class="form-control" id="identity-id" name="identity_id" placeholder="Optional - auto-generated if blank">
                            </div>
                            <div class="mb-3">
                                <label for="identity-image" class="form-label">Face Image</label>
                                <input type="file" class="form-control" id="identity-image" name="file" accept="image/*" required>
                            </div>
                            <button type="submit" class="btn btn-primary">Add to Database</button>
                        </form>
                        <div class="spinner-border text-primary spinner" id="add-spinner" role="status">
                            <span class="visually-hidden">Loading...</span>
                        </div>
                        <div id="add-result" class="mt-3"></div>
                    </div>
                </div>
                
                <div class="card mt-4">
                    <div class="card-header">
                        <h4>Database Information</h4>
                    </div>
                    <div class="card-body">
                        <button id="refresh-db" class="btn btn-secondary">Refresh</button>
                        <div id="db-info" class="mt-3">
                            <p>Database size: <span id="db-size">Loading...</span></p>
                            <div id="identities-list"></div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h4>Secure Identification</h4>
                    </div>
                    <div class="card-body">
                        <form id="identify-form" enctype="multipart/form-data">
                            <div class="mb-3">
                                <label for="query-image" class="form-label">Query Image</label>
                                <input type="file" class="form-control" id="query-image" name="file" accept="image/*" required>
                            </div>
                            <button type="submit" class="btn btn-success">Identify</button>
                            <button type="button" id="visualize-btn" class="btn btn-info">Visualize</button>
                        </form>
                        <div class="spinner-border text-primary spinner" id="identify-spinner" role="status">
                            <span class="visually-hidden">Loading...</span>
                        </div>
                        <div id="result-container" class="mt-3">
                            <h5>Results:</h5>
                            <div id="result"></div>
                            <div id="timing-info" class="mt-2">
                                <h6>Performance:</h6>
                                <ul id="timing-list" class="list-group"></ul>
                            </div>
                            <div id="matches-container" class="mt-3">
                                <h6>Matches:</h6>
                                <ul id="matches-list" class="list-group"></ul>
                            </div>
                        </div>
                        <img id="visualization" style="display: none;">
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Add identity form handler
        document.getElementById('add-form').addEventListener('submit', function(e) {
            e.preventDefault();
            
            const formData = new FormData(this);
            const spinner = document.getElementById('add-spinner');
            const resultDiv = document.getElementById('add-result');
            
            spinner.style.display = 'block';
            resultDiv.innerHTML = '';
            
            fetch('/add_identity', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                spinner.style.display = 'none';
                if (data.error) {
                    resultDiv.innerHTML = `<div class="alert alert-danger">${data.error}</div>`;
                } else {
                    resultDiv.innerHTML = `<div class="alert alert-success">
                        Identity added successfully!<br>
                        ID: ${data.identity_id}<br>
                        Database size: ${data.database_size}
                    </div>`;
                    // Clear the form
                    document.getElementById('identity-id').value = '';
                    document.getElementById('identity-image').value = '';
                    // Refresh database info
                    refreshDatabaseInfo();
                }
            })
            .catch(error => {
                spinner.style.display = 'none';
                resultDiv.innerHTML = `<div class="alert alert-danger">Error: ${error.message}</div>`;
            });
        });
        
        // Identify form handler
        document.getElementById('identify-form').addEventListener('submit', function(e) {
            e.preventDefault();
            
            const formData = new FormData(this);
            const spinner = document.getElementById('identify-spinner');
            const resultContainer = document.getElementById('result-container');
            const resultDiv = document.getElementById('result');
            const timingList = document.getElementById('timing-list');
            const matchesList = document.getElementById('matches-list');
            
            spinner.style.display = 'block';
            resultContainer.style.display = 'none';
            document.getElementById('visualization').style.display = 'none';
            
            fetch('/identify', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                spinner.style.display = 'none';
                resultContainer.style.display = 'block';
                
                if (data.error) {
                    resultDiv.innerHTML = `<div class="alert alert-danger">${data.error}</div>`;
                    timingList.innerHTML = '';
                    matchesList.innerHTML = '';
                } else {
                    resultDiv.innerHTML = `<div class="alert alert-info">${data.result}</div>`;
                    
                    // Display timing information
                    timingList.innerHTML = '';
                    if (data.embedding_time) {
                        timingList.innerHTML += `<li class="list-group-item">Embedding extraction: ${data.embedding_time}</li>`;
                    }
                    if (data.encryption_time) {
                        timingList.innerHTML += `<li class="list-group-item">Encryption: ${data.encryption_time}</li>`;
                    }
                    if (data.matching_time) {
                        timingList.innerHTML += `<li class="list-group-item">Secure matching: ${data.matching_time}</li>`;
                    }
                    if (data.total_time) {
                        timingList.innerHTML += `<li class="list-group-item">Total time: ${data.total_time}</li>`;
                    }
                    
                    // Display matches
                    matchesList.innerHTML = '';
                    if (data.matches && data.matches.length > 0) {
                        data.matches.forEach(match => {
                            matchesList.innerHTML += `<li class="list-group-item">
                                ID: ${match.identity} - Score: ${match.score.toFixed(4)}
                            </li>`;
                        });
                    } else {
                        matchesList.innerHTML = `<li class="list-group-item">No matches found</li>`;
                    }
                }
            })
            .catch(error => {
                spinner.style.display = 'none';
                resultContainer.style.display = 'block';
                resultDiv.innerHTML = `<div class="alert alert-danger">Error: ${error.message}</div>`;
            });
        });
        
        // Visualize button handler
        document.getElementById('visualize-btn').addEventListener('click', function() {
            const formData = new FormData(document.getElementById('identify-form'));
            const spinner = document.getElementById('identify-spinner');
            const visualization = document.getElementById('visualization');
            
            if (!formData.get('file') || formData.get('file').size === 0) {
                alert('Please select an image first');
                return;
            }
            
            spinner.style.display = 'block';
            visualization.style.display = 'none';
            
            fetch('/visualize', {
                method: 'POST',
                body: formData
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Visualization failed');
                }
                return response.blob();
            })
            .then(blob => {
                spinner.style.display = 'none';
                const imageUrl = URL.createObjectURL(blob);
                visualization.src = imageUrl;
                visualization.style.display = 'block';
            })
            .catch(error => {
                spinner.style.display = 'none';
                alert('Error: ' + error.message);
            });
        });
        
        // Database info refresh handler
        function refreshDatabaseInfo() {
            fetch('/database_info')
            .then(response => response.json())
            .then(data => {
                document.getElementById('db-size').textContent = data.size;
                
                const identitiesDiv = document.getElementById('identities-list');
                if (data.identities && data.identities.length > 0) {
                    let html = '<div class="mt-2"><strong>Identities:</strong></div><ul class="list-group mt-1">';
                    data.identities.forEach(id => {
                        html += `<li class="list-group-item">${id}</li>`;
                    });
                    html += '</ul>';
                    identitiesDiv.innerHTML = html;
                } else {
                    identitiesDiv.innerHTML = '<p class="text-muted">No identities in database</p>';
                }
            })
            .catch(error => {
                console.error('Error fetching database info:', error);
            });
        }
        
        document.getElementById('refresh-db').addEventListener('click', refreshDatabaseInfo);
        
        // Initial load
        refreshDatabaseInfo();
    </script>
</body>
</html>
        """)
    
    # Run the app
    app.run(host=args.host, port=args.port, debug=args.debug) 