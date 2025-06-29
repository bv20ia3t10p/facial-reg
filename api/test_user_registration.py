#!/usr/bin/env python3
"""
Test script for user registration functionality
Demonstrates adding a new user with face images to the biometric system
"""

import os
import json
import requests
import logging
from pathlib import Path
from PIL import Image, ImageDraw
import io
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_image(size=(224, 224), color='lightblue', user_id="test_user"):
    """Create a test face image"""
    img = Image.new('RGB', size, color)
    draw = ImageDraw.Draw(img)
    
    # Draw a simple face
    # Head
    draw.ellipse([50, 50, 174, 174], fill='tan', outline='black')
    
    # Eyes
    draw.ellipse([70, 80, 90, 100], fill='white', outline='black')
    draw.ellipse([134, 80, 154, 100], fill='white', outline='black')
    draw.ellipse([75, 85, 85, 95], fill='black')
    draw.ellipse([139, 85, 149, 95], fill='black')
    
    # Nose
    draw.ellipse([105, 110, 119, 130], fill='tan', outline='black')
    
    # Mouth
    draw.arc([90, 140, 134, 160], 0, 180, fill='black', width=3)
    
    # Add user ID text
    draw.text((10, 10), f"User: {user_id}", fill='black')
    
    return img

def test_user_registration():
    """Test the user registration endpoint"""
    
    # Configuration
    api_base_url = "http://localhost:8000"  # Client API
    coordinator_url = "http://localhost:9000"  # Coordinator API
    
    # Test user data
    test_user = {
        "name": "John Test User",
        "email": "john.test@example.com",
        "department": "IT",
        "role": "employee"
    }
    
    logger.info("Starting user registration test...")
    
    try:
        # Step 1: Create test images
        logger.info("Creating test face images...")
        test_images = []
        for i in range(3):
            img = create_test_image(user_id=f"test_{i}")
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='JPEG')
            img_bytes.seek(0)
            test_images.append(('images', (f'test_face_{i}.jpg', img_bytes, 'image/jpeg')))
        
        # Step 2: Get authentication token (if needed)
        # For this test, we'll assume the HR user is authenticated
        # In a real scenario, you'd need to authenticate first
        
        logger.info("Attempting user registration...")
        
        # Step 3: Prepare form data
        form_data = {
            'name': test_user['name'],
            'email': test_user['email'],
            'department': test_user['department'],
            'role': test_user['role']
        }
        
        # Step 4: Make the registration request
        response = requests.post(
            f"{api_base_url}/api/users/register",
            data=form_data,
            files=test_images,
            headers={
                'Authorization': 'Bearer test_token'  # Replace with actual token
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"✅ User registration successful!")
            logger.info(f"User ID: {result.get('user_id')}")
            logger.info(f"Message: {result.get('message')}")
            
            # Step 5: Verify user was added to mapping
            logger.info("Checking if user was added to mapping...")
            mapping_response = requests.get(f"{coordinator_url}/api/mapping")
            
            if mapping_response.status_code == 200:
                mapping_data = mapping_response.json()
                user_id = result.get('user_id')
                
                if user_id and user_id in mapping_data.get('mapping', {}):
                    logger.info(f"✅ User {user_id} found in global mapping!")
                    logger.info(f"Mapping index: {mapping_data['mapping'][user_id]}")
                else:
                    logger.warning(f"⚠️ User {user_id} not found in global mapping")
            else:
                logger.warning(f"⚠️ Could not verify mapping: {mapping_response.status_code}")
            
            # Step 6: Check data directory
            user_id = result.get('user_id')
            if user_id:
                data_dir = Path(f"/app/data/partitioned/client1/{user_id}")
                if data_dir.exists():
                    image_count = len(list(data_dir.glob("*.jpg")))
                    logger.info(f"✅ User data directory created with {image_count} images")
                else:
                    logger.warning(f"⚠️ User data directory not found: {data_dir}")
            
            return True
            
        else:
            logger.error(f"❌ User registration failed: {response.status_code}")
            logger.error(f"Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        logger.error("❌ Could not connect to API. Make sure the services are running.")
        logger.info("To start services:")
        logger.info("1. Start coordinator: docker compose up fl-coordinator")
        logger.info("2. Start client API: docker compose up client1-api")
        return False
        
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        return False

def test_mapping_debug():
    """Test the mapping debug endpoint"""
    try:
        coordinator_url = "http://localhost:9000"
        
        logger.info("Testing mapping debug endpoint...")
        response = requests.get(f"{coordinator_url}/api/mapping/debug")
        
        if response.status_code == 200:
            debug_data = response.json()
            logger.info("✅ Mapping debug information:")
            logger.info(f"  - File exists: {debug_data.get('file_exists')}")
            logger.info(f"  - Mapping count: {debug_data.get('mapping_count')}")
            logger.info(f"  - Cache active: {debug_data.get('cache_active')}")
            logger.info(f"  - Sample entries: {debug_data.get('sample_entries')}")
            return True
        else:
            logger.error(f"❌ Mapping debug failed: {response.status_code}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Mapping debug test failed: {e}")
        return False

def check_services():
    """Check if required services are running"""
    services = [
        ("Client API", "http://localhost:8000/health"),
        ("Coordinator", "http://localhost:9000/health")
    ]
    
    all_running = True
    
    for service_name, url in services:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                logger.info(f"✅ {service_name} is running")
            else:
                logger.error(f"❌ {service_name} returned {response.status_code}")
                all_running = False
        except requests.exceptions.ConnectionError:
            logger.error(f"❌ {service_name} is not running")
            all_running = False
        except Exception as e:
            logger.error(f"❌ Error checking {service_name}: {e}")
            all_running = False
    
    return all_running

if __name__ == "__main__":
    logger.info("🧪 Testing User Registration Functionality")
    logger.info("=" * 50)
    
    # Check if services are running
    if not check_services():
        logger.error("❌ Required services are not running. Please start them first.")
        logger.info("\nTo start services:")
        logger.info("cd /path/to/facial-reg")
        logger.info("docker compose up fl-coordinator client1-api")
        exit(1)
    
    # Test mapping debug
    logger.info("\n📊 Testing mapping debug...")
    test_mapping_debug()
    
    # Test user registration
    logger.info("\n👤 Testing user registration...")
    success = test_user_registration()
    
    if success:
        logger.info("\n🎉 All tests passed! User registration is working correctly.")
    else:
        logger.error("\n💥 Some tests failed. Check the logs above for details.")
    
    logger.info("\n📋 Next steps:")
    logger.info("1. Check the frontend AddUserModal component")
    logger.info("2. Test with actual face images")
    logger.info("3. Verify federated training integration")
    logger.info("4. Test authentication with the new user") 