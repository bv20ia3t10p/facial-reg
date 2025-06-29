# User Registration System Guide

This guide explains how to add new users to the facial recognition system through the frontend interface.

## Overview

The user registration system allows HR personnel and administrators to add new users to the biometric authentication system. When a new user is registered:

1. **Face images are processed** and stored for training
2. **Identity mapping is updated** to include the new user
3. **Database record is created** with user details and face encoding
4. **Federated training is triggered** to update the recognition model
5. **Data directories are created** in the proper partition structure

## Architecture

```
Frontend (bioemo-web)
    ↓ POST /api/users/register
Client API (port 8000)
    ↓ Process images & create user
    ↓ POST /api/mapping/add-user  
Coordinator (port 9000)
    ↓ Update global mapping
    ↓ Trigger federated training
Updated Models
```

## Frontend Usage

### Adding a User via Web Interface

1. **Navigate to Add User page** (HR or Admin role required)
2. **Fill in user details**:
   - Full Name
   - Email Address
   - Department (Engineering, HR, Marketing, etc.)
   - Role (employee, manager, admin)

3. **Upload face images**:
   - Upload 3-5 clear face images
   - Images should show different angles
   - Ensure good lighting and clear facial features

4. **Submit registration**:
   - Form will validate all fields
   - Images will be processed
   - User will be added to the system
   - Default password is set to "demo"

### API Call Example

The frontend makes this API call:

```typescript
const formData = new FormData();
formData.append('name', userData.name);
formData.append('email', userData.email);
formData.append('department', userData.department);
formData.append('role', userData.role);

// Append multiple images
userData.images.forEach((file) => {
  formData.append('images', file.originFileObj);
});

const response = await fetch('/api/users/register', {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${authToken}`
  },
  body: formData
});
```

## Backend Processing

### 1. User Registration Endpoint

**Endpoint**: `POST /api/users/register`

**Required Fields**:
- `name`: Full name of the user
- `email`: Email address (must be unique)
- `department`: User's department
- `role`: User's role in the organization
- `images`: Array of face image files

**Authorization**: HR department or admin role required

### 2. Image Processing Pipeline

```python
def process_user_images_for_database(images: List[UploadFile]) -> Optional[bytes]:
    """
    Process uploaded images and create face encoding for database storage
    """
    # 1. Validate image format (convert to RGB if needed)
    # 2. Use BiometricService to preprocess image
    # 3. Extract facial features using the trained model
    # 4. Convert features to bytes for database storage
    # 5. Return face encoding for user record
```

### 3. Data Directory Creation

New users get their own data directory:
```
/app/data/partitioned/client1/{user_id}/
├── user_image_000.jpg
├── user_image_001.jpg
└── user_image_002.jpg
```

### 4. Identity Mapping Update

The global identity mapping is updated:
```json
{
  "version": "1.0.0",
  "mapping": {
    "101": 114,
    "1009": 103,
    "new_user_id": 301  // ← New user added
  },
  "total_identities": 301,
  "hash": "updated_hash"
}
```

### 5. Database Record Creation

```sql
INSERT INTO users (
  id, name, email, department, role, 
  password_hash, face_encoding, created_at
) VALUES (
  'new_user_id', 'John Doe', 'john@example.com', 
  'IT', 'employee', 'hashed_demo_password', 
  face_encoding_bytes, current_timestamp
);
```

## Federated Learning Integration

### Training Trigger

After user registration, federated training is triggered:

1. **Background task starts** federated training script
2. **New data is included** in the training process
3. **Model is updated** to recognize the new user
4. **Updated model is distributed** to all clients

### Training Process

```bash
# Triggered automatically after user registration
python /app/train_federated.py
```

This updates the model to include the new user's face patterns.

## Testing the System

### Automated Testing

Run the test script to verify functionality:

```bash
cd /path/to/facial-reg/api
python test_user_registration.py
```

This script:
- ✅ Checks if services are running
- ✅ Creates test face images
- ✅ Submits registration request
- ✅ Verifies user in mapping
- ✅ Confirms data directory creation

### Manual Testing

1. **Start services**:
   ```bash
   docker compose up fl-coordinator client1-api
   ```

2. **Open frontend**: Navigate to Add User page

3. **Register test user** with actual face images

4. **Verify authentication** works with new user

## Error Handling

### Common Issues

1. **"BiometricService not initialized"**
   - Ensure the client API started properly
   - Check model files are present

2. **"Failed to process face images"**
   - Verify images contain clear faces
   - Check image format (JPG/PNG supported)
   - Ensure proper lighting and angle

3. **"User with email already exists"**
   - Email addresses must be unique
   - Check if user was previously registered

4. **"Only HR department or admin users can register"**
   - Ensure authenticated user has proper role
   - Check JWT token is valid

### Debugging

Enable debug logging:
```python
logging.getLogger('api.src.routes.users').setLevel(logging.DEBUG)
```

Check mapping status:
```bash
curl http://localhost:9000/api/mapping/debug
```

## Security Considerations

### Access Control
- Only HR and admin users can register new users
- JWT authentication required for all requests

### Data Protection
- Face encodings are stored securely in database
- Images are processed and stored in isolated directories
- Default password ("demo") should be changed on first login

### Privacy
- Face processing uses privacy-preserving techniques
- Differential privacy can be enabled for federated training
- Images are only accessible by authorized services

## Integration with Frontend

### AddUserModal Component

The frontend component handles:
- Form validation
- Image upload with preview
- Progress indication during registration
- Success/error message display

### Key Features
- Multiple image upload support
- Real-time validation
- Responsive design
- Integration with theme system

## Monitoring and Maintenance

### Log Monitoring

Key log messages to watch:
```
INFO: Generated unique user ID 'abc12345' for new user registration
INFO: Successfully processed image 0 for face encoding  
INFO: Added user abc12345 to identity mapping with index 301
INFO: Successfully registered new user abc12345 (John Doe)
INFO: Federated training completed successfully
```

### Periodic Tasks

1. **Monitor storage usage** for user images
2. **Backup identity mapping** regularly
3. **Verify model accuracy** after adding users
4. **Clean up failed registrations** if any

## Best Practices

### For HR Personnel

1. **Collect quality images**:
   - Multiple angles (front, slight left/right)
   - Good lighting conditions
   - Clear facial features
   - No obstructions (glasses, masks, etc.)

2. **Verify user information**:
   - Double-check email spelling
   - Confirm department and role
   - Use consistent naming conventions

3. **Test authentication**:
   - Verify new user can authenticate
   - Check recognition accuracy
   - Update user if needed

### For Developers

1. **Monitor performance**:
   - Track registration response times
   - Monitor federated training duration
   - Watch for memory usage spikes

2. **Handle errors gracefully**:
   - Provide clear error messages
   - Implement retry mechanisms
   - Log detailed debugging information

3. **Maintain data consistency**:
   - Verify mapping updates
   - Check database integrity
   - Ensure proper cleanup on failures

## Future Enhancements

### Planned Features

1. **Bulk user registration** from CSV files
2. **Advanced image quality checking** before processing
3. **User role management** and permission updates
4. **Integration with HR systems** (LDAP, Active Directory)
5. **Automated model retraining** scheduling
6. **Enhanced privacy controls** and consent management

### Performance Optimizations

1. **Async image processing** for faster responses
2. **Caching mechanisms** for frequently accessed data
3. **Distributed storage** for large-scale deployments
4. **Model compression** for faster inference 