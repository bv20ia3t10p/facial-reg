# Python 3.11 Setup for Facial Recognition Project

This project is now configured to use Python 3.11 for maximum compatibility with all required libraries.

## Using the Python 3.11 Environment

1. **Activate the environment**:
   ```
   .\venv311\Scripts\activate
   ```
   
   Or simply run:
   ```
   .\use_python311.bat
   ```

2. **Verify Python version**:
   ```
   python --version
   ```
   Should show: `Python 3.11.x`

3. **Run project scripts**:
   Once the environment is activated, you can run all project scripts normally:
   ```
   python run.py [command]
   ```

## Why Python 3.11?

As detailed in `PYTHON_COMPATIBILITY.md`, Python 3.11 is required for this project because:

- TensorFlow Federated (critical for our federated learning implementation) is not compatible with Python 3.12+
- All other libraries (TensorFlow, TensorFlow Privacy, TenSEAL, etc.) work well with Python 3.11
- Python 3.11 offers good performance and stability

## Environment Information

- Python version: 3.11.9
- Virtual environment: `venv311`
- Dependencies: See `requirements-py311.txt`

## Troubleshooting

If you encounter issues:

1. Make sure you've activated the Python 3.11 environment first
2. Ensure all packages are properly installed:
   ```
   pip install -r requirements-py311.txt
   ```
3. Try recreating the environment if needed:
   ```
   py -3.11 -m venv venv311_new
   .\venv311_new\Scripts\activate
   pip install -r requirements-py311.txt
   ``` 