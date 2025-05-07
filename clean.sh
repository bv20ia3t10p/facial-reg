#!/bin/bash
# Script to clean up cache files, build artifacts, and other temporary files

echo "Cleaning up project files..."

# Clean Python cache files
echo "Removing Python cache files..."
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
find . -type f -name "*.pyo" -delete
find . -type f -name "*.pyd" -delete
find . -type f -name ".coverage" -delete
find . -type d -name ".pytest_cache" -exec rm -rf {} +
find . -type d -name "*.egg-info" -exec rm -rf {} +
find . -type d -name "*.eggs" -exec rm -rf {} +

# Clean Docker artifacts (but preserve docker-compose.yml)
echo "Cleaning Docker artifacts..."
rm -rf .docker/
rm -f .env

# Clean build artifacts
echo "Removing build artifacts..."
rm -rf build/
rm -rf dist/

# Clean temporary runtime files 
echo "Removing temporary runtime files..."
rm -rf logs/
rm -rf output/
rm -rf runs/
rm -rf .cached_extractions*
find . -type f -name "*.log" -delete
find . -type f -name "*.tfevents.*" -delete

# Optional: Remove shared Docker volume data (uncomment if needed)
# echo "Removing shared Docker volume data..."
# rm -rf shared/

# Optional: Clean model files (uncomment if needed)
# echo "Removing saved models..."
# rm -rf models/

echo "Cleanup complete!" 