#!/bin/bash
set -e

# Get the script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Ensure we're in the project root
cd "$PROJECT_ROOT"

# Create the database directory if it doesn't exist
DATABASE_DIR="$PROJECT_ROOT/database"
mkdir -p "$DATABASE_DIR"

# Run the Python script to create databases
echo "Creating client databases..."
python scripts/create_client_dbs.py

if [ $? -eq 0 ]; then
    echo -e "\e[32m✓ Client databases created successfully\e[0m"
else
    echo -e "\e[31m✗ Failed to create client databases\e[0m"
    exit 1
fi

# List created databases
echo -e "\nCreated databases:"
for db in "$DATABASE_DIR"/*.db; do
    echo -e "\e[36m  - $(basename "$db")\e[0m"
done

echo -e "\nDone!" 