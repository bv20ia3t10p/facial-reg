#!/usr/bin/env python
"""
Facial Recognition System - Main Entry Point

This script is a simple wrapper that forwards all arguments to the 
src/scripts/run_wrapper.py script, which provides the full command-line interface.
"""

import os
import sys
from pathlib import Path

# Make sure we can import from src
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Import the main function from the wrapper script
from src.scripts.run_wrapper import main

if __name__ == "__main__":
    main() 