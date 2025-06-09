#!/bin/bash

# Navigate to the project root
cd "$(dirname "$0")/.."

echo "Cleaning folder ending with '__pycache__'."

# Find and delete all matching files
rm -rf BA__Programmierung/__pycache__/

echo "Cleanup complete."