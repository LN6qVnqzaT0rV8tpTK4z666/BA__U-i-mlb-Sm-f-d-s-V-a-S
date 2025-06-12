#!/bin/bash
# BA__Projekt/scripts/clean____pycache__.sh

# Navigate to the project root (directory of the script's parent)
cd "$(dirname "$0")/.." || exit 1
echo "Cleaning folder ending with '__pycache__'."

# Find and remove all __pycache__ folders in the project
find . -type d -name "__pycache__" -exec rm -rf {} +

echo "Cleanup complete."
