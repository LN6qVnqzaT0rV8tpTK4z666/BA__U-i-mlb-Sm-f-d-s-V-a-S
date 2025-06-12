#!/bin/bash

# Navigate to the project root (one level up from script location)
cd "$(dirname "$0")/.."

echo "Cleaning folder: assets/docs/"

# Delete all contents in the viz directory (but not the directory itself)
rm -rf assets/docs/*

echo "Cleanup complete."
