#!/bin/bash

# Navigate to the project root (one level up from script location)
cd "$(dirname "$0")/.."

echo "Cleaning folder: data/processed/"

# Delete all contents in the data/processed directory (but not the directory itself)
rm -rf data/processed/*

echo "Cleanup complete."