#!/bin/bash
# BA__Projekt/scripts/clean__data-processed.sh

# Navigate to the project root (one level up from script location)
cd "$(dirname "$0")/.."

echo "Cleaning folder: assets/data/processed/"

# Delete all contents in the data/processed directory (but not the directory itself)
rm -rf assets/data/processed/*

echo "Cleanup complete."