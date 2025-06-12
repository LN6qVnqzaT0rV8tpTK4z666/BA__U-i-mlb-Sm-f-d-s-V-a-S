#!/bin/bash
# BA__Projekt/scripts/clean__dbs.sh

# Navigate to the project root (one level up from script location)
cd "$(dirname "$0")/.."

echo "Cleaning folder: assets/dbs/"

# Delete all contents in the dbs directory (but not the directory itself)
rm -rf assets/dbs/*

echo "Cleanup complete."