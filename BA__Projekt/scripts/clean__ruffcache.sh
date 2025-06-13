#!/bin/bash
# BA__Projekt/scripts/clean__ruffcache.sh

# Navigate to the project root (one level up from script location)
cd "$(dirname "$0")/.."

echo "Cleaning folder: ruff_cache/"

# Delete all contents in the ruff_cache directory
rm -rf .ruff_cache/

echo "Cleanup complete."
