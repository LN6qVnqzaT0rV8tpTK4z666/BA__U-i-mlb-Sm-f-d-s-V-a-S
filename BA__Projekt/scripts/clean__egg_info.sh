#!/bin/bash

# Navigate to the project root
cd "$(dirname "$0")/.."

echo "Cleaning folder ending with 'BA__Programmierung.egg-info'."

# Find and delete all matching files
rm -rf BA__Programmierung/BA__Programmierung.egg-info

echo "Cleanup complete."