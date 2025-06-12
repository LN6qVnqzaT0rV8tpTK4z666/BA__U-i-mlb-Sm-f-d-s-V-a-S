#!/bin/bash
# BA__Projekt/scripts/clean__egg-info.sh

# Navigate to the project root
cd "$(dirname "$0")/.."

echo "Cleaning folder ending with 'BA__Programmierung.egg-info'."

# Find and delete all matching files
rm -rf BA__Projekt.egg-info/
sleep 3
rm -rf BA__Programmierung/BA__Projekt.egg-info

echo "Cleanup complete."