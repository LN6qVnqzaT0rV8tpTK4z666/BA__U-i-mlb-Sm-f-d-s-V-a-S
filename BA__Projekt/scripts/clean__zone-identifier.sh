#!/bin/bash
# BA__Projekt/scripts/clean__zone-identifier.sh

# Navigate to the project root
cd "$(dirname "$0")/.."

echo "Cleaning all files ending with 'Zone.Identifier'..."

# Find and delete all matching files
find . -type f -name '*Zone.Identifier' -print -delete

echo "Cleanup complete."