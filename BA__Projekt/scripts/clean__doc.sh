#!/bin/bash
# BA__Projekt/scripts/clean__doc.sh

# Navigate to the project root (one level up from script location)
cd "$(dirname "$0")/.." || exit 1

# Parse quiet flag
QUIET=false
for arg in "$@"; do
    if [[ "$arg" == "--quiet" || "$arg" == "--q" ]]; then
        QUIET=true
        break
    fi
done

if [ "$QUIET" = false ]; then
    echo "Cleaning folder: assets/docs/"
fi

# Delete all contents in the docs directory (but not the directory itself)
rm -rf assets/docs/*

if [ "$QUIET" = false ]; then
    echo "Cleanup complete."
fi
