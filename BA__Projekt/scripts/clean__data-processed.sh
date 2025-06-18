#!/bin/bash
# BA__Projekt/scripts/clean__data-processed.sh

# Navigate to the project root
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
    echo "Cleaning folder: assets/data/processed/"
fi

# Delete all contents but not the directory itself
rm -rf assets/data/processed/*

if [ "$QUIET" = false ]; then
    echo "Cleanup complete."
fi
