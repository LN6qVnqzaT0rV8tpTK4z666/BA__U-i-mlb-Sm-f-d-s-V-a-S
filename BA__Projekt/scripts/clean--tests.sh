#!/bin/bash
# BA__Projekt/scripts/clean--tests.sh

# Navigate to the project root (one level up from script location)
cd "$(dirname "$0")/.."

# Parse quiet flag
QUIET=false
for arg in "$@"; do
    if [[ "$arg" == "--quiet" || "$arg" == "--q" ]]; then
        QUIET=true
        break
    fi
done

if [ "$QUIET" = false ]; then
    echo "Cleaning folder: tests/"
fi

# Delete all contents in the tests directory (but not the directory itself)
rm -rf tests/*

if [ "$QUIET" = false ]; then
    echo "Cleanup complete."
fi
