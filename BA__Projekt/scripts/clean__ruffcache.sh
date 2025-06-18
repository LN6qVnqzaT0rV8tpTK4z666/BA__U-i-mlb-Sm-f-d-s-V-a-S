#!/bin/bash
# BA__Projekt/scripts/clean__ruffcache.sh

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
    echo "Cleaning folder: .ruff_cache/"
fi

# Delete all contents in the ruff_cache directory
rm -rf .ruff_cache/

if [ "$QUIET" = false ]; then
    echo "Cleanup complete."
fi
