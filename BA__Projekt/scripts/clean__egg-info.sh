#!/bin/bash
# BA__Projekt/scripts/clean__egg-info.sh

# Navigate to the project root
cd "$(dirname "$0")/.." || exit 1

# Check for quiet flag
QUIET=false
for arg in "$@"; do
    if [[ "$arg" == "--quiet" || "$arg" == "--q" ]]; then
        QUIET=true
        break
    fi
done

if [ "$QUIET" = false ]; then
    echo "Cleaning folder ending with 'BA__Programmierung.egg-info'."
fi

# Find and delete all matching files
rm -rf BA__Projekt.egg-info/
sleep 3
rm -rf BA__Programmierung/BA__Projekt.egg-info

if [ "$QUIET" = false ]; then
    echo "Cleanup complete."
fi
