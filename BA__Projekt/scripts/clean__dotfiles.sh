#!/bin/bash
# BA__Projekt/scripts/clean__dotfiles.sh

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
    echo "Cleaning all files beginning with . and certain extensions"
fi

# Find and delete all matching files
find . -type f -name '*.DS_Store' -print -delete
find . -type f -name '*._*' -print -delete

if [ "$QUIET" = false ]; then
    echo "Cleanup complete."
fi
