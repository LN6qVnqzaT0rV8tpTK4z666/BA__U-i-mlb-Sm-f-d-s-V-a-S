#!/bin/bash
set -e
# BA__Projekt/scripts/data-source__extract_all.sh

# Detect --quiet / --q flag
QUIET=false
for arg in "$@"; do
    if [[ "$arg" == "--quiet" || "$arg" == "--q" ]]; then
        QUIET=true
        break
    fi
done

# Navigate to the project root (one level up from script location)
cd "$(dirname "$0")" || exit 1

if [ "$QUIET" = false ]; then
    echo "Current directory: $(pwd)"
    echo "Found files:"
    ls data_source__*.sh
fi

# Define script execution order
ordered_scripts=(
  "data-source__rename.sh"
  "data-source__extract-archives.sh"
  "data-source__move.sh"
  "data-source__flatten.sh"
  "data-source__convert.sh"
  "data-source__rename-csv.sh"
)

for script in "${ordered_scripts[@]}"; do
  if [ -f "$script" ] && [ "$script" != "data_source__all.sh" ]; then
    [ "$QUIET" = false ] && echo "Running $script"
    bash "$script" "$@"  # Forward flags to inner scripts
  else
    [ "$QUIET" = false ] && echo "Skipping $script (not found or excluded)"
  fi
done
