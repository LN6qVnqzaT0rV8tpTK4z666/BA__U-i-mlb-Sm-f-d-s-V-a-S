#!/bin/bash
set -e
# BA__Projekt/scripts/check__all.sh

# Navigate to the project root (one level up from script location)
cd "$(dirname "$0")" || exit 1

# Detect --quiet or --q flag
QUIET_FLAG=""
for arg in "$@"; do
  if [[ "$arg" == "--quiet" || "$arg" == "--q" ]]; then
    QUIET_FLAG="--quiet"
    break
  fi
done

if [ -z "$QUIET_FLAG" ]; then
  echo "Current directory: $(pwd)"
  echo "Found files:"
  ls check__*.sh
fi

# Run all check__*.sh scripts except check__all.sh
for script in check__*.sh; do
  if [ "$script" != "check__all.sh" ]; then
    [ -z "$QUIET_FLAG" ] && echo "Running $script"
    bash "$script" "$QUIET_FLAG"
  else
    [ -z "$QUIET_FLAG" ] && echo "Skipping $script"
  fi
done
