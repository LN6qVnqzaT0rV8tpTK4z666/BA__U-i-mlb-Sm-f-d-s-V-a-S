#!/bin/bash
set -e

# Navigate to the project root (one level up from script location)
cd "$(dirname "$0")" || exit 1  # ins Skriptverzeichnis wechseln
echo "Current directory: $(pwd)"
echo "Found files:"
ls data_source__*.sh

# Reihenfolge der auszuführenden Skripte definieren
ordered_scripts=(
  "data_source__rename.sh"
  "data_source__extract_archives.sh"
  "data_source__move.sh"
  "data_source__flatten.sh"
)

for script in "${ordered_scripts[@]}"; do
  # Prüfen, ob die Datei existiert und nicht data_source__all.sh ist
  if [ -f "$script" ] && [ "$script" != "data_source__all.sh" ]; then
    echo "Running $script"
    bash "$script"
  else
    echo "Skipping $script (not found or excluded)"
  fi
done
