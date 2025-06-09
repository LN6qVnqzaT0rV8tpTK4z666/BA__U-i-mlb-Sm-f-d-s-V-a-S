#!/bin/bash
set -e

# Navigate to the project root (one level up from script location)
cd "$(dirname "$0")" || exit 1  # ins Skriptverzeichnis wechseln
echo "Current directory: $(pwd)"
echo "Found files:"
ls clean__*.sh

# Führe alle clean__*.sh Skripte im aktuellen Ordner aus
for script in clean__*.sh; do
  # Prüfen, ob die Datei nicht clean__all.sh ist
  if [ "$script" != "clean__all.sh" ]; then
    echo "Running $script"
    bash "$script"
  else
    echo "Skipping $script"
  fi
done