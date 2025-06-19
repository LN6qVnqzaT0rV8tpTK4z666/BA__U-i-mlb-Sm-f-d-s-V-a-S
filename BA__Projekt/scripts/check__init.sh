#!/bin/bash
# BA__Projekt/scripts/check__init__.py.sh

# Detect --quiet or --q flag
QUIET_FLAG=""
for arg in "$@"; do
  if [[ "$arg" == "--quiet" || "$arg" == "--q" ]]; then
    QUIET_FLAG="--quiet"
    break
  fi
done

# Basis-Pfad für das Projektverzeichnis
PROJECT_BASE="/root/BA__U-i-mlb-Sm-f-d-s-V-a-S/BA__Projekt"

# Basis-Ordner BA__Programmierung
PROGRAMMING_BASE="$PROJECT_BASE/BA__Programmierung"

# Liste der vor dem Build existierenden Ordner unter BA__Programmierung (ohne tests)
EXISTING_FOLDERS=(
    "$PROGRAMMING_BASE"
    "$PROGRAMMING_BASE/conv"
    "$PROGRAMMING_BASE/db"
    "$PROGRAMMING_BASE/gen"
    "$PROGRAMMING_BASE/ml"
    "$PROGRAMMING_BASE/ml/datasets"
    "$PROGRAMMING_BASE/ml/losses"
    "$PROGRAMMING_BASE/ml/metrics"
    "$PROGRAMMING_BASE/ml/utils"
    "$PROGRAMMING_BASE/util"
    "$PROGRAMMING_BASE/viz"
)

# Liste von Dateimustern oder Dateinamen, die ignoriert werden sollen
IGNORE_LIST=(
    "__init__.py"   # __init__.py soll ignoriert werden
)

# Alle Verzeichnisse im gesamten Projektordner überprüfen
for folder in "${EXISTING_FOLDERS[@]}"; do
    # Prüfen, ob die __init__.py Datei bereits vorhanden ist
    if [ ! -f "$folder/__init__.py" ]; then
        # Wenn __init__.py fehlt, erstellen wir es
        if [[ -z "$QUIET_FLAG" ]]; then
            echo "Erstelle '__init__.py' in: $folder"
        fi
        touch "$folder/__init__.py"
        if [[ -z "$QUIET_FLAG" ]]; then
            echo "__init__.py wurde erstellt in: $folder"
        fi
    else
        if [[ -z "$QUIET_FLAG" ]]; then
            echo "__init__.py bereits vorhanden in: $folder"
        fi
    fi
done
