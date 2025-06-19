#!/bin/bash
# BA__Projekt/scripts/check__spacing.sh

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

# Liste der vor dem Build existierenden Ordner unter BA__Programmierung und BA__Projekt/tests
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
    "$PROJECT_BASE/tests"  # Der Tests-Ordner wird jetzt auch berücksichtigt
)

# Liste von Dateimustern oder Dateinamen, die ignoriert werden sollen
IGNORE_LIST=(
    "__init__.py"   # Beispiel: __init__.py soll ignoriert werden
    "config.py"     # Beispiel: config.py wird auch ignoriert (kann nach Bedarf ergänzt werden)
    "*.tmp"          # Beispiel: Alle .tmp-Dateien werden ignoriert
)

# Alle Python-Dateien im gesamten Projektordner überprüfen
find "$PROJECT_BASE" -type f -name "*.py" | while read -r file; do
    # Prüfen, ob die Datei in der Ignore-Liste enthalten ist
    skip_file=false
    for ignore_pattern in "${IGNORE_LIST[@]}"; do
        if [[ "$file" == *$ignore_pattern* ]]; then
            skip_file=true
            break
        fi
    done

    # Wenn die Datei ignoriert werden soll, überspringen
    if [[ "$skip_file" == true ]]; then
        continue
    fi

    # Den Ordner der aktuellen Datei ermitteln
    file_dir=$(dirname "$file")
    
    # Prüfen, ob der Ordner Teil der Liste der vor dem Build existierenden Ordner unter BA__Programmierung und BA__Projekt/tests ist
    folder_exists=false
    for folder in "${EXISTING_FOLDERS[@]}"; do
        if [[ "$file_dir" == "$folder"* ]]; then
            folder_exists=true
            break
        fi
    done

    # Wenn der Ordner nicht vor dem Build existiert hat, überspringen
    if [[ "$folder_exists" == false ]]; then
        continue
    fi

    # Sicherstellen, dass die zweite Zeile leer ist
    second_line=$(sed -n '2p' "$file")
    if [[ -n "$second_line" ]]; then
        # Wenn die zweite Zeile nicht leer ist, verschiebe den Inhalt nach der ersten Zeile
        # Die erste Zeile wird beibehalten, der Rest wird nach unten verschoben
        sed -i '2s/.*/\n&/' "$file"
    fi

    # Sicherstellen, dass die letzte Zeile leer ist
    last_line=$(tail -n 1 "$file")
    if [[ -n "$last_line" ]]; then
        # Wenn die letzte Zeile nicht leer ist, füge eine leere Zeile hinzu
        echo "" >> "$file"
    fi

    if [[ -z "$QUIET_FLAG" ]]; then
        echo "Spacing überprüft und angepasst in: $file"
    fi
done
