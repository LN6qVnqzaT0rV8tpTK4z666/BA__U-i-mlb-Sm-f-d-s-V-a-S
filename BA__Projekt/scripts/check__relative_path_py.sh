#!/bin/bash
# BA__Projekt/scripts/check__relative_path_py.sh

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

    # Die erste Zeile der Datei lesen
    first_line=$(head -n 1 "$file")
    
    # Prüfen, ob die erste Zeile einen relativen Pfad enthält
    if [[ ! "$first_line" =~ ^#.*BA__Projekt/ ]]; then
        if [[ -z "$QUIET_FLAG" ]]; then
            echo "Kein relativer Pfad in der ersten Zeile gefunden. Füge hinzu: $file"
        fi
        
        # Relativen Pfad berechnen, der mit BA__Projekt beginnt
        relative_path=$(realpath --relative-to="$PROJECT_BASE" "$file")
        
        # Neuen relativen Pfad in der ersten Zeile einfügen
        sed -i "1s|^|# BA__Projekt/$relative_path\n|" "$file"
        
        if [[ -z "$QUIET_FLAG" ]]; then
            echo "Relativer Pfad wurde hinzugefügt: $relative_path"
        fi
    else
        if [[ -z "$QUIET_FLAG" ]]; then
            echo "Relativer Pfad bereits vorhanden in: $file"
        fi
    fi
done
