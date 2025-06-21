#!/bin/bash
# BA__Projekt/scripts/check__spacing.sh

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
    "$PROJECT_BASE/tests"
    "$PROJECT_BASE/tests/conv"
    "$PROJECT_BASE/tests/db"
    "$PROJECT_BASE/tests/gen"
    "$PROJECT_BASE/tests/ml"
    "$PROJECT_BASE/tests/ml/datasets"
    "$PROJECT_BASE/tests/ml/losses"
    "$PROJECT_BASE/tests/ml/metrics"
    "$PROJECT_BASE/tests/ml/utils"
    "$PROJECT_BASE/tests/util"
    "$PROJECT_BASE/tests/viz"
)

# Liste von Dateimustern oder Dateinamen, die ignoriert werden sollen
IGNORE_LIST=(
    "__init__.py"
    "config.py"
    "*.tmp"
)

# Alle Dateien im gesamten Projektordner überprüfen
find "$PROJECT_BASE" -type f | while read -r file; do
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

    # Zweite Zeile überprüfen (ob sie leer ist)
    second_line=$(sed -n '2p' "$file")
    if [[ -n "$second_line" ]]; then
        # Wenn die zweite Zeile nicht leer ist, fügen wir einen Zeilenumbruch nach der ersten Zeile hinzu
        sed -i '2i\' "$file"
        echo "⚠️ Zeilenumbruch nach der ersten Zeile hinzugefügt: $file"
    fi

    # Letzte Zeile überprüfen (ob sie leer ist)
    last_line=$(tail -n 1 "$file")
    if [[ -n "$last_line" ]]; then
        # Wenn die letzte Zeile nicht leer ist, fügen wir einen Zeilenumbruch am Ende hinzu, aber nur, wenn noch keine leere Zeile vorhanden ist
        if [[ $(tail -n 2 "$file" | head -n 1) != "" ]]; then
            echo "" >> "$file"
            echo "⚠️ Letzte Zeile nicht leer, Leere Zeile hinzugefügt am Ende: $file"
        fi
    fi

    # Entfernen von Zeilen nach der letzten leeren Zeile
    last_empty_line_index=$(awk 'NR > 1 {if ($0 == "") print NR}' "$file" | tail -n 1)
    if [[ -n "$last_empty_line_index" ]]; then
        # Löschen aller Zeilen nach der letzten leeren Zeile
        sed -i "$((last_empty_line_index + 1)),\$d" "$file"
        echo "⚠️ Zeilen nach der letzten leeren Zeile entfernt: $file"
    fi
done
