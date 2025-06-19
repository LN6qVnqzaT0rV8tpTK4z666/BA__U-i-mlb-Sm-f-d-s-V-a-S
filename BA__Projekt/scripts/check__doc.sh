#!/bin/bash
# BA__Projekt/scripts/check__sphinx_docs.sh

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

# Den Pfad zur .gitignore-Datei definieren
GITIGNORE_PATH="$PROJECT_BASE/.gitignore"

# Alle Python-Dateien im gesamten Projektordner überprüfen
find "$PROJECT_BASE" -type f -name "*.py" | while read -r file; do
    # Prüfen, ob die Datei in der Ignore-Liste enthalten ist
    skip_file=false

    # Prüfen, ob die Datei in der .gitignore enthalten ist
    if grep -Fxq "$file" "$GITIGNORE_PATH"; then
        skip_file=true
    fi

    # Prüfen, ob die Datei in der Ignore-Liste enthalten ist
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

    # Die dritte Zeile der Datei lesen
    third_line=$(sed -n '3p' "$file")
    
    # Prüfen, ob die dritte Zeile die Sphinx-Dokumentation startet (""" ist vorhanden)
    third_line_check=false
    if [[ "$third_line" == *'"""'* ]]; then
        third_line_check=true
    fi

    # Prüfen, ob die Klasse eine Docstring nach der Klassendefinition hat
    class_docstring_check=false
    class_line=$(grep -n 'class ' "$file" | cut -d: -f1)  # Zeilenummer der class Definition
    if [[ -n "$class_line" ]]; then
        class_docstring=$(sed -n "$((class_line + 1))p" "$file")  # Docstring nach der Klassendefinition
        if [[ "$class_docstring" == *'"""'* ]]; then
            class_docstring_check=true
        fi
    fi

    # Überprüfen, ob eines der Kriterien erfüllt ist (Dritte Zeile oder Klassendocstring)
    if [[ "$third_line_check" == true || "$class_docstring_check" == true ]]; then
        if [[ -z "$QUIET_FLAG" ]]; then
            echo "✅ Sphinx-Dokumentation: $file"
        fi
    else
        if [[ -z "$QUIET_FLAG" ]]; then
            echo "❌ Keine Sphinx-Dokumentation: $file"
        fi
    fi
done
