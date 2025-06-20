#!/bin/bash
# BA__Projekt/scripts/create__test_folders.sh

# Projektverzeichnis bestimmen
PROJECT_BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROGRAMMING_BASE="$PROJECT_BASE/BA__Programmierung"
TESTS_BASE="$PROJECT_BASE/tests"

# Quiet-Modus
QUIET=false
for arg in "$@"; do
  if [[ "$arg" == "--quiet" || "$arg" == "--q" ]]; then
    QUIET=true
    break
  fi
done

# Liste der Ordnernamen, die ignoriert werden sollen
IGNORE_NAMES=(
  ".venv"
  ".ruff_cache"
  "__pycache__"
  "build"
  "scripts"
  "tests"
  "assets"
)

created_count=0

# Alle Verzeichnisse unterhalb von BA__Programmierung durchgehen
find "$PROGRAMMING_BASE" -type d | while read -r src_dir; do
  # Relativen Pfad ermitteln
  rel_path="${src_dir#$PROGRAMMING_BASE/}"

  # Falls leer (BA__Programmierung selbst), überspringen
  if [[ -z "$rel_path" ]]; then
    continue
  fi

  # Prüfen, ob eines der Teile im Pfad in der Ignore-Liste ist
  skip=false
  IFS='/' read -ra path_parts <<< "$rel_path"
  for part in "${path_parts[@]}"; do
    for ignore in "${IGNORE_NAMES[@]}"; do
      if [[ "$part" == "$ignore" ]]; then
        skip=true
        break 2
      fi
    done
  done

  if [[ "$skip" == true ]]; then
    continue
  fi

  # Zielverzeichnis im tests/-Ordner
  target_dir="$TESTS_BASE/$rel_path"

  # Nur erstellen, wenn noch nicht vorhanden
  if [[ ! -d "$target_dir" ]]; then
    mkdir -p "$target_dir"
    ((created_count++))
    $QUIET || echo "[+] Test-Ordner erstellt: tests/$rel_path"
  fi
done

# Zusammenfassung
if [[ "$created_count" -eq 0 ]]; then
  $QUIET || echo "✅ Alle Test-Ordner sind bereits aktuell."
fi
