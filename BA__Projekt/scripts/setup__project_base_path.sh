#!/bin/bash

# Definiere die Zeilen, die hinzugefügt werden sollen
export_project_base_path='export PROJECT_BASE_PATH="/root/BA__U-i-mlb-Sm-f-d-s-V-a-S/BA__Projekt/"'
export_pythonpath='export PYTHONPATH=/root/BA__U-i-mlb-Sm-f-d-s-V-a-S/BA__Projekt:$PYTHONPATH'

# Funktion zum Hinzufügen, falls noch nicht vorhanden
add_line_if_missing() {
  local line="$1"
  local file="$2"
  if ! grep -Fxq "$line" "$file"; then
    echo "$line" >> "$file"
    echo "Zeile hinzugefügt: $line"
  else
    echo "Zeile bereits vorhanden: $line"
  fi
}

# Füge beide Zeilen zur ~/.bashrc hinzu (wenn nicht vorhanden)
add_line_if_missing "$export_project_base_path" ~/.bashrc
add_line_if_missing "$export_pythonpath" ~/.bashrc