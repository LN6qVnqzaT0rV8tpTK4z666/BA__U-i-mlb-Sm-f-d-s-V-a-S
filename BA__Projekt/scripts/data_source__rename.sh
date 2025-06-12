#!/bin/bash
# BA__Projekt/scripts/data-source__rename.sh

# Absoluter oder relativer Pfad zum source-Datenordner
DATA_DIR="$(dirname "$0")/../assets/data/source"

# Sicherstellen, dass der Ordner existiert
if [ ! -d "$DATA_DIR" ]; then
  echo "Ordner $DATA_DIR existiert nicht."
  exit 1
fi

# Dateien im Ordner durchiterieren
for file in "$DATA_DIR"/*; do
  # Nur reguläre Dateien (nicht Verzeichnisse)
  if [ -f "$file" ]; then
    filename=$(basename "$file")

    # Nur fortfahren, wenn die Datei nicht bereits korrekt benannt ist
    if ! echo "$filename" | grep -q "^dataset__"; then
      # Ersetze Sonderzeichen wie "+" durch "_"
      sanitized_name=$(echo "$filename" | sed 's/+/_/g')

      # Präfix hinzufügen
      new_name="dataset__${sanitized_name}"

      mv "$file" "$DATA_DIR/$new_name"
      echo "Renamed: $filename → $new_name"
    fi
  fi
done
