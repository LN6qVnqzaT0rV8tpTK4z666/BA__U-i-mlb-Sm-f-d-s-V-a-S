#!/bin/bash

# Pfad zum Verzeichnis mit den Rohdaten
DATA_DIR="$(dirname "$0")/../data/raw"

# Prüfen, ob Verzeichnis existiert
if [ ! -d "$DATA_DIR" ]; then
  echo "Verzeichnis $DATA_DIR existiert nicht."
  exit 1
fi

echo "Lösche alle Verzeichnisse in $DATA_DIR, die mit 'dataset__' beginnen..."

# Verzeichnisse finden und löschen
find "$DATA_DIR" -maxdepth 1 -type d -name 'dataset__*' | while read -r dir; do
  echo "Lösche Ordner: $dir"
  rm -rf "$dir"
done

echo "Fertig."
