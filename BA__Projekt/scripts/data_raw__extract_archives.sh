#!/bin/bash

# Zielverzeichnis
DATA_DIR="../assets/data/raw"

cd "$DATA_DIR" || exit 1

echo "Starte Entpackung in $DATA_DIR"

# === 1. ZIP-Dateien ===
for zipfile in *.zip; do
    [ -f "$zipfile" ] || continue
    dirname="${zipfile%.zip}"
    echo "Entpacke ZIP: $zipfile → $dirname/"
    mkdir -p "$dirname"
    unzip -q "$zipfile" -d "$dirname"
done

# === 2. TAR.GZ-Dateien ===
for tarfile in *.tar.gz; do
    [ -f "$tarfile" ] || continue
    dirname="${tarfile%.tar.gz}"
    echo "Entpacke TAR.GZ: $tarfile → $dirname/"
    mkdir -p "$dirname"
    tar -xzf "$tarfile" -C "$dirname"
done

# === 3. Alle anderen Dateien (z. B. .arff) ===
for file in *; do
    # Nur reguläre Dateien (keine Verzeichnisse) und keine .zip/.tar.gz
    if [[ -f "$file" && "$file" != *.zip && "$file" != *.tar.gz ]]; then
        name="${file%.*}"  # Basisname ohne Erweiterung
        echo "Verschiebe Datei: $file → $name/"
        mkdir -p "$name"
        mv "$file" "$name/"
    fi
done

echo "Fertig!"
