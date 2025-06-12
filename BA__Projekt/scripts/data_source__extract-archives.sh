#!/bin/bash
# BA__Projekt/scripts/data-source__extract-archives.sh

# Zielverzeichnis
DATA_DIR="$(dirname "$0")/../assets/data/source"

cd "$DATA_DIR" || exit 1

echo "Starte Verarbeitung in $DATA_DIR"

# === 1. ZIP-Dateien ===
for zipfile in *.zip; do
    [ -f "$zipfile" ] || continue
    dirname="${zipfile%.zip}"
    target_dir="$dirname"
    echo "Entpacke ZIP: $zipfile → $target_dir/"
    mkdir -p "$target_dir"
    unzip -q "$zipfile" -d "$target_dir"
    if [ $? -ne 0 ]; then
        echo "Fehler beim Entpacken von $zipfile"
    fi
    mv "$zipfile" "$target_dir/"
done

# === 2. TAR.GZ-Dateien ===
for tarfile in *.tar.gz; do
    [ -f "$tarfile" ] || continue
    dirname="${tarfile%.tar.gz}"
    target_dir="$dirname"
    echo "Entpacke TAR.GZ: $tarfile → $target_dir/"
    mkdir -p "$target_dir"
    tar -xzf "$tarfile" -C "$target_dir"
    if [ $? -ne 0 ]; then
        echo "Fehler beim Entpacken von $tarfile"
    fi
    mv "$tarfile" "$target_dir/"
done

# === 3. CSV- und ARFF-Dateien ===
for file in *.csv *.arff; do
    [ -f "$file" ] || continue
    filename=$(basename "$file")
    name="${filename%.*}"  # Datei ohne Extension
    target_dir="$name"
    echo "Verschiebe $filename → $target_dir/"
    mkdir -p "$target_dir"
    mv "$file" "$target_dir/"
done

echo "Fertig!"
