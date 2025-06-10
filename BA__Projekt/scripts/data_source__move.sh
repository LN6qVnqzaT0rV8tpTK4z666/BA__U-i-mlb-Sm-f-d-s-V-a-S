#!/bin/bash

# Paths
SOURCE_DIR="../assets/data/source"
TARGET_DIR="../assets/data/raw"

echo "Verschiebe Verzeichnisse von $SOURCE_DIR nach $TARGET_DIR"

# Ensure target directory exists
mkdir -p "$TARGET_DIR"

# Iterate over all subdirectories in SOURCE_DIR
for dir in "$SOURCE_DIR"/*/; do
    # Remove trailing slash and get the directory name
    dir=${dir%/}
    folder_name=$(basename "$dir")

    src="$SOURCE_DIR/$folder_name"
    dst="$TARGET_DIR/$folder_name"

    echo "→ Verschiebe $folder_name"

    # If destination exists, remove it first to allow overwrite
    if [ -d "$dst" ]; then
        echo "Zielverzeichnis $dst existiert bereits – wird ersetzt."
        rm -rf "$dst"
    fi

    # Move folder
    mv "$src" "$dst"
done

echo "Fertig mit dem Verschieben."
