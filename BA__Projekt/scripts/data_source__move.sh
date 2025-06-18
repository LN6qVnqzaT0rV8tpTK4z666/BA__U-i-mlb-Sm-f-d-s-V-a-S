#!/bin/bash
# BA__Projekt/scripts/data-source__move.sh

# Parse --quiet or --q flag
QUIET=false
for arg in "$@"; do
    if [[ "$arg" == "--quiet" || "$arg" == "--q" ]]; then
        QUIET=true
        break
    fi
done

# Paths
SOURCE_DIR="../assets/data/source"
TARGET_DIR="../assets/data/raw"

[ "$QUIET" = false ] && echo "Verschiebe Verzeichnisse von $SOURCE_DIR nach $TARGET_DIR"

mkdir -p "$TARGET_DIR"

for dir in "$SOURCE_DIR"/*/; do
    dir=${dir%/}
    folder_name=$(basename "$dir")

    src="$SOURCE_DIR/$folder_name"
    dst="$TARGET_DIR/$folder_name"

    [ "$QUIET" = false ] && echo "→ Verschiebe $folder_name"

    if [ -d "$dst" ]; then
        [ "$QUIET" = false ] && echo "Zielverzeichnis $dst existiert bereits – wird ersetzt."
        rm -rf "$dst"
    fi

    mv "$src" "$dst"
done

[ "$QUIET" = false ] && echo "Fertig mit dem Verschieben."
