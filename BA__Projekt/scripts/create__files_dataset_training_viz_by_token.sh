#!/bin/bash
# BA__Projekt/scripts/create__files_dataset_training_viz_by_token.sh

# Check for --quiet / --q flag
QUIET=false
for arg in "$@"; do
    if [[ "$arg" == "--quiet" || "$arg" == "--q" ]]; then
        QUIET=true
        set -- "${@/--quiet/}"
        set -- "${@/--q/}"
    fi
done

# Prompt for token
if [ "$QUIET" = false ]; then
    echo "Enter token (used for all files):"
fi
read token

while true; do
    if [ "$QUIET" = false ]; then
        echo "Enter a name (or press Enter to quit):"
    fi
    read name

    if [[ -z "$name" ]]; then
        [ "$QUIET" = false ] && echo "Done."
        break
    fi

    # File paths
    dataset_file="BA__Projekt/BA__Programmierung/ml/datasets/dataset__${name}__${token}.py"
    model_file="BA__Projekt/BA__Programmierung/ml/${name}__${token}.py"
    test_file="BA__Projekt/tests/test__${name}__${token}.py"

    # Create directories
    mkdir -p "$(dirname "$dataset_file")"
    mkdir -p "$(dirname "$model_file")"
    mkdir -p "$(dirname "$test_file")"

    for file in "$dataset_file" "$model_file" "$test_file"; do
        if [[ -e "$file" ]]; then
            [ "$QUIET" = false ] && echo "⚠️  File already exists: $file"
        else
            touch "$file"
            [ "$QUIET" = false ] && echo "✅ Created: $file"
        fi
    done

    [ "$QUIET" = false ] && echo ""
done
