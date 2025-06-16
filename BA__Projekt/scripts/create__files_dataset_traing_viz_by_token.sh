#!/bin/bash
# BA__Projekt/scripts/create__files_dataset_traing_viz_by_token.sh

echo "Enter token (used for all files):"
read token

while true; do
    echo "Enter a name (or press Enter to quit):"
    read name

    # Exit condition
    if [[ -z "$name" ]]; then
        echo "Done."
        break
    fi

    # File paths
    dataset_file="BA__Projekt/BA__Programmierung/ml/datasets/dataset__${name}__${token}.py"
    model_file="BA__Projekt/BA__Programmierung/ml/${name}__${token}.py"
    test_file="BA__Projekt/tests/test__${name}__${token}.py"

    # Create directories if they don't exist
    mkdir -p "$(dirname "$dataset_file")"
    mkdir -p "$(dirname "$model_file")"
    mkdir -p "$(dirname "$test_file")"

    # Create files if they don't exist
    for file in "$dataset_file" "$model_file" "$test_file"; do
        if [[ -e "$file" ]]; then
            echo "⚠️  File already exists: $file"
        else
            touch "$file"
            echo "✅ Created: $file"
        fi
    done

    echo ""
done
