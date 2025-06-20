#!/bin/bash
set -e
# BA__Projekt/scripts/create__test_folders.sh

# Function to create mirrored test files ignoring __init__.py
create_tests_for() {
    src_dir="$1"
    dst_dir="$2"
    pattern="${3:-*.py}"  # optional file pattern (default: *.py)

    mkdir -p "$dst_dir"
    for file in "$src_dir"/$pattern; do
        base=$(basename "$file")
        if [[ "$base" == "__init__.py" ]]; then
            continue
        fi
        touch "$dst_dir/test__${base}"
    done
}

create_tests_for "BA__Programmierung/conv"              "tests/BA__Programmierung/conv" "conversion__*.py"
create_tests_for "BA__Programmierung/db"                "tests/BA__Programmierung/db"
create_tests_for "BA__Programmierung/gen"               "tests/BA__Programmierung/gen"
create_tests_for "BA__Programmierung/ml/datasets"       "tests/BA__Programmierung/ml/datasets"
create_tests_for "BA__Programmierung/ml/losses"         "tests/BA__Programmierung/ml/losses"
create_tests_for "BA__Programmierung/ml/metrics"        "tests/BA__Programmierung/ml/metrics"
create_tests_for "BA__Programmierung/ml/utils"          "tests/BA__Programmierung/ml/utils"
create_tests_for "BA__Programmierung/util"              "tests/BA__Programmierung/util"
create_tests_for "BA__Programmierung/viz"               "tests/BA__Programmierung/viz"

echo "✅ Test file scaffolding completed (excluding __init__.py files)."
