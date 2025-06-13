#!/bin/bash

set -e # Exit on error

TEST_DIR="/root/BA__U-i-mlb-Sm-f-d-s-V-a-S/BA__Projekt/tests"

echo "Running all unittests in $TEST_DIR using unittest discovery..."

python3 -m unittest discover -s "$TEST_DIR" -p 'test__*.py'

echo "All unittests completed successfully."
