#!/bin/bash
# BA__Projekt/scripts/create__doc.sh

# Check for --quiet / --q flags
QUIET=false
for arg in "$@"; do
    if [[ "$arg" == "--quiet" || "$arg" == "--q" ]]; then
        QUIET=true
        break
    fi
done

# Step 1: Initialize Sphinx
if [ "$QUIET" = false ]; then echo "==> Preparing Sphinx documentation..."; fi

# Step 2: Run sphinx-quickstart
if [ "$QUIET" = false ]; then echo "==> Writing conf.py..."; fi
sphinx-quickstart \
  --project "BA__U-i-mlb-Sm-f-d-s-V-a-S" \
  --author "Marten Windler" \
  --release "0.1.0" \
  --language "en" \
  --extensions "sphinx.ext.autodoc,sphinx.ext.napoleon,sphinx.ext.viewcode" \
  --sep \
  --makefile \
  --batchfile \
  ./assets/docs/

sleep 3

# Step 3: Generate .rst files
if [ "$QUIET" = false ]; then echo "==> Generating .rst files from source..."; fi
sphinx-apidoc -o assets/docs/source BA__Programmierung --force --separate

sleep 3

# Step 4: Build HTML
if [ "$QUIET" = false ]; then echo "==> Building HTML documentation..."; fi
sphinx-build -b html assets/docs/source assets/docs/build/html
