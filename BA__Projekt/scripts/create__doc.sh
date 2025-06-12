#!/bin/bash
# BA__Projekt/scripts/create__doc.sh
echo "==> Preparing Sphinx documentation..."
#mkdir -p assets/docs/source assets/docs/build

echo "==> Writing conf.py..."
sphinx-quickstart \
  --project "BA__U-i-mlb-Sm-f-d-s-V-a-S" \
  --author "Marten Windler" \
  --release "0.1.0" \
  --language "de" \
  --extensions "sphinx.ext.autodoc,sphinx.ext.napoleon,sphinx.ext.viewcode" \
  --sep \
  --makefile \
  --batchfile \
  ./assets/docs/

sleep 3

echo "==> Generating .rst files from source..."
sphinx-apidoc -o assets/docs/source BA__Programmierung --force --separate

sleep 3

echo "==> Building HTML documentation..."
sphinx-build -b html assets/docs/source assets/docs/build/html
