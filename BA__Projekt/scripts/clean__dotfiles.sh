
#!/bin/bash
# BA__Projekt/scripts/clean__dotfiles.sh

# Navigate to the project root
cd "$(dirname "$0")/.."

echo "Cleaning all files beginning with . and certain extensions"

# Find and delete all matching files
find . -type f -name '*.DS_Store' -print -delete
find . -type f -name '*._*' -print -delete

echo "Cleanup complete."
