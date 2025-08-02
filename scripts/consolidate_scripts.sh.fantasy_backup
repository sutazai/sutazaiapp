#!/bin/bash

# Script to consolidate all scripts to /opt/sutazaiapp/scripts and remove cursor

set -e

SCRIPTS_DIR="/opt/sutazaiapp/scripts"
BASE_DIR="/opt/sutazaiapp"

echo "Starting script consolidation..."

# Move shell scripts from root directory
echo "Moving shell scripts from root..."
find "$BASE_DIR" -maxdepth 1 -name "*.sh" -type f | while read -r file; do
    filename=$(basename "$file")
    echo "Moving $filename to scripts/"
    mv "$file" "$SCRIPTS_DIR/" 2>/dev/null || true
done

# Move shell scripts from other directories
echo "Moving shell scripts from subdirectories..."
find "$BASE_DIR" -name "*.sh" ! -path "*/scripts/*" ! -path "*/.git/*" ! -path "*/venv/*" ! -path "*/.venv/*" ! -path "*/site-packages/*" -type f | while read -r file; do
    filename=$(basename "$file")
    dir=$(dirname "$file")
    dir_name=$(echo "$dir" | sed "s|$BASE_DIR/||" | tr '/' '_')
    
    # Create a descriptive name to avoid conflicts
    if [ "$dir" != "$BASE_DIR" ]; then
        new_name="${dir_name}_${filename}"
    else
        new_name="$filename"
    fi
    
    echo "Moving $file to scripts/$new_name"
    mv "$file" "$SCRIPTS_DIR/$new_name" 2>/dev/null || true
done

# Move Python scripts that look like utility scripts
echo "Moving Python utility scripts..."
find "$BASE_DIR" -name "*.py" ! -path "*/scripts/*" ! -path "*/.git/*" ! -path "*/venv/*" ! -path "*/.venv/*" ! -path "*/site-packages/*" ! -path "*/backend/*" ! -path "*/frontend/*" ! -path "*/brain/*" -type f | while read -r file; do
    # Check if it's likely a script (not a module)
    if grep -q "if __name__ == ['\"]__main__['\"]" "$file" 2>/dev/null; then
        filename=$(basename "$file")
        dir=$(dirname "$file")
        dir_name=$(echo "$dir" | sed "s|$BASE_DIR/||" | tr '/' '_')
        
        if [ "$dir" != "$BASE_DIR" ]; then
            new_name="${dir_name}_${filename}"
        else
            new_name="$filename"
        fi
        
        echo "Moving Python script $file to scripts/$new_name"
        mv "$file" "$SCRIPTS_DIR/$new_name" 2>/dev/null || true
    fi
done

# Remove cursor directory
echo "Removing cursor directory..."
rm -rf "$BASE_DIR/.cursor"

# Remove any other cursor-related files
echo "Removing other cursor-related files..."
find "$BASE_DIR" -name "*cursor*" -o -name "*Cursor*" | grep -v "/.git/" | while read -r file; do
    echo "Removing $file"
    rm -rf "$file" 2>/dev/null || true
done

echo "Script consolidation complete!"
echo "Total scripts in scripts directory:"
ls -1 "$SCRIPTS_DIR"/*.sh "$SCRIPTS_DIR"/*.py 2>/dev/null | wc -l

echo -e "\nScript types:"
echo "Shell scripts: $(ls -1 "$SCRIPTS_DIR"/*.sh 2>/dev/null | wc -l)"
echo "Python scripts: $(ls -1 "$SCRIPTS_DIR"/*.py 2>/dev/null | wc -l)"