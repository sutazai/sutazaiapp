#!/bin/bash

# Script to fix indentation in test_sync_manager_complete_coverage.py

echo "Fixing indentation in test_sync_manager_complete_coverage.py..."

# Create a temporary file
tempfile=$(mktemp)

# Process the file line by line
while IFS= read -r line; do
    # Check if line contains the decorator and not properly indented
    if [[ "$line" == "@pytest.mark.asyncio" && "$line" != "    @pytest.mark.asyncio"* ]]; then
        # Add proper indentation
        echo "    @pytest.mark.asyncio" >> "$tempfile"
    else
        echo "$line" >> "$tempfile"
    fi
done < "tests/test_sync_manager_complete_coverage.py"

# Replace the original file with the fixed one
mv "$tempfile" "tests/test_sync_manager_complete_coverage.py"

# Make the file executable (optional)
chmod +x "tests/test_sync_manager_complete_coverage.py"

echo "Indentation fixed in test_sync_manager_complete_coverage.py" 