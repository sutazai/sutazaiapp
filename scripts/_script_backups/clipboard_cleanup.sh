#!/bin/bash

# Comprehensive Clipboard and Memory Cleanup Script

# Clear system clipboard
echo "Clearing system clipboard..."
for clip in clipboard primary secondary; do
    echo "" | xclip -selection "$clip" 2>/dev/null
done

# Clear temporary files
echo "Cleaning temporary files..."
find /tmp -type f \( -name "*clipboard*" -o -name "*paste*" -o -name "*cursor*" \) -delete 2>/dev/null

# Clear memory cache
echo "Clearing memory cache..."
sync
echo 3 > /proc/sys/vm/drop_caches

# Reset Cursor AI configuration
echo "Resetting Cursor AI configuration..."
rm -rf ~/.config/Cursor ~/.cursor 2>/dev/null

echo "Cleanup complete. Please restart Cursor AI." 