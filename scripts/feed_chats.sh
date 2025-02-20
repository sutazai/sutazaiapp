#!/bin/bash

# Check if input file is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <chat_file.txt>"
    exit 1
fi

CHAT_FILE="$1"

# Feed chats to the Python script
cat "$CHAT_FILE" | python3 scripts/feed_chats.py 