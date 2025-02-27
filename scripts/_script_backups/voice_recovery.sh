#!/bin/bash

# Voice Recovery Script
source "${SCRIPT_DIR}/voice_verification.sh"

echo "Voice Recovery Options:"
echo "1. Recreate Voice Signature"
echo "2. Verify Current Voice Signature"
read -p "Select option: " option

case $option in
    1)
        ./create_voice_signature.sh
        ;;
    2)
        if voice_verification; then
            echo "Voice verification successful"
        else
            echo "Voice verification failed"
        fi
        ;;
    *)
        echo "Invalid option"
        ;;
esac 