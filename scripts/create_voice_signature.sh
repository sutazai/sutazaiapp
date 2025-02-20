#!/bin/bash

# Voice Signature Creation Script
VOICE_SIGNATURE_FILE="/etc/voice_signature.dat"

echo "Creating voice signature..."
echo "Please say 'K@roLin@04142013073806!!##' clearly into the microphone"

# Record voice sample
arecord -d 5 -f cd -t wav /tmp/voice_sample.wav

# Create voice signature
sox /tmp/voice_sample.wav -n stat 2>&1 | grep "Mean    norm" > $VOICE_SIGNATURE_FILE

# Secure the signature file
chmod 600 $VOICE_SIGNATURE_FILE
chown root:root $VOICE_SIGNATURE_FILE

echo "Voice signature created and stored securely" 