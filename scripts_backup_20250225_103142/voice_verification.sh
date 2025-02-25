#!/bin/bash

# Voice Verification Script
VOICE_SIGNATURE_FILE="/etc/voice_signature.dat"
TEMP_SIGNATURE="/tmp/temp_voice_signature.dat"
MAX_VOICE_ATTEMPTS=3
VOICE_LOCK_FILE="/tmp/voice_lock"
VOICE_ATTEMPT_COUNT=0

# Verify voice signature exists
if [ ! -f "$VOICE_SIGNATURE_FILE" ]; then
    echo "ERROR: Voice signature not found"
    exit 1
fi

if [ -f "$VOICE_LOCK_FILE" ]; then
    VOICE_ATTEMPT_COUNT=$(cat "$VOICE_LOCK_FILE")
    if [ "$VOICE_ATTEMPT_COUNT" -ge "$MAX_VOICE_ATTEMPTS" ]; then
        echo "Voice verification locked. Too many failed attempts."
        exit 1
    fi
fi

echo "Please say 'K@roLin@04142013073806!!##' to verify your identity"

# Record verification sample
arecord -d 5 -f cd -t wav /tmp/verification_sample.wav

# Create temporary signature
sox /tmp/verification_sample.wav -n stat 2>&1 | grep "Mean    norm" > $TEMP_SIGNATURE

# Compare signatures
DIFF=$(diff $VOICE_SIGNATURE_FILE $TEMP_SIGNATURE | wc -l)

# Clean up
rm -f /tmp/verification_sample.wav $TEMP_SIGNATURE

# Add to the script
log_voice_attempt() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >> /var/log/voice_verification.log
}

if [ $DIFF -ne 0 ]; then
    VOICE_ATTEMPT_COUNT=$((VOICE_ATTEMPT_COUNT + 1))
    echo "$VOICE_ATTEMPT_COUNT" > "$VOICE_LOCK_FILE"
    if [ "$VOICE_ATTEMPT_COUNT" -ge "$MAX_VOICE_ATTEMPTS" ]; then
        echo "Voice verification locked. Too many failed attempts."
        exit 1
    fi
    log_voice_attempt "Failed voice verification attempt"
    return 1
else
    rm -f "$VOICE_LOCK_FILE"
    log_voice_attempt "Successful voice verification"
    return 0
fi 