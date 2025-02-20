#!/bin/bash

# Add at the beginning of the script
source "${SCRIPT_DIR}/voice_verification.sh"

if ! voice_verification; then
    echo "Voice verification failed"
    exit 1
fi

# Password Manager Script
MASTER_PASSWORD="K@roLin@04142013073806!!##"
ENCRYPTED_PASSWORD=$(echo -n "$MASTER_PASSWORD" | openssl enc -aes-256-cbc -a -salt -pass pass:secret -pbkdf2)

# Add at the beginning of the script
PASSWORD_EXPIRE_DAYS=90
LAST_CHANGE_DATE=$(date -d "2023-01-01" +%s) # Replace with actual last change date
CURRENT_DATE=$(date +%s)
DAYS_SINCE_CHANGE=$(( (CURRENT_DATE - LAST_CHANGE_DATE) / 86400 ))

if [ $DAYS_SINCE_CHANGE -gt $PASSWORD_EXPIRE_DAYS ]; then
    echo "WARNING: Password has expired. Please change it."
    ./password_recovery.sh
    exit 1
fi

# Add at the beginning of the script
MAX_ATTEMPTS=3
LOCK_FILE="/tmp/password_lock"
ATTEMPT_COUNT=0

if [ -f "$LOCK_FILE" ]; then
    ATTEMPT_COUNT=$(cat "$LOCK_FILE")
    if [ "$ATTEMPT_COUNT" -ge "$MAX_ATTEMPTS" ]; then
        echo "Account locked. Too many failed attempts."
        exit 1
    fi
fi

verify_password() {
    echo -n "Enter master password: "
    read -s entered_password
    echo
    
    decrypted_password=$(echo "$ENCRYPTED_PASSWORD" | openssl enc -d -aes-256-cbc -a -pass pass:secret -pbkdf2)
    
    log_message() {
        echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >> /var/log/password_access.log
    }
    
    if [ "$entered_password" != "$decrypted_password" ]; then
        ATTEMPT_COUNT=$((ATTEMPT_COUNT + 1))
        echo "$ATTEMPT_COUNT" > "$LOCK_FILE"
        if [ "$ATTEMPT_COUNT" -ge "$MAX_ATTEMPTS" ]; then
            echo "Account locked. Too many failed attempts."
            exit 1
        fi
        log_message "Failed password verification attempt"
        return 1
    else
        rm -f "$LOCK_FILE"
        log_message "Successful password verification"
        return 0
    fi
}

# Example usage
if verify_password; then
    echo "Access granted"
    # Add your protected commands here
else
    echo "Access denied"
    exit 1
fi 