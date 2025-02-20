#!/bin/bash

# Password Recovery Script
source "${SCRIPT_DIR}/password_manager.sh"

echo "Password Recovery Options:"
echo "1. Change Master Password"
echo "2. Verify Current Password"
read -p "Select option: " option

case $option in
    1)
        if verify_password; then
            read -sp "Enter new master password: " new_password
            echo
            read -sp "Confirm new master password: " confirm_password
            echo
            
            if [ "$new_password" == "$confirm_password" ]; then
                ENCRYPTED_PASSWORD=$(echo -n "$new_password" | openssl enc -aes-256-cbc -a -salt -pass pass:secret -pbkdf2)
                echo "Password changed successfully"
            else
                echo "Passwords do not match"
            fi
        else
            echo "Incorrect current password"
        fi
        ;;
    2)
        if verify_password; then
            echo "Password verification successful"
        else
            echo "Incorrect password"
        fi
        ;;
    *)
        echo "Invalid option"
        ;;
esac 