#!/bin/bash
# Automatically add/remove users
USER_LIST=("user1" "user2" "user3")
for user in "${USER_LIST[@]}"; do
    if ! id $user &>/dev/null; then
        useradd $user
        echo "User $user added"
    fi
done 