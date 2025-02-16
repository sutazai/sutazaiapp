#!/bin/bash
case $1 in
    start)
        sutazai-cli chat --security-level=maximum
        ;;
    auth)
        sutazai-cli auth-request --method $2
        ;;
    history)
        sutazai-cli chat-history --encrypted
        ;;
    *)
        echo "Usage: $0 {start|auth|history}"
        exit 1
esac 