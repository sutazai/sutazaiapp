#!/bin/bash
case $1 in
    start)
        ;;
    list)
        sutazai-cli research list
        ;;
    integrate)
        sutazai-cli research integrate --package $2
        ;;
    *)
        echo "Usage: $0 {start|list|integrate}"
        exit 1
esac 