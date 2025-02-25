#!/bin/bash
case $1 in
    create)
        sutazai-cli agents create --type $2
        ;;
    list)
        sutazai-cli agents list
        ;;
    terminate)
        sutazai-cli agents terminate --id $2
        ;;
    *)
        echo "Usage: $0 {create|list|terminate}"
        exit 1
esac 