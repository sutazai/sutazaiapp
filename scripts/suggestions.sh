#!/bin/bash
case $1 in
    list)
        sutazai-cli suggestions list
        ;;
    approve)
        sutazai-cli suggestions approve --id $2
        ;;
    reject)
        sutazai-cli suggestions reject --id $2
        ;;
    *)
        echo "Usage: $0 {list|approve|reject}"
        exit 1
esac 