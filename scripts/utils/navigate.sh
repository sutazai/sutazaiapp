#!/bin/bash
# Quick navigation helper for SutazAI project

case "$1" in
    deploy)
        cd scripts/deployment/system
        ;;
    agents)
        cd scripts/agents
        ;;
    models)
        cd scripts/models
        ;;
    docs)
        cd docs
        ;;
    *)
        echo "Usage: ./navigate.sh [deploy|agents|models|docs]"
        echo "Quick navigation to common directories"
        ;;
esac
