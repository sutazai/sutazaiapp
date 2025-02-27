#!/bin/bash
# Validate system configurations
docker-compose config -q && echo "Config is valid" || echo "Config is invalid" 