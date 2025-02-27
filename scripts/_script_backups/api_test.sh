#!/bin/bash
# Test API endpoints
curl -sSf http://localhost:8000/health && echo "API is healthy" || echo "API is down" 