#!/bin/bash
# Health check script for MCP Manager

curl -sf http://localhost:8081/health || exit 1