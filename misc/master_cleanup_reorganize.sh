#!/bin/bash

# Master Cleanup and Reorganization Script for SutazAI
# Created on Tue Feb 25 09:27:56 PM UTC 2025
# Configuration
PROJECT_ROOT="/opt/SutazAI"
LOG_DIR="${PROJECT_ROOT}/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MASTER_LOG="${LOG_DIR}/master_cleanup_${TIMESTAMP}.log"
