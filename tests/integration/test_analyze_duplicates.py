#!/usr/bin/env python3
"""Direct test of duplicate analysis function"""

import logging

logger = logging.getLogger(__name__)
import sys
import os
# Path handled by pytest configuration

from app import HardwareResourceOptimizerAgent

# Create agent instance
agent = HardwareResourceOptimizerAgent()

# Test path safety
test_path = '/tmp/storage_test_environment/duplicates'
logger.info(f"Testing path: {test_path}")
logger.info(f"Path exists: {os.path.exists(test_path)}")
logger.info(f"Path is safe: {agent._is_safe_path(test_path)}")

if agent._is_safe_path(test_path) and os.path.exists(test_path):
    logger.info("Running duplicate analysis...")
    result = agent._analyze_duplicates(test_path)
    logger.info("Result:")
    import json
    logger.info(json.dumps(result, indent=2))
else:
    logger.info("Path is not safe or doesn't exist, skipping analysis")