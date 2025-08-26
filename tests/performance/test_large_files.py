#!/usr/bin/env python3
"""Direct test of large file analysis function"""

import logging

logger = logging.getLogger(__name__)
import sys
import os
import json
# Path handled by pytest configuration

from app import HardwareResourceOptimizerAgent

# Create agent instance
agent = HardwareResourceOptimizerAgent()

# Test large file detection with different thresholds
test_path = '/tmp/storage_test_environment/large_files'
logger.info(f"Testing large file detection at: {test_path}")
logger.info(f"Path exists: {os.path.exists(test_path)}")
logger.info(f"Path is safe: {agent._is_safe_path(test_path)}")

if agent._is_safe_path(test_path) and os.path.exists(test_path):
    thresholds = [1, 10, 50, 100, 200]  # MB
    
    for threshold in thresholds:
        logger.info(f"\n=== Testing with threshold: {threshold} MB ===")
        result = agent._analyze_large_files(test_path, threshold)
        
        if result['status'] == 'success':
            logger.info(f"Found {result['large_files_count']} files >= {threshold} MB")
            logger.info(f"Total size: {result['total_size_mb']:.1f} MB")
            
            if result['large_files']:
                logger.info("Large files found:")
                for file_info in result['large_files'][:5]:  # Show first 5
                    logger.info(f"  - {file_info['name']}: {file_info['size_mb']:.1f} MB")
        else:
            logger.error(f"Error: {result.get('error', 'Unknown error')}")

else:
    logger.info("Path is not safe or doesn't exist, skipping analysis")