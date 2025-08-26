#!/usr/bin/env python3
"""Direct test of storage analysis function"""

import logging

logger = logging.getLogger(__name__)
import sys
import os
import json
# Path handled by pytest configuration

from app import HardwareResourceOptimizerAgent

# Create agent instance
agent = HardwareResourceOptimizerAgent()

# Test storage analysis on different test directories
test_paths = [
    '/tmp/storage_test_environment/duplicates',
    '/tmp/storage_test_environment/large_files',
    '/tmp/storage_test_environment/old_files',
    '/tmp/storage_test_environment/compressible_files',
    '/tmp/storage_test_environment'  # Root test directory
]

logger.info("Testing storage analysis on various directories...")

for test_path in test_paths:
    logger.info(f"\n{'='*60}")
    logger.info(f"Analyzing: {test_path}")
    logger.info(f"{'='*60}")
    
    if os.path.exists(test_path):
        result = agent._analyze_storage(test_path)
        
        if result['status'] == 'success':
            logger.info(f"Total files: {result['total_files']}")
            logger.info(f"Total size: {result['total_size_mb']:.2f} MB")
            
            logger.info("\nTop file extensions by size:")
            for ext, stats in list(result['extension_breakdown'].items())[:5]:
                logger.info(f"  {ext}: {stats['count']} files, {stats['total_size']/(1024*1024):.2f} MB")
            
            logger.info("\nSize distribution:")
            for size_range, count in result['size_distribution'].items():
                logger.info(f"  {size_range}: {count} files")
            
            logger.info("\nAge distribution:")
            for age_range, count in result['age_distribution'].items():
                logger.info(f"  {age_range}: {count} files")
        else:
            logger.error(f"Error: {result.get('error', 'Unknown error')}")
    else:
        logger.info("Path does not exist")

# Also test the comprehensive storage report
logger.info(f"\n{'='*60}")
logger.info("Testing comprehensive storage report...")
logger.info(f"{'='*60}")

report = agent._generate_storage_report()
if report['status'] == 'success':
    logger.info(f"Disk usage: {report['disk_usage']['usage_percent']:.1f}%")
    logger.info(f"Free space: {report['disk_usage']['free_gb']:.1f} GB")
    
    logger.info(f"\nDuplicate summary: {report['duplicate_summary']['groups']} groups, {report['duplicate_summary']['space_wasted_mb']:.2f} MB wasted")
    logger.info(f"Large files summary: {report['large_files_summary']['count']} files, {report['large_files_summary']['total_size_mb']:.1f} MB")
    
    logger.info("\nPath analysis:")
    for path, analysis in report['path_analysis'].items():
        logger.info(f"  {path}: {analysis['total_files']} files, {analysis['total_size_mb']:.2f} MB")
else:
    logger.error(f"Report error: {report.get('error', 'Unknown error')}")