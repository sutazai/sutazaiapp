#!/usr/bin/env python3
"""Direct test of storage analysis function"""

import sys
import os
import json
sys.path.insert(0, '.')

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

print("Testing storage analysis on various directories...")

for test_path in test_paths:
    print(f"\n{'='*60}")
    print(f"Analyzing: {test_path}")
    print(f"{'='*60}")
    
    if os.path.exists(test_path):
        result = agent._analyze_storage(test_path)
        
        if result['status'] == 'success':
            print(f"Total files: {result['total_files']}")
            print(f"Total size: {result['total_size_mb']:.2f} MB")
            
            print("\nTop file extensions by size:")
            for ext, stats in list(result['extension_breakdown'].items())[:5]:
                print(f"  {ext}: {stats['count']} files, {stats['total_size']/(1024*1024):.2f} MB")
            
            print("\nSize distribution:")
            for size_range, count in result['size_distribution'].items():
                print(f"  {size_range}: {count} files")
            
            print("\nAge distribution:")
            for age_range, count in result['age_distribution'].items():
                print(f"  {age_range}: {count} files")
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")
    else:
        print("Path does not exist")

# Also test the comprehensive storage report
print(f"\n{'='*60}")
print("Testing comprehensive storage report...")
print(f"{'='*60}")

report = agent._generate_storage_report()
if report['status'] == 'success':
    print(f"Disk usage: {report['disk_usage']['usage_percent']:.1f}%")
    print(f"Free space: {report['disk_usage']['free_gb']:.1f} GB")
    
    print(f"\nDuplicate summary: {report['duplicate_summary']['groups']} groups, {report['duplicate_summary']['space_wasted_mb']:.2f} MB wasted")
    print(f"Large files summary: {report['large_files_summary']['count']} files, {report['large_files_summary']['total_size_mb']:.1f} MB")
    
    print("\nPath analysis:")
    for path, analysis in report['path_analysis'].items():
        print(f"  {path}: {analysis['total_files']} files, {analysis['total_size_mb']:.2f} MB")
else:
    print(f"Report error: {report.get('error', 'Unknown error')}")