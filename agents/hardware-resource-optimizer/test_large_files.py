#!/usr/bin/env python3
"""Direct test of large file analysis function"""

import sys
import os
import json
sys.path.insert(0, '.')

from app import HardwareResourceOptimizerAgent

# Create agent instance
agent = HardwareResourceOptimizerAgent()

# Test large file detection with different thresholds
test_path = '/tmp/storage_test_environment/large_files'
print(f"Testing large file detection at: {test_path}")
print(f"Path exists: {os.path.exists(test_path)}")
print(f"Path is safe: {agent._is_safe_path(test_path)}")

if agent._is_safe_path(test_path) and os.path.exists(test_path):
    thresholds = [1, 10, 50, 100, 200]  # MB
    
    for threshold in thresholds:
        print(f"\n=== Testing with threshold: {threshold} MB ===")
        result = agent._analyze_large_files(test_path, threshold)
        
        if result['status'] == 'success':
            print(f"Found {result['large_files_count']} files >= {threshold} MB")
            print(f"Total size: {result['total_size_mb']:.1f} MB")
            
            if result['large_files']:
                print("Large files found:")
                for file_info in result['large_files'][:5]:  # Show first 5
                    print(f"  - {file_info['name']}: {file_info['size_mb']:.1f} MB")
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")

else:
    print("Path is not safe or doesn't exist, skipping analysis")