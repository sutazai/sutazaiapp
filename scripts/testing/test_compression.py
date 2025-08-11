#!/usr/bin/env python3
"""Direct test of file compression functionality"""

import sys
import os
import json
import time
sys.path.insert(0, '.')

from app import HardwareResourceOptimizerAgent

# Create agent instance
agent = HardwareResourceOptimizerAgent()

def get_directory_files_info(directory):
    """Get detailed info about files in directory"""
    files_info = []
    if not os.path.exists(directory):
        return files_info
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            filepath = os.path.join(root, file)
            try:
                stat_info = os.stat(filepath)
                files_info.append({
                    'path': filepath,
                    'name': file,
                    'size': stat_info.st_size,
                    'compressed': file.endswith('.gz')
                })
            except (OSError, FileNotFoundError):
                pass
    return files_info

print("Testing file compression functionality...")

# Test compression on our compressible test files
test_path = '/tmp/storage_test_environment/compressible_files'

print(f"\nAnalyzing files in: {test_path}")
before_files = get_directory_files_info(test_path)

print("\nBefore compression:")
total_size_before = 0
for file_info in before_files:
    print(f"  {file_info['name']}: {file_info['size']} bytes ({'compressed' if file_info['compressed'] else 'uncompressed'})")
    total_size_before += file_info['size']

print(f"\nTotal size before: {total_size_before} bytes ({total_size_before/(1024*1024):.2f} MB)")

# Test compression with different age thresholds
age_thresholds = [20, 30, 60, 90]  # days

for days_old in age_thresholds:
    print(f"\n{'='*60}")
    print(f"Testing compression for files older than {days_old} days")
    print(f"{'='*60}")
    
    result = agent._optimize_compress(test_path, days_old)
    
    if result['status'] == 'success':
        print(f"Compression completed for {days_old} day threshold!")
        print(f"Path: {result['path']}")
        print(f"Age threshold: {result['days_old']} days")
        print(f"Files compressed: {result['files_compressed']}")
        print(f"Space saved: {result['space_saved_mb']:.2f} MB")
        
        if result['actions_taken']:
            print(f"\nActions taken:")
            for action in result['actions_taken']:
                print(f"  - {action}")
        else:
            print("\nNo actions taken (no files met compression criteria)")
            
        # Check current state after compression
        after_files = get_directory_files_info(test_path)
        print(f"\nFiles after compression attempt:")
        total_size_after = 0
        compressed_count = 0
        for file_info in after_files:
            if file_info['compressed']:
                compressed_count += 1
            print(f"  {file_info['name']}: {file_info['size']} bytes ({'compressed' if file_info['compressed'] else 'uncompressed'})")
            total_size_after += file_info['size']
        
        print(f"\nSummary after {days_old}-day threshold:")
        print(f"  Total files: {len(after_files)}")
        print(f"  Compressed files: {compressed_count}")
        print(f"  Total size: {total_size_after} bytes ({total_size_after/(1024*1024):.2f} MB)")
        print(f"  Space saved: {total_size_before - total_size_after} bytes")
        
        # Update for next iteration
        total_size_before = total_size_after
        
    else:
        print(f"Compression failed: {result.get('error', 'Unknown error')}")

print(f"\n{'='*60}")
print("Testing log optimization (includes compression)...")
print(f"{'='*60}")

# Test log optimization which includes compression
log_result = agent._optimize_logs()

if log_result['status'] == 'success':
    print("Log optimization completed!")
    print(f"Actions taken: {len(log_result['actions_taken'])}")
    print(f"Space freed: {log_result['space_freed_mb']:.2f} MB")
    
    if log_result['actions_taken']:
        print(f"\nLog optimization actions:")
        for action in log_result['actions_taken'][:10]:  # Show first 10
            print(f"  - {action}")
        
        if len(log_result['actions_taken']) > 10:
            print(f"  ... and {len(log_result['actions_taken']) - 10} more actions")
else:
    print(f"Log optimization failed: {log_result.get('error', 'Unknown error')}")

print(f"\nCompression testing completed!")