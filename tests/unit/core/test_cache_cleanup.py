#!/usr/bin/env python3
"""Direct test of cache cleanup functionality"""

import sys
import os
import json
import time
# Path handled by pytest configuration

from app import HardwareResourceOptimizerAgent

# Create agent instance
agent = HardwareResourceOptimizerAgent()

def count_files_in_directory(directory):
    """Count files in directory recursively"""
    if not os.path.exists(directory):
        return 0
    
    count = 0
    for root, dirs, files in os.walk(directory):
        count += len(files)
    return count

def get_directory_size(directory):
    """Get total size of directory in bytes"""
    if not os.path.exists(directory):
        return 0
    
    total_size = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            filepath = os.path.join(root, file)
            try:
                total_size += os.path.getsize(filepath)
            except (OSError, FileNotFoundError):
                pass
    return total_size

print("Testing cache cleanup functionality...")

# Test cache directories we created
cache_base = '/tmp/storage_test_environment/cache_directories'
cache_dirs = ['pip_cache', 'npm_cache', 'browser_cache', 'app_cache', 'system_cache']

print("\nBefore cache cleanup:")
initial_stats = {}
for cache_dir in cache_dirs:
    cache_path = os.path.join(cache_base, cache_dir)
    file_count = count_files_in_directory(cache_path)
    dir_size = get_directory_size(cache_path)
    initial_stats[cache_dir] = {'files': file_count, 'size_mb': dir_size / (1024*1024)}
    print(f"  {cache_dir}: {file_count} files, {dir_size/(1024*1024):.2f} MB")

# Test the cache optimization function
print("\n" + "="*50)
print("Running cache optimization...")
print("="*50)

result = agent._optimize_cache()

if result['status'] == 'success':
    print("Cache optimization completed successfully!")
    print(f"Actions taken: {len(result['actions_taken'])}")
    
    print("\nActions performed:")
    for action in result['actions_taken'][:10]:  # Show first 10 actions
        print(f"  - {action}")
    
    if len(result['actions_taken']) > 10:
        print(f"  ... and {len(result['actions_taken']) - 10} more actions")
        
else:
    print(f"Cache optimization failed: {result.get('error', 'Unknown error')}")

print("\nAfter cache cleanup:")
final_stats = {}
for cache_dir in cache_dirs:
    cache_path = os.path.join(cache_base, cache_dir)
    file_count = count_files_in_directory(cache_path)
    dir_size = get_directory_size(cache_path)
    final_stats[cache_dir] = {'files': file_count, 'size_mb': dir_size / (1024*1024)}
    print(f"  {cache_dir}: {file_count} files, {dir_size/(1024*1024):.2f} MB")

print("\nCache cleanup summary:")
total_files_before = sum(stats['files'] for stats in initial_stats.values())
total_files_after = sum(stats['files'] for stats in final_stats.values())
total_size_before = sum(stats['size_mb'] for stats in initial_stats.values())
total_size_after = sum(stats['size_mb'] for stats in final_stats.values())

print(f"Files: {total_files_before} -> {total_files_after} (removed {total_files_before - total_files_after})")
print(f"Size: {total_size_before:.2f} MB -> {total_size_after:.2f} MB (freed {total_size_before - total_size_after:.2f} MB)")

# Test comprehensive storage optimization with dry run
print("\n" + "="*50)
print("Testing comprehensive storage optimization (dry run)...")
print("="*50)

storage_result = agent._optimize_storage_comprehensive(dry_run=True)

if storage_result['status'] == 'success':
    print("Comprehensive storage optimization (dry run) completed!")
    print(f"Dry run: {storage_result['dry_run']}")
    print(f"Actions that would be taken: {len(storage_result['actions_taken'])}")
    print(f"Estimated space freed: {storage_result['estimated_space_freed_mb']:.2f} MB")
    
    print("\nActions that would be performed:")
    for action in storage_result['actions_taken'][:10]:
        print(f"  - {action}")
    
    if len(storage_result['actions_taken']) > 10:
        print(f"  ... and {len(storage_result['actions_taken']) - 10} more actions")
else:
    print(f"Storage optimization failed: {storage_result.get('error', 'Unknown error')}")