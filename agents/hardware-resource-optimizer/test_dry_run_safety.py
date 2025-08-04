#!/usr/bin/env python3
"""Test dry run functionality and safety features"""

import sys
import os
import json
import time
import shutil
sys.path.insert(0, '.')

from app import HardwareResourceOptimizerAgent

# Create agent instance
agent = HardwareResourceOptimizerAgent()

def count_files_recursive(directory):
    """Count all files in directory recursively"""
    if not os.path.exists(directory):
        return 0
    
    count = 0
    for root, dirs, files in os.walk(directory):
        count += len(files)
    return count

def create_test_safety_scenario():
    """Create a test scenario for safety testing"""
    test_dir = '/tmp/storage_safety_test'
    os.makedirs(test_dir, exist_ok=True)
    
    # Create files that should be safe to delete (old temp files)
    safe_files = []
    for i in range(5):
        filename = f'temp_file_{i}.tmp'
        filepath = os.path.join(test_dir, filename)
        with open(filepath, 'w') as f:
            f.write(f'Temporary test file {i}\n' + 'X' * 100)
        
        # Make files old (4 days ago)
        old_time = time.time() - (4 * 86400)
        os.utime(filepath, (old_time, old_time))
        safe_files.append(filepath)
    
    # Create files that should NOT be deleted (recent files)
    recent_files = []
    for i in range(3):
        filename = f'recent_file_{i}.txt'
        filepath = os.path.join(test_dir, filename)
        with open(filepath, 'w') as f:
            f.write(f'Recent important file {i}\n' + 'Y' * 100)
        recent_files.append(filepath)
    
    return test_dir, safe_files, recent_files

print("Testing dry run functionality and safety features...")

# Test 1: Dry Run vs Real Run for Duplicate Removal
print(f"\n{'='*60}")
print("Test 1: Dry Run vs Real Run - Duplicate Removal")
print(f"{'='*60}")

# Create a test duplicate scenario
test_dup_dir = '/tmp/test_duplicates'
os.makedirs(test_dup_dir, exist_ok=True)

# Create original and duplicate files
original_content = "This is a test file for duplicate testing with unique content."
original_file = os.path.join(test_dup_dir, 'original.txt')
duplicate_file = os.path.join(test_dup_dir, 'duplicate.txt')

with open(original_file, 'w') as f:
    f.write(original_content)
with open(duplicate_file, 'w') as f:
    f.write(original_content)

# Set different modification times
os.utime(original_file, (time.time(), time.time()))
os.utime(duplicate_file, (time.time() - 3600, time.time() - 3600))  # 1 hour older

print(f"Created test files:")
print(f"  Original: {original_file} (exists: {os.path.exists(original_file)})")
print(f"  Duplicate: {duplicate_file} (exists: {os.path.exists(duplicate_file)})")

# Test dry run first
print(f"\nTesting DRY RUN duplicate removal...")
dry_result = agent._optimize_duplicates(test_dup_dir, dry_run=True)

if dry_result['status'] == 'success':
    print(f"Dry run completed successfully!")
    print(f"Dry run flag: {dry_result['dry_run']}")
    print(f"Actions that would be taken: {len(dry_result['actions_taken'])}")
    print(f"Duplicates that would be removed: {dry_result['duplicates_removed']}")
    print(f"Space that would be freed: {dry_result['space_freed_mb']:.4f} MB")
    
    for action in dry_result['actions_taken']:
        print(f"  Would do: {action}")
    
    # Verify files still exist after dry run
    print(f"\nAfter DRY RUN:")
    print(f"  Original exists: {os.path.exists(original_file)}")
    print(f"  Duplicate exists: {os.path.exists(duplicate_file)}")
    
    if not os.path.exists(original_file) or not os.path.exists(duplicate_file):
        print("ERROR: Dry run deleted files! This is a critical safety violation!")
    else:
        print("PASS: Dry run preserved all files as expected")
else:
    print(f"Dry run failed: {dry_result.get('error', 'Unknown error')}")

# Test real run
print(f"\nTesting REAL RUN duplicate removal...")
real_result = agent._optimize_duplicates(test_dup_dir, dry_run=False)

if real_result['status'] == 'success':
    print(f"Real run completed successfully!")
    print(f"Dry run flag: {real_result['dry_run']}")
    print(f"Actions taken: {len(real_result['actions_taken'])}")
    print(f"Duplicates removed: {real_result['duplicates_removed']}")
    print(f"Space freed: {real_result['space_freed_mb']:.4f} MB")
    
    for action in real_result['actions_taken']:
        print(f"  Did: {action}")
    
    # Verify expected behavior after real run
    print(f"\nAfter REAL RUN:")
    print(f"  Original exists: {os.path.exists(original_file)}")
    print(f"  Duplicate exists: {os.path.exists(duplicate_file)}")
    
    # Check safety backup
    safety_dir = agent.safe_temp_location
    print(f"  Safety backup directory: {safety_dir}")
    if os.path.exists(safety_dir):
        safety_files = os.listdir(safety_dir)
        print(f"  Files in safety backup: {len(safety_files)}")
        for safety_file in safety_files:
            print(f"    - {safety_file}")
    
else:
    print(f"Real run failed: {real_result.get('error', 'Unknown error')}")

# Test 2: Protected Path Safety
print(f"\n{'='*60}")
print("Test 2: Protected Path Safety")
print(f"{'='*60}")

protected_test_paths = [
    '/etc/test_file.txt',
    '/usr/test_file.txt',
    '/bin/test_file.txt',
    '/boot/test_file.txt',
    '/sys/test_file.txt',
    '/proc/test_file.txt'
]

print("Testing protected paths (should all be blocked):")
for test_path in protected_test_paths:
    is_safe = agent._is_safe_path(test_path)
    print(f"  {test_path}: {'BLOCKED' if not is_safe else 'ALLOWED (ERROR!)'}")

# Test 3: Comprehensive Storage Optimization Safety
print(f"\n{'='*60}")
print("Test 3: Comprehensive Storage Optimization Safety")
print(f"{'='*60}")

# Create safety test scenario
safety_test_dir, safe_files, recent_files = create_test_safety_scenario()

print(f"Created safety test scenario in: {safety_test_dir}")
print(f"  Safe files (old): {len(safe_files)}")
print(f"  Recent files (should be preserved): {len(recent_files)}")

# Count files before
files_before = count_files_recursive(safety_test_dir)
print(f"  Total files before: {files_before}")

# Test comprehensive optimization with dry run
print(f"\nTesting comprehensive storage optimization (DRY RUN)...")
comp_dry_result = agent._optimize_storage_comprehensive(dry_run=True)

if comp_dry_result['status'] == 'success':
    print(f"Comprehensive dry run completed!")
    print(f"Dry run flag: {comp_dry_result['dry_run']}")
    print(f"Actions that would be taken: {len(comp_dry_result['actions_taken'])}")
    print(f"Estimated space freed: {comp_dry_result['estimated_space_freed_mb']:.2f} MB")
    
    # Verify no files were actually deleted
    files_after_dry = count_files_recursive(safety_test_dir)
    print(f"Files before dry run: {files_before}")
    print(f"Files after dry run: {files_after_dry}")
    
    if files_before == files_after_dry:
        print("PASS: Dry run preserved all files")
    else:
        print(f"ERROR: Dry run deleted {files_before - files_after_dry} files!")
        
else:
    print(f"Comprehensive dry run failed: {comp_dry_result.get('error', 'Unknown error')}")

# Test 4: Hash Caching Performance
print(f"\n{'='*60}")
print("Test 4: Hash Caching Performance")
print(f"{'='*60}")

# Test hash caching by analyzing the same directory twice
print("Testing hash caching performance...")

test_file = '/tmp/storage_test_environment/large_files/huge_file.data'
if os.path.exists(test_file):
    print(f"Testing hash calculation for: {test_file}")
    
    # First calculation (should cache)
    start_time = time.time()
    hash1 = agent._get_file_hash(test_file)
    time1 = time.time() - start_time
    print(f"First hash calculation: {hash1} (took {time1:.4f} seconds)")
    
    # Second calculation (should use cache)
    start_time = time.time()
    hash2 = agent._get_file_hash(test_file)
    time2 = time.time() - start_time
    print(f"Second hash calculation: {hash2} (took {time2:.4f} seconds)")
    
    if hash1 == hash2:
        print("PASS: Hash values match")
        if time2 < time1:
            print(f"PASS: Cached lookup was faster ({time1/time2:.1f}x speedup)")
        else:
            print("INFO: Cache lookup not significantly faster (may be due to small file or system caching)")
    else:
        print("ERROR: Hash values don't match!")
else:
    print("Test file not found, skipping hash caching test")

# Test 5: Audit Logging
print(f"\n{'='*60}")
print("Test 5: Audit Logging Verification")
print(f"{'='*60}")

print("Testing audit logging...")
print(f"Safety backup location: {agent.safe_temp_location}")

if os.path.exists(agent.safe_temp_location):
    safety_files = os.listdir(agent.safe_temp_location)
    print(f"Files in safety backup: {len(safety_files)}")
    
    if safety_files:
        print("Safety backup files (audit trail):")
        for safety_file in safety_files:
            filepath = os.path.join(agent.safe_temp_location, safety_file)
            stat_info = os.stat(filepath)
            mod_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stat_info.st_mtime))
            print(f"  - {safety_file} (backed up: {mod_time}, size: {stat_info.st_size} bytes)")
    else:
        print("No files in safety backup (no deletions performed yet)")
else:
    print("Safety backup directory doesn't exist")

print(f"\nSafety and dry run testing completed!")

# Cleanup test files
try:
    if os.path.exists(test_dup_dir):
        shutil.rmtree(test_dup_dir)
    if os.path.exists(safety_test_dir):
        shutil.rmtree(safety_test_dir)
    print("Test cleanup completed.")
except Exception as e:
    print(f"Cleanup error: {e}")