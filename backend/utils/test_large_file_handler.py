#!/usr/bin/env python3
"""
Test script for the large file handler utility.
Demonstrates how to use the large file handler for files that exceed the 256KB limit.
"""

import os
import sys
from large_file_handler import (
    read_large_file, 
    read_file_lines, 
    search_large_file, 
    get_file_info,
    LargeFileHandler
)


def demo_basic_usage():
    """Demonstrate basic usage of the large file handler."""
    print("=== Large File Handler Demo ===\n")
    
    # Example file path (you can change this to test with your own large file)
    test_file = "/opt/sutazaiapp/backend/app/working_main.py"
    
    # 1. Get file information
    print("1. Getting file information:")
    info = get_file_info(test_file)
    print(f"   File: {info['path']}")
    print(f"   Size: {info['size_mb']:.2f} MB ({info['size_bytes']:,} bytes)")
    print(f"   Exceeds 256KB limit: {info['exceeds_256kb_limit']}")
    print(f"   Estimated lines: {info['estimated_lines']:,}")
    print(f"   Recommended chunk size: {info['recommended_chunk_size']}")
    print()
    
    # 2. Read first 50 lines
    print("2. Reading first 50 lines:")
    content = read_large_file(test_file, max_lines=50)
    print(content[:500] + "...\n")  # Show first 500 chars
    
    # 3. Read specific lines (e.g., lines 100-110)
    print("3. Reading lines 100-110:")
    lines = read_file_lines(test_file, start_line=100, num_lines=10)
    print(lines)
    print()
    
    # 4. Search for a term
    print("4. Searching for 'async' in file:")
    results = search_large_file(test_file, "async", context_lines=1)
    print(results[:1000] + "...\n" if len(results) > 1000 else results + "\n")


def demo_advanced_usage():
    """Demonstrate advanced usage with the LargeFileHandler class."""
    print("=== Advanced Usage Demo ===\n")
    
    handler = LargeFileHandler(chunk_size=500)
    test_file = "/opt/sutazaiapp/backend/app/working_main.py"
    
    # Process file in chunks
    print("Processing file in chunks:")
    total_lines = 0
    total_chunks = 0
    
    for chunk_lines, start_line, end_line in handler.read_file_in_chunks(test_file):
        total_chunks += 1
        total_lines += len(chunk_lines)
        print(f"  Chunk {total_chunks}: lines {start_line}-{end_line} ({len(chunk_lines)} lines)")
        
        # Stop after 5 chunks for demo
        if total_chunks >= 5:
            print("  ... (stopping after 5 chunks for demo)")
            break
    
    print(f"\nProcessed {total_chunks} chunks, {total_lines} lines total")


def demo_error_handling():
    """Demonstrate error handling."""
    print("\n=== Error Handling Demo ===\n")
    
    # Try to read a non-existent file
    try:
        content = read_large_file("/non/existent/file.txt")
    except Exception as e:
        print(f"Expected error for non-existent file: {e}")
    
    # Create a handler with file size check
    handler = LargeFileHandler()
    
    # Check a large file (this is just a demo, file might not exist)
    large_file = "/opt/sutazaiapp/scripts/deploy_complete_system.sh"
    if os.path.exists(large_file):
        size_bytes, exceeds = handler.check_file_size(large_file)
        print(f"\nFile '{large_file}':")
        print(f"  Size: {size_bytes:,} bytes")
        print(f"  Exceeds 256KB: {exceeds}")


if __name__ == "__main__":
    # Run demos
    demo_basic_usage()
    demo_advanced_usage()
    demo_error_handling()
    
    print("\n=== Demo Complete ===")
    print("You can now use these functions to handle large files in your code!")
    print("Import them like: from backend.utils.large_file_handler import read_large_file")