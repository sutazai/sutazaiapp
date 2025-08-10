#!/usr/bin/env python3
"""
Large File Handler Demo

Purpose: Demonstrates usage of `backend.utils.large_file_handler` for files exceeding 256KB.
Author: Repo Maintenance
Date: 2025-08-09
Usage:
  python3 scripts/utils/large_file_handler_demo.py

Notes:
- Reads and searches demo files if present; safe to run locally.
"""

import os
from backend.utils.large_file_handler import (
    read_large_file,
    read_file_lines,
    search_large_file,
    get_file_info,
    LargeFileHandler,
)


def demo_basic_usage():
    print("=== Large File Handler Demo ===\n")

    test_file = "/opt/sutazaiapp/backend/app/working_main.py"

    print("1. Getting file information:")
    info = get_file_info(test_file)
    print(f"   File: {info['path']}")
    print(f"   Size: {info['size_mb']:.2f} MB ({info['size_bytes']:,} bytes)")
    print(f"   Exceeds 256KB limit: {info['exceeds_256kb_limit']}")
    print(f"   Estimated lines: {info['estimated_lines']:,}")
    print(f"   Recommended chunk size: {info['recommended_chunk_size']}")
    print()

    print("2. Reading first 50 lines:")
    content = read_large_file(test_file, max_lines=50)
    print((content[:500] + "...\n") if content else "(no content)\n")

    print("3. Reading lines 100-110:")
    lines = read_file_lines(test_file, start_line=100, num_lines=10)
    print(lines)
    print()

    print("4. Searching for 'async' in file:")
    results = search_large_file(test_file, "async", context_lines=1)
    preview = results[:1000] + "...\n" if len(results) > 1000 else results + "\n"
    print(preview)


def demo_advanced_usage():
    print("=== Advanced Usage Demo ===\n")

    handler = LargeFileHandler(chunk_size=500)
    test_file = "/opt/sutazaiapp/backend/app/working_main.py"

    print("Processing file in chunks:")
    total_lines = 0
    total_chunks = 0

    for chunk_lines, start_line, end_line in handler.read_file_in_chunks(test_file):
        total_chunks += 1
        total_lines += len(chunk_lines)
        print(f"  Chunk {total_chunks}: lines {start_line}-{end_line} ({len(chunk_lines)} lines)")

        if total_chunks >= 5:
            print("  ... (stopping after 5 chunks for demo)")
            break

    print(f"\nProcessed {total_chunks} chunks, {total_lines} lines total")


def demo_error_handling():
    print("\n=== Error Handling Demo ===\n")

    try:
        _ = read_large_file("/non/existent/file.txt")
    except Exception as e:
        print(f"Expected error for non-existent file: {e}")

    handler = LargeFileHandler()
    large_file = "/opt/sutazaiapp/scripts/deploy_complete_system.sh"
    if os.path.exists(large_file):
        size_bytes, exceeds = handler.check_file_size(large_file)
        print(f"\nFile '{large_file}':")
        print(f"  Size: {size_bytes:,} bytes")
        print(f"  Exceeds 256KB: {exceeds}")


if __name__ == "__main__":
    demo_basic_usage()
    demo_advanced_usage()
    demo_error_handling()

    print("\n=== Demo Complete ===")
    print("You can now use these functions to handle large files in your code!")
    print("Import them like: from backend.utils.large_file_handler import read_large_file")

