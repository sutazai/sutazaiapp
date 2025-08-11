#!/usr/bin/env python3
"""
Command-line tool for reading large files.

Usage:
    python readlarge.py <file_path> [options]
    
Options:
    --lines N       Read only first N lines
    --offset N      Start reading from line N
    --search TERM   Search for TERM in the file
    --info          Show file information only
"""

import sys
import argparse
from pathlib import Path
from file_reader import read_file, read_file_info, search_in_file


def main():
    parser = argparse.ArgumentParser(
        description="Read large files with automatic chunking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Read entire file
    python readlarge.py /path/to/large/file.txt
    
    # Read first 1000 lines
    python readlarge.py /path/to/file.txt --lines 1000
    
    # Read lines 5000-6000
    python readlarge.py /path/to/file.txt --offset 5000 --lines 1000
    
    # Search for pattern
    python readlarge.py /path/to/file.txt --search "error"
    
    # Get file info only
    python readlarge.py /path/to/file.txt --info
        """
    )
    
    parser.add_argument('file_path', help='Path to the file to read')
    parser.add_argument('--lines', '-n', type=int, help='Number of lines to read')
    parser.add_argument('--offset', '-o', type=int, help='Starting line number (1-indexed)')
    parser.add_argument('--search', '-s', help='Search for pattern in file')
    parser.add_argument('--info', '-i', action='store_true', help='Show file info only')
    parser.add_argument('--case-sensitive', '-c', action='store_true', 
                       help='Case-sensitive search (default: case-insensitive)')
    parser.add_argument('--context', '-C', type=int, default=2,
                       help='Lines of context for search results (default: 2)')
    
    args = parser.parse_args()
    
    # Validate file path
    file_path = Path(args.file_path)
    if not file_path.exists():
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        sys.exit(1)
    
    try:
        # Show file info
        if args.info or (not args.search and not args.lines and not args.offset):
            info = read_file_info(str(file_path))
            print(f"File: {info['path']}")
            print(f"Size: {info['size_mb']:.2f} MB ({info['size_bytes']:,} bytes)")
            print(f"Exceeds 256KB limit: {info['exceeds_256kb_limit']}")
            print(f"Estimated lines: {info['estimated_lines']:,}")
            print(f"Recommended chunk size: {info['recommended_chunk_size']}")
            
            if args.info:
                return
            
            if info['exceeds_256kb_limit']:
                print("Use --lines to read a portion, or --search to find specific content.")
                if not args.lines and not args.offset and not args.search:
                    print("\nReading first 100 lines as preview...")
                    args.lines = 100
                    args.offset = 1
        
        # Search in file
        if args.search:
            print(f"Searching for '{args.search}' in {file_path}...")
            results = search_in_file(
                str(file_path), 
                args.search,
                case_sensitive=args.case_sensitive,
                context_lines=args.context
            )
            
            if not results:
                print(f"No matches found for '{args.search}'")
            else:
                print(f"\nFound {len(results)} matches:\n")
                
                prev_line = -999
                for result in results[:50]:  # Limit to first 50 results
                    line_num = result['line_number']
                    content = result['content']
                    
                    # Add separator for non-consecutive lines
                    if line_num > prev_line + 1:
                        print("...")
                    
                    # Highlight the search term
                    if not args.case_sensitive:
                        # Case-insensitive highlighting
                        import re
                        highlighted = re.sub(
                            f'({re.escape(args.search)})', 
                            r'>>>\1<<<', 
                            content, 
                            flags=re.IGNORECASE
                        )
                    else:
                        highlighted = content.replace(args.search, f'>>>{args.search}<<<')
                    
                    print(f"{line_num:6d}\t{highlighted}")
                    prev_line = line_num
                
                if len(results) > 50:
                    print(f"\n... and {len(results) - 50} more matches")
        
        # Read file content
        elif args.lines or args.offset:
            offset = args.offset or 1
            limit = args.lines
            
            print(f"Reading from line {offset}" + 
                  (f" ({limit} lines)" if limit else ""))
            
            content = read_file(str(file_path), offset=offset, limit=limit)
            print(content)
        
        else:
            # Read entire file (if not too large)
            info = read_file_info(str(file_path))
            if info['size_mb'] > 5:
                print(f"Warning: File is {info['size_mb']:.1f}MB. Reading entire file...")
                response = input("Continue? (y/N): ")
                if response.lower() != 'y':
                    print("Aborted.")
                    return
            
            content = read_file(str(file_path))
            print(content)
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
