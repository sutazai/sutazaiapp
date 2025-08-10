"""
Large File Handler Utility

This module provides utilities for handling large files that exceed the 256KB limit
of the standard Read tool. It implements automatic chunking and transparent pagination.
"""

import os
from typing import Optional, Generator, List, Tuple
from pathlib import Path


class LargeFileHandler:
    """Handler for reading large files with automatic chunking."""
    
    # Default chunk size in lines (adjust based on typical line length)
    DEFAULT_CHUNK_SIZE = 1000
    
    # Maximum file size we'll handle (5MB)
    MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB in bytes
    
    # Estimated bytes per line (used for size estimation)
    ESTIMATED_BYTES_PER_LINE = 100
    
    def __init__(self, chunk_size: int = DEFAULT_CHUNK_SIZE):
        """
        Initialize the large file handler.
        
        Args:
            chunk_size: Number of lines to read per chunk
        """
        self.chunk_size = chunk_size
    
    def check_file_size(self, file_path: str) -> Tuple[int, bool]:
        """
        Check if a file exceeds the size limit.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Tuple of (file_size_bytes, exceeds_limit)
        """
        try:
            size = os.path.getsize(file_path)
            # 256KB limit from the Read tool
            exceeds_limit = size > 256 * 1024
            return size, exceeds_limit
        except OSError:
            return 0, False
    
    def estimate_total_lines(self, file_path: str) -> int:
        """
        Estimate the total number of lines in a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Estimated number of lines
        """
        size_bytes, _ = self.check_file_size(file_path)
        return max(1, size_bytes // self.ESTIMATED_BYTES_PER_LINE)
    
    def read_file_in_chunks(self, file_path: str, 
                           start_line: Optional[int] = None,
                           end_line: Optional[int] = None) -> Generator[Tuple[List[str], int, int], None, None]:
        """
        Read a file in chunks, yielding each chunk with line numbers.
        
        Args:
            file_path: Path to the file
            start_line: Starting line number (1-indexed), None for beginning
            end_line: Ending line number (1-indexed), None for end
            
        Yields:
            Tuple of (lines, chunk_start_line, chunk_end_line)
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Check file size
        size_bytes, exceeds_limit = self.check_file_size(file_path)
        if size_bytes > self.MAX_FILE_SIZE:
            raise ValueError(f"File too large ({size_bytes / 1024 / 1024:.1f}MB). Maximum supported size is {self.MAX_FILE_SIZE / 1024 / 1024}MB")
        
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            current_line = 1
            chunk_lines = []
            chunk_start = current_line
            
            for line in f:
                # Skip lines before start_line
                if start_line and current_line < start_line:
                    current_line += 1
                    continue
                
                # Stop if we've reached end_line
                if end_line and current_line > end_line:
                    break
                
                chunk_lines.append(line.rstrip('\n'))
                
                # Yield chunk when it reaches chunk_size
                if len(chunk_lines) >= self.chunk_size:
                    yield chunk_lines, chunk_start, current_line
                    chunk_lines = []
                    chunk_start = current_line + 1
                
                current_line += 1
            
            # Yield any remaining lines
            if chunk_lines:
                yield chunk_lines, chunk_start, current_line - 1
    
    def read_entire_file(self, file_path: str, max_lines: Optional[int] = None) -> str:
        """
        Read an entire file, handling large files automatically.
        
        Args:
            file_path: Path to the file
            max_lines: Maximum number of lines to read (None for all)
            
        Returns:
            File content as a string with line numbers
        """
        lines_read = 0
        result_lines = []
        
        try:
            for chunk_lines, start_line, end_line in self.read_file_in_chunks(file_path):
                for i, line in enumerate(chunk_lines):
                    if max_lines and lines_read >= max_lines:
                        result_lines.append(f"\n... (truncated at {max_lines} lines)")
                        return '\n'.join(result_lines)
                    
                    # Format with line numbers similar to 'cat -n'
                    line_num = start_line + i
                    result_lines.append(f"{line_num:6d}\t{line}")
                    lines_read += 1
                    
        except Exception as e:
            raise RuntimeError(f"Error reading file: {e}")
        
        return '\n'.join(result_lines)
    
    def search_in_file(self, file_path: str, search_term: str, 
                      case_sensitive: bool = False,
                      context_lines: int = 0) -> List[Tuple[int, str]]:
        """
        Search for a term in a large file.
        
        Args:
            file_path: Path to the file
            search_term: Term to search for
            case_sensitive: Whether search should be case-sensitive
            context_lines: Number of context lines before/after match
            
        Returns:
            List of tuples (line_number, line_content) for matching lines
        """
        matches = []
        search_lower = search_term if case_sensitive else search_term.lower()
        
        # Store lines for context
        line_buffer = []
        
        for chunk_lines, start_line, end_line in self.read_file_in_chunks(file_path):
            for i, line in enumerate(chunk_lines):
                line_num = start_line + i
                line_to_search = line if case_sensitive else line.lower()
                
                # Add to buffer for context
                line_buffer.append((line_num, line))
                if len(line_buffer) > context_lines * 2 + 1:
                    line_buffer.pop(0)
                
                if search_lower in line_to_search:
                    # Add context lines before
                    start_idx = max(0, len(line_buffer) - context_lines - 1)
                    end_idx = len(line_buffer)
                    
                    for idx in range(start_idx, end_idx):
                        if idx < len(line_buffer):
                            matches.append(line_buffer[idx])
        
        return matches


# Convenience functions for direct use

def read_large_file(file_path: str, chunk_size: int = 1000, max_lines: Optional[int] = None) -> str:
    """
    Read a potentially large file with automatic chunking.
    
    Args:
        file_path: Path to the file
        chunk_size: Lines per chunk (default 1000)
        max_lines: Maximum lines to read (None for all)
        
    Returns:
        File content with line numbers
    """
    handler = LargeFileHandler(chunk_size=chunk_size)
    return handler.read_entire_file(file_path, max_lines=max_lines)


def read_file_lines(file_path: str, start_line: int, num_lines: int) -> str:
    """
    Read specific lines from a file.
    
    Args:
        file_path: Path to the file
        start_line: Starting line number (1-indexed)
        num_lines: Number of lines to read
        
    Returns:
        Requested lines with line numbers
    """
    handler = LargeFileHandler()
    end_line = start_line + num_lines - 1
    
    result_lines = []
    for chunk_lines, chunk_start, chunk_end in handler.read_file_in_chunks(
        file_path, start_line=start_line, end_line=end_line
    ):
        for i, line in enumerate(chunk_lines):
            line_num = chunk_start + i
            result_lines.append(f"{line_num:6d}\t{line}")
    
    return '\n'.join(result_lines)


def search_large_file(file_path: str, search_term: str, 
                     case_sensitive: bool = False,
                     context_lines: int = 2) -> str:
    """
    Search in a potentially large file.
    
    Args:
        file_path: Path to the file
        search_term: Term to search for
        case_sensitive: Whether search is case-sensitive
        context_lines: Lines of context around matches
        
    Returns:
        Search results with line numbers
    """
    handler = LargeFileHandler()
    matches = handler.search_in_file(file_path, search_term, 
                                   case_sensitive=case_sensitive,
                                   context_lines=context_lines)
    
    if not matches:
        return f"No matches found for '{search_term}'"
    
    result_lines = [f"Found {len(matches)} matches for '{search_term}':\n"]
    
    prev_line_num = -999
    for line_num, line_content in matches:
        # Add separator between non-consecutive matches
        if line_num > prev_line_num + 1:
            result_lines.append("...")
        
        result_lines.append(f"{line_num:6d}\t{line_content}")
        prev_line_num = line_num
    
    return '\n'.join(result_lines)


def get_file_info(file_path: str) -> dict:
    """
    Get information about a file including size and estimated lines.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Dictionary with file information
    """
    handler = LargeFileHandler()
    size_bytes, exceeds_limit = handler.check_file_size(file_path)
    estimated_lines = handler.estimate_total_lines(file_path)
    
    return {
        'path': file_path,
        'size_bytes': size_bytes,
        'size_mb': size_bytes / 1024 / 1024,
        'exceeds_256kb_limit': exceeds_limit,
        'estimated_lines': estimated_lines,
        'recommended_chunk_size': min(1000, estimated_lines // 10)
    }