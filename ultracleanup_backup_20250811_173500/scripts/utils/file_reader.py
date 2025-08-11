"""
Enhanced File Reader with automatic large file handling.

This module provides a unified interface for reading files of any size,
automatically handling the 256KB limitation by using chunking when needed.
"""

from typing import Optional, Union, List, Dict, Any
from pathlib import Path
from .large_file_handler import LargeFileHandler, get_file_info


class SmartFileReader:
    """
    Smart file reader that automatically handles files of any size.
    
    This reader checks file size and automatically uses the appropriate
    method to read files, whether they're small (< 256KB) or large.
    """
    
    # 256KB limit (in bytes)
    STANDARD_READ_LIMIT = 256 * 1024
    
    def __init__(self):
        self.large_handler = LargeFileHandler()
    
    def read(self, 
             file_path: Union[str, Path], 
             lines: Optional[int] = None,
             start_line: Optional[int] = None,
             end_line: Optional[int] = None) -> str:
        """
        Read a file intelligently, handling large files automatically.
        
        Args:
            file_path: Path to the file
            lines: Maximum number of lines to read (takes precedence over end_line)
            start_line: Starting line number (1-indexed)
            end_line: Ending line number (1-indexed)
            
        Returns:
            File content with line numbers
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If parameters are invalid
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Get file info
        info = get_file_info(str(file_path))
        
        # If file is small enough, we could use standard read
        # But we'll use our handler for consistency
        
        if lines and start_line:
            # Read specific number of lines from start_line
            end_line = start_line + lines - 1
        elif lines and not start_line:
            # Read first N lines
            start_line = 1
            end_line = lines
        
        # Use the large file handler
        return self._read_with_handler(str(file_path), start_line, end_line)
    
    def _read_with_handler(self, 
                          file_path: str, 
                          start_line: Optional[int] = None,
                          end_line: Optional[int] = None) -> str:
        """Internal method to read using the large file handler."""
        result_lines = []
        
        for chunk_lines, chunk_start, chunk_end in self.large_handler.read_file_in_chunks(
            file_path, start_line=start_line, end_line=end_line
        ):
            for i, line in enumerate(chunk_lines):
                line_num = chunk_start + i
                result_lines.append(f"{line_num:6d}\t{line}")
        
        return '\n'.join(result_lines)
    
    def read_full(self, file_path: Union[str, Path]) -> str:
        """
        Read entire file content regardless of size.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Complete file content with line numbers
            
        Warning:
            Use with caution on very large files (> 5MB)
        """
        return self.large_handler.read_entire_file(str(file_path))
    
    def search(self, 
               file_path: Union[str, Path], 
               pattern: str,
               case_sensitive: bool = False,
               context: int = 2,
               max_results: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Search for pattern in file.
        
        Args:
            file_path: Path to the file
            pattern: Pattern to search for
            case_sensitive: Whether search is case-sensitive
            context: Lines of context around matches
            max_results: Maximum number of results to return
            
        Returns:
            List of match dictionaries with line numbers and content
        """
        matches = self.large_handler.search_in_file(
            str(file_path), 
            pattern, 
            case_sensitive=case_sensitive,
            context_lines=context
        )
        
        # Format results
        results = []
        for line_num, line_content in matches:
            results.append({
                'line_number': line_num,
                'content': line_content,
                'file': str(file_path)
            })
            
            if max_results and len(results) >= max_results:
                break
        
        return results
    
    def get_info(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get detailed information about a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary with file information
        """
        return get_file_info(str(file_path))


# Global instance for convenience
_reader = SmartFileReader()


# Convenience functions that mirror the standard Read tool interface
def read_file(file_path: Union[str, Path], 
              offset: Optional[int] = None,
              limit: Optional[int] = None) -> str:
    """
    Read a file with automatic large file handling.
    
    This function provides a similar interface to the standard Read tool
    but handles large files automatically.
    
    Args:
        file_path: Path to the file
        offset: Starting line number (1-indexed), similar to Read tool
        limit: Maximum number of lines to read
        
    Returns:
        File content with line numbers
        
    Example:
        # Read entire file (auto-handles large files)
        content = read_file("/path/to/large/file.txt")
        
        # Read first 1000 lines
        content = read_file("/path/to/large/file.txt", offset=1, limit=1000)
        
        # Read lines 5000-6000
        content = read_file("/path/to/large/file.txt", offset=5000, limit=1000)
    """
    return _reader.read(file_path, start_line=offset, lines=limit)


def read_file_info(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Get information about a file including size and line count estimates."""
    return _reader.get_info(file_path)


def search_in_file(file_path: Union[str, Path], 
                  pattern: str,
                  case_sensitive: bool = False,
                  context_lines: int = 2) -> List[Dict[str, Any]]:
    """Search for a pattern in a file, handling large files automatically."""
    return _reader.search(file_path, pattern, 
                         case_sensitive=case_sensitive, 
                         context=context_lines)


# For backward compatibility and as a drop-in replacement
def safe_read_file(file_path: Union[str, Path]) -> str:
    """
    Safely read any file, regardless of size.
    
    This function can be used as a drop-in replacement for standard file reading
    when you're not sure about file size.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File content with line numbers
    """
    try:
        info = read_file_info(file_path)
        
        if info['exceeds_256kb_limit']:
            print(f"Note: File is {info['size_mb']:.1f}MB, using chunked reading...")
            
        return read_file(file_path)
        
    except Exception as e:
        raise RuntimeError(f"Error reading file {file_path}: {e}")


if __name__ == "__main__":
    # Quick test
    import sys
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        print(f"Reading file: {file_path}")
        
        info = read_file_info(file_path)
        print(f"File size: {info['size_mb']:.2f}MB")
        print(f"Exceeds limit: {info['exceeds_256kb_limit']}")
        print(f"Estimated lines: {info['estimated_lines']:,}")
        
        print("\nFirst 20 lines:")
        print(read_file(file_path, offset=1, limit=20))
    else:
        print("Usage: python file_reader.py <file_path>")