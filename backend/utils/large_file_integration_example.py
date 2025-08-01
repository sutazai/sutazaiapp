"""
Example integration of the large file handler into existing backend services.

This shows how to integrate the large file handling utilities into your
existing API endpoints and services.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List, Dict, Any
from pathlib import Path

from .file_reader import read_file, read_file_info, search_in_file, safe_read_file


# Example API router for file operations
router = APIRouter(prefix="/api/v1/files", tags=["files"])


@router.get("/read")
async def read_file_endpoint(
    file_path: str,
    offset: Optional[int] = Query(None, description="Starting line number (1-indexed)"),
    limit: Optional[int] = Query(None, description="Maximum number of lines to read")
):
    """
    Read a file with automatic large file handling.
    
    This endpoint automatically handles files larger than 256KB by using
    chunked reading behind the scenes.
    """
    try:
        # Validate file path (add your own validation logic)
        path = Path(file_path)
        if not path.is_absolute():
            raise HTTPException(status_code=400, detail="File path must be absolute")
        
        # Get file info first
        info = read_file_info(file_path)
        
        # Read the file
        content = read_file(file_path, offset=offset, limit=limit)
        
        return {
            "file_path": file_path,
            "size_mb": info['size_mb'],
            "exceeds_limit": info['exceeds_256kb_limit'],
            "content": content,
            "lines_returned": len(content.split('\n')) if content else 0
        }
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")


@router.get("/info")
async def get_file_info_endpoint(file_path: str):
    """Get information about a file including size and line estimates."""
    try:
        info = read_file_info(file_path)
        return info
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting file info: {str(e)}")


@router.get("/search")
async def search_file_endpoint(
    file_path: str,
    pattern: str,
    case_sensitive: bool = Query(False, description="Case-sensitive search"),
    context_lines: int = Query(2, description="Lines of context around matches"),
    max_results: int = Query(100, description="Maximum results to return")
):
    """Search for a pattern in a file, handling large files automatically."""
    try:
        results = search_in_file(file_path, pattern, 
                               case_sensitive=case_sensitive,
                               context_lines=context_lines)
        
        # Limit results
        if len(results) > max_results:
            results = results[:max_results]
            
        return {
            "file_path": file_path,
            "pattern": pattern,
            "matches_found": len(results),
            "results": results
        }
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching file: {str(e)}")


# Example service class that uses the file reader
class DocumentService:
    """Example service that processes documents of any size."""
    
    def process_document(self, file_path: str) -> Dict[str, Any]:
        """Process a document, handling large files automatically."""
        try:
            # Get file info
            info = read_file_info(file_path)
            
            # For large files, process in chunks
            if info['exceeds_256kb_limit']:
                return self._process_large_document(file_path, info)
            else:
                # Small file, read it all
                content = read_file(file_path)
                return self._analyze_content(content, info)
                
        except Exception as e:
            raise RuntimeError(f"Error processing document: {e}")
    
    def _process_large_document(self, file_path: str, info: Dict[str, Any]) -> Dict[str, Any]:
        """Process a large document in chunks."""
        # Example: Count lines, search for patterns, etc.
        total_lines = 0
        error_count = 0
        warning_count = 0
        
        # Read in chunks of 1000 lines
        chunk_size = 1000
        offset = 1
        
        while offset <= info['estimated_lines']:
            try:
                content = read_file(file_path, offset=offset, limit=chunk_size)
                lines = content.split('\n')
                total_lines += len(lines)
                
                # Count errors and warnings
                for line in lines:
                    if 'error' in line.lower():
                        error_count += 1
                    if 'warning' in line.lower():
                        warning_count += 1
                
                offset += chunk_size
                
            except Exception:
                break  # Reached end of file
        
        return {
            "file_path": file_path,
            "size_mb": info['size_mb'],
            "total_lines": total_lines,
            "errors_found": error_count,
            "warnings_found": warning_count
        }
    
    def _analyze_content(self, content: str, info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze file content."""
        lines = content.split('\n')
        return {
            "file_path": info['path'],
            "size_mb": info['size_mb'],
            "total_lines": len(lines),
            "first_line": lines[0] if lines else "",
            "last_line": lines[-1] if lines else ""
        }


# Example: Integration with existing file processing code
def upgrade_legacy_file_reader(file_path: str) -> str:
    """
    Drop-in replacement for legacy file reading code.
    
    Before:
        with open(file_path, 'r') as f:
            content = f.read()  # This fails for large files!
            
    After:
        content = upgrade_legacy_file_reader(file_path)
    """
    return safe_read_file(file_path)


# Example: Batch processing with progress
def process_multiple_files(file_paths: List[str]) -> List[Dict[str, Any]]:
    """Process multiple files, handling large files automatically."""
    results = []
    
    for i, file_path in enumerate(file_paths):
        print(f"Processing file {i+1}/{len(file_paths)}: {file_path}")
        
        try:
            info = read_file_info(file_path)
            
            if info['exceeds_256kb_limit']:
                print(f"  Large file detected ({info['size_mb']:.1f}MB), using chunked processing...")
            
            # Process based on file type
            if file_path.endswith('.log'):
                # Search for errors in log files
                errors = search_in_file(file_path, "error", case_sensitive=False)
                results.append({
                    "file": file_path,
                    "type": "log",
                    "errors_found": len(errors)
                })
            else:
                # Read first 1000 lines for other files
                content = read_file(file_path, offset=1, limit=1000)
                results.append({
                    "file": file_path,
                    "type": "other",
                    "preview_lines": len(content.split('\n'))
                })
                
        except Exception as e:
            results.append({
                "file": file_path,
                "error": str(e)
            })
    
    return results


if __name__ == "__main__":
    # Example usage
    print("Large File Integration Examples")
    print("=" * 50)
    
    # Test with a sample file
    test_file = "/opt/sutazaiapp/scripts/deploy_complete_system.sh"
    
    # 1. Get file info
    info = read_file_info(test_file)
    print(f"\nFile: {test_file}")
    print(f"Size: {info['size_mb']:.2f}MB")
    print(f"Exceeds 256KB limit: {info['exceeds_256kb_limit']}")
    
    # 2. Read first 10 lines
    print("\nFirst 10 lines:")
    content = read_file(test_file, offset=1, limit=10)
    print(content)
    
    # 3. Search for pattern
    print("\nSearching for 'docker':")
    results = search_in_file(test_file, "docker", context_lines=1)
    print(f"Found {len(results)} matches")
    if results:
        print(f"First match at line {results[0]['line_number']}: {results[0]['content']}")