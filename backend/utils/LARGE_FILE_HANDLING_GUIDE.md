# Large File Handling Guide

When encountering the error:
```
Error: File content (XXX KB) exceeds maximum allowed size (256KB). 
Please use offset and limit parameters to read specific portions of the file, 
or use the GrepTool to search for specific content.
```

## NEW: Automatic Large File Handling Solution

We now have utilities that handle large files automatically! Use these instead of manual pagination:

### Quick Start

```python
from backend.utils.file_reader import read_file, read_file_info, search_in_file

# Read any file, regardless of size (auto-handles chunking)
content = read_file("/path/to/large/file.txt")

# Get file information
info = read_file_info("/path/to/large/file.txt")
print(f"File size: {info['size_mb']:.2f}MB")

# Read specific portions (same as Read tool interface)
content = read_file("/path/to/file.txt", offset=1000, limit=500)

# Search in large files
results = search_in_file("/path/to/file.txt", "error", context_lines=3)
```

### Available Functions

1. **read_file(file_path, offset=None, limit=None)**
   - Automatically handles files of any size
   - Same interface as the Read tool
   - No manual chunking needed!

2. **read_file_info(file_path)**
   - Get file size, line count estimates
   - Check if file exceeds 256KB limit

3. **search_in_file(file_path, pattern, case_sensitive=False, context_lines=2)**
   - Search large files efficiently
   - Returns results with line numbers

### Using the SmartFileReader Class

For more control:

```python
from backend.utils.file_reader import SmartFileReader

reader = SmartFileReader()

# Read entire file
content = reader.read_full("/path/to/large/file.txt")

# Read with specific lines
content = reader.read("/path/to/file.txt", lines=1000, start_line=5000)

# Search with options
results = reader.search("/path/to/file.txt", "pattern", 
                       case_sensitive=True, context=5, max_results=10)
```

## Solution 1: Use Read with offset and limit

The Read tool accepts `offset` and `limit` parameters:
- `offset`: Line number to start reading from (1-indexed)
- `limit`: Maximum number of lines to read

### Examples:

```python
# Read first 100 lines
Read(file_path, offset=1, limit=100)

# Read lines 101-200
Read(file_path, offset=101, limit=100)

# Read lines 1000-1500
Read(file_path, offset=1000, limit=500)
```

### Reading a large file sequentially:

```python
# Step 1: Read first chunk
chunk1 = Read(file_path, offset=1, limit=1000)

# Step 2: Read next chunk
chunk2 = Read(file_path, offset=1001, limit=1000)

# Step 3: Continue as needed
chunk3 = Read(file_path, offset=2001, limit=1000)
```

## Solution 2: Use Grep to search

When you need specific content from a large file, use Grep:

```python
# Search for a specific pattern
Grep(pattern="error|warning", path=file_path, output_mode="content")

# Search with context
Grep(pattern="function_name", path=file_path, output_mode="content", -B=5, -A=5)

# Count occurrences
Grep(pattern="TODO", path=file_path, output_mode="count")
```

## Best Practices

1. **For Code Files**: Use Grep to find specific functions, classes, or patterns
2. **For Log Files**: Use Grep to search for errors, timestamps, or specific events
3. **For Data Files**: Read in chunks using offset/limit
4. **For Configuration Files**: Usually small enough to read entirely, but use Grep if needed

## Example: Reading a 1.7MB file

```python
# Error when trying to read entire file:
Read("/path/to/large_file.txt")
# Error: File content (1.7MB) exceeds maximum allowed size (256KB)

# Solution: Read in chunks
# Chunk 1: Lines 1-1000
Read("/path/to/large_file.txt", offset=1, limit=1000)

# Chunk 2: Lines 1001-2000
Read("/path/to/large_file.txt", offset=1001, limit=1000)

# Or search for specific content
Grep("specific_pattern", path="/path/to/large_file.txt", output_mode="content")
```

## Tips

1. Start with smaller chunks (100-500 lines) to avoid hitting limits
2. Use Grep first to locate relevant sections, then Read specific portions
3. For very large files, consider using head_limit with Grep
4. Remember that offset is 1-indexed (starts at 1, not 0)