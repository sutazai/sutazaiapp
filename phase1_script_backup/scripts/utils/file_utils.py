"""
Shared utilities for scripts to reduce duplication.
"""
from pathlib import Path
import hashlib


def calculate_checksum(file_path: Union[str, Path], algo: str = 'sha256', chunk_size: int = 8192) -> str:
    """
    Calculate a checksum for a file.

    - Uses streaming reads to support large files
    - Defaults to SHA-256
    """
    path = Path(file_path)
    h = hashlib.new(algo)
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            h.update(chunk)
    return h.hexdigest()

