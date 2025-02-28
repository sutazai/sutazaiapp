# New utility module
import logging
from typing import Any


def safe_file_read(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def safe_file_write(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


# Create shared utilities to reduce duplication
def safe_file_op(file_path: str, mode: str, operation: callable) -> Any:
    try:
        with open(file_path, mode, encoding="utf-8") as f:
            return operation(f)
    except Exception as e:
        logging.error(f"File operation failed on {file_path}: {str(e)}")
        raise
