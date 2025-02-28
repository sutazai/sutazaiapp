"""
Type stubs for external libraries used in the document processor module.
This file helps MyPy understand the types of external libraries that don't provide their own type stubs.
"""

from typing import Any, Dict, List, Tuple, Callable

# Type stubs for fitz (PyMuPDF)
class Rect:
    """Rectangle in a PDF page."""

    x0: float
    y0: float
    x1: float
    y1: float
    width: float
    height: float

class Page:
    """PDF page object."""

    rect: Rect

    def get_text(self, option: str = ...) -> str: ...
    def get_images(self) -> List[Tuple[int, ...]]: ...
    def get_image_bbox(self, xref: int) -> Rect: ...
    def insert_text(self, point: Tuple[float, float], text: str, **kwargs) -> None: ...

class Document:
    """PDF document object."""

    metadata: Dict[str, Any]

    def __init__(self) -> None: ...
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> Page: ...
    def save(self, filename: str) -> None: ...
    def close(self) -> None: ...
    def new_page(self, width: float = ..., height: float = ...) -> Page: ...

def open(filename: str = ...) -> Document: ...

# Type stubs for cv2
def imread(filename: str) -> Any: ...
def imwrite(filename: str, img: Any) -> bool: ...
def cvtColor(src: Any, code: int) -> Any: ...
def fastNlMeansDenoising(src: Any, **kwargs) -> Any: ...
def threshold(src: Any, thresh: float, maxval: float, type: int) -> Tuple[float, Any]: ...
def putText(
    img: Any,
    text: str,
    org: Tuple[int, int],
    fontFace: int,
    fontScale: float,
    color: Tuple[int, int, int],
    thickness: int = ...,
) -> None: ...

# Constants for cv2
COLOR_BGR2GRAY: int
FONT_HERSHEY_SIMPLEX: int
THRESH_BINARY: int
THRESH_OTSU: int

# Type stubs for numpy
class ndarray:
    """NumPy array."""

    shape: Tuple[int, ...]
    dtype: Any

def zeros(shape: Tuple[int, ...], dtype: Any = ...) -> ndarray: ...

# Type stubs for pytesseract
def image_to_string(image: Any, lang: str = ..., **kwargs) -> str: ...

# Type stubs for pytest
class FixtureRequest:
    """Pytest fixture request object."""

    param: Any

class Config:
    """Pytest configuration object."""

    def addinivalue_line(self, name: str, line: str) -> None: ...

def fixture(scope: str = ..., params: List[Any] = ..., autouse: bool = ..., ids: List[str] = ...) -> Callable: ...
def raises(expected_exception: Any) -> Any: ...
def main(args: List[str] = ...) -> int: ...

# Type stubs for loguru
class Logger:
    """Loguru logger object."""

    def info(self, message: str) -> None: ...
    def error(self, message: str) -> None: ...
    def warning(self, message: str) -> None: ...
    def debug(self, message: str) -> None: ...
    def add(self, sink: Any, **kwargs) -> int: ...

logger: Logger

# Type stubs for magic
class Magic:
    """Magic file type detection."""

    def __init__(self, mime: bool = ...) -> None: ...
    def from_file(self, filename: str) -> str: ...
