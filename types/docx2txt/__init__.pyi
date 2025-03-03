from typing import Any, Dict, Optional, Union, List

def process(docx_file: str, img_dir: Optional[str] = None) -> str:
    """
    Extract text from a .docx file.
    
    Args:
        docx_file: Path to the .docx file to extract text from.
        img_dir: Directory to extract images to. If None, images will not be extracted.
    
    Returns:
        The extracted text content.
    """
    ...

# Allow the module to be None in conditional imports
docx2txt: Optional['_Docx2txtModule']

class _Docx2txtModule:
    """Type stub for docx2txt module when it's imported"""
    
    @staticmethod
    def process(docx_file: str, img_dir: Optional[str] = None) -> str: ... 