from typing import Dict

#!/usr/bin/env python3.11
"""
Custom Exceptions for SutazAI Agents
"""

from typing import Optional, dict


class DocumentProcessingError(Exception):
    """Exception raised for errors during document processing."""




class OCRFailureError(Exception):
    """Exception raised when OCR processing fails"""




class PDFExtractionError(Exception):
    """Exception raised during PDF text extraction"""




class AgentError(Exception):
    """Exception raised for errors in agent processing."""

    def __init__(self, message: str, task: Optional[Dict] = None):
        super().__init__(message)
        self.task = task or {}
