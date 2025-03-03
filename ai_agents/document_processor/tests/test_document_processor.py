#!/usr/bin/env python3.11
import os
import tempfile
from collections.abc import Generator
import cv2  # type: ignore
import numpy as np
import pytest
from ai_agents.document_processor.src import DocumentProcessorAgent
from ai_agents.document_processor.utils.document_utils import DocumentUtils
from ai_agents.exceptions import AgentError

@pytest.fixture
def sample_pdf_path() -> Generator[str, None, None]:
    """
    Fixture to generate a sample PDF for testing
    Returns:
        str: Path to the sample PDF
    """
    import fitz  # type: ignore
    # Create a temporary PDF
    temp_pdf = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)  # A4 size
    page.insert_text((50, 50), "SutazAI Document Processing Test")
    doc.save(temp_pdf.name)
    doc.close()
    yield temp_pdf.name
    # Cleanup
    os.unlink(temp_pdf.name)

@pytest.fixture
def sample_image_path() -> Generator[str, None, None]:
    """
    Fixture to generate a sample image for testing
    Returns:
        str: Path to the sample image
    """
    # Create a temporary image
    temp_img = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    image = np.zeros((200, 200), dtype=np.uint8)
    cv2.putText(
        image,
        "SutazAI OCR Test",
        (10, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),  # Use a tuple for color in BGR format
        2,
    )
    cv2.imwrite(temp_img.name, image)
    yield temp_img.name
    # Cleanup
    os.unlink(temp_img.name)

class TestDocumentProcessorAgent:
    """
    Comprehensive Test Suite for Document Processor Agent
    Covers various scenarios and edge cases for document processing
    """
    def test_document_extraction(self, sample_pdf_path: str):
        """Test PDF text extraction functionality"""
        agent = DocumentProcessorAgent()
        task = {
            "document_path": sample_pdf_path,
            "operation": "extract_text",
            "parameters": {},
        }
        result = agent.execute(task)
        assert result["status"] == "success"
        assert len(result["text"]) > 0
        assert result["total_pages"] == 1
        assert "SutazAI Document Processing Test" in result["text"][0]

    def test_ocr_processing(self, sample_image_path: str):
        """Test OCR processing functionality"""
        agent = DocumentProcessorAgent()
        task = {
            "document_path": sample_image_path,
            "operation": "ocr",
            "parameters": {"languages": ["eng"]},
        }
        result = agent.execute(task)
        assert result["status"] == "success"
        assert "SutazAI OCR Test" in result["text"]

    def test_document_analysis(self, sample_pdf_path: str):
        """Test advanced document analysis functionality"""
        agent = DocumentProcessorAgent()
        task = {
            "document_path": sample_pdf_path,
            "operation": "analyze",
            "parameters": {},
        }
        result = agent.execute(task)
        assert result["status"] == "success"
        assert result["analysis"]["total_pages"] == 1
        assert len(result["analysis"]["text_blocks"]) > 0

    def test_invalid_document_path(self):
        """Test handling of invalid document path"""
        agent = DocumentProcessorAgent()
        task = {
            "document_path": "/path/to/nonexistent/document.pdf",
            "operation": "extract_text",
            "parameters": {},
        }
        with pytest.raises(AgentError) as excinfo:
            agent.execute(task)
        assert "Document processing failed" in str(excinfo.value)

    def test_unsupported_task_type(self, sample_pdf_path: str):
        """Test handling of unsupported task type"""
        agent = DocumentProcessorAgent()
        task = {
            "document_path": sample_pdf_path,
            "operation": "unsupported_task",
            "parameters": {},
        }
        with pytest.raises(AgentError) as excinfo:
            agent.execute(task)
        assert "Unsupported operation" in str(excinfo.value)

    def test_document_utils_validation(self, sample_pdf_path: str):
        """Test document validation utility"""
        validation_result = DocumentUtils.validate_document(sample_pdf_path)
        assert validation_result["status"] == "success"
        assert "metadata" in validation_result
        assert "file_path" in validation_result["metadata"]
        assert "mime_type" in validation_result["metadata"]
        assert "file_size_bytes" in validation_result["metadata"]
        assert "file_hash" in validation_result["metadata"]

    def test_document_utils_metadata(self, sample_pdf_path: str):
        """Test document metadata extraction utility"""
        metadata_result = DocumentUtils.extract_document_metadata(sample_pdf_path)
        assert metadata_result["status"] == "success"
        assert "metadata" in metadata_result
        assert "total_pages" in metadata_result["metadata"]
        assert "page_dimensions" in metadata_result["metadata"]

    def test_document_language_detection(self):
        """Test document language detection utility"""
        test_text = "This is a test document in English."
        languages = DocumentUtils.detect_document_language(test_text)
        assert isinstance(languages, list)
        assert "en" in languages[0]


def pytest_configure(config):
    """Configure pytest for comprehensive reporting"""
    config.addinivalue_line(
        "markers",
        "document_processor: mark test as a document processor agent test",
    )


if __name__ == "__main__":
    pytest.main(
        [
            "-v",  # Verbose output
            "--tb=short",  # Shorter traceback format
        ],
    )

            
