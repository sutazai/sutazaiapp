#!/usr/bin/env python
"""
SutazaiApp Installation Test Script
This script performs basic tests to verify that the SutazaiApp installation
is working correctly on the test server.
"""

import os
import sys
import logging
import requests
import tempfile

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("test_installation")

# Define API base URL
API_BASE_URL = "http://localhost:8000"


def test_health_endpoint():
    """Test the health endpoint to verify the API is running."""
    try:
        logger.info("Testing health endpoint...")
        response = requests.get(f"{API_BASE_URL}/health", timeout=30)
        response.raise_for_status()
        data = response.json()

        assert data["status"] == "ok", "Health status is not 'ok'"
        logger.info("✓ Health endpoint test passed")

        # Log services status
        if "services" in data:
            for service, status in data["services"].items():
                logger.info(
                    f"  - {service}: {'✓ Available' if status else '✗ Not available'}"
                )

        return True
    except Exception as e:
        logger.error(f"✗ Health endpoint test failed: {e}")
        return False


def test_vector_search():
    """Test the vector search endpoint."""
    try:
        logger.info("Testing vector search endpoint...")
        payload = {"query": "test query", "limit": 3}
        response = requests.post(
            f"{API_BASE_URL}/vector/search", json=payload, timeout=30
        )
        response.raise_for_status()
        data = response.json()

        assert "results" in data, "Results key missing in response"
        assert "search_time_ms" in data, "Search time missing in response"

        logger.info(
            f"✓ Vector search test passed - found {len(data['results'])} results in {data['search_time_ms']}ms"
        )
        return True
    except Exception as e:
        logger.error(f"✗ Vector search test failed: {e}")
        return False


def test_code_generation():
    """Test the code generation endpoint if available."""
    try:
        logger.info("Testing code generation endpoint...")
        payload = {
            "spec_text": "Create a function that calculates factorial of a number",
            "language": "python",
        }
        response = requests.post(
            f"{API_BASE_URL}/code/generate", json=payload, timeout=60
        )
        response.raise_for_status()
        data = response.json()

        assert "generated_code" in data, "Generated code missing in response"
        logger.info(
            f"✓ Code generation test passed - generated code in {data.get('generation_time_ms', 0)}ms"
        )
        return True
    except Exception as e:
        logger.error(f"✗ Code generation test failed: {e}")
        return False


def create_test_pdf():
    """Create a simple test PDF file for document processing."""
    try:
        import fitz  # PyMuPDF

        # Create a simple PDF with text
        doc = fitz.open()
        page = doc.new_page()
        text = "This is a test PDF document for SutazaiApp."
        page.insert_text((50, 50), text, fontsize=12)

        # Save to temp file
        temp_file = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        doc.save(temp_file.name)
        doc.close()

        logger.info(f"Created test PDF at {temp_file.name}")
        return temp_file.name
    except Exception as e:
        logger.error(f"Failed to create test PDF: {e}")
        return None


def test_document_processing():
    """Test the document processing endpoint."""
    pdf_path = create_test_pdf()
    if not pdf_path:
        return False

    try:
        logger.info("Testing document processing endpoint...")
        with open(pdf_path, "rb") as pdf:
            files = {"file": ("test.pdf", pdf, "application/pdf")}
            response = requests.post(
                f"{API_BASE_URL}/documents/process",
                files=files,
                data={"store_vectors": "true"},
                timeout=120,
            )
        response.raise_for_status()
        data = response.json()

        assert "document_id" in data, "Document ID missing in response"
        assert "metadata" in data, "Metadata missing in response"
        assert "full_text" in data, "Full text missing in response"

        logger.info(
            f"✓ Document processing test passed - Document ID: {data['document_id']}"
        )
        return True
    except Exception as e:
        logger.error(f"✗ Document processing test failed: {e}")
        return False
    finally:
        # Clean up temp file
        if pdf_path and os.path.exists(pdf_path):
            os.unlink(pdf_path)


def main():
    """Run all installation tests."""
    logger.info("Starting SutazaiApp installation tests...")

    tests = [
        ("API Health", test_health_endpoint),
        ("Vector Search", test_vector_search),
        ("Document Processing", test_document_processing),
        ("Code Generation", test_code_generation),
    ]

    results = []
    for test_name, test_func in tests:
        logger.info(f"\n=== Testing {test_name} ===")
        success = test_func()
        results.append((test_name, success))

    # Print summary
    logger.info("\n=== Test Summary ===")
    all_passed = True
    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        logger.info(f"{status}: {test_name}")
        if not success:
            all_passed = False

    if all_passed:
        logger.info("\n✓ All tests passed! SutazaiApp is installed correctly.")
        return 0
    else:
        logger.error("\n✗ Some tests failed. Please check the logs for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
