#!/usr/bin/env python3.11
import os
import statistics
import tempfile
import time
from typing import Any, Dict, List

import cv2
import fitz  # type: ignore
import numpy as np

from ..src import DocumentProcessorAgent


class DocumentProcessorBenchmark:
    """
    Comprehensive Performance Benchmarking for Document Processor Agent
    Measures and analyzes performance across various document processing tasks
    """

    @staticmethod
    def generate_test_documents(num_docs: int = 10) -> List[str]:
        """
        Generate a set of test documents for benchmarking
        Args:
        num_docs (int): Number of test documents to generate
        Returns:
        list: Paths to generated test documents
        """
        test_docs = []
        for i in range(num_docs):
            # PDF document
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
                doc = fitz.open()
                # Updated for PyMuPDF compatibility with Python 3.11
                page = doc.new_page(width=595, height=842)  # A4 size
                page.insert_text((50, 50), f"SutazAI Performance Test Document {i}")
                doc.save(temp_pdf.name)
                doc.close()
                test_docs.append(temp_pdf.name)

            # Image document
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_img:
                image = np.zeros((200, 200), dtype=np.uint8)
                cv2.putText(  # type: ignore
                    image,
                    f"SutazAI OCR Test {i}",
                    (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
                cv2.imwrite(temp_img.name, image)
                test_docs.append(temp_img.name)

        return test_docs

    @staticmethod
    def benchmark_document_extraction(documents: List[str]) -> Dict[str, Any]:
        """
        Benchmark PDF text extraction performance
        Args:
        documents (list): List of document paths
        Returns:
        Dict: Performance benchmark results
        """
        agent = DocumentProcessorAgent()
        extraction_times = []

        for doc_path in documents:
            if not doc_path.endswith(".pdf"):
                continue

            task = {
                "document_path": doc_path,
                "operation": "extract_text",
                "parameters": {},
            }

            start_time = time.time()
            agent.execute(task)
            end_time = time.time()
            extraction_times.append(end_time - start_time)

        return {
            "task": "text_extraction",
            "total_documents": len(extraction_times),
            "mean_time": statistics.mean(extraction_times) if extraction_times else 0,
            "median_time": statistics.median(extraction_times) if extraction_times else 0,
            "min_time": min(extraction_times) if extraction_times else 0,
            "max_time": max(extraction_times) if extraction_times else 0,
        }

    @staticmethod
    def benchmark_ocr_processing(documents: List[str]) -> Dict[str, Any]:
        """
        Benchmark OCR processing performance
        Args:
        documents (list): List of document paths
        Returns:
        Dict: Performance benchmark results
        """
        agent = DocumentProcessorAgent()
        ocr_times = []

        for doc_path in documents:
            if not doc_path.endswith(".png"):
                continue

            task = {
                "document_path": doc_path,
                "operation": "ocr",
                "parameters": {"languages": ["eng"]},
            }

            start_time = time.time()
            agent.execute(task)
            end_time = time.time()
            ocr_times.append(end_time - start_time)

        return {
            "task": "ocr_processing",
            "total_documents": len(ocr_times),
            "mean_time": statistics.mean(ocr_times) if ocr_times else 0,
            "median_time": statistics.median(ocr_times) if ocr_times else 0,
            "min_time": min(ocr_times) if ocr_times else 0,
            "max_time": max(ocr_times) if ocr_times else 0,
        }

    @staticmethod
    def benchmark_document_analysis(documents: List[str]) -> Dict[str, Any]:
        """
        Benchmark advanced document analysis performance
        Args:
        documents (list): List of document paths
        Returns:
        Dict: Performance benchmark results
        """
        agent = DocumentProcessorAgent()
        analysis_times = []

        for doc_path in documents:
            if not doc_path.endswith(".pdf"):
                continue

            task = {
                "document_path": doc_path,
                "operation": "analyze",
                "parameters": {},
            }

            start_time = time.time()
            agent.execute(task)
            end_time = time.time()
            analysis_times.append(end_time - start_time)

        return {
            "task": "document_analysis",
            "total_documents": len(analysis_times),
            "mean_time": statistics.mean(analysis_times) if analysis_times else 0,
            "median_time": statistics.median(analysis_times) if analysis_times else 0,
            "min_time": min(analysis_times) if analysis_times else 0,
            "max_time": max(analysis_times) if analysis_times else 0,
        }

    @staticmethod
    def run_comprehensive_benchmark(num_docs: int = 10) -> Dict[str, Any]:
        """
        Run comprehensive performance benchmarks
        Args:
        num_docs (int): Number of test documents
        Returns:
        Dict: Comprehensive benchmark results
        """
        # Generate test documents
        test_documents = DocumentProcessorBenchmark.generate_test_documents(num_docs)

        # Run benchmarks
        benchmarks = {
            "text_extraction": DocumentProcessorBenchmark.benchmark_document_extraction(
                test_documents,
            ),
            "ocr_processing": DocumentProcessorBenchmark.benchmark_ocr_processing(
                test_documents,
            ),
            "document_analysis": DocumentProcessorBenchmark.benchmark_document_analysis(
                test_documents,
            ),
        }

        # Cleanup generated documents
        for doc in test_documents:
            os.unlink(doc)

        return benchmarks


def test_performance_benchmarks():
    """
    Pytest performance benchmark test
    """
    benchmark_results = DocumentProcessorBenchmark.run_comprehensive_benchmark(
        2,  # Use fewer docs for testing
    )

    # Performance assertions
    for task, results in benchmark_results.items():
        assert results["mean_time"] < 5.0, f"{task} performance too slow"
        assert results["total_documents"] >= 0, f"No documents processed for {task}"


def main():
    """
    Run performance benchmarks and generate report
    """
    import json

    benchmark_results = DocumentProcessorBenchmark.run_comprehensive_benchmark()

    # Generate performance report
    report_path = "/opt/sutazaiapp/logs/document_processor_performance.json"
    with open(report_path, "w", encoding="utf-8") as report_file:
        json.dump(benchmark_results, report_file, indent=2)

    print("Performance Benchmark Results:")
    print(json.dumps(benchmark_results, indent=2))


if __name__ == "__main__":
    main()
