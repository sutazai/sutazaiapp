import os
import statistics
import tempfile
import time

import cv2
import fitz
import numpy as np

from ai_agents.document_processor.src import DocumentProcessorAgent


class DocumentProcessorBenchmark:
    """
    Comprehensive Performance Benchmarking for Document Processor Agent

    Measures and analyzes performance across various document processing tasks
    """

    @staticmethod
    def generate_test_documents(num_docs: int = 10) -> list:
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
                page = doc.new_page()
                page.insert_text((50, 50), f"SutazAI Performance Test Document {i}")
                doc.save(temp_pdf.name)
                doc.close()
                test_docs.append(temp_pdf.name)

            # Image document
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_img:
                image = np.zeros((200, 200), dtype=np.uint8)
                cv2.putText(
                    image,
                    f"SutazAI OCR Test {i}",
                    (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    255,
                    2,
                )
                cv2.imwrite(temp_img.name, image)
                test_docs.append(temp_img.name)

        return test_docs

    @staticmethod
    def benchmark_document_extraction(documents: list) -> Dict[str, Any]:
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

            task = {"type": "extract_text", "document": doc_path, "params": {}}

            start_time = time.time()
            result = agent.execute(task)
            end_time = time.time()

            extraction_times.append(end_time - start_time)

        return {
            "task": "text_extraction",
            "total_documents": len(documents),
            "mean_time": statistics.mean(extraction_times),
            "median_time": statistics.median(extraction_times),
            "min_time": min(extraction_times),
            "max_time": max(extraction_times),
        }

    @staticmethod
    def benchmark_ocr_processing(documents: list) -> Dict[str, Any]:
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
                "type": "ocr_processing",
                "document": doc_path,
                "params": {"languages": ["eng"]},
            }

            start_time = time.time()
            result = agent.execute(task)
            end_time = time.time()

            ocr_times.append(end_time - start_time)

        return {
            "task": "ocr_processing",
            "total_documents": len(documents),
            "mean_time": statistics.mean(ocr_times),
            "median_time": statistics.median(ocr_times),
            "min_time": min(ocr_times),
            "max_time": max(ocr_times),
        }

    @staticmethod
    def benchmark_document_analysis(documents: list) -> Dict[str, Any]:
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
                "type": "document_analysis",
                "document": doc_path,
                "params": {},
            }

            start_time = time.time()
            result = agent.execute(task)
            end_time = time.time()

            analysis_times.append(end_time - start_time)

        return {
            "task": "document_analysis",
            "total_documents": len(documents),
            "mean_time": statistics.mean(analysis_times),
            "median_time": statistics.median(analysis_times),
            "min_time": min(analysis_times),
            "max_time": max(analysis_times),
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
                test_documents
            ),
            "ocr_processing": DocumentProcessorBenchmark.benchmark_ocr_processing(
                test_documents
            ),
            "document_analysis": DocumentProcessorBenchmark.benchmark_document_analysis(
                test_documents
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
    benchmark_results = DocumentProcessorBenchmark.run_comprehensive_benchmark()

    # Performance assertions
    for task, results in benchmark_results.items():
        assert results["mean_time"] < 2.0, f"{task} performance too slow"
        assert results["total_documents"] > 0, f"No documents processed for {task}"


def main():
    """
    Run performance benchmarks and generate report
    """
    import json

    benchmark_results = DocumentProcessorBenchmark.run_comprehensive_benchmark()

    # Generate performance report
    report_path = (
        "/opt/sutazai_project/SutazAI/logs/document_processor_performance.json"
    )
    with open(report_path, "w") as report_file:
        json.dump(benchmark_results, report_file, indent=2)

    print("Performance Benchmark Results:")
    print(json.dumps(benchmark_results, indent=2))


if __name__ == "__main__":
    main()
