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
            List[str]: Paths to generated test documents
        """
        test_documents = []
        with tempfile.TemporaryDirectory() as temp_dir:
            for i in range(num_docs):
                # Create a simple PDF with multiple pages
                doc = fitz.open()  # type: ignore
                for page_num in range(5):  # 5 pages per document
                    page = doc.new_page()  # type: ignore
                    text = f"Test Document {i} - Page {page_num}\n" * 50
                    page.insert_text((100, 100), text)  # type: ignore
                
                doc_path = os.path.join(temp_dir, f"test_doc_{i}.pdf")
                doc.save(doc_path)  # type: ignore
                doc.close()  # type: ignore
                test_documents.append(doc_path)
        
        return test_documents

    def benchmark_text_extraction(self, documents: List[str]) -> Dict[str, Any]:
        """
        Benchmark text extraction performance
        Args:
            documents (List[str]): List of document paths
        Returns:
            Dict[str, Any]: Benchmark results
        """
        agent = DocumentProcessorAgent()
        extraction_times = []
        
        for doc_path in documents:
            start_time = time.time()
            
            task = {
                "document_path": doc_path,
                "operation": "extract_text",
                "parameters": {}
            }
            
            result = agent.execute(task)
            end_time = time.time()
            
            if result["status"] == "success":
                extraction_times.append(end_time - start_time)
        
        return {
            "operation": "text_extraction",
            "documents_processed": len(extraction_times),
            "avg_time": statistics.mean(extraction_times) if extraction_times else 0,
            "min_time": min(extraction_times) if extraction_times else 0,
            "max_time": max(extraction_times) if extraction_times else 0,
            "total_time": sum(extraction_times)
        }

    def benchmark_document_analysis(self, documents: List[str]) -> Dict[str, Any]:
        """
        Benchmark document analysis performance
        Args:
            documents (List[str]): List of document paths
        Returns:
            Dict[str, Any]: Benchmark results
        """
        agent = DocumentProcessorAgent()
        analysis_times = []
        
        for doc_path in documents:
            start_time = time.time()
            
            task = {
                "document_path": doc_path,
                "operation": "analyze",
                "parameters": {"analysis_type": "structure"}
            }
            
            result = agent.execute(task)
            end_time = time.time()
            
            if result["status"] == "success":
                analysis_times.append(end_time - start_time)
        
        return {
            "operation": "document_analysis",
            "documents_processed": len(analysis_times),
            "avg_time": statistics.mean(analysis_times) if analysis_times else 0,
            "min_time": min(analysis_times) if analysis_times else 0,
            "max_time": max(analysis_times) if analysis_times else 0,
            "total_time": sum(analysis_times)
        }

    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """
        Run a comprehensive benchmark suite
        Returns:
            Dict[str, Any]: Complete benchmark results
        """
        print("🏃 Starting Document Processor Benchmark...")
        
        # Generate test documents
        test_docs = self.generate_test_documents(5)
        
        # Run benchmarks
        text_results = self.benchmark_text_extraction(test_docs)
        analysis_results = self.benchmark_document_analysis(test_docs)
        
        comprehensive_results = {
            "benchmark_timestamp": time.time(),
            "test_documents_count": len(test_docs),
            "text_extraction": text_results,
            "document_analysis": analysis_results
        }
        
        print("📊 Benchmark Complete!")
        return comprehensive_results

    def generate_performance_report(self, results: Dict[str, Any]) -> str:
        """
        Generate a formatted performance report
        Args:
            results (Dict[str, Any]): Benchmark results
        Returns:
            str: Formatted report
        """
        report = "📈 Document Processor Performance Report\n"
        report += "=" * 50 + "\n"
        
        for operation, data in results.items():
            if operation in ["text_extraction", "document_analysis"]:
                report += f"\n{operation.upper()}\n"
                report += f"Documents Processed: {data['documents_processed']}\n"
                report += f"Average Time: {data['avg_time']:.4f}s\n"
                report += f"Min Time: {data['min_time']:.4f}s\n"
                report += f"Max Time: {data['max_time']:.4f}s\n"
                report += f"Total Time: {data['total_time']:.4f}s\n"
        
        return report


def main():
    """Run the performance benchmark"""
    benchmark = DocumentProcessorBenchmark()
    results = benchmark.run_comprehensive_benchmark()
    report = benchmark.generate_performance_report(results)
    print(report)


if __name__ == "__main__":
    main()
