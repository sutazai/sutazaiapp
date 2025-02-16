import os
import time
import asyncio
import statistics
import json
import logging
from typing import List, Dict, Any, Optional
import psutil
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from document_processor import AdvancedDocumentProcessor

class DocumentProcessorBenchmark:
    """
    Comprehensive benchmarking and performance monitoring for document processing.
    """
    
    def __init__(self, 
                 processor: Optional[AdvancedDocumentProcessor] = None,
                 log_dir: str = '/var/log/sutazai/benchmarks'):
        """
        Initialize the document processor benchmark.
        
        Args:
            processor (AdvancedDocumentProcessor): Document processor instance
            log_dir (str): Directory to store benchmark logs and reports
        """
        self.processor = processor or AdvancedDocumentProcessor()
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, 'document_processor_benchmark.log')),
                logging.StreamHandler()
            ]
        )
    
    def benchmark_document_processing(self, 
                                      file_paths: List[str], 
                                      iterations: int = 3) -> Dict[str, Any]:
        """
        Comprehensive performance benchmarking for document processing.
        
        Args:
            file_paths (List[str]): List of document file paths to process
            iterations (int): Number of benchmark iterations
        
        Returns:
            Detailed benchmark results
        """
        benchmark_results = {
            'total_processing_times': [],
            'per_file_processing_times': {},
            'system_metrics': {
                'cpu_usage': [],
                'memory_usage': [],
                'gpu_memory_usage': []
            },
            'statistical_summary': {}
        }
        
        for iteration in range(iterations):
            self.logger.info(f"Benchmark Iteration {iteration + 1}")
            
            # Track system metrics
            start_cpu = psutil.cpu_percent(interval=None)
            start_memory = psutil.virtual_memory().percent
            
            # Measure total processing time
            start_time = time.time()
            results = self.processor.batch_process_documents(file_paths)
            total_processing_time = time.time() - start_time
            
            # Collect system metrics
            end_cpu = psutil.cpu_percent(interval=None)
            end_memory = psutil.virtual_memory().percent
            
            # GPU memory usage (if available)
            gpu_memory_usage = 0
            if torch.cuda.is_available():
                gpu_memory_usage = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
                torch.cuda.reset_peak_memory_stats()
            
            # Store results
            benchmark_results['total_processing_times'].append(total_processing_time)
            benchmark_results['system_metrics']['cpu_usage'].append((start_cpu + end_cpu) / 2)
            benchmark_results['system_metrics']['memory_usage'].append((start_memory + end_memory) / 2)
            benchmark_results['system_metrics']['gpu_memory_usage'].append(gpu_memory_usage)
            
            # Per-file processing times
            for result in results:
                file_path = result.get('file_path', 'unknown')
                processing_time = result.get('processing_time', 0)
                
                if file_path not in benchmark_results['per_file_processing_times']:
                    benchmark_results['per_file_processing_times'][file_path] = []
                
                benchmark_results['per_file_processing_times'][file_path].append(processing_time)
        
        # Calculate statistical summary
        benchmark_results['statistical_summary'] = {
            'total_processing_time': {
                'mean': statistics.mean(benchmark_results['total_processing_times']),
                'median': statistics.median(benchmark_results['total_processing_times']),
                'stdev': statistics.stdev(benchmark_results['total_processing_times']) if len(benchmark_results['total_processing_times']) > 1 else 0
            },
            'cpu_usage': {
                'mean': statistics.mean(benchmark_results['system_metrics']['cpu_usage']),
                'median': statistics.median(benchmark_results['system_metrics']['cpu_usage']),
                'stdev': statistics.stdev(benchmark_results['system_metrics']['cpu_usage']) if len(benchmark_results['system_metrics']['cpu_usage']) > 1 else 0
            },
            'memory_usage': {
                'mean': statistics.mean(benchmark_results['system_metrics']['memory_usage']),
                'median': statistics.median(benchmark_results['system_metrics']['memory_usage']),
                'stdev': statistics.stdev(benchmark_results['system_metrics']['memory_usage']) if len(benchmark_results['system_metrics']['memory_usage']) > 1 else 0
            }
        }
        
        return benchmark_results
    
    def generate_performance_report(self, benchmark_results: Dict[str, Any]):
        """
        Generate a comprehensive performance report with visualizations.
        
        Args:
            benchmark_results (Dict[str, Any]): Benchmark results to analyze
        """
        report_path = os.path.join(self.log_dir, 'performance_report.json')
        with open(report_path, 'w') as f:
            json.dump(benchmark_results, f, indent=2)
        
        # Generate performance visualizations
        self._create_performance_plots(benchmark_results)
        
        # Print summary report
        print("\n🚀 Document Processor Performance Report 🚀")
        print(f"Total Processing Time: {benchmark_results['statistical_summary']['total_processing_time']['mean']:.4f} ± {benchmark_results['statistical_summary']['total_processing_time']['stdev']:.4f} seconds")
        print(f"Average CPU Usage: {benchmark_results['statistical_summary']['cpu_usage']['mean']:.2f}%")
        print(f"Average Memory Usage: {benchmark_results['statistical_summary']['memory_usage']['mean']:.2f}%")
        print(f"\nDetailed report saved to: {report_path}")
    
    def _create_performance_plots(self, benchmark_results: Dict[str, Any]):
        """
        Create performance visualization plots.
        
        Args:
            benchmark_results (Dict[str, Any]): Benchmark results to visualize
        """
        plt.figure(figsize=(15, 5))
        
        # Processing Time Distribution
        plt.subplot(131)
        sns.histplot(benchmark_results['total_processing_times'], kde=True)
        plt.title('Processing Time Distribution')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Frequency')
        
        # CPU Usage
        plt.subplot(132)
        sns.boxplot(data=benchmark_results['system_metrics']['cpu_usage'])
        plt.title('CPU Usage')
        plt.ylabel('Percentage')
        
        # Memory Usage
        plt.subplot(133)
        sns.boxplot(data=benchmark_results['system_metrics']['memory_usage'])
        plt.title('Memory Usage')
        plt.ylabel('Percentage')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, 'performance_metrics.png'))
        plt.close()

def main():
    """
    Run document processor benchmarking.
    """
    # Example document paths (replace with actual paths)
    test_files = [
        '/path/to/document1.pdf',
        '/path/to/document2.docx',
        '/path/to/document3.txt'
    ]
    
    # Initialize benchmark
    benchmark = DocumentProcessorBenchmark()
    
    # Run benchmarks
    results = benchmark.benchmark_document_processing(test_files)
    
    # Generate performance report
    benchmark.generate_performance_report(results)

if __name__ == "__main__":
    main() 