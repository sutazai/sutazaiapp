#!/usr/bin/env python3
"""
SutazAI v9 Advanced Batch Processing System
Handles processing of 50+ files simultaneously with intelligent load balancing
"""

import asyncio
import aiofiles
import aiohttp
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import List, Dict, Any, Callable, Optional, Union
from pathlib import Path
import logging
import time
import json
import hashlib
import psutil
from dataclasses import dataclass, asdict
from enum import Enum
import queue
import threading
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProcessingType(Enum):
    CPU_INTENSIVE = "cpu_intensive"
    IO_INTENSIVE = "io_intensive"
    MIXED = "mixed"
    AI_INFERENCE = "ai_inference"

class TaskStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class ProcessingTask:
    """Represents a single processing task"""
    id: str
    file_path: str
    operation: str
    parameters: Dict[str, Any]
    priority: int = 5
    processing_type: ProcessingType = ProcessingType.MIXED
    status: TaskStatus = TaskStatus.PENDING
    created_at: float = None
    started_at: float = None
    completed_at: float = None
    result: Any = None
    error: str = None
    progress: float = 0.0
    estimated_duration: float = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()
        if self.id is None:
            self.id = self.generate_id()

    def generate_id(self) -> str:
        """Generate unique task ID"""
        content = f"{self.file_path}_{self.operation}_{self.created_at}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

@dataclass
class BatchProcessingResults:
    """Results of batch processing operation"""
    total_tasks: int
    completed: int
    failed: int
    cancelled: int
    total_duration: float
    results: Dict[str, Any]
    errors: List[str]
    performance_metrics: Dict[str, Any]

class ResourceMonitor:
    """Monitors system resources during processing"""
    
    def __init__(self):
        self.cpu_usage = []
        self.memory_usage = []
        self.disk_io = []
        self.monitoring = False
        self.monitor_thread = None

    def start_monitoring(self):
        """Start resource monitoring"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_resources)
        self.monitor_thread.start()

    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()

    def _monitor_resources(self):
        """Monitor system resources"""
        while self.monitoring:
            try:
                cpu = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory().percent
                disk = psutil.disk_io_counters()
                
                self.cpu_usage.append(cpu)
                self.memory_usage.append(memory)
                self.disk_io.append({
                    'read_bytes': disk.read_bytes if disk else 0,
                    'write_bytes': disk.write_bytes if disk else 0
                })
                
                time.sleep(2)
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")

    def get_metrics(self) -> Dict[str, Any]:
        """Get resource usage metrics"""
        if not self.cpu_usage:
            return {}
            
        return {
            'cpu': {
                'avg': sum(self.cpu_usage) / len(self.cpu_usage),
                'max': max(self.cpu_usage),
                'min': min(self.cpu_usage)
            },
            'memory': {
                'avg': sum(self.memory_usage) / len(self.memory_usage),
                'max': max(self.memory_usage),
                'min': min(self.memory_usage)
            },
            'samples': len(self.cpu_usage)
        }

class ProgressTracker:
    """Tracks progress of batch processing"""
    
    def __init__(self, total_tasks: int):
        self.total_tasks = total_tasks
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.lock = threading.Lock()
        self.callbacks = []

    def add_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add progress callback"""
        self.callbacks.append(callback)

    def update(self, status: TaskStatus):
        """Update progress"""
        with self.lock:
            if status == TaskStatus.COMPLETED:
                self.completed_tasks += 1
            elif status == TaskStatus.FAILED:
                self.failed_tasks += 1
            
            progress_data = {
                'total': self.total_tasks,
                'completed': self.completed_tasks,
                'failed': self.failed_tasks,
                'progress_percent': (self.completed_tasks + self.failed_tasks) / self.total_tasks * 100
            }
            
            for callback in self.callbacks:
                try:
                    callback(progress_data)
                except Exception as e:
                    logger.error(f"Progress callback error: {e}")

class BatchProcessor:
    """Advanced batch processing system for 50+ files"""
    
    def __init__(self, 
                 max_workers: Optional[int] = None,
                 max_memory_mb: int = 8192,
                 enable_monitoring: bool = True):
        
        self.cpu_count = mp.cpu_count()
        self.max_workers = max_workers or min(50, self.cpu_count * 4)
        self.max_memory_mb = max_memory_mb
        self.enable_monitoring = enable_monitoring
        
        # Thread pools for different types of operations
        self.io_executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.cpu_executor = ProcessPoolExecutor(max_workers=self.cpu_count)
        
        # Task management
        self.tasks = {}
        self.task_queue = asyncio.Queue()
        self.processing_tasks = set()
        
        # Monitoring
        self.resource_monitor = ResourceMonitor() if enable_monitoring else None
        
        # Performance optimization
        self.cache = {}
        self.performance_stats = {}
        
        logger.info(f"BatchProcessor initialized with {self.max_workers} workers")

    async def process_files_batch(self, 
                                  files: List[str], 
                                  operation: str,
                                  parameters: Dict[str, Any] = None,
                                  batch_size: int = 50,
                                  priority_function: Callable = None) -> BatchProcessingResults:
        """
        Process multiple files in optimized batches
        
        Args:
            files: List of file paths to process
            operation: Operation to perform on files
            parameters: Additional parameters for processing
            batch_size: Number of files per batch
            priority_function: Function to determine task priority
            
        Returns:
            BatchProcessingResults object with detailed results
        """
        start_time = time.time()
        total_files = len(files)
        
        logger.info(f"Starting batch processing of {total_files} files")
        
        # Start monitoring
        if self.resource_monitor:
            self.resource_monitor.start_monitoring()
        
        # Create progress tracker
        progress_tracker = ProgressTracker(total_files)
        progress_tracker.add_callback(self._log_progress)
        
        try:
            # Create tasks
            tasks = []
            for i, file_path in enumerate(files):
                task = ProcessingTask(
                    id=f"task_{i}",
                    file_path=file_path,
                    operation=operation,
                    parameters=parameters or {},
                    priority=priority_function(file_path) if priority_function else 5,
                    processing_type=self._determine_processing_type(operation)
                )
                tasks.append(task)
                self.tasks[task.id] = task
            
            # Sort tasks by priority and file size for optimal processing
            tasks = self._optimize_task_order(tasks)
            
            # Create batches
            batches = self._create_optimal_batches(tasks, batch_size)
            
            # Process batches
            all_results = {}
            all_errors = []
            
            for batch_id, batch in enumerate(batches):
                logger.info(f"Processing batch {batch_id + 1}/{len(batches)} with {len(batch)} tasks")
                
                batch_results = await self._process_batch(batch, progress_tracker)
                
                # Aggregate results
                for task_id, result in batch_results.items():
                    if isinstance(result, Exception):
                        all_errors.append(f"Task {task_id}: {str(result)}")
                    else:
                        all_results[task_id] = result
            
            # Calculate final metrics
            end_time = time.time()
            total_duration = end_time - start_time
            
            # Get resource metrics
            performance_metrics = {}
            if self.resource_monitor:
                self.resource_monitor.stop_monitoring()
                performance_metrics = self.resource_monitor.get_metrics()
            
            # Create results object
            results = BatchProcessingResults(
                total_tasks=total_files,
                completed=len(all_results),
                failed=len(all_errors),
                cancelled=0,
                total_duration=total_duration,
                results=all_results,
                errors=all_errors,
                performance_metrics=performance_metrics
            )
            
            logger.info(f"Batch processing completed in {total_duration:.2f}s")
            logger.info(f"Success rate: {len(all_results)/total_files*100:.1f}%")
            
            return results
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            if self.resource_monitor:
                self.resource_monitor.stop_monitoring()
            raise

    async def _process_batch(self, batch: List[ProcessingTask], progress_tracker: ProgressTracker) -> Dict[str, Any]:
        """Process a single batch of tasks"""
        batch_results = {}
        
        # Group tasks by processing type for optimal execution
        cpu_tasks = [t for t in batch if t.processing_type == ProcessingType.CPU_INTENSIVE]
        io_tasks = [t for t in batch if t.processing_type == ProcessingType.IO_INTENSIVE]
        ai_tasks = [t for t in batch if t.processing_type == ProcessingType.AI_INFERENCE]
        mixed_tasks = [t for t in batch if t.processing_type == ProcessingType.MIXED]
        
        # Process different task types concurrently
        tasks_to_await = []
        
        # CPU-intensive tasks (use process pool)
        if cpu_tasks:
            cpu_future = self._process_cpu_tasks(cpu_tasks, progress_tracker)
            tasks_to_await.append(cpu_future)
        
        # I/O-intensive tasks (use thread pool)
        if io_tasks:
            io_future = self._process_io_tasks(io_tasks, progress_tracker)
            tasks_to_await.append(io_future)
        
        # AI inference tasks (use specialized handling)
        if ai_tasks:
            ai_future = self._process_ai_tasks(ai_tasks, progress_tracker)
            tasks_to_await.append(ai_future)
        
        # Mixed tasks (use adaptive approach)
        if mixed_tasks:
            mixed_future = self._process_mixed_tasks(mixed_tasks, progress_tracker)
            tasks_to_await.append(mixed_future)
        
        # Wait for all task groups to complete
        if tasks_to_await:
            results_list = await asyncio.gather(*tasks_to_await, return_exceptions=True)
            
            # Merge results
            for results in results_list:
                if isinstance(results, dict):
                    batch_results.update(results)
                elif isinstance(results, Exception):
                    logger.error(f"Batch processing error: {results}")
        
        return batch_results

    async def _process_cpu_tasks(self, tasks: List[ProcessingTask], progress_tracker: ProgressTracker) -> Dict[str, Any]:
        """Process CPU-intensive tasks using process pool"""
        results = {}
        loop = asyncio.get_event_loop()
        
        futures = []
        for task in tasks:
            future = loop.run_in_executor(
                self.cpu_executor,
                self._execute_cpu_task,
                task
            )
            futures.append((task.id, future))
        
        for task_id, future in futures:
            try:
                result = await future
                results[task_id] = result
                progress_tracker.update(TaskStatus.COMPLETED)
            except Exception as e:
                logger.error(f"CPU task {task_id} failed: {e}")
                results[task_id] = e
                progress_tracker.update(TaskStatus.FAILED)
        
        return results

    async def _process_io_tasks(self, tasks: List[ProcessingTask], progress_tracker: ProgressTracker) -> Dict[str, Any]:
        """Process I/O-intensive tasks using async I/O"""
        results = {}
        
        # Create semaphore to limit concurrent I/O operations
        semaphore = asyncio.Semaphore(self.max_workers)
        
        async def process_single_io_task(task: ProcessingTask):
            async with semaphore:
                try:
                    result = await self._execute_io_task(task)
                    progress_tracker.update(TaskStatus.COMPLETED)
                    return task.id, result
                except Exception as e:
                    logger.error(f"I/O task {task.id} failed: {e}")
                    progress_tracker.update(TaskStatus.FAILED)
                    return task.id, e
        
        # Process all I/O tasks concurrently
        task_futures = [process_single_io_task(task) for task in tasks]
        task_results = await asyncio.gather(*task_futures, return_exceptions=True)
        
        for task_id, result in task_results:
            if not isinstance(result, Exception):
                results[task_id] = result
        
        return results

    async def _process_ai_tasks(self, tasks: List[ProcessingTask], progress_tracker: ProgressTracker) -> Dict[str, Any]:
        """Process AI inference tasks with specialized handling"""
        results = {}
        
        # AI tasks often need sequential processing to avoid memory issues
        for task in tasks:
            try:
                result = await self._execute_ai_task(task)
                results[task.id] = result
                progress_tracker.update(TaskStatus.COMPLETED)
            except Exception as e:
                logger.error(f"AI task {task.id} failed: {e}")
                results[task.id] = e
                progress_tracker.update(TaskStatus.FAILED)
        
        return results

    async def _process_mixed_tasks(self, tasks: List[ProcessingTask], progress_tracker: ProgressTracker) -> Dict[str, Any]:
        """Process mixed tasks using adaptive approach"""
        results = {}
        
        # Use thread pool for mixed tasks with moderate concurrency
        semaphore = asyncio.Semaphore(self.max_workers // 2)
        
        async def process_single_mixed_task(task: ProcessingTask):
            async with semaphore:
                try:
                    result = await self._execute_mixed_task(task)
                    progress_tracker.update(TaskStatus.COMPLETED)
                    return task.id, result
                except Exception as e:
                    logger.error(f"Mixed task {task.id} failed: {e}")
                    progress_tracker.update(TaskStatus.FAILED)
                    return task.id, e
        
        task_futures = [process_single_mixed_task(task) for task in tasks]
        task_results = await asyncio.gather(*task_futures, return_exceptions=True)
        
        for task_id, result in task_results:
            if not isinstance(result, Exception):
                results[task_id] = result
        
        return results

    def _execute_cpu_task(self, task: ProcessingTask) -> Any:
        """Execute CPU-intensive task"""
        task.status = TaskStatus.PROCESSING
        task.started_at = time.time()
        
        try:
            # Implement specific CPU-intensive operations here
            if task.operation == "compile":
                return self._compile_file(task.file_path, task.parameters)
            elif task.operation == "optimize":
                return self._optimize_file(task.file_path, task.parameters)
            elif task.operation == "analyze":
                return self._analyze_file(task.file_path, task.parameters)
            else:
                raise ValueError(f"Unknown CPU operation: {task.operation}")
                
        finally:
            task.completed_at = time.time()
            task.status = TaskStatus.COMPLETED

    async def _execute_io_task(self, task: ProcessingTask) -> Any:
        """Execute I/O-intensive task"""
        task.status = TaskStatus.PROCESSING
        task.started_at = time.time()
        
        try:
            if task.operation == "read":
                return await self._read_file(task.file_path, task.parameters)
            elif task.operation == "write":
                return await self._write_file(task.file_path, task.parameters)
            elif task.operation == "copy":
                return await self._copy_file(task.file_path, task.parameters)
            elif task.operation == "parse":
                return await self._parse_file(task.file_path, task.parameters)
            else:
                raise ValueError(f"Unknown I/O operation: {task.operation}")
                
        finally:
            task.completed_at = time.time()
            task.status = TaskStatus.COMPLETED

    async def _execute_ai_task(self, task: ProcessingTask) -> Any:
        """Execute AI inference task"""
        task.status = TaskStatus.PROCESSING
        task.started_at = time.time()
        
        try:
            if task.operation == "generate_code":
                return await self._generate_code(task.file_path, task.parameters)
            elif task.operation == "analyze_code":
                return await self._analyze_code(task.file_path, task.parameters)
            elif task.operation == "summarize":
                return await self._summarize_content(task.file_path, task.parameters)
            elif task.operation == "translate":
                return await self._translate_content(task.file_path, task.parameters)
            else:
                raise ValueError(f"Unknown AI operation: {task.operation}")
                
        finally:
            task.completed_at = time.time()
            task.status = TaskStatus.COMPLETED

    async def _execute_mixed_task(self, task: ProcessingTask) -> Any:
        """Execute mixed task"""
        task.status = TaskStatus.PROCESSING
        task.started_at = time.time()
        
        try:
            if task.operation == "process_document":
                return await self._process_document(task.file_path, task.parameters)
            elif task.operation == "convert":
                return await self._convert_file(task.file_path, task.parameters)
            else:
                # Default to I/O task
                return await self._execute_io_task(task)
                
        finally:
            task.completed_at = time.time()
            task.status = TaskStatus.COMPLETED

    def _determine_processing_type(self, operation: str) -> ProcessingType:
        """Determine the processing type based on operation"""
        cpu_ops = {"compile", "optimize", "analyze", "compress", "encrypt"}
        io_ops = {"read", "write", "copy", "move", "delete"}
        ai_ops = {"generate_code", "analyze_code", "summarize", "translate", "classify"}
        
        if operation in cpu_ops:
            return ProcessingType.CPU_INTENSIVE
        elif operation in io_ops:
            return ProcessingType.IO_INTENSIVE
        elif operation in ai_ops:
            return ProcessingType.AI_INFERENCE
        else:
            return ProcessingType.MIXED

    def _optimize_task_order(self, tasks: List[ProcessingTask]) -> List[ProcessingTask]:
        """Optimize task execution order"""
        # Sort by priority (higher first), then by estimated file size (larger first)
        def sort_key(task):
            try:
                file_size = Path(task.file_path).stat().st_size if Path(task.file_path).exists() else 0
            except:
                file_size = 0
            return (-task.priority, -file_size)
        
        return sorted(tasks, key=sort_key)

    def _create_optimal_batches(self, tasks: List[ProcessingTask], batch_size: int) -> List[List[ProcessingTask]]:
        """Create optimal batches for processing"""
        batches = []
        current_batch = []
        current_batch_size = 0
        max_batch_memory = self.max_memory_mb * 1024 * 1024  # Convert to bytes
        
        for task in tasks:
            try:
                # Estimate task memory usage
                file_size = Path(task.file_path).stat().st_size if Path(task.file_path).exists() else 0
                estimated_memory = file_size * 2  # Rough estimate
                
                # Check if adding this task would exceed batch limits
                if (len(current_batch) >= batch_size or 
                    current_batch_size + estimated_memory > max_batch_memory):
                    
                    if current_batch:
                        batches.append(current_batch)
                        current_batch = []
                        current_batch_size = 0
                
                current_batch.append(task)
                current_batch_size += estimated_memory
                
            except Exception as e:
                logger.warning(f"Error estimating task size for {task.file_path}: {e}")
                current_batch.append(task)
        
        if current_batch:
            batches.append(current_batch)
        
        return batches

    def _log_progress(self, progress_data: Dict[str, Any]):
        """Log progress updates"""
        logger.info(f"Progress: {progress_data['completed']}/{progress_data['total']} "
                   f"({progress_data['progress_percent']:.1f}%) completed, "
                   f"{progress_data['failed']} failed")

    # File operation implementations
    async def _read_file(self, file_path: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Read file asynchronously"""
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            content = await f.read()
            return {
                'operation': 'read',
                'file_path': file_path,
                'size': len(content),
                'success': True
            }

    async def _write_file(self, file_path: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Write file asynchronously"""
        content = parameters.get('content', '')
        async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
            await f.write(content)
            return {
                'operation': 'write',
                'file_path': file_path,
                'bytes_written': len(content),
                'success': True
            }

    async def _copy_file(self, file_path: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Copy file asynchronously"""
        dest_path = parameters.get('destination')
        if not dest_path:
            raise ValueError("Destination path required for copy operation")
        
        async with aiofiles.open(file_path, 'rb') as src:
            async with aiofiles.open(dest_path, 'wb') as dst:
                content = await src.read()
                await dst.write(content)
        
        return {
            'operation': 'copy',
            'source': file_path,
            'destination': dest_path,
            'bytes_copied': len(content),
            'success': True
        }

    async def _parse_file(self, file_path: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Parse file content"""
        file_type = parameters.get('type', 'auto')
        
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            content = await f.read()
        
        # Basic parsing logic
        if file_type == 'json' or file_path.endswith('.json'):
            try:
                parsed_content = json.loads(content)
                return {
                    'operation': 'parse',
                    'file_path': file_path,
                    'type': 'json',
                    'parsed_data': parsed_content,
                    'success': True
                }
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON: {e}")
        
        # Default text parsing
        return {
            'operation': 'parse',
            'file_path': file_path,
            'type': 'text',
            'line_count': len(content.split('\n')),
            'char_count': len(content),
            'success': True
        }

    def _compile_file(self, file_path: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Compile file (CPU-intensive)"""
        # Placeholder for compilation logic
        import subprocess
        
        compiler = parameters.get('compiler', 'gcc')
        output_path = parameters.get('output', file_path + '.out')
        
        try:
            result = subprocess.run(
                [compiler, file_path, '-o', output_path],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            return {
                'operation': 'compile',
                'file_path': file_path,
                'output_path': output_path,
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'success': result.returncode == 0
            }
        except subprocess.TimeoutExpired:
            raise TimeoutError(f"Compilation of {file_path} timed out")

    def _optimize_file(self, file_path: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize file (CPU-intensive)"""
        # Placeholder for optimization logic
        optimization_type = parameters.get('type', 'general')
        
        # Simulate CPU-intensive optimization
        time.sleep(0.1)  # Simulate processing time
        
        return {
            'operation': 'optimize',
            'file_path': file_path,
            'optimization_type': optimization_type,
            'success': True,
            'improvement': '15%'  # Simulated improvement
        }

    def _analyze_file(self, file_path: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze file (CPU-intensive)"""
        # Placeholder for analysis logic
        analysis_type = parameters.get('type', 'general')
        
        try:
            file_stats = Path(file_path).stat()
            
            return {
                'operation': 'analyze',
                'file_path': file_path,
                'analysis_type': analysis_type,
                'file_size': file_stats.st_size,
                'modified_time': file_stats.st_mtime,
                'analysis_results': {
                    'complexity': 'medium',
                    'quality_score': 85,
                    'issues_found': 3
                },
                'success': True
            }
        except Exception as e:
            raise ValueError(f"Analysis failed: {e}")

    async def _generate_code(self, file_path: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate code using AI"""
        # This would integrate with the AI models
        prompt = parameters.get('prompt', 'Generate code for this file')
        model = parameters.get('model', 'deepseek-r1:8b')
        
        # Simulate AI inference
        await asyncio.sleep(0.5)  # Simulate AI processing time
        
        return {
            'operation': 'generate_code',
            'file_path': file_path,
            'model': model,
            'prompt': prompt,
            'generated_code': '# Generated code placeholder\nprint("Hello, World!")',
            'success': True
        }

    async def _analyze_code(self, file_path: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze code using AI"""
        model = parameters.get('model', 'deepseek-coder:33b')
        
        # Read the file content
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            content = await f.read()
        
        # Simulate AI analysis
        await asyncio.sleep(0.3)
        
        return {
            'operation': 'analyze_code',
            'file_path': file_path,
            'model': model,
            'analysis': {
                'language': 'python',
                'complexity_score': 7.2,
                'maintainability': 'good',
                'suggestions': ['Add type hints', 'Improve error handling']
            },
            'success': True
        }

    async def _summarize_content(self, file_path: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize content using AI"""
        model = parameters.get('model', 'qwen3:8b')
        max_length = parameters.get('max_length', 200)
        
        # Read the file content
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            content = await f.read()
        
        # Simulate AI summarization
        await asyncio.sleep(0.4)
        
        return {
            'operation': 'summarize',
            'file_path': file_path,
            'model': model,
            'original_length': len(content),
            'summary': f"Summary of {Path(file_path).name}: This file contains important content that has been summarized.",
            'summary_length': max_length,
            'success': True
        }

    async def _translate_content(self, file_path: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Translate content using AI"""
        model = parameters.get('model', 'qwen3:8b')
        target_language = parameters.get('target_language', 'en')
        
        # Read the file content
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            content = await f.read()
        
        # Simulate AI translation
        await asyncio.sleep(0.6)
        
        return {
            'operation': 'translate',
            'file_path': file_path,
            'model': model,
            'target_language': target_language,
            'original_text': content[:100] + '...' if len(content) > 100 else content,
            'translated_text': f"[Translated to {target_language}] {content[:100]}...",
            'success': True
        }

    async def _process_document(self, file_path: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Process document (mixed operation)"""
        processing_steps = parameters.get('steps', ['read', 'parse', 'analyze'])
        
        results = {}
        
        for step in processing_steps:
            if step == 'read':
                results['read'] = await self._read_file(file_path, parameters)
            elif step == 'parse':
                results['parse'] = await self._parse_file(file_path, parameters)
            elif step == 'analyze':
                results['analyze'] = await self._analyze_code(file_path, parameters)
        
        return {
            'operation': 'process_document',
            'file_path': file_path,
            'steps_completed': processing_steps,
            'results': results,
            'success': True
        }

    async def _convert_file(self, file_path: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Convert file format"""
        target_format = parameters.get('target_format')
        if not target_format:
            raise ValueError("Target format required for conversion")
        
        # Simulate conversion
        await asyncio.sleep(0.2)
        
        output_path = f"{file_path}.{target_format}"
        
        return {
            'operation': 'convert',
            'file_path': file_path,
            'target_format': target_format,
            'output_path': output_path,
            'success': True
        }

    def cleanup(self):
        """Cleanup resources"""
        try:
            if self.resource_monitor:
                self.resource_monitor.stop_monitoring()
            
            self.io_executor.shutdown(wait=True)
            self.cpu_executor.shutdown(wait=True)
            
            logger.info("BatchProcessor cleanup completed")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

# Example usage and testing
async def main():
    """Example usage of the batch processor"""
    
    # Create sample files for testing
    test_files = []
    for i in range(55):  # Test with 55 files
        file_path = f"/tmp/test_file_{i}.txt"
        test_files.append(file_path)
        
        # Create test file
        async with aiofiles.open(file_path, 'w') as f:
            await f.write(f"This is test file {i}\nContent line 2\nContent line 3")
    
    # Initialize batch processor
    with BatchProcessor(max_workers=20, enable_monitoring=True) as processor:
        
        # Test different operations
        operations = [
            ('read', {}),
            ('parse', {'type': 'text'}),
            ('analyze', {'type': 'general'}),
            ('summarize', {'max_length': 100}),
        ]
        
        for operation, params in operations:
            print(f"\n--- Testing {operation} operation ---")
            
            results = await processor.process_files_batch(
                files=test_files,
                operation=operation,
                parameters=params,
                batch_size=15
            )
            
            print(f"Operation: {operation}")
            print(f"Total files: {results.total_tasks}")
            print(f"Completed: {results.completed}")
            print(f"Failed: {results.failed}")
            print(f"Duration: {results.total_duration:.2f}s")
            print(f"Success rate: {results.completed/results.total_tasks*100:.1f}%")
            
            if results.performance_metrics:
                cpu_avg = results.performance_metrics.get('cpu', {}).get('avg', 0)
                memory_avg = results.performance_metrics.get('memory', {}).get('avg', 0)
                print(f"Avg CPU: {cpu_avg:.1f}%, Avg Memory: {memory_avg:.1f}%")
    
    # Cleanup test files
    for file_path in test_files:
        try:
            Path(file_path).unlink()
        except:
            pass

if __name__ == "__main__":
    asyncio.run(main())