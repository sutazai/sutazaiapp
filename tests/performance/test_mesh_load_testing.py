"""
Performance tests for mesh system load scenarios.
Tests system performance under various load conditions.
"""
import json
import time
import statistics
import threading
import pytest
from typing import Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

# Import mesh components
from app.mesh.redis_bus import (
    get_redis, enqueue_task, tail_results, register_agent,
    list_agents, create_consumer_group, read_group, ack,
    task_stream, result_stream
)

class PerformanceMetrics:
    """Collect and analyze performance metrics."""
    
    def __init__(self):
        self.enqueue_times = []
        self.retrieval_times = []
        self.processing_times = []
        self.throughput_samples = []
        self.error_count = 0
        self.start_time = None
        self.end_time = None
    
    def record_enqueue_time(self, duration: float):
        """Record time taken to enqueue a task."""
        self.enqueue_times.append(duration)
    
    def record_retrieval_time(self, duration: float):
        """Record time taken to retrieve results."""
        self.retrieval_times.append(duration)
    
    def record_processing_time(self, duration: float):
        """Record end-to-end processing time."""
        self.processing_times.append(duration)
    
    def record_throughput(self, tasks_per_second: float):
        """Record throughput measurement."""
        self.throughput_samples.append(tasks_per_second)
    
    def record_error(self):
        """Record an error occurrence."""
        self.error_count += 1
    
    def start_timing(self):
        """Start overall timing."""
        self.start_time = time.time()
    
    def end_timing(self):
        """End overall timing."""
        self.end_time = time.time()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        return {
            "enqueue_metrics": self._get_stats(self.enqueue_times),
            "retrieval_metrics": self._get_stats(self.retrieval_times),
            "processing_metrics": self._get_stats(self.processing_times),
            "throughput_metrics": self._get_stats(self.throughput_samples),
            "error_count": self.error_count,
            "total_duration": self.end_time - self.start_time if self.start_time and self.end_time else 0
        }
    
    def _get_stats(self, data: List[float]) -> Dict[str, float]:
        """Calculate statistics for a data set."""
        if not data:
            return {"count": 0}
        
        return {
            "count": len(data),
            "min": min(data),
            "max": max(data),
            "mean": statistics.mean(data),
            "median": statistics.median(data),
            "p95": self._percentile(data, 95),
            "p99": self._percentile(data, 99)
        }
    
    def _percentile(self, data: List[float], p: int) -> float:
        """Calculate percentile."""
        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * (p / 100)
        f = int(k)
        c = k - f
        if f == len(sorted_data) - 1:
            return sorted_data[f]
        return sorted_data[f] * (1 - c) + sorted_data[f + 1] * c

# Test fixtures
@pytest.fixture
def redis_client():
    """Get Redis client for performance tests."""
    try:
        client = get_redis()
        client.ping()
        return client
    except Exception as e:
        pytest.skip(f"Redis not available: {e}")

@pytest.fixture
def test_topic():
    """Test topic with timestamp."""
    return f"perf_test_{int(time.time())}"

@pytest.fixture
def performance_metrics():
    """Performance metrics collector."""
    return PerformanceMetrics()

@pytest.fixture(autouse=True)
def cleanup_test_data(redis_client, test_topic):
    """Cleanup test data."""
    # Cleanup before
    try:
        redis_client.delete(task_stream(test_topic))
        redis_client.delete(result_stream(test_topic))
    except:
        pass
    
    yield
    
    # Cleanup after
    try:
        redis_client.delete(task_stream(test_topic))
        redis_client.delete(result_stream(test_topic))
        
        # Clean up consumer groups
        try:
            redis_client.xgroup_destroy(task_stream(test_topic), "load_test_group")
        except:
            pass
    except:
        pass

class TestBasicLoadScenarios:
    """Test basic load scenarios."""
    
    def test_sequential_task_enqueuing(self, redis_client, test_topic, performance_metrics):
        """Test sequential task enqueuing performance."""
        num_tasks = 100
        task_size_kb = 1  # 1KB per task
        
        # Generate test payload
        base_payload = {
            "task_type": "load_test",
            "data": "x" * (task_size_kb * 1024),
            "timestamp": time.time()
        }
        
        # Reset enqueue cache
        if hasattr(enqueue_task, '_stream_cache'):
            enqueue_task._stream_cache = {}
        
        performance_metrics.start_timing()
        
        for i in range(num_tasks):
            payload = base_payload.copy()
            payload["task_id"] = f"seq_task_{i}"
            
            start_time = time.time()
            msg_id = enqueue_task(test_topic, payload)
            end_time = time.time()
            
            assert msg_id is not None
            performance_metrics.record_enqueue_time(end_time - start_time)
        
        performance_metrics.end_timing()
        
        # Verify all tasks enqueued
        messages = redis_client.xlen(task_stream(test_topic))
        assert messages == num_tasks
        
        # Performance assertions
        summary = performance_metrics.get_summary()
        enqueue_metrics = summary["enqueue_metrics"]
        
        # Average enqueue time should be reasonable
        assert enqueue_metrics["mean"] < 0.1  # Less than 100ms average
        assert enqueue_metrics["p95"] < 0.2   # 95th percentile less than 200ms
        
        # Calculate throughput
        total_time = summary["total_duration"]
        throughput = num_tasks / total_time
        assert throughput > 50  # At least 50 tasks/second
        
        print(f"Sequential enqueue performance: {throughput:.2f} tasks/sec")
        print(f"Average enqueue time: {enqueue_metrics['mean']*1000:.2f}ms")
    
    def test_concurrent_task_enqueuing(self, redis_client, test_topic, performance_metrics):
        """Test concurrent task enqueuing performance."""
        num_tasks = 200
        num_workers = 10
        tasks_per_worker = num_tasks // num_workers
        
        def enqueue_worker(worker_id: int) -> List[float]:
            """Worker function for enqueueing tasks."""
            worker_times = []
            
            # Reset enqueue cache per worker
            if hasattr(enqueue_task, '_stream_cache'):
                enqueue_task._stream_cache = {}
            
            for i in range(tasks_per_worker):
                payload = {
                    "task_type": "concurrent_load_test",
                    "worker_id": worker_id,
                    "task_index": i,
                    "data": "x" * 512,  # 512 bytes
                    "timestamp": time.time()
                }
                
                start_time = time.time()
                msg_id = enqueue_task(test_topic, payload)
                end_time = time.time()
                
                assert msg_id is not None
                worker_times.append(end_time - start_time)
            
            return worker_times
        
        performance_metrics.start_timing()
        
        # Run concurrent enqueueing
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for worker_id in range(num_workers):
                future = executor.submit(enqueue_worker, worker_id)
                futures.append(future)
            
            # Collect results
            all_times = []
            for future in as_completed(futures):
                try:
                    worker_times = future.result()
                    all_times.extend(worker_times)
                except Exception as e:
                    performance_metrics.record_error()
                    pytest.fail(f"Worker failed: {e}")
        
        performance_metrics.end_timing()
        
        # Record all times
        for t in all_times:
            performance_metrics.record_enqueue_time(t)
        
        # Verify all tasks enqueued
        messages = redis_client.xlen(task_stream(test_topic))
        assert messages == num_tasks
        
        # Performance assertions
        summary = performance_metrics.get_summary()
        enqueue_metrics = summary["enqueue_metrics"]
        
        # Concurrent should be faster overall
        total_time = summary["total_duration"]
        throughput = num_tasks / total_time
        assert throughput > 100  # At least 100 tasks/second with concurrency
        
        # Individual enqueue times might be higher due to contention
        assert enqueue_metrics["mean"] < 0.5  # Still reasonable
        assert enqueue_metrics["p95"] < 1.0   # 95th percentile less than 1s
        
        print(f"Concurrent enqueue performance: {throughput:.2f} tasks/sec")
        print(f"Average enqueue time: {enqueue_metrics['mean']*1000:.2f}ms")
    
    def test_bulk_result_retrieval(self, redis_client, test_topic, performance_metrics):
        """Test bulk result retrieval performance."""
        num_results = 500
        
        # Populate result stream
        for i in range(num_results):
            result_data = {
                "task_id": f"result_task_{i}",
                "status": "completed",
                "result": f"result_data_{i}",
                "data": "x" * 256,  # 256 bytes per result
                "timestamp": time.time()
            }
            
            redis_client.xadd(
                result_stream(test_topic),
                {"json": json.dumps(result_data)}
            )
        
        # Test various retrieval sizes
        retrieval_sizes = [10, 50, 100, 200]
        
        for size in retrieval_sizes:
            start_time = time.time()
            results = tail_results(test_topic, count=size)
            end_time = time.time()
            
            assert len(results) == size
            performance_metrics.record_retrieval_time(end_time - start_time)
        
        # Performance assertions
        summary = performance_metrics.get_summary()
        retrieval_metrics = summary["retrieval_metrics"]
        
        # Retrieval should be fast
        assert retrieval_metrics["mean"] < 1.0  # Less than 1 second average
        assert retrieval_metrics["max"] < 2.0   # Maximum less than 2 seconds
        
        print(f"Average retrieval time: {retrieval_metrics['mean']*1000:.2f}ms")
        print(f"Max retrieval time: {retrieval_metrics['max']*1000:.2f}ms")

class TestHighVolumeScenarios:
    """Test high-volume scenarios."""
    
    def test_sustained_load(self, redis_client, test_topic, performance_metrics):
        """Test sustained load over time."""
        duration_seconds = 30
        target_tps = 50  # Target tasks per second
        
        def sustained_enqueuer():
            """Continuously enqueue tasks."""
            task_count = 0
            start_time = time.time()
            
            # Reset enqueue cache
            if hasattr(enqueue_task, '_stream_cache'):
                enqueue_task._stream_cache = {}
            
            while time.time() - start_time < duration_seconds:
                payload = {
                    "task_type": "sustained_load",
                    "task_id": f"sustained_{task_count}",
                    "timestamp": time.time(),
                    "data": "x" * 100
                }
                
                enqueue_start = time.time()
                msg_id = enqueue_task(test_topic, payload)
                enqueue_end = time.time()
                
                assert msg_id is not None
                performance_metrics.record_enqueue_time(enqueue_end - enqueue_start)
                
                task_count += 1
                
                # Control rate
                target_interval = 1.0 / target_tps
                actual_interval = enqueue_end - enqueue_start
                if actual_interval < target_interval:
                    time.sleep(target_interval - actual_interval)
            
            return task_count
        
        performance_metrics.start_timing()
        total_tasks = sustained_enqueuer()
        performance_metrics.end_timing()
        
        # Verify sustained performance
        summary = performance_metrics.get_summary()
        actual_tps = total_tasks / duration_seconds
        
        # Should maintain target TPS
        assert actual_tps >= target_tps * 0.9  # Within 90% of target
        
        # Enqueue times should remain stable
        enqueue_metrics = summary["enqueue_metrics"]
        assert enqueue_metrics["mean"] < 0.1  # Stable performance
        
        print(f"Sustained load: {actual_tps:.2f} TPS for {duration_seconds}s")
        print(f"Total tasks: {total_tasks}")
    
    def test_memory_usage_under_load(self, redis_client, test_topic, performance_metrics):
        """Test memory usage doesn't grow unbounded under load."""
        num_batches = 10
        batch_size = 100
        
        # Reset enqueue cache
        if hasattr(enqueue_task, '_stream_cache'):
            enqueue_task._stream_cache = {}
        
        memory_samples = []
        
        for batch in range(num_batches):
            # Enqueue batch
            for i in range(batch_size):
                payload = {
                    "task_type": "memory_test",
                    "batch": batch,
                    "task_index": i,
                    "data": "x" * 1024,  # 1KB per task
                    "timestamp": time.time()
                }
                
                msg_id = enqueue_task(test_topic, payload)
                assert msg_id is not None
            
            # Sample memory usage (stream length as proxy)
            stream_len = redis_client.xlen(task_stream(test_topic))
            memory_samples.append(stream_len)
            
            # Brief pause between batches
            time.sleep(0.1)
        
        # Memory usage should be bounded by stream maxlen
        max_memory = max(memory_samples)
        final_memory = memory_samples[-1]
        
        # Should not grow unbounded (maxlen=10000 by default)
        assert max_memory <= 11000  # Allow some buffer
        
        # Memory growth should stabilize
        recent_samples = memory_samples[-3:]
        memory_variance = statistics.variance(recent_samples) if len(recent_samples) > 1 else 0
        assert memory_variance < 1000  # Low variance in recent samples
        
        print(f"Max memory usage: {max_memory} messages")
        print(f"Final memory usage: {final_memory} messages")
    
    def test_large_payload_performance(self, redis_client, test_topic, performance_metrics):
        """Test performance with large payloads."""
        payload_sizes = [1, 10, 50, 100]  # KB
        tasks_per_size = 20
        
        for size_kb in payload_sizes:
            # Reset enqueue cache
            if hasattr(enqueue_task, '_stream_cache'):
                enqueue_task._stream_cache = {}
            
            size_times = []
            
            for i in range(tasks_per_size):
                payload = {
                    "task_type": "large_payload_test",
                    "size_kb": size_kb,
                    "task_index": i,
                    "large_data": "x" * (size_kb * 1024),
                    "timestamp": time.time()
                }
                
                start_time = time.time()
                msg_id = enqueue_task(test_topic, payload)
                end_time = time.time()
                
                assert msg_id is not None
                size_times.append(end_time - start_time)
            
            # Record average time for this payload size
            avg_time = statistics.mean(size_times)
            performance_metrics.record_enqueue_time(avg_time)
            
            print(f"Payload {size_kb}KB: avg {avg_time*1000:.2f}ms")
        
        # Performance should degrade gracefully with size
        # But should still be reasonable for large payloads
        summary = performance_metrics.get_summary()
        enqueue_metrics = summary["enqueue_metrics"]
        
        # Even large payloads should be handled reasonably
        assert enqueue_metrics["max"] < 1.0  # Less than 1 second even for 100KB

class TestConcurrencyScenarios:
    """Test concurrency scenarios."""
    
    def test_reader_writer_concurrency(self, redis_client, test_topic, performance_metrics):
        """Test concurrent readers and writers."""
        duration_seconds = 15
        num_writers = 3
        num_readers = 2
        
        # Shared state
        write_count = threading.Value('i', 0)
        read_count = threading.Value('i', 0)
        errors = threading.Value('i', 0)
        
        def writer_worker(writer_id: int):
            """Writer worker function."""
            local_count = 0
            start_time = time.time()
            
            # Reset enqueue cache per worker
            if hasattr(enqueue_task, '_stream_cache'):
                enqueue_task._stream_cache = {}
            
            while time.time() - start_time < duration_seconds:
                try:
                    payload = {
                        "task_type": "concurrency_test",
                        "writer_id": writer_id,
                        "local_count": local_count,
                        "timestamp": time.time()
                    }
                    
                    msg_id = enqueue_task(test_topic, payload)
                    assert msg_id is not None
                    
                    with write_count.get_lock():
                        write_count.value += 1
                    
                    local_count += 1
                    time.sleep(0.01)  # 100 TPS per writer
                    
                except Exception as e:
                    with errors.get_lock():
                        errors.value += 1
        
        def reader_worker(reader_id: int):
            """Reader worker function."""
            local_count = 0
            start_time = time.time()
            
            while time.time() - start_time < duration_seconds:
                try:
                    results = tail_results(test_topic, count=10)
                    local_count += len(results)
                    
                    with read_count.get_lock():
                        read_count.value += len(results)
                    
                    time.sleep(0.1)  # Read every 100ms
                    
                except Exception as e:
                    with errors.get_lock():
                        errors.value += 1
        
        performance_metrics.start_timing()
        
        # Start all workers
        workers = []
        
        # Start writers
        for i in range(num_writers):
            worker = threading.Thread(target=writer_worker, args=(i,))
            worker.start()
            workers.append(worker)
        
        # Start readers
        for i in range(num_readers):
            worker = threading.Thread(target=reader_worker, args=(i,))
            worker.start()
            workers.append(worker)
        
        # Wait for completion
        for worker in workers:
            worker.join()
        
        performance_metrics.end_timing()
        
        # Verify concurrent operations
        assert write_count.value > 0
        assert read_count.value >= 0  # Might be 0 if no writes completed before reads
        assert errors.value == 0  # No errors during concurrent operations
        
        # Calculate throughput
        summary = performance_metrics.get_summary()
        total_time = summary["total_duration"]
        write_tps = write_count.value / total_time
        
        print(f"Concurrent performance: {write_tps:.2f} writes/sec")
        print(f"Total writes: {write_count.value}, reads: {read_count.value}")
        print(f"Errors: {errors.value}")
    
    def test_multiple_consumer_groups(self, redis_client, test_topic, performance_metrics):
        """Test performance with multiple consumer groups."""
        num_groups = 5
        num_consumers_per_group = 2
        num_tasks = 100
        
        # Create consumer groups
        groups = [f"load_test_group_{i}" for i in range(num_groups)]
        for group in groups:
            try:
                create_consumer_group(test_topic, group)
            except:
                pass  # Group might exist
        
        # Enqueue tasks
        # Reset enqueue cache
        if hasattr(enqueue_task, '_stream_cache'):
            enqueue_task._stream_cache = {}
        
        start_time = time.time()
        for i in range(num_tasks):
            payload = {
                "task_type": "multi_group_test",
                "task_id": f"task_{i}",
                "timestamp": time.time()
            }
            
            msg_id = enqueue_task(test_topic, payload)
            assert msg_id is not None
        
        enqueue_time = time.time() - start_time
        
        # Test reading from multiple groups concurrently
        def consumer_worker(group: str, consumer_id: int) -> int:
            """Consumer worker function."""
            messages_read = 0
            consumer_name = f"consumer_{group}_{consumer_id}"
            
            for _ in range(10):  # Attempt to read up to 10 times
                try:
                    messages = read_group(test_topic, group, consumer_name, count=5, block_ms=100)
                    
                    for msg_id, data in messages:
                        # Acknowledge message
                        ack(test_topic, group, msg_id)
                        messages_read += 1
                    
                    if not messages:
                        break  # No more messages
                        
                except Exception:
                    break
            
            return messages_read
        
        performance_metrics.start_timing()
        
        # Start all consumers
        with ThreadPoolExecutor(max_workers=num_groups * num_consumers_per_group) as executor:
            futures = []
            for group in groups:
                for consumer_id in range(num_consumers_per_group):
                    future = executor.submit(consumer_worker, group, consumer_id)
                    futures.append(future)
            
            # Collect results
            total_consumed = 0
            for future in as_completed(futures):
                try:
                    consumed = future.result()
                    total_consumed += consumed
                except Exception as e:
                    performance_metrics.record_error()
        
        performance_metrics.end_timing()
        
        # Verify consumption
        # Each group should receive all messages, so total consumed should be
        # num_tasks * num_groups (each group gets copy of all messages)
        expected_min = num_tasks * num_groups * 0.8  # Allow 80% success
        assert total_consumed >= expected_min
        
        # Performance metrics
        summary = performance_metrics.get_summary()
        consumption_time = summary["total_duration"]
        consumption_tps = total_consumed / consumption_time
        
        print(f"Multi-group consumption: {consumption_tps:.2f} messages/sec")
        print(f"Total consumed: {total_consumed}")
        print(f"Enqueue time: {enqueue_time:.2f}s, consumption time: {consumption_time:.2f}s")

class TestStressScenarios:
    """Test stress scenarios pushing system limits."""
    
    @pytest.mark.slow
    def test_maximum_throughput(self, redis_client, test_topic, performance_metrics):
        """Test maximum throughput capacity."""
        duration_seconds = 60
        num_workers = 20
        
        # Shared counters
        total_enqueued = threading.Value('i', 0)
        total_errors = threading.Value('i', 0)
        
        def max_throughput_worker(worker_id: int):
            """Worker pushing maximum throughput."""
            local_count = 0
            start_time = time.time()
            
            # Reset enqueue cache per worker
            if hasattr(enqueue_task, '_stream_cache'):
                enqueue_task._stream_cache = {}
            
            while time.time() - start_time < duration_seconds:
                try:
                    payload = {
                        "task_type": "max_throughput",
                        "worker_id": worker_id,
                        "local_count": local_count,
                        "timestamp": time.time()
                    }
                    
                    enqueue_start = time.time()
                    msg_id = enqueue_task(test_topic, payload)
                    enqueue_end = time.time()
                    
                    assert msg_id is not None
                    
                    with total_enqueued.get_lock():
                        total_enqueued.value += 1
                    
                    performance_metrics.record_enqueue_time(enqueue_end - enqueue_start)
                    local_count += 1
                    
                except Exception as e:
                    with total_errors.get_lock():
                        total_errors.value += 1
        
        performance_metrics.start_timing()
        
        # Start all workers
        workers = []
        for i in range(num_workers):
            worker = threading.Thread(target=max_throughput_worker, args=(i,))
            worker.start()
            workers.append(worker)
        
        # Wait for completion
        for worker in workers:
            worker.join()
        
        performance_metrics.end_timing()
        
        # Calculate maximum throughput achieved
        summary = performance_metrics.get_summary()
        total_time = summary["total_duration"]
        max_throughput = total_enqueued.value / total_time
        
        print(f"Maximum throughput achieved: {max_throughput:.2f} tasks/sec")
        print(f"Total enqueued: {total_enqueued.value}")
        print(f"Total errors: {total_errors.value}")
        print(f"Error rate: {(total_errors.value / total_enqueued.value * 100):.2f}%")
        
        # Stress test assertions
        assert max_throughput > 500  # Should achieve at least 500 TPS
        error_rate = total_errors.value / (total_enqueued.value + total_errors.value)
        assert error_rate < 0.01  # Less than 1% error rate

class TestPerformanceRegression:
    """Test for performance regression detection."""
    
    def test_baseline_performance(self, redis_client, test_topic, performance_metrics):
        """Establish baseline performance metrics."""
        # Standard test scenario
        num_tasks = 100
        payload_size = 512  # bytes
        
        # Reset enqueue cache
        if hasattr(enqueue_task, '_stream_cache'):
            enqueue_task._stream_cache = {}
        
        performance_metrics.start_timing()
        
        for i in range(num_tasks):
            payload = {
                "task_type": "baseline_test",
                "task_id": f"baseline_{i}",
                "data": "x" * payload_size,
                "timestamp": time.time()
            }
            
            start_time = time.time()
            msg_id = enqueue_task(test_topic, payload)
            end_time = time.time()
            
            assert msg_id is not None
            performance_metrics.record_enqueue_time(end_time - start_time)
        
        performance_metrics.end_timing()
        
        # Baseline expectations
        summary = performance_metrics.get_summary()
        enqueue_metrics = summary["enqueue_metrics"]
        total_time = summary["total_duration"]
        throughput = num_tasks / total_time
        
        # Baseline performance requirements
        assert enqueue_metrics["mean"] < 0.05  # Average < 50ms
        assert enqueue_metrics["p95"] < 0.1    # 95th percentile < 100ms
        assert throughput > 100                # At least 100 TPS
        
        # Store baseline metrics for future comparison
        baseline_metrics = {
            "throughput": throughput,
            "avg_latency": enqueue_metrics["mean"],
            "p95_latency": enqueue_metrics["p95"],
            "test_config": {
                "num_tasks": num_tasks,
                "payload_size": payload_size
            }
        }
        
        print("Baseline performance metrics:")
        print(json.dumps(baseline_metrics, indent=2))
        
        return baseline_metrics

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])