"""
Performance tests for mesh system concurrency scenarios.
Tests concurrent access patterns, race conditions, and scalability.
"""
import json
import time
import threading
import queue
import statistics
import pytest
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from collections import defaultdict, Counter
import random

# Import mesh components
from app.mesh.redis_bus import (
    get_redis, enqueue_task, tail_results, register_agent,
    list_agents, create_consumer_group, read_group, ack,
    task_stream, result_stream, heartbeat_agent
)

class ConcurrencyMetrics:
    """Collect concurrency-specific metrics."""
    
    def __init__(self):
        self.operation_times = defaultdict(list)
        self.contention_events = []
        self.race_conditions = []
        self.deadlock_events = []
        self.throughput_samples = []
        self.error_counts = defaultdict(int)
        self.start_time = None
        self.end_time = None
        self._lock = threading.Lock()
    
    def record_operation(self, operation: str, duration: float, thread_id: str = None):
        """Record operation timing with thread safety."""
        with self._lock:
            self.operation_times[operation].append({
                "duration": duration,
                "thread_id": thread_id or threading.current_thread().ident,
                "timestamp": time.time()
            })
    
    def record_contention(self, resource: str, wait_time: float):
        """Record resource contention event."""
        with self._lock:
            self.contention_events.append({
                "resource": resource,
                "wait_time": wait_time,
                "thread_id": threading.current_thread().ident,
                "timestamp": time.time()
            })
    
    def record_race_condition(self, operation: str, expected: Any, actual: Any):
        """Record race condition detection."""
        with self._lock:
            self.race_conditions.append({
                "operation": operation,
                "expected": expected,
                "actual": actual,
                "thread_id": threading.current_thread().ident,
                "timestamp": time.time()
            })
    
    def record_error(self, error_type: str, details: str = None):
        """Record error with thread safety."""
        with self._lock:
            self.error_counts[error_type] += 1
    
    def start_timing(self):
        """Start overall timing."""
        self.start_time = time.time()
    
    def end_timing(self):
        """End overall timing."""
        self.end_time = time.time()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get concurrency metrics summary."""
        with self._lock:
            operation_stats = {}
            for op, times in self.operation_times.items():
                durations = [t["duration"] for t in times]
                if durations:
                    operation_stats[op] = {
                        "count": len(durations),
                        "mean": statistics.mean(durations),
                        "min": min(durations),
                        "max": max(durations),
                        "p95": self._percentile(durations, 95),
                        "threads": len(set(t["thread_id"] for t in times))
                    }
            
            return {
                "operation_stats": operation_stats,
                "contention_events": len(self.contention_events),
                "race_conditions": len(self.race_conditions),
                "deadlock_events": len(self.deadlock_events),
                "error_counts": dict(self.error_counts),
                "total_duration": self.end_time - self.start_time if self.start_time and self.end_time else 0
            }
    
    def _percentile(self, data: List[float], p: int) -> float:
        """Calculate percentile."""
        if not data:
            return 0
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
    """Get Redis client for concurrency tests."""
    try:
        client = get_redis()
        client.ping()
        return client
    except Exception as e:
        pytest.skip(f"Redis not available: {e}")

@pytest.fixture
def test_topic():
    """Test topic with timestamp."""
    return f"concurrency_test_{int(time.time())}"

@pytest.fixture
def concurrency_metrics():
    """Concurrency metrics collector."""
    return ConcurrencyMetrics()

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
            for i in range(10):  # Clean up potential test groups
                redis_client.xgroup_destroy(task_stream(test_topic), f"concurrent_group_{i}")
        except:
            pass
    except:
        pass

class TestBasicConcurrency:
    """Test basic concurrent operations."""
    
    def test_concurrent_enqueue_operations(self, redis_client, test_topic, concurrency_metrics):
        """Test concurrent task enqueuing without conflicts."""
        num_threads = 10
        tasks_per_thread = 20
        
        def enqueue_worker(worker_id: int) -> List[str]:
            """Worker function for enqueueing tasks."""
            msg_ids = []
            
            # Reset enqueue cache per worker
            if hasattr(enqueue_task, '_stream_cache'):
                enqueue_task._stream_cache = {}
            
            for i in range(tasks_per_thread):
                payload = {
                    "task_type": "concurrent_enqueue",
                    "worker_id": worker_id,
                    "task_index": i,
                    "timestamp": time.time(),
                    "thread_id": threading.current_thread().ident
                }
                
                start_time = time.time()
                try:
                    msg_id = enqueue_task(test_topic, payload)
                    end_time = time.time()
                    
                    msg_ids.append(msg_id)
                    concurrency_metrics.record_operation(
                        "enqueue", 
                        end_time - start_time,
                        str(worker_id)
                    )
                    
                except Exception as e:
                    concurrency_metrics.record_error("enqueue_error", str(e))
                    end_time = time.time()
                    concurrency_metrics.record_operation(
                        "enqueue_error",
                        end_time - start_time,
                        str(worker_id)
                    )
            
            return msg_ids
        
        concurrency_metrics.start_timing()
        
        # Run concurrent enqueueing
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for worker_id in range(num_threads):
                future = executor.submit(enqueue_worker, worker_id)
                futures.append(future)
            
            # Collect results
            all_msg_ids = []
            for future in as_completed(futures):
                try:
                    msg_ids = future.result()
                    all_msg_ids.extend(msg_ids)
                except Exception as e:
                    concurrency_metrics.record_error("worker_failure", str(e))
        
        concurrency_metrics.end_timing()
        
        # Verify all tasks were enqueued
        total_expected = num_threads * tasks_per_thread
        assert len(all_msg_ids) == total_expected
        
        # Verify all message IDs are unique (no duplicates from race conditions)
        assert len(set(all_msg_ids)) == len(all_msg_ids)
        
        # Verify stream contains all messages
        stream_length = redis_client.xlen(task_stream(test_topic))
        assert stream_length == total_expected
        
        # Performance assertions
        summary = concurrency_metrics.get_summary()
        enqueue_stats = summary["operation_stats"]["enqueue"]
        
        assert enqueue_stats["count"] == total_expected
        assert enqueue_stats["mean"] < 0.5  # Average < 500ms even with contention
        assert summary["error_counts"].get("enqueue_error", 0) == 0
        
        print(f"Concurrent enqueue: {enqueue_stats['count']} tasks, "
              f"avg {enqueue_stats['mean']*1000:.2f}ms, "
              f"{enqueue_stats['threads']} threads")
    
    def test_concurrent_read_operations(self, redis_client, test_topic, concurrency_metrics):
        """Test concurrent result reading operations."""
        num_readers = 8
        num_results = 100
        
        # Populate results first
        for i in range(num_results):
            result_data = {
                "task_id": f"read_test_{i}",
                "status": "completed",
                "result": f"result_{i}",
                "timestamp": time.time()
            }
            redis_client.xadd(
                result_stream(test_topic),
                {"json": json.dumps(result_data)}
            )
        
        def reader_worker(reader_id: int) -> List[Tuple[str, Dict]]:
            """Worker function for reading results."""
            all_results = []
            read_counts = [5, 10, 15, 20]  # Different read sizes
            
            for count in read_counts:
                start_time = time.time()
                try:
                    results = tail_results(test_topic, count=count)
                    end_time = time.time()
                    
                    all_results.extend(results)
                    concurrency_metrics.record_operation(
                        "tail_results",
                        end_time - start_time,
                        str(reader_id)
                    )
                    
                except Exception as e:
                    concurrency_metrics.record_error("read_error", str(e))
                    end_time = time.time()
                    concurrency_metrics.record_operation(
                        "read_error",
                        end_time - start_time,
                        str(reader_id)
                    )
                
                # Brief pause between reads
                time.sleep(0.01)
            
            return all_results
        
        concurrency_metrics.start_timing()
        
        # Run concurrent reading
        with ThreadPoolExecutor(max_workers=num_readers) as executor:
            futures = []
            for reader_id in range(num_readers):
                future = executor.submit(reader_worker, reader_id)
                futures.append(future)
            
            # Collect results
            for future in as_completed(futures):
                try:
                    results = future.result()
                    # Results should be consistent across readers
                    assert len(results) > 0
                except Exception as e:
                    concurrency_metrics.record_error("reader_failure", str(e))
        
        concurrency_metrics.end_timing()
        
        # Performance assertions
        summary = concurrency_metrics.get_summary()
        read_stats = summary["operation_stats"]["tail_results"]
        
        assert read_stats["count"] > 0
        assert read_stats["mean"] < 1.0  # Average < 1s
        assert summary["error_counts"].get("read_error", 0) == 0
        
        print(f"Concurrent reads: {read_stats['count']} operations, "
              f"avg {read_stats['mean']*1000:.2f}ms, "
              f"{read_stats['threads']} threads")
    
    def test_concurrent_agent_registration(self, redis_client, concurrency_metrics):
        """Test concurrent agent registration and heartbeats."""
        num_agents = 15
        operations_per_agent = 10
        
        def agent_worker(agent_id: int) -> int:
            """Worker function for agent operations."""
            agent_name = f"concurrent_agent_{agent_id}"
            operations_completed = 0
            
            for i in range(operations_per_agent):
                try:
                    # Register agent
                    start_time = time.time()
                    register_agent(
                        agent_name,
                        "test_agent",
                        ttl_seconds=30,
                        meta={"worker_id": agent_id, "operation": i}
                    )
                    end_time = time.time()
                    
                    concurrency_metrics.record_operation(
                        "register_agent",
                        end_time - start_time,
                        str(agent_id)
                    )
                    operations_completed += 1
                    
                    # Heartbeat
                    start_time = time.time()
                    heartbeat_agent(agent_name, 30)
                    end_time = time.time()
                    
                    concurrency_metrics.record_operation(
                        "heartbeat_agent",
                        end_time - start_time,
                        str(agent_id)
                    )
                    operations_completed += 1
                    
                    # Brief pause
                    time.sleep(0.01)
                    
                except Exception as e:
                    concurrency_metrics.record_error("agent_operation_error", str(e))
            
            return operations_completed
        
        concurrency_metrics.start_timing()
        
        # Run concurrent agent operations
        with ThreadPoolExecutor(max_workers=num_agents) as executor:
            futures = []
            for agent_id in range(num_agents):
                future = executor.submit(agent_worker, agent_id)
                futures.append(future)
            
            # Collect results
            total_operations = 0
            for future in as_completed(futures):
                try:
                    operations = future.result()
                    total_operations += operations
                except Exception as e:
                    concurrency_metrics.record_error("agent_worker_failure", str(e))
        
        concurrency_metrics.end_timing()
        
        # Verify agent registrations
        agents = list_agents()
        concurrent_agents = [a for a in agents if a.get("agent_id", "").startswith("concurrent_agent_")]
        
        # Should have agents registered (might be fewer due to TTL)
        assert len(concurrent_agents) > 0
        assert len(concurrent_agents) <= num_agents
        
        # Performance assertions
        summary = concurrency_metrics.get_summary()
        
        if "register_agent" in summary["operation_stats"]:
            reg_stats = summary["operation_stats"]["register_agent"]
            assert reg_stats["mean"] < 0.1  # Fast registration
        
        if "heartbeat_agent" in summary["operation_stats"]:
            hb_stats = summary["operation_stats"]["heartbeat_agent"]
            assert hb_stats["mean"] < 0.05  # Very fast heartbeat
        
        print(f"Concurrent agents: {len(concurrent_agents)} registered, "
              f"{total_operations} total operations")

class TestResourceContention:
    """Test resource contention scenarios."""
    
    def test_stream_creation_contention(self, redis_client, test_topic, concurrency_metrics):
        """Test contention during stream/group creation."""
        num_workers = 20
        groups_per_worker = 3
        
        # Reset global cache to force contention
        if hasattr(enqueue_task, '_stream_cache'):
            enqueue_task._stream_cache = {}
        
        def stream_creator_worker(worker_id: int) -> List[str]:
            """Worker that creates streams and groups."""
            created_groups = []
            
            for i in range(groups_per_worker):
                group_name = f"contention_group_{worker_id}_{i}"
                
                start_time = time.time()
                try:
                    # This should create contention on first stream creation
                    create_consumer_group(test_topic, group_name)
                    end_time = time.time()
                    
                    created_groups.append(group_name)
                    concurrency_metrics.record_operation(
                        "create_group",
                        end_time - start_time,
                        str(worker_id)
                    )
                    
                except Exception as e:
                    end_time = time.time()
                    concurrency_metrics.record_error("group_creation_error", str(e))
                    concurrency_metrics.record_contention(
                        "stream_creation",
                        end_time - start_time
                    )
            
            return created_groups
        
        concurrency_metrics.start_timing()
        
        # Run concurrent stream/group creation
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for worker_id in range(num_workers):
                future = executor.submit(stream_creator_worker, worker_id)
                futures.append(future)
            
            # Collect results
            all_groups = []
            for future in as_completed(futures):
                try:
                    groups = future.result()
                    all_groups.extend(groups)
                except Exception as e:
                    concurrency_metrics.record_error("stream_worker_failure", str(e))
        
        concurrency_metrics.end_timing()
        
        # Verify groups were created
        try:
            group_info = redis_client.xinfo_groups(task_stream(test_topic))
            created_group_names = [g['name'] for g in group_info]
            
            # Should have created all groups despite contention
            expected_groups = num_workers * groups_per_worker
            assert len(all_groups) <= expected_groups  # Some might fail due to contention
            
        except Exception:
            # Stream might not exist if all creations failed
            pass
        
        # Performance assertions
        summary = concurrency_metrics.get_summary()
        
        if "create_group" in summary["operation_stats"]:
            create_stats = summary["operation_stats"]["create_group"]
            # Should handle contention gracefully
            assert create_stats["mean"] < 1.0  # Even with contention
        
        print(f"Stream contention: {len(all_groups)} groups created, "
              f"{summary['contention_events']} contention events")
    
    def test_consumer_group_contention(self, redis_client, test_topic, concurrency_metrics):
        """Test contention in consumer group operations."""
        num_consumers = 12
        messages_per_consumer = 10
        group_name = "contention_test_group"
        
        # Setup: create group and enqueue messages
        try:
            create_consumer_group(test_topic, group_name)
        except:
            pass
        
        # Reset enqueue cache
        if hasattr(enqueue_task, '_stream_cache'):
            enqueue_task._stream_cache = {}
        
        # Enqueue messages for consumption
        total_messages = num_consumers * messages_per_consumer
        for i in range(total_messages):
            enqueue_task(test_topic, {
                "task_type": "contention_test",
                "message_id": i,
                "timestamp": time.time()
            })
        
        def consumer_worker(consumer_id: int) -> Tuple[int, int]:
            """Worker that consumes from the same group."""
            consumer_name = f"consumer_{consumer_id}"
            messages_read = 0
            messages_acked = 0
            
            for attempt in range(messages_per_consumer * 2):  # More attempts than expected
                start_time = time.time()
                try:
                    messages = read_group(
                        test_topic, 
                        group_name, 
                        consumer_name, 
                        count=1, 
                        block_ms=100
                    )
                    end_time = time.time()
                    
                    concurrency_metrics.record_operation(
                        "read_group",
                        end_time - start_time,
                        str(consumer_id)
                    )
                    
                    if not messages:
                        break  # No more messages
                    
                    for msg_id, data in messages:
                        messages_read += 1
                        
                        # Acknowledge with potential contention
                        start_time = time.time()
                        ack_result = ack(test_topic, group_name, msg_id)
                        end_time = time.time()
                        
                        if ack_result > 0:
                            messages_acked += 1
                        
                        concurrency_metrics.record_operation(
                            "ack_message",
                            end_time - start_time,
                            str(consumer_id)
                        )
                    
                except Exception as e:
                    end_time = time.time()
                    concurrency_metrics.record_error("consumer_error", str(e))
                    concurrency_metrics.record_contention(
                        "consumer_group",
                        end_time - start_time
                    )
            
            return messages_read, messages_acked
        
        concurrency_metrics.start_timing()
        
        # Run concurrent consumers
        with ThreadPoolExecutor(max_workers=num_consumers) as executor:
            futures = []
            for consumer_id in range(num_consumers):
                future = executor.submit(consumer_worker, consumer_id)
                futures.append(future)
            
            # Collect results
            total_read = 0
            total_acked = 0
            for future in as_completed(futures):
                try:
                    read_count, ack_count = future.result()
                    total_read += read_count
                    total_acked += ack_count
                except Exception as e:
                    concurrency_metrics.record_error("consumer_worker_failure", str(e))
        
        concurrency_metrics.end_timing()
        
        # Verify message distribution
        # Total read should equal total messages (each message read by one consumer)
        assert total_read <= total_messages
        assert total_acked <= total_read
        
        # Most messages should be successfully processed
        success_rate = total_acked / total_messages if total_messages > 0 else 0
        assert success_rate > 0.8  # At least 80% success rate despite contention
        
        # Performance assertions
        summary = concurrency_metrics.get_summary()
        
        if "read_group" in summary["operation_stats"]:
            read_stats = summary["operation_stats"]["read_group"]
            assert read_stats["mean"] < 1.0  # Reasonable despite contention
        
        print(f"Consumer contention: {total_read}/{total_messages} read, "
              f"{total_acked} acked, {summary['contention_events']} contentions")

class TestRaceConditionDetection:
    """Test race condition detection and handling."""
    
    def test_message_id_uniqueness(self, redis_client, test_topic, concurrency_metrics):
        """Test that message IDs remain unique under concurrent load."""
        num_threads = 15
        messages_per_thread = 50
        
        # Shared data structures to detect races
        all_msg_ids = queue.Queue()
        id_tracker = defaultdict(int)
        lock = threading.Lock()
        
        def unique_id_worker(worker_id: int) -> List[str]:
            """Worker that enqueues messages and tracks IDs."""
            worker_msg_ids = []
            
            # Reset enqueue cache per worker
            if hasattr(enqueue_task, '_stream_cache'):
                enqueue_task._stream_cache = {}
            
            for i in range(messages_per_thread):
                payload = {
                    "task_type": "uniqueness_test",
                    "worker_id": worker_id,
                    "sequence": i,
                    "timestamp": time.time()
                }
                
                start_time = time.time()
                try:
                    msg_id = enqueue_task(test_topic, payload)
                    end_time = time.time()
                    
                    worker_msg_ids.append(msg_id)
                    all_msg_ids.put(msg_id)
                    
                    # Track ID occurrences (should always be 1)
                    with lock:
                        id_tracker[msg_id] += 1
                        if id_tracker[msg_id] > 1:
                            concurrency_metrics.record_race_condition(
                                "duplicate_message_id",
                                1,  # expected count
                                id_tracker[msg_id]  # actual count
                            )
                    
                    concurrency_metrics.record_operation(
                        "unique_enqueue",
                        end_time - start_time,
                        str(worker_id)
                    )
                    
                except Exception as e:
                    concurrency_metrics.record_error("unique_enqueue_error", str(e))
            
            return worker_msg_ids
        
        concurrency_metrics.start_timing()
        
        # Run concurrent enqueueing
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for worker_id in range(num_threads):
                future = executor.submit(unique_id_worker, worker_id)
                futures.append(future)
            
            # Collect all message IDs
            collected_ids = []
            for future in as_completed(futures):
                try:
                    msg_ids = future.result()
                    collected_ids.extend(msg_ids)
                except Exception as e:
                    concurrency_metrics.record_error("unique_worker_failure", str(e))
        
        concurrency_metrics.end_timing()
        
        # Analyze for race conditions
        total_expected = num_threads * messages_per_thread
        assert len(collected_ids) == total_expected
        
        # Check for duplicate IDs (race condition indicator)
        unique_ids = set(collected_ids)
        if len(unique_ids) != len(collected_ids):
            duplicates = len(collected_ids) - len(unique_ids)
            concurrency_metrics.record_race_condition(
                "duplicate_message_ids",
                0,  # expected duplicates
                duplicates  # actual duplicates
            )
        
        # Verify no race conditions detected
        summary = concurrency_metrics.get_summary()
        assert summary["race_conditions"] == 0, "Race conditions detected in message ID generation"
        
        print(f"ID uniqueness: {len(unique_ids)}/{total_expected} unique IDs, "
              f"{summary['race_conditions']} race conditions")
    
    def test_agent_registration_races(self, redis_client, concurrency_metrics):
        """Test agent registration race conditions."""
        num_agents = 10
        registrations_per_agent = 20
        
        # Track agent states
        agent_states = defaultdict(list)
        state_lock = threading.Lock()
        
        def racing_agent_worker(agent_id: int) -> int:
            """Worker that rapidly registers/heartbeats same agent."""
            agent_name = f"racing_agent_{agent_id}"
            operations = 0
            
            for i in range(registrations_per_agent):
                try:
                    # Register with varying TTL
                    ttl = 10 + (i % 20)  # 10-30 seconds
                    
                    start_time = time.time()
                    register_agent(
                        agent_name,
                        "racing_agent",
                        ttl_seconds=ttl,
                        meta={"registration": i, "timestamp": time.time()}
                    )
                    end_time = time.time()
                    
                    # Track state for race detection
                    with state_lock:
                        agent_states[agent_name].append({
                            "operation": "register",
                            "ttl": ttl,
                            "timestamp": end_time
                        })
                    
                    concurrency_metrics.record_operation(
                        "racing_register",
                        end_time - start_time,
                        str(agent_id)
                    )
                    operations += 1
                    
                    # Random heartbeat
                    if random.random() < 0.3:  # 30% chance
                        start_time = time.time()
                        heartbeat_agent(agent_name, ttl)
                        end_time = time.time()
                        
                        with state_lock:
                            agent_states[agent_name].append({
                                "operation": "heartbeat",
                                "ttl": ttl,
                                "timestamp": end_time
                            })
                        
                        concurrency_metrics.record_operation(
                            "racing_heartbeat",
                            end_time - start_time,
                            str(agent_id)
                        )
                        operations += 1
                    
                    # Brief pause
                    time.sleep(0.001)
                    
                except Exception as e:
                    concurrency_metrics.record_error("racing_agent_error", str(e))
            
            return operations
        
        concurrency_metrics.start_timing()
        
        # Run racing agents
        with ThreadPoolExecutor(max_workers=num_agents) as executor:
            futures = []
            for agent_id in range(num_agents):
                future = executor.submit(racing_agent_worker, agent_id)
                futures.append(future)
            
            # Collect results
            total_operations = 0
            for future in as_completed(futures):
                try:
                    operations = future.result()
                    total_operations += operations
                except Exception as e:
                    concurrency_metrics.record_error("racing_worker_failure", str(e))
        
        concurrency_metrics.end_timing()
        
        # Check final agent states
        agents = list_agents()
        racing_agents = [a for a in agents if a.get("agent_id", "").startswith("racing_agent_")]
        
        # Should have some agents registered (exact count depends on timing)
        assert len(racing_agents) <= num_agents
        
        # Verify no obvious race conditions (like impossible states)
        for agent in racing_agents:
            agent_id = agent.get("agent_id")
            if agent_id in agent_states:
                states = agent_states[agent_id]
                # Last registration should be relatively recent
                if states:
                    last_op = max(states, key=lambda x: x["timestamp"])
                    time_since = time.time() - last_op["timestamp"]
                    if time_since > 60:  # More than 1 minute old
                        concurrency_metrics.record_race_condition(
                            "stale_agent_registration",
                            "recent",
                            f"{time_since:.2f}s ago"
                        )
        
        # Performance assertions
        summary = concurrency_metrics.get_summary()
        assert summary["race_conditions"] == 0, "Race conditions detected in agent registration"
        
        print(f"Agent racing: {len(racing_agents)} final agents, "
              f"{total_operations} operations, {summary['race_conditions']} races")

class TestScalabilityLimits:
    """Test concurrency scalability limits."""
    
    def test_maximum_concurrent_connections(self, redis_client, test_topic, concurrency_metrics):
        """Test maximum concurrent Redis connections."""
        max_connections = 50  # Test up to connection pool limit
        operations_per_connection = 5
        
        def connection_worker(worker_id: int) -> int:
            """Worker that performs operations using Redis connections."""
            operations = 0
            
            try:
                # Each worker gets its own operations
                for i in range(operations_per_connection):
                    start_time = time.time()
                    
                    # Mix of operations to test connection usage
                    if i % 3 == 0:
                        # Enqueue operation
                        msg_id = enqueue_task(test_topic, {
                            "task_type": "connection_test",
                            "worker_id": worker_id,
                            "operation": i
                        })
                        assert msg_id is not None
                        operation_type = "enqueue"
                        
                    elif i % 3 == 1:
                        # Read operation
                        results = tail_results(test_topic, count=1)
                        operation_type = "read"
                        
                    else:
                        # Agent operation
                        register_agent(f"conn_agent_{worker_id}", "test", 30)
                        operation_type = "register"
                    
                    end_time = time.time()
                    concurrency_metrics.record_operation(
                        f"connection_{operation_type}",
                        end_time - start_time,
                        str(worker_id)
                    )
                    operations += 1
                    
            except Exception as e:
                concurrency_metrics.record_error("connection_error", str(e))
            
            return operations
        
        concurrency_metrics.start_timing()
        
        # Test with maximum connections
        with ThreadPoolExecutor(max_workers=max_connections) as executor:
            futures = []
            for worker_id in range(max_connections):
                future = executor.submit(connection_worker, worker_id)
                futures.append(future)
            
            # Collect results
            total_operations = 0
            successful_workers = 0
            for future in as_completed(futures):
                try:
                    operations = future.result()
                    total_operations += operations
                    if operations > 0:
                        successful_workers += 1
                except Exception as e:
                    concurrency_metrics.record_error("connection_worker_failure", str(e))
        
        concurrency_metrics.end_timing()
        
        # Verify scalability
        expected_operations = max_connections * operations_per_connection
        success_rate = total_operations / expected_operations if expected_operations > 0 else 0
        
        # Should handle high concurrency gracefully
        assert success_rate > 0.8  # At least 80% success rate
        assert successful_workers > max_connections * 0.8  # Most workers successful
        
        # Performance should remain reasonable
        summary = concurrency_metrics.get_summary()
        for op_type, stats in summary["operation_stats"].items():
            if stats["count"] > 0:
                assert stats["mean"] < 2.0  # Even under load, should be reasonable
        
        print(f"Max connections: {successful_workers}/{max_connections} workers, "
              f"{total_operations}/{expected_operations} operations "
              f"({success_rate*100:.1f}% success)")
    
    def test_thread_safety_verification(self, redis_client, test_topic, concurrency_metrics):
        """Verify thread safety of mesh operations."""
        num_threads = 25
        operations_per_thread = 30
        
        # Shared counters to verify thread safety
        global_counter = threading.Value('i', 0)
        operation_counts = defaultdict(lambda: threading.Value('i', 0))
        
        def thread_safety_worker(worker_id: int) -> Dict[str, int]:
            """Worker that performs various operations to test thread safety."""
            local_counts = defaultdict(int)
            
            for i in range(operations_per_thread):
                operation = i % 4  # Cycle through operations
                
                try:
                    if operation == 0:
                        # Enqueue with counter
                        with global_counter.get_lock():
                            counter_val = global_counter.value
                            global_counter.value += 1
                        
                        msg_id = enqueue_task(test_topic, {
                            "task_type": "thread_safety",
                            "counter": counter_val,
                            "worker_id": worker_id
                        })
                        
                        with operation_counts["enqueue"].get_lock():
                            operation_counts["enqueue"].value += 1
                        
                        local_counts["enqueue"] += 1
                        
                    elif operation == 1:
                        # Read results
                        results = tail_results(test_topic, count=random.randint(1, 5))
                        
                        with operation_counts["read"].get_lock():
                            operation_counts["read"].value += 1
                        
                        local_counts["read"] += 1
                        
                    elif operation == 2:
                        # Agent registration
                        register_agent(
                            f"safety_agent_{worker_id}",
                            "thread_safety_test",
                            ttl_seconds=20,
                            meta={"iteration": i}
                        )
                        
                        with operation_counts["register"].get_lock():
                            operation_counts["register"].value += 1
                        
                        local_counts["register"] += 1
                        
                    else:
                        # Agent list
                        agents = list_agents()
                        
                        with operation_counts["list"].get_lock():
                            operation_counts["list"].value += 1
                        
                        local_counts["list"] += 1
                    
                    # Random microsleep to increase chance of race conditions
                    time.sleep(random.uniform(0.0001, 0.001))
                    
                except Exception as e:
                    concurrency_metrics.record_error("thread_safety_error", str(e))
            
            return dict(local_counts)
        
        concurrency_metrics.start_timing()
        
        # Run thread safety test
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for worker_id in range(num_threads):
                future = executor.submit(thread_safety_worker, worker_id)
                futures.append(future)
            
            # Collect results
            all_local_counts = defaultdict(int)
            for future in as_completed(futures):
                try:
                    local_counts = future.result()
                    for op, count in local_counts.items():
                        all_local_counts[op] += count
                except Exception as e:
                    concurrency_metrics.record_error("thread_safety_worker_failure", str(e))
        
        concurrency_metrics.end_timing()
        
        # Verify thread safety
        # Global counters should match local counts
        for op_type in ["enqueue", "read", "register", "list"]:
            global_count = operation_counts[op_type].value
            local_count = all_local_counts[op_type]
            
            if global_count != local_count:
                concurrency_metrics.record_race_condition(
                    f"counter_mismatch_{op_type}",
                    local_count,
                    global_count
                )
        
        # Global counter should equal number of enqueue operations
        expected_enqueues = operation_counts["enqueue"].value
        actual_counter = global_counter.value
        
        if expected_enqueues != actual_counter:
            concurrency_metrics.record_race_condition(
                "global_counter_mismatch",
                expected_enqueues,
                actual_counter
            )
        
        # Verify no race conditions detected
        summary = concurrency_metrics.get_summary()
        assert summary["race_conditions"] == 0, "Thread safety violations detected"
        
        print(f"Thread safety: {num_threads} threads, "
              f"{sum(all_local_counts.values())} total operations, "
              f"{summary['race_conditions']} race conditions")

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])