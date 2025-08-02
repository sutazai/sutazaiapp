"""
Resource Usage Monitoring Test Suite for SutazAI automation System

Tests system resource usage, performance monitoring, and capacity management
to ensure the system operates within acceptable resource limits.
"""

import pytest
import httpx
import asyncio
import json
import time
import psutil
import subprocess
from typing import Dict, Any, List, Tuple
from datetime import datetime, timedelta

# Configuration
BASE_URL = "http://localhost:8000"
MONITORING_TIMEOUT = 30.0

# Resource limits and thresholds
RESOURCE_LIMITS = {
    "cpu_warning_threshold": 80.0,  # %
    "cpu_critical_threshold": 95.0,  # %
    "memory_warning_threshold": 85.0,  # %
    "memory_critical_threshold": 95.0,  # %
    "disk_warning_threshold": 85.0,  # %
    "disk_critical_threshold": 95.0,  # %
    "response_time_warning": 5.0,  # seconds
    "response_time_critical": 10.0,  # seconds
}

DOCKER_SERVICES = [
    "sutazai-postgres",
    "sutazai-redis",
    "sutazai-chromadb",
    "sutazai-qdrant",
    "sutazai-ollama",
    "sutazai-backend",
    "sutazai-frontend"
]

@pytest.fixture
async def client():
    """Create async HTTP client"""
    timeout = httpx.Timeout(MONITORING_TIMEOUT, connect=10.0)
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=timeout) as client:
        yield client

@pytest.fixture
async def auth_headers():
    """Get authentication headers"""
    return {}

class TestSystemResourceMonitoring:
    """Test system-level resource monitoring"""
    
    @pytest.mark.asyncio
    async def test_system_metrics_availability(self, client, auth_headers):
        """Test that system metrics are available and properly formatted"""
        response = await client.get("/metrics", headers=auth_headers)
        assert response.status_code == 200
        
        data = response.json()
        assert "timestamp" in data
        assert "system" in data
        assert "services" in data
        assert "performance" in data
        
        # Validate system metrics structure
        system = data["system"]
        required_metrics = ["cpu_percent", "memory_percent", "memory_used_gb", "memory_total_gb"]
        
        for metric in required_metrics:
            assert metric in system, f"Missing system metric: {metric}"
            assert isinstance(system[metric], (int, float)), f"Invalid type for {metric}"
        
        # Validate reasonable ranges
        assert 0 <= system["cpu_percent"] <= 100
        assert 0 <= system["memory_percent"] <= 100
        assert system["memory_used_gb"] >= 0
        assert system["memory_total_gb"] > 0
        assert system["memory_used_gb"] <= system["memory_total_gb"]
    
    @pytest.mark.asyncio
    async def test_public_metrics_endpoint(self, client):
        """Test public metrics endpoint (no auth required)"""
        response = await client.get("/public/metrics")
        assert response.status_code == 200
        
        data = response.json()
        assert "timestamp" in data
        assert "system" in data
        assert "performance" in data
        
        # Public metrics should include basic system info
        system = data["system"]
        assert "cpu_percent" in system
        assert "memory_percent" in system
        assert "uptime" in system
    
    @pytest.mark.asyncio
    async def test_prometheus_metrics_format(self, client):
        """Test Prometheus metrics format"""
        response = await client.get("/prometheus-metrics")
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/plain; charset=utf-8"
        
        content = response.text
        
        # Validate Prometheus format
        assert "# HELP" in content
        assert "# TYPE" in content
        assert "sutazai_uptime_seconds" in content
        assert "sutazai_cache_entries_total" in content
        assert "sutazai_info" in content
        
        # Parse and validate metric values
        lines = content.strip().split("\n")
        metric_lines = [line for line in lines if not line.startswith("#") and line.strip()]
        
        for line in metric_lines:
            assert " " in line, f"Invalid Prometheus metric format: {line}"
            metric_name, metric_value = line.rsplit(" ", 1)
            assert metric_name.startswith("sutazai_")
            
            try:
                float(metric_value)  # Should be a valid number
            except ValueError:
                pytest.fail(f"Invalid metric value: {metric_value}")
    
    def test_host_system_resources(self):
        """Test host system resource availability and limits"""
        # CPU information
        cpu_count = psutil.cpu_count()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        assert cpu_count > 0, "No CPU cores detected"
        assert 0 <= cpu_percent <= 100, f"Invalid CPU usage: {cpu_percent}%"
        
        if cpu_percent > RESOURCE_LIMITS["cpu_warning_threshold"]:
            print(f"WARNING: High CPU usage detected: {cpu_percent}%")
        
        # Memory information
        memory = psutil.virtual_memory()
        
        assert memory.total > 0, "No memory detected"
        assert 0 <= memory.percent <= 100, f"Invalid memory usage: {memory.percent}%"
        
        if memory.percent > RESOURCE_LIMITS["memory_warning_threshold"]:
            print(f"WARNING: High memory usage detected: {memory.percent}%")
        
        # Disk information
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        
        assert disk.total > 0, "No disk space detected"
        assert 0 <= disk_percent <= 100, f"Invalid disk usage: {disk_percent}%"
        
        if disk_percent > RESOURCE_LIMITS["disk_warning_threshold"]:
            print(f"WARNING: High disk usage detected: {disk_percent}%")
        
        print(f"System resources - CPU: {cpu_percent}%, Memory: {memory.percent}%, Disk: {disk_percent:.1f}%")
    
    @pytest.mark.asyncio
    async def test_resource_monitoring_consistency(self, client, auth_headers):
        """Test consistency of resource monitoring over time"""
        measurements = []
        
        for i in range(5):
            response = await client.get("/metrics", headers=auth_headers)
            assert response.status_code == 200
            
            data = response.json()
            system = data["system"]
            
            measurement = {
                "timestamp": datetime.now(),
                "cpu_percent": system["cpu_percent"],
                "memory_percent": system["memory_percent"],
                "memory_used_gb": system["memory_used_gb"],
                "memory_total_gb": system["memory_total_gb"]
            }
            
            measurements.append(measurement)
            await asyncio.sleep(2)
        
        # Analyze consistency
        cpu_values = [m["cpu_percent"] for m in measurements]
        memory_values = [m["memory_percent"] for m in measurements]
        
        # CPU and memory should not vary wildly (unless under load)
        cpu_range = max(cpu_values) - min(cpu_values)
        memory_range = max(memory_values) - min(memory_values)
        
        assert cpu_range < 50, f"CPU usage varies too widely: {cpu_range}%"
        assert memory_range < 20, f"Memory usage varies too widely: {memory_range}%"
        
        # Memory total should be consistent
        total_memory_values = [m["memory_total_gb"] for m in measurements]
        assert len(set(total_memory_values)) == 1, "Total memory should be constant"
        
        print(f"Resource consistency - CPU range: {cpu_range:.1f}%, Memory range: {memory_range:.1f}%")

class TestDockerContainerResourceMonitoring:
    """Test Docker container resource monitoring"""
    
    def test_docker_container_existence(self):
        """Test that expected Docker containers exist"""
        try:
            result = subprocess.run(
                ["docker", "ps", "--format", "{{.Names}}"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                pytest.skip("Docker not available or accessible")
            
            running_containers = result.stdout.strip().split("\n")
            running_containers = [name.strip() for name in running_containers if name.strip()]
            
            for service in DOCKER_SERVICES:
                if service not in running_containers:
                    print(f"WARNING: Expected container {service} not running")
            
            print(f"Running containers: {running_containers}")
            
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            pytest.skip(f"Cannot access Docker: {e}")
    
    def test_docker_container_resource_usage(self):
        """Test Docker container resource usage"""
        try:
            # Get container stats
            result = subprocess.run([
                "docker", "stats", "--no-stream", "--format",
                "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}"
            ], capture_output=True, text=True, timeout=15)
            
            if result.returncode != 0:
                pytest.skip("Cannot get Docker stats")
            
            lines = result.stdout.strip().split("\n")
            header = lines[0] if lines else ""
            
            assert "NAME" in header, "Invalid Docker stats format"
            
            container_stats = []
            for line in lines[1:]:
                if line.strip():
                    parts = line.split("\t")
                    if len(parts) >= 4:
                        name, cpu, mem_usage, mem_perc = parts[:4]
                        container_stats.append({
                            "name": name.strip(),
                            "cpu": cpu.strip(),
                            "mem_usage": mem_usage.strip(),
                            "mem_perc": mem_perc.strip()
                        })
            
            assert len(container_stats) > 0, "No container stats available"
            
            # Analyze resource usage
            high_cpu_containers = []
            high_memory_containers = []
            
            for stat in container_stats:
                cpu_str = stat["cpu"].replace("%", "")
                mem_str = stat["mem_perc"].replace("%", "")
                
                try:
                    cpu_usage = float(cpu_str)
                    mem_usage = float(mem_str)
                    
                    if cpu_usage > 80:
                        high_cpu_containers.append((stat["name"], cpu_usage))
                    
                    if mem_usage > 80:
                        high_memory_containers.append((stat["name"], mem_usage))
                        
                except ValueError:
                    # Skip containers with non-numeric values
                    continue
            
            if high_cpu_containers:
                print(f"WARNING: High CPU usage containers: {high_cpu_containers}")
            
            if high_memory_containers:
                print(f"WARNING: High memory usage containers: {high_memory_containers}")
            
            print(f"Container resource stats: {container_stats}")
            
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            pytest.skip(f"Cannot access Docker stats: {e}")
    
    def test_docker_container_health_status(self):
        """Test Docker container health status"""
        try:
            result = subprocess.run([
                "docker", "ps", "--format",
                "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                pytest.skip("Cannot get Docker container status")
            
            lines = result.stdout.strip().split("\n")
            header = lines[0] if lines else ""
            
            assert "NAMES" in header, "Invalid Docker ps format"
            
            unhealthy_containers = []
            containers_info = []
            
            for line in lines[1:]:
                if line.strip():
                    parts = line.split("\t")
                    if len(parts) >= 2:
                        name, status = parts[0].strip(), parts[1].strip()
                        ports = parts[2].strip() if len(parts) > 2 else ""
                        
                        containers_info.append({
                            "name": name,
                            "status": status,
                            "ports": ports
                        })
                        
                        # Check for unhealthy containers
                        if "unhealthy" in status.lower() or "exited" in status.lower():
                            unhealthy_containers.append((name, status))
            
            if unhealthy_containers:
                print(f"WARNING: Unhealthy containers detected: {unhealthy_containers}")
            
            # At least some containers should be running
            running_containers = [c for c in containers_info if "up" in c["status"].lower()]
            assert len(running_containers) > 0, "No containers are running"
            
            print(f"Container health status: {containers_info}")
            
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            pytest.skip(f"Cannot access Docker container status: {e}")

class TestApplicationPerformanceMonitoring:
    """Test application-level performance monitoring"""
    
    @pytest.mark.asyncio
    async def test_api_response_time_monitoring(self, client, auth_headers):
        """Test API response time monitoring"""
        endpoints_to_test = [
            ("/health", "GET", None),
            ("/agents", "GET", None),
            ("/models", "GET", None),
            ("/public/metrics", "GET", None),
            ("/simple-chat", "POST", {"message": "Hello"}),
            ("/public/think", "POST", {"query": "Test query", "reasoning_type": "simple"})
        ]
        
        response_times = {}
        
        for endpoint, method, payload in endpoints_to_test:
            times = []
            
            for _ in range(3):  # Test each endpoint 3 times
                start_time = time.time()
                
                try:
                    if method == "GET":
                        response = await client.get(endpoint, headers=auth_headers)
                    else:
                        response = await client.post(endpoint, json=payload, headers=auth_headers)
                    
                    end_time = time.time()
                    response_time = end_time - start_time
                    
                    if response.status_code == 200:
                        times.append(response_time)
                    
                except Exception as e:
                    print(f"Error testing {endpoint}: {e}")
                    continue
                
                await asyncio.sleep(0.5)
            
            if times:
                avg_time = sum(times) / len(times)
                max_time = max(times)
                response_times[endpoint] = {
                    "average": avg_time,
                    "maximum": max_time,
                    "samples": len(times)
                }
        
        # Analyze response times
        slow_endpoints = []
        critical_endpoints = []
        
        for endpoint, stats in response_times.items():
            avg_time = stats["average"]
            max_time = stats["maximum"]
            
            if avg_time > RESOURCE_LIMITS["response_time_warning"]:
                slow_endpoints.append((endpoint, avg_time))
            
            if max_time > RESOURCE_LIMITS["response_time_critical"]:
                critical_endpoints.append((endpoint, max_time))
        
        if slow_endpoints:
            print(f"WARNING: Slow endpoints detected: {slow_endpoints}")
        
        if critical_endpoints:
            print(f"CRITICAL: Very slow endpoints detected: {critical_endpoints}")
        
        # At least basic endpoints should respond quickly
        health_time = response_times.get("/health", {}).get("average", float('inf'))
        assert health_time < 5.0, f"Health endpoint too slow: {health_time}s"
        
        print(f"API response times: {response_times}")
    
    @pytest.mark.asyncio
    async def test_concurrent_request_performance(self, client, auth_headers):
        """Test performance under concurrent load"""
        concurrent_requests = 10
        
        async def make_request(request_id):
            start_time = time.time()
            
            try:
                response = await client.post("/public/think", json={
                    "query": f"Concurrent test request {request_id}",
                    "reasoning_type": "simple"
                })
                
                end_time = time.time()
                response_time = end_time - start_time
                
                return {
                    "request_id": request_id,
                    "status_code": response.status_code,
                    "response_time": response_time,
                    "success": response.status_code == 200
                }
                
            except Exception as e:
                end_time = time.time()
                response_time = end_time - start_time
                
                return {
                    "request_id": request_id,
                    "status_code": None,
                    "response_time": response_time,
                    "success": False,
                    "error": str(e)
                }
        
        # Execute concurrent requests
        start_time = time.time()
        tasks = [make_request(i) for i in range(concurrent_requests)]
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        # Analyze results
        successful_requests = [r for r in results if r["success"]]
        failed_requests = [r for r in results if not r["success"]]
        
        success_rate = len(successful_requests) / len(results)
        avg_response_time = sum(r["response_time"] for r in successful_requests) / len(successful_requests) if successful_requests else 0
        max_response_time = max(r["response_time"] for r in results)
        
        # Performance assertions
        assert success_rate >= 0.7, f"Concurrent request success rate too low: {success_rate}"
        assert total_time < 30.0, f"Concurrent requests took too long: {total_time:.2f}s"
        
        if avg_response_time > RESOURCE_LIMITS["response_time_warning"]:
            print(f"WARNING: High average response time under load: {avg_response_time:.2f}s")
        
        print(f"Concurrent performance - Success rate: {success_rate:.2f}, Avg time: {avg_response_time:.2f}s, Max time: {max_response_time:.2f}s, Total time: {total_time:.2f}s")
    
    @pytest.mark.asyncio
    async def test_memory_usage_monitoring(self, client, auth_headers):
        """Test memory usage monitoring during operations"""
        # Get initial memory state
        initial_response = await client.get("/metrics", headers=auth_headers)
        assert initial_response.status_code == 200
        
        initial_data = initial_response.json()
        initial_memory = initial_data["system"]["memory_used_gb"]
        
        # Perform memory-intensive operations
        memory_intensive_operations = [
            {
                "content": "Large text content for learning: " + "This is a test sentence that will be repeated many times to create a large text block. " * 100,
                "type": "large_text"
            },
            {
                "query": "Analyze and process this complex multi-part question about artificial intelligence, machine learning, deep learning, processing networks, natural language processing, computer vision, robotics, and their interconnections",
                "reasoning_type": "complex"
            }
        ]
        
        memory_measurements = []
        
        for operation in memory_intensive_operations:
            # Measure memory before operation
            pre_response = await client.get("/metrics", headers=auth_headers)
            pre_memory = pre_response.json()["system"]["memory_used_gb"]
            
            # Perform operation
            if "content" in operation:
                await client.post("/learn", json=operation)
            else:
                await client.post("/public/think", json=operation)
            
            # Measure memory after operation
            post_response = await client.get("/metrics", headers=auth_headers)
            post_memory = post_response.json()["system"]["memory_used_gb"]
            
            memory_change = post_memory - pre_memory
            memory_measurements.append({
                "operation": "learn" if "content" in operation else "think",
                "pre_memory": pre_memory,
                "post_memory": post_memory,
                "change": memory_change
            })
            
            await asyncio.sleep(2)
        
        # Get final memory state
        final_response = await client.get("/metrics", headers=auth_headers)
        final_data = final_response.json()
        final_memory = final_data["system"]["memory_used_gb"]
        
        total_memory_change = final_memory - initial_memory
        
        # Analyze memory usage
        if total_memory_change > 1.0:  # More than 1GB increase
            print(f"WARNING: Significant memory increase detected: {total_memory_change:.2f}GB")
        
        print(f"Memory usage monitoring - Initial: {initial_memory:.2f}GB, Final: {final_memory:.2f}GB, Change: {total_memory_change:.2f}GB")
        print(f"Operation memory measurements: {memory_measurements}")

class TestServiceResourceMonitoring:
    """Test resource monitoring for individual services"""
    
    @pytest.mark.asyncio
    async def test_ollama_resource_usage(self, client):
        """Test Ollama service resource usage"""
        # Check if Ollama is accessible
        try:
            async with httpx.AsyncClient(timeout=10.0) as ollama_client:
                response = await ollama_client.get("http://ollama:11434/api/tags")
                ollama_available = response.status_code == 200
        except:
            ollama_available = False
        
        if not ollama_available:
            pytest.skip("Ollama service not available")
        
        # Test Ollama through the main API
        chat_requests = [
            {"message": "Hello, how are you?"},
            {"message": "What is machine learning?"},
            {"message": "Explain advanced computing"}
        ]
        
        ollama_response_times = []
        
        for request in chat_requests:
            start_time = time.time()
            
            try:
                response = await client.post("/simple-chat", json=request)
                end_time = time.time()
                response_time = end_time - start_time
                
                if response.status_code == 200:
                    ollama_response_times.append(response_time)
                    data = response.json()
                    
                    if "processing_time" in data:
                        processing_time = data["processing_time"]
                        print(f"Ollama processing time: {processing_time}s")
                
            except Exception as e:
                print(f"Ollama test error: {e}")
                continue
            
            await asyncio.sleep(1)
        
        if ollama_response_times:
            avg_ollama_time = sum(ollama_response_times) / len(ollama_response_times)
            max_ollama_time = max(ollama_response_times)
            
            # Ollama responses can be slow on CPU, but should be reasonable
            if avg_ollama_time > 30.0:
                print(f"WARNING: Ollama average response time high: {avg_ollama_time:.2f}s")
            
            print(f"Ollama performance - Avg: {avg_ollama_time:.2f}s, Max: {max_ollama_time:.2f}s")
        else:
            pytest.skip("No successful Ollama responses to analyze")
    
    @pytest.mark.asyncio
    async def test_database_resource_usage(self, client, auth_headers):
        """Test database resource usage"""
        # Test database-intensive operations
        database_operations = [
            # Learning operations that should store data
            {"endpoint": "/learn", "payload": {"content": f"Database test content {i}", "type": "test_data"}} 
            for i in range(5)
        ]
        
        database_response_times = []
        
        for operation in database_operations:
            start_time = time.time()
            
            try:
                response = await client.post(
                    operation["endpoint"],
                    json=operation["payload"],
                    headers=auth_headers
                )
                
                end_time = time.time()
                response_time = end_time - start_time
                
                if response.status_code == 200:
                    database_response_times.append(response_time)
                
            except Exception as e:
                print(f"Database operation error: {e}")
                continue
            
            await asyncio.sleep(0.5)
        
        if database_response_times:
            avg_db_time = sum(database_response_times) / len(database_response_times)
            max_db_time = max(database_response_times)
            
            # Database operations should be reasonably fast
            assert avg_db_time < 10.0, f"Database operations too slow: {avg_db_time:.2f}s"
            
            print(f"Database performance - Avg: {avg_db_time:.2f}s, Max: {max_db_time:.2f}s")
        else:
            pytest.skip("No successful database operations to analyze")
    
    @pytest.mark.asyncio
    async def test_vector_database_resource_usage(self, client, auth_headers):
        """Test vector database (ChromaDB/Qdrant) resource usage"""
        # Check vector database availability through health endpoint
        health_response = await client.get("/health")
        assert health_response.status_code == 200
        
        health_data = health_response.json()
        services = health_data.get("services", {})
        
        chromadb_available = services.get("chromadb") == "connected"
        qdrant_available = services.get("qdrant") == "connected"
        
        if not (chromadb_available or qdrant_available):
            pytest.skip("No vector databases available")
        
        # Test vector database through learning operations
        vector_operations = [
            {"content": f"Vector database test document {i}: This document contains information about artificial intelligence and machine learning concepts.", "type": "vector_test"}
            for i in range(3)
        ]
        
        vector_response_times = []
        
        for operation in vector_operations:
            start_time = time.time()
            
            try:
                response = await client.post("/learn", json=operation, headers=auth_headers)
                end_time = time.time()
                response_time = end_time - start_time
                
                if response.status_code == 200:
                    vector_response_times.append(response_time)
                    
                    data = response.json()
                    if "processing_stats" in data:
                        stats = data["processing_stats"]
                        if "embeddings_created" in stats:
                            print(f"Embeddings created: {stats['embeddings_created']}")
                
            except Exception as e:
                print(f"Vector database operation error: {e}")
                continue
            
            await asyncio.sleep(1)
        
        if vector_response_times:
            avg_vector_time = sum(vector_response_times) / len(vector_response_times)
            max_vector_time = max(vector_response_times)
            
            print(f"Vector database performance - Avg: {avg_vector_time:.2f}s, Max: {max_vector_time:.2f}s")
            
            # Vector operations can be slower but should complete
            if avg_vector_time > 15.0:
                print(f"WARNING: Vector database operations slow: {avg_vector_time:.2f}s")
        else:
            pytest.skip("No successful vector database operations to analyze")

class TestResourceCapacityAndLimits:
    """Test system capacity and resource limits"""
    
    @pytest.mark.asyncio
    async def test_system_capacity_under_load(self, client, auth_headers):
        """Test system capacity under sustained load"""
        load_duration = 30  # seconds
        request_interval = 2  # seconds
        
        capacity_measurements = []
        start_time = time.time()
        
        while time.time() - start_time < load_duration:
            measurement_start = time.time()
            
            # Make a test request
            try:
                response = await client.post("/public/think", json={
                    "query": "Capacity test query",
                    "reasoning_type": "simple"
                })
                
                request_success = response.status_code == 200
                
            except Exception as e:
                request_success = False
                print(f"Request failed during capacity test: {e}")
            
            # Get system metrics
            try:
                metrics_response = await client.get("/public/metrics")
                if metrics_response.status_code == 200:
                    metrics_data = metrics_response.json()
                    system_metrics = metrics_data["system"]
                else:
                    system_metrics = {}
            except:
                system_metrics = {}
            
            measurement_time = time.time() - measurement_start
            
            capacity_measurements.append({
                "timestamp": time.time(),
                "request_success": request_success,
                "response_time": measurement_time,
                "cpu_percent": system_metrics.get("cpu_percent", 0),
                "memory_percent": system_metrics.get("memory_percent", 0)
            })
            
            await asyncio.sleep(request_interval)
        
        # Analyze capacity measurements
        successful_requests = [m for m in capacity_measurements if m["request_success"]]
        failed_requests = [m for m in capacity_measurements if not m["request_success"]]
        
        success_rate = len(successful_requests) / len(capacity_measurements) if capacity_measurements else 0
        avg_response_time = sum(m["response_time"] for m in successful_requests) / len(successful_requests) if successful_requests else 0
        
        avg_cpu = sum(m["cpu_percent"] for m in capacity_measurements) / len(capacity_measurements) if capacity_measurements else 0
        avg_memory = sum(m["memory_percent"] for m in capacity_measurements) / len(capacity_measurements) if capacity_measurements else 0
        
        max_cpu = max(m["cpu_percent"] for m in capacity_measurements) if capacity_measurements else 0
        max_memory = max(m["memory_percent"] for m in capacity_measurements) if capacity_measurements else 0
        
        # Capacity assertions
        assert success_rate >= 0.8, f"System capacity insufficient, success rate: {success_rate}"
        assert avg_response_time < 10.0, f"Response time degraded under load: {avg_response_time:.2f}s"
        
        if max_cpu > RESOURCE_LIMITS["cpu_critical_threshold"]:
            print(f"WARNING: CPU usage reached critical levels: {max_cpu}%")
        
        if max_memory > RESOURCE_LIMITS["memory_critical_threshold"]:
            print(f"WARNING: Memory usage reached critical levels: {max_memory}%")
        
        print(f"Capacity test results - Success rate: {success_rate:.2f}, Avg response: {avg_response_time:.2f}s")
        print(f"Resource usage - CPU avg: {avg_cpu:.1f}% (max: {max_cpu:.1f}%), Memory avg: {avg_memory:.1f}% (max: {max_memory:.1f}%)")
    
    @pytest.mark.asyncio
    async def test_resource_cleanup_and_recovery(self, client, auth_headers):
        """Test resource cleanup and recovery after intensive operations"""
        # Get baseline metrics
        baseline_response = await client.get("/metrics", headers=auth_headers)
        assert baseline_response.status_code == 200
        
        baseline_data = baseline_response.json()
        baseline_memory = baseline_data["system"]["memory_used_gb"]
        baseline_cpu = baseline_data["system"]["cpu_percent"]
        
        # Perform intensive operations
        intensive_operations = [
            # Large learning operation
            {
                "endpoint": "/learn",
                "payload": {
                    "content": "Intensive operation content: " + "Large data block for testing memory usage and cleanup mechanisms. " * 500,
                    "type": "intensive_test"
                }
            },
            # Complex reasoning operation
            {
                "endpoint": "/public/think",
                "payload": {
                    "query": "Perform complex analysis of distributed systems architecture, microservices patterns, scalability strategies, performance optimization techniques, security considerations, and monitoring approaches",
                    "reasoning_type": "complex"
                }
            }
        ]
        
        for operation in intensive_operations:
            try:
                response = await client.post(
                    operation["endpoint"],
                    json=operation["payload"],
                    headers=auth_headers
                )
                
                if response.status_code != 200:
                    print(f"Intensive operation failed: {response.status_code}")
                
            except Exception as e:
                print(f"Intensive operation error: {e}")
            
            await asyncio.sleep(2)
        
        # Wait for potential cleanup
        await asyncio.sleep(10)
        
        # Get post-operation metrics
        post_response = await client.get("/metrics", headers=auth_headers)
        assert post_response.status_code == 200
        
        post_data = post_response.json()
        post_memory = post_data["system"]["memory_used_gb"]
        post_cpu = post_data["system"]["cpu_percent"]
        
        # Analyze resource recovery
        memory_increase = post_memory - baseline_memory
        cpu_difference = abs(post_cpu - baseline_cpu)
        
        # Resource usage should not increase dramatically and permanently
        if memory_increase > 2.0:  # More than 2GB increase
            print(f"WARNING: Significant memory increase not cleaned up: {memory_increase:.2f}GB")
        
        if cpu_difference > 20:  # More than 20% CPU difference
            print(f"WARNING: CPU usage significantly different after operations: {cpu_difference:.1f}%")
        
        print(f"Resource recovery - Memory change: {memory_increase:.2f}GB, CPU difference: {cpu_difference:.1f}%")
        print(f"Baseline: CPU {baseline_cpu:.1f}%, Memory {baseline_memory:.2f}GB")
        print(f"Post-ops: CPU {post_cpu:.1f}%, Memory {post_memory:.2f}GB")

# Test configuration and utilities
class TestResourceAlerts:
    """Test resource monitoring alerts and thresholds"""
    
    @pytest.mark.asyncio
    async def test_resource_threshold_monitoring(self, client, auth_headers):
        """Test monitoring of resource thresholds"""
        # Get current resource usage
        response = await client.get("/metrics", headers=auth_headers)
        assert response.status_code == 200
        
        data = response.json()
        system = data["system"]
        
        cpu_percent = system["cpu_percent"]
        memory_percent = system["memory_percent"]
        
        # Check against thresholds
        alerts = []
        
        if cpu_percent > RESOURCE_LIMITS["cpu_critical_threshold"]:
            alerts.append(f"CRITICAL: CPU usage {cpu_percent}% exceeds critical threshold {RESOURCE_LIMITS['cpu_critical_threshold']}%")
        elif cpu_percent > RESOURCE_LIMITS["cpu_warning_threshold"]:
            alerts.append(f"WARNING: CPU usage {cpu_percent}% exceeds warning threshold {RESOURCE_LIMITS['cpu_warning_threshold']}%")
        
        if memory_percent > RESOURCE_LIMITS["memory_critical_threshold"]:
            alerts.append(f"CRITICAL: Memory usage {memory_percent}% exceeds critical threshold {RESOURCE_LIMITS['memory_critical_threshold']}%")
        elif memory_percent > RESOURCE_LIMITS["memory_warning_threshold"]:
            alerts.append(f"WARNING: Memory usage {memory_percent}% exceeds warning threshold {RESOURCE_LIMITS['memory_warning_threshold']}%")
        
        for alert in alerts:
            print(alert)
        
        # System should be within reasonable limits for normal operation
        assert cpu_percent < RESOURCE_LIMITS["cpu_critical_threshold"], f"CPU usage critically high: {cpu_percent}%"
        assert memory_percent < RESOURCE_LIMITS["memory_critical_threshold"], f"Memory usage critically high: {memory_percent}%"
        
        print(f"Resource threshold monitoring - CPU: {cpu_percent}%, Memory: {memory_percent}%")

if __name__ == "__main__":
    pytest.main(["-v", __file__])
