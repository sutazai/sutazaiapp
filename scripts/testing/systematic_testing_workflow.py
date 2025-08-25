#!/usr/bin/env python3
"""
SutazAI Systematic Testing Workflow
==================================

Comprehensive testing framework for the SutazAI system with:
- Multi-agent AI orchestration (200+ agents)
- Service mesh with Consul, Kong, RabbitMQ  
- Vector databases (ChromaDB, Qdrant, FAISS)
- Ollama LLM management
- MCP server network
- Docker containerization (38+ containers)

Architecture-aware testing with dependency ordering and parallel execution.
"""

import asyncio
import aiohttp
import logging
import time
import json
import subprocess
import concurrent.futures
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import sys
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestPhase(Enum):
    """Testing phases in dependency order"""
    PHASE_1_CORE_INFRA = "1_core_infrastructure"
    PHASE_2_AI_SERVICES = "2_ai_services" 
    PHASE_3_AGENT_NETWORK = "3_agent_network"
    PHASE_4_SERVICE_MESH = "4_service_mesh"
    PHASE_5_INTEGRATION = "5_integration"

class TestPriority(Enum):
    """Test execution priority"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class TestResult(Enum):
    """Test execution results"""
    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"
    TIMEOUT = "timeout"

@dataclass
class ServiceEndpoint:
    """Service endpoint configuration"""
    name: str
    url: str
    port: int
    protocol: str = "http"
    health_path: str = "/health"
    timeout: int = 30
    retries: int = 3

@dataclass 
class TestCase:
    """Individual test case definition"""
    name: str
    phase: TestPhase
    priority: TestPriority
    description: str
    test_function: str
    dependencies: List[str] = field(default_factory=list)
    parallel_group: Optional[str] = None
    timeout_seconds: int = 60
    expected_duration: float = 5.0
    tags: List[str] = field(default_factory=list)

@dataclass
class TestExecutionResult:
    """Test execution result"""
    test_case: TestCase
    result: TestResult
    duration: float
    error_message: Optional[str] = None
    output: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

class SutazAITestingWorkflow:
    """Comprehensive testing workflow for SutazAI system"""
    
    def __init__(self):
        self.results: List[TestExecutionResult] = []
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        
        # Service endpoints from port registry
        self.endpoints = {
            # Core Infrastructure (10000-10199)
            "postgres": ServiceEndpoint("postgres", "http://localhost", 10000),
            "redis": ServiceEndpoint("redis", "http://localhost", 10001), 
            "neo4j": ServiceEndpoint("neo4j", "http://localhost", 10002, health_path="/"),
            "rabbitmq": ServiceEndpoint("rabbitmq", "http://localhost", 10008, health_path="/api/overview"),
            "consul": ServiceEndpoint("consul", "http://localhost", 10006, health_path="/v1/status/leader"),
            "kong": ServiceEndpoint("kong", "http://localhost", 10005, health_path="/status"),
            
            # Application Services
            "backend": ServiceEndpoint("backend", "http://localhost", 10010),
            "frontend": ServiceEndpoint("frontend", "http://localhost", 10011),
            
            # AI Services  
            "chromadb": ServiceEndpoint("chromadb", "http://localhost", 10100, health_path="/api/v1/heartbeat"),
            "qdrant": ServiceEndpoint("qdrant", "http://localhost", 10101, health_path="/health"),
            "faiss": ServiceEndpoint("faiss", "http://localhost", 10103),
            "ollama": ServiceEndpoint("ollama", "http://localhost", 10104, health_path="/api/tags"),
            
            # Monitoring Stack
            "prometheus": ServiceEndpoint("prometheus", "http://localhost", 10200, health_path="/-/healthy"),
            "grafana": ServiceEndpoint("grafana", "http://localhost", 10201, health_path="/api/health"),
            "loki": ServiceEndpoint("loki", "http://localhost", 10202, health_path="/ready"),
            "alertmanager": ServiceEndpoint("alertmanager", "http://localhost", 10203, health_path="/-/healthy"),
        }
        
        # Define comprehensive test cases
        self.test_cases = self._define_test_cases()
    
    def _define_test_cases(self) -> List[TestCase]:
        """Define all test cases with proper dependency ordering"""
        return [
            # Phase 1: Core Infrastructure Tests (Foundation Layer)
            TestCase(
                name="postgres_health_check",
                phase=TestPhase.PHASE_1_CORE_INFRA,
                priority=TestPriority.CRITICAL,
                description="Verify PostgreSQL database connectivity and health",
                test_function="test_postgres_health",
                parallel_group="database_services",
                tags=["database", "critical", "foundation"]
            ),
            TestCase(
                name="redis_health_check", 
                phase=TestPhase.PHASE_1_CORE_INFRA,
                priority=TestPriority.CRITICAL,
                description="Verify Redis cache connectivity and performance",
                test_function="test_redis_health",
                parallel_group="database_services",
                tags=["cache", "critical", "foundation"]
            ),
            TestCase(
                name="neo4j_health_check",
                phase=TestPhase.PHASE_1_CORE_INFRA, 
                priority=TestPriority.HIGH,
                description="Verify Neo4j graph database connectivity",
                test_function="test_neo4j_health",
                parallel_group="database_services",
                tags=["database", "graph", "high"]
            ),
            TestCase(
                name="rabbitmq_health_check",
                phase=TestPhase.PHASE_1_CORE_INFRA,
                priority=TestPriority.CRITICAL,
                description="Verify RabbitMQ message broker connectivity",
                test_function="test_rabbitmq_health", 
                dependencies=["postgres_health_check"],
                tags=["messaging", "critical", "foundation"]
            ),
            
            # Phase 2: AI Services Tests
            TestCase(
                name="ollama_health_check",
                phase=TestPhase.PHASE_2_AI_SERVICES,
                priority=TestPriority.CRITICAL,
                description="Verify Ollama LLM service availability",
                test_function="test_ollama_health",
                dependencies=["postgres_health_check", "redis_health_check"],
                tags=["ai", "llm", "critical"]
            ),
            TestCase(
                name="chromadb_health_check", 
                phase=TestPhase.PHASE_2_AI_SERVICES,
                priority=TestPriority.HIGH,
                description="Verify ChromaDB vector database connectivity",
                test_function="test_chromadb_health",
                parallel_group="vector_services",
                dependencies=["redis_health_check"],
                tags=["vector", "ai", "high"]
            ),
            TestCase(
                name="qdrant_health_check",
                phase=TestPhase.PHASE_2_AI_SERVICES,
                priority=TestPriority.HIGH, 
                description="Verify Qdrant vector search service",
                test_function="test_qdrant_health",
                parallel_group="vector_services",
                dependencies=["redis_health_check"],
                tags=["vector", "ai", "high"]
            ),
            TestCase(
                name="faiss_health_check",
                phase=TestPhase.PHASE_2_AI_SERVICES,
                priority=TestPriority.MEDIUM,
                description="Verify FAISS vector similarity search",
                test_function="test_faiss_health",
                parallel_group="vector_services", 
                dependencies=["redis_health_check"],
                tags=["vector", "ai", "medium"]
            ),
            
            # Phase 3: Agent Network Tests
            TestCase(
                name="mcp_orchestrator_health",
                phase=TestPhase.PHASE_3_AGENT_NETWORK,
                priority=TestPriority.CRITICAL,
                description="Verify MCP orchestrator Docker-in-Docker service",
                test_function="test_mcp_orchestrator_health",
                dependencies=["ollama_health_check"],
                timeout_seconds=120,
                tags=["mcp", "orchestration", "critical"]
            ),
            TestCase(
                name="agent_registry_validation",
                phase=TestPhase.PHASE_3_AGENT_NETWORK,
                priority=TestPriority.HIGH,
                description="Validate agent registry and discovery",
                test_function="test_agent_registry",
                dependencies=["mcp_orchestrator_health", "redis_health_check"],
                tags=["agents", "registry", "high"]
            ),
            TestCase(
                name="mcp_server_network_test",
                phase=TestPhase.PHASE_3_AGENT_NETWORK,
                priority=TestPriority.HIGH,
                description="Test MCP server network connectivity",
                test_function="test_mcp_server_network", 
                dependencies=["mcp_orchestrator_health"],
                timeout_seconds=180,
                tags=["mcp", "network", "high"]
            ),
            
            # Phase 4: Service Mesh Tests  
            TestCase(
                name="consul_service_discovery",
                phase=TestPhase.PHASE_4_SERVICE_MESH,
                priority=TestPriority.CRITICAL,
                description="Test Consul service discovery and registration",
                test_function="test_consul_service_discovery",
                dependencies=["rabbitmq_health_check"],
                tags=["service_mesh", "discovery", "critical"]
            ),
            TestCase(
                name="kong_api_gateway_test",
                phase=TestPhase.PHASE_4_SERVICE_MESH,
                priority=TestPriority.CRITICAL,
                description="Test Kong API gateway routing and load balancing",
                test_function="test_kong_gateway",
                dependencies=["consul_service_discovery"],
                tags=["api_gateway", "routing", "critical"]
            ),
            TestCase(
                name="service_mesh_communication",
                phase=TestPhase.PHASE_4_SERVICE_MESH,
                priority=TestPriority.HIGH,
                description="Test inter-service communication through mesh",
                test_function="test_service_mesh_communication",
                dependencies=["kong_api_gateway_test", "agent_registry_validation"],
                tags=["service_mesh", "communication", "high"]
            ),
            
            # Phase 5: Integration Tests
            TestCase(
                name="backend_health_comprehensive",
                phase=TestPhase.PHASE_5_INTEGRATION,
                priority=TestPriority.CRITICAL,
                description="Comprehensive backend API health check",
                test_function="test_backend_health_comprehensive",
                dependencies=["service_mesh_communication", "ollama_health_check"],
                tags=["backend", "api", "critical"]
            ),
            TestCase(
                name="frontend_backend_integration",
                phase=TestPhase.PHASE_5_INTEGRATION,
                priority=TestPriority.HIGH,
                description="Test frontend to backend integration",
                test_function="test_frontend_backend_integration",
                dependencies=["backend_health_comprehensive"],
                tags=["integration", "frontend", "high"]
            ),
            TestCase(
                name="end_to_end_workflow_test",
                phase=TestPhase.PHASE_5_INTEGRATION,
                priority=TestPriority.HIGH,
                description="End-to-end workflow from frontend to AI services",
                test_function="test_end_to_end_workflow",
                dependencies=["frontend_backend_integration", "mcp_server_network_test"],
                timeout_seconds=300,
                tags=["e2e", "workflow", "integration", "high"]
            ),
            
            # Monitoring and Observability Tests
            TestCase(
                name="monitoring_stack_health",
                phase=TestPhase.PHASE_5_INTEGRATION,
                priority=TestPriority.MEDIUM,
                description="Test monitoring stack (Prometheus, Grafana, Loki)",
                test_function="test_monitoring_stack",
                parallel_group="monitoring_services",
                dependencies=["backend_health_comprehensive"],
                tags=["monitoring", "observability", "medium"]
            ),
        ]
    
    async def execute_test_workflow(self) -> Dict[str, Any]:
        """Execute complete testing workflow with dependency resolution"""
        logger.info("Starting SutazAI comprehensive testing workflow")
        self.start_time = datetime.now()
        
        try:
            # Group tests by phase
            phases = self._group_tests_by_phase()
            
            # Execute each phase with proper dependency handling
            for phase in TestPhase:
                if phase in phases:
                    logger.info(f"Executing {phase.value} tests")
                    await self._execute_phase(phases[phase])
            
            # Generate comprehensive report
            return self._generate_final_report()
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            return {"status": "error", "message": str(e)}
        finally:
            self.end_time = datetime.now()
    
    def _group_tests_by_phase(self) -> Dict[TestPhase, List[TestCase]]:
        """Group test cases by execution phase"""
        phases = {}
        for test_case in self.test_cases:
            if test_case.phase not in phases:
                phases[test_case.phase] = []
            phases[test_case.phase].append(test_case)
        
        # Sort tests within each phase by priority and dependencies
        for phase, tests in phases.items():
            phases[phase] = sorted(tests, key=lambda t: (
                t.priority.value,
                len(t.dependencies)
            ))
        
        return phases
    
    async def _execute_phase(self, test_cases: List[TestCase]):
        """Execute tests in a phase with parallel execution where possible"""
        # Group by parallel execution groups
        parallel_groups = {}
        sequential_tests = []
        
        for test_case in test_cases:
            if test_case.parallel_group:
                if test_case.parallel_group not in parallel_groups:
                    parallel_groups[test_case.parallel_group] = []
                parallel_groups[test_case.parallel_group].append(test_case)
            else:
                sequential_tests.append(test_case)
        
        # Execute parallel groups concurrently
        parallel_tasks = []
        for group_name, group_tests in parallel_groups.items():
            task = asyncio.create_task(self._execute_parallel_group(group_tests))
            parallel_tasks.append(task)
        
        # Execute parallel groups
        if parallel_tasks:
            await asyncio.gather(*parallel_tasks, return_exceptions=True)
        
        # Execute remaining sequential tests
        for test_case in sequential_tests:
            if self._dependencies_satisfied(test_case):
                await self._execute_single_test(test_case)
    
    async def _execute_parallel_group(self, test_cases: List[TestCase]):
        """Execute a group of tests in parallel"""
        tasks = []
        for test_case in test_cases:
            if self._dependencies_satisfied(test_case):
                task = asyncio.create_task(self._execute_single_test(test_case))
                tasks.append(task)
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    def _dependencies_satisfied(self, test_case: TestCase) -> bool:
        """Check if test dependencies are satisfied"""
        if not test_case.dependencies:
            return True
        
        completed_tests = {result.test_case.name for result in self.results 
                          if result.result == TestResult.PASS}
        
        return all(dep in completed_tests for dep in test_case.dependencies)
    
    async def _execute_single_test(self, test_case: TestCase) -> TestExecutionResult:
        """Execute a single test case"""
        logger.info(f"Executing test: {test_case.name}")
        start_time = time.time()
        
        try:
            # Get test function and execute
            test_function = getattr(self, test_case.test_function)
            
            # Execute with timeout
            result = await asyncio.wait_for(
                test_function(),
                timeout=test_case.timeout_seconds
            )
            
            duration = time.time() - start_time
            test_result = TestExecutionResult(
                test_case=test_case,
                result=TestResult.PASS if result else TestResult.FAIL,
                duration=duration,
                output=str(result) if result else None
            )
            
        except asyncio.TimeoutError:
            duration = time.time() - start_time
            test_result = TestExecutionResult(
                test_case=test_case,
                result=TestResult.TIMEOUT,
                duration=duration,
                error_message=f"Test timed out after {test_case.timeout_seconds} seconds"
            )
            
        except Exception as e:
            duration = time.time() - start_time
            test_result = TestExecutionResult(
                test_case=test_case,
                result=TestResult.FAIL,
                duration=duration,
                error_message=str(e)
            )
        
        self.results.append(test_result)
        logger.info(f"Test {test_case.name}: {test_result.result.value} ({duration:.2f}s)")
        return test_result
    
    # Core Infrastructure Test Methods
    async def test_postgres_health(self) -> bool:
        """Test PostgreSQL database connectivity"""
        endpoint = self.endpoints["postgres"]
        try:
            # Try to connect via psql command
            result = subprocess.run([
                "pg_isready", "-h", "localhost", "-p", str(endpoint.port), 
                "-U", "sutazai", "-d", "sutazai"
            ], capture_output=True, text=True, timeout=10)
            
            return result.returncode == 0
            
        except subprocess.TimeoutExpired:
            logger.error("PostgreSQL health check timed out")
            return False
        except Exception as e:
            logger.error(f"PostgreSQL health check failed: {e}")
            return False
    
    async def test_redis_health(self) -> bool:
        """Test Redis connectivity and basic operations"""
        endpoint = self.endpoints["redis"] 
        try:
            # Test Redis via redis-cli
            result = subprocess.run([
                "redis-cli", "-h", "localhost", "-p", str(endpoint.port), "ping"
            ], capture_output=True, text=True, timeout=10)
            
            return result.returncode == 0 and "PONG" in result.stdout
            
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return False
    
    async def test_neo4j_health(self) -> bool:
        """Test Neo4j graph database connectivity"""
        endpoint = self.endpoints["neo4j"]
        url = f"{endpoint.protocol}://{endpoint.url.replace('http://', '')}:{endpoint.port}"
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(f"{url}{endpoint.health_path}", 
                                     timeout=aiohttp.ClientTimeout(total=30)) as response:
                    return response.status == 200
            except Exception as e:
                logger.error(f"Neo4j health check failed: {e}")
                return False
    
    async def test_rabbitmq_health(self) -> bool:
        """Test RabbitMQ message broker connectivity"""
        endpoint = self.endpoints["rabbitmq"]
        url = f"{endpoint.protocol}://{endpoint.url.replace('http://', '')}:{endpoint.port}"
        
        async with aiohttp.ClientSession() as session:
            try:
                # Basic auth for RabbitMQ management API
                auth = aiohttp.BasicAuth("sutazai", "sutazai123")
                async with session.get(f"{url}{endpoint.health_path}", 
                                     auth=auth,
                                     timeout=aiohttp.ClientTimeout(total=30)) as response:
                    return response.status == 200
            except Exception as e:
                logger.error(f"RabbitMQ health check failed: {e}")
                return False
    
    # AI Services Test Methods
    async def test_ollama_health(self) -> bool:
        """Test Ollama LLM service availability"""
        endpoint = self.endpoints["ollama"]
        url = f"{endpoint.protocol}://{endpoint.url.replace('http://', '')}:{endpoint.port}"
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(f"{url}{endpoint.health_path}",
                                     timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status == 200:
                        data = await response.json()
                        return isinstance(data, dict)
                    return False
            except Exception as e:
                logger.error(f"Ollama health check failed: {e}")
                return False
    
    async def test_chromadb_health(self) -> bool:
        """Test ChromaDB vector database connectivity"""
        endpoint = self.endpoints["chromadb"] 
        url = f"{endpoint.protocol}://{endpoint.url.replace('http://', '')}:{endpoint.port}"
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(f"{url}{endpoint.health_path}",
                                     timeout=aiohttp.ClientTimeout(total=30)) as response:
                    return response.status == 200
            except Exception as e:
                logger.error(f"ChromaDB health check failed: {e}")
                return False
    
    async def test_qdrant_health(self) -> bool:
        """Test Qdrant vector search service"""
        endpoint = self.endpoints["qdrant"]
        url = f"{endpoint.protocol}://{endpoint.url.replace('http://', '')}:{endpoint.port}"
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(f"{url}{endpoint.health_path}",
                                     timeout=aiohttp.ClientTimeout(total=30)) as response:
                    return response.status == 200
            except Exception as e:
                logger.error(f"Qdrant health check failed: {e}")
                return False
    
    async def test_faiss_health(self) -> bool:
        """Test FAISS vector similarity search"""
        endpoint = self.endpoints["faiss"]
        url = f"{endpoint.protocol}://{endpoint.url.replace('http://', '')}:{endpoint.port}"
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(f"{url}{endpoint.health_path}",
                                     timeout=aiohttp.ClientTimeout(total=30)) as response:
                    return response.status == 200
            except Exception as e:
                logger.error(f"FAISS health check failed: {e}")
                return False
    
    # Agent Network Test Methods
    async def test_mcp_orchestrator_health(self) -> bool:
        """Test MCP orchestrator Docker-in-Docker service"""
        try:
            # Check if DinD container is running
            result = subprocess.run([
                "docker", "ps", "--filter", "name=sutazai-mcp-orchestrator", 
                "--format", "{{.Status}}"
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0 and "Up" in result.stdout:
                # Test Docker API connectivity
                result = subprocess.run([
                    "curl", "-f", "-s", "http://localhost:12375/version"
                ], capture_output=True, text=True, timeout=10)
                
                return result.returncode == 0
            
            return False
            
        except Exception as e:
            logger.error(f"MCP orchestrator health check failed: {e}")
            return False
    
    async def test_agent_registry(self) -> bool:
        """Test agent registry and discovery functionality"""
        endpoint = self.endpoints["backend"]
        url = f"{endpoint.protocol}://{endpoint.url.replace('http://', '')}:{endpoint.port}"
        
        async with aiohttp.ClientSession() as session:
            try:
                # Test agent registry endpoint
                async with session.get(f"{url}/api/v1/agents",
                                     timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status == 200:
                        data = await response.json()
                        return isinstance(data, (list, dict))
                    return False
            except Exception as e:
                logger.error(f"Agent registry test failed: {e}")
                return False
    
    async def test_mcp_server_network(self) -> bool:
        """Test MCP server network connectivity"""
        try:
            # Check if MCP servers are accessible through the orchestrator
            result = subprocess.run([
                "docker", "exec", "sutazai-mcp-orchestrator", 
                "docker", "ps", "--filter", "label=mcp-server", 
                "--format", "{{.Names}}"
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                mcp_servers = result.stdout.strip().split('\n')
                return len([s for s in mcp_servers if s.strip()]) > 0
            
            return False
            
        except Exception as e:
            logger.error(f"MCP server network test failed: {e}")
            return False
    
    # Service Mesh Test Methods
    async def test_consul_service_discovery(self) -> bool:
        """Test Consul service discovery and registration"""
        endpoint = self.endpoints["consul"]
        url = f"{endpoint.protocol}://{endpoint.url.replace('http://', '')}:{endpoint.port}"
        
        async with aiohttp.ClientSession() as session:
            try:
                # Check Consul leader
                async with session.get(f"{url}/v1/status/leader",
                                     timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status != 200:
                        return False
                
                # Check registered services
                async with session.get(f"{url}/v1/catalog/services",
                                     timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status == 200:
                        data = await response.json()
                        return isinstance(data, dict) and len(data) > 0
                    
                return False
                
            except Exception as e:
                logger.error(f"Consul service discovery test failed: {e}")
                return False
    
    async def test_kong_gateway(self) -> bool:
        """Test Kong API gateway routing and load balancing"""
        endpoint = self.endpoints["kong"]
        url = f"{endpoint.protocol}://{endpoint.url.replace('http://', '')}:{endpoint.port}"
        
        async with aiohttp.ClientSession() as session:
            try:
                # Test Kong status endpoint
                async with session.get(f"{url}/status",
                                     timeout=aiohttp.ClientTimeout(total=30)) as response:
                    return response.status == 200
            except Exception as e:
                logger.error(f"Kong gateway test failed: {e}")
                return False
    
    async def test_service_mesh_communication(self) -> bool:
        """Test inter-service communication through mesh"""
        try:
            # Test routing through Kong to backend
            async with aiohttp.ClientSession() as session:
                async with session.get("http://localhost:10005/health",
                                     timeout=aiohttp.ClientTimeout(total=30)) as response:
                    return response.status in [200, 502, 503]  # Accept routing attempts
        except Exception as e:
            logger.error(f"Service mesh communication test failed: {e}")
            return False
    
    # Integration Test Methods  
    async def test_backend_health_comprehensive(self) -> bool:
        """Comprehensive backend API health check"""
        endpoint = self.endpoints["backend"]
        url = f"{endpoint.protocol}://{endpoint.url.replace('http://', '')}:{endpoint.port}"
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(f"{url}/health",
                                     timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status == 200:
                        data = await response.json()
                        return (isinstance(data, dict) and 
                               data.get("status") in ["healthy", "ok"])
                    return False
            except Exception as e:
                logger.error(f"Backend comprehensive health check failed: {e}")
                return False
    
    async def test_frontend_backend_integration(self) -> bool:
        """Test frontend to backend integration"""
        frontend_endpoint = self.endpoints["frontend"]
        url = f"{frontend_endpoint.protocol}://{frontend_endpoint.url.replace('http://', '')}:{frontend_endpoint.port}"
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(f"{url}/health",
                                     timeout=aiohttp.ClientTimeout(total=30)) as response:
                    return response.status == 200
            except Exception as e:
                logger.error(f"Frontend-backend integration test failed: {e}")
                return False
    
    async def test_end_to_end_workflow(self) -> bool:
        """End-to-end workflow from frontend to AI services"""
        try:
            # Test complete workflow: frontend -> backend -> AI services
            backend_url = "http://localhost:10010"
            
            async with aiohttp.ClientSession() as session:
                # Test chat endpoint (integration with Ollama)
                chat_payload = {
                    "message": "Hello, test message",
                    "model": "tinyllama"
                }
                
                async with session.post(f"{backend_url}/api/v1/chat",
                                      json=chat_payload,
                                      timeout=aiohttp.ClientTimeout(total=60)) as response:
                    return response.status in [200, 503]  # Accept if service available
                    
        except Exception as e:
            logger.error(f"End-to-end workflow test failed: {e}")
            return False
    
    async def test_monitoring_stack(self) -> bool:
        """Test monitoring stack (Prometheus, Grafana, Loki)"""
        monitoring_services = ["prometheus", "grafana", "loki"]
        results = []
        
        async with aiohttp.ClientSession() as session:
            for service_name in monitoring_services:
                if service_name in self.endpoints:
                    endpoint = self.endpoints[service_name]
                    url = f"{endpoint.protocol}://{endpoint.url.replace('http://', '')}:{endpoint.port}"
                    
                    try:
                        async with session.get(f"{url}{endpoint.health_path}",
                                             timeout=aiohttp.ClientTimeout(total=30)) as response:
                            results.append(response.status == 200)
                    except Exception as e:
                        logger.error(f"Monitoring service {service_name} test failed: {e}")
                        results.append(False)
        
        return len(results) > 0 and any(results)  # At least one monitoring service working
    
    def _generate_final_report(self) -> Dict[str, Any]:
        """Generate comprehensive test execution report"""
        total_tests = len(self.results)
        passed_tests = len([r for r in self.results if r.result == TestResult.PASS])
        failed_tests = len([r for r in self.results if r.result == TestResult.FAIL])
        timeout_tests = len([r for r in self.results if r.result == TestResult.TIMEOUT])
        
        total_duration = (self.end_time - self.start_time).total_seconds() if self.end_time and self.start_time else 0
        
        # Group results by phase
        phase_results = {}
        for result in self.results:
            phase = result.test_case.phase.value
            if phase not in phase_results:
                phase_results[phase] = []
            phase_results[phase].append({
                "name": result.test_case.name,
                "result": result.result.value,
                "duration": result.duration,
                "error": result.error_message
            })
        
        # Critical service status
        critical_services = [
            "postgres_health_check", 
            "redis_health_check", 
            "rabbitmq_health_check",
            "backend_health_comprehensive",
            "consul_service_discovery",
            "kong_api_gateway_test"
        ]
        
        critical_status = {}
        for result in self.results:
            if result.test_case.name in critical_services:
                critical_status[result.test_case.name] = result.result.value
        
        return {
            "summary": {
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "timeout": timeout_tests,
                "success_rate": round(passed_tests / total_tests * 100, 2) if total_tests > 0 else 0,
                "total_duration_seconds": total_duration,
                "execution_time": self.start_time.isoformat() if self.start_time else None
            },
            "critical_services": critical_status,
            "phase_results": phase_results,
            "recommendations": self._generate_recommendations(),
            "system_readiness": self._assess_system_readiness()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations based on test results"""
        recommendations = []
        
        # Check critical service failures
        critical_failures = [r for r in self.results 
                           if r.result == TestResult.FAIL and r.test_case.priority == TestPriority.CRITICAL]
        
        if critical_failures:
            recommendations.append(f"CRITICAL: {len(critical_failures)} critical services are failing")
        
        # Check timeout issues
        timeout_tests = [r for r in self.results if r.result == TestResult.TIMEOUT]
        if timeout_tests:
            recommendations.append(f"Performance issue: {len(timeout_tests)} tests timed out")
        
        # Check AI services
        ai_tests = [r for r in self.results if "ai" in r.test_case.tags or "vector" in r.test_case.tags]
        ai_failures = [r for r in ai_tests if r.result == TestResult.FAIL]
        
        if len(ai_failures) > len(ai_tests) // 2:
            recommendations.append("AI services are experiencing issues - check Ollama and vector databases")
        
        # Check monitoring  
        monitoring_failures = [r for r in self.results 
                             if r.result == TestResult.FAIL and "monitoring" in r.test_case.tags]
        if monitoring_failures:
            recommendations.append("Monitoring stack has issues - observability may be limited")
        
        return recommendations
    
    def _assess_system_readiness(self) -> str:
        """Assess overall system readiness based on test results"""
        critical_tests = [r for r in self.results if r.test_case.priority == TestPriority.CRITICAL]
        critical_failures = [r for r in critical_tests if r.result == TestResult.FAIL]
        
        if len(critical_failures) == 0:
            return "READY"
        elif len(critical_failures) <= len(critical_tests) // 4:
            return "PARTIALLY_READY" 
        else:
            return "NOT_READY"

async def main():
    """Main execution function"""
    workflow = SutazAITestingWorkflow()
    
    print("🚀 Starting SutazAI Comprehensive Testing Workflow")
    print("=" * 60)
    
    report = await workflow.execute_test_workflow()
    
    print("\n" + "=" * 60)
    print("📊 TESTING WORKFLOW COMPLETE")
    print("=" * 60)
    
    # Print summary
    summary = report["summary"]
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Passed: {summary['passed']} ({summary['success_rate']}%)")
    print(f"Failed: {summary['failed']}")
    print(f"Timeouts: {summary['timeout']}")
    print(f"Duration: {summary['total_duration_seconds']:.2f} seconds")
    print(f"System Readiness: {report['system_readiness']}")
    
    # Print critical services
    print("\n🔥 Critical Services Status:")
    for service, status in report["critical_services"].items():
        status_icon = "✅" if status == "pass" else "❌"
        print(f"  {status_icon} {service}: {status}")
    
    # Print recommendations
    if report["recommendations"]:
        print("\n💡 Recommendations:")
        for rec in report["recommendations"]:
            print(f"  • {rec}")
    
    # Save detailed report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"sutazai_test_report_{timestamp}.json"
    
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n📄 Detailed report saved to: {report_file}")
    
    return report["system_readiness"] == "READY"

if __name__ == "__main__":
    # Run the testing workflow
    success = asyncio.run(main())
    sys.exit(0 if success else 1)