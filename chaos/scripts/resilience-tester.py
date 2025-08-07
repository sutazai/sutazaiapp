#!/usr/bin/env python3
"""
SutazAI Resilience Testing Suite
Comprehensive testing for system resilience, recovery time measurement, and cascade failure detection
"""

import sys
import os
import json
import time
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import docker
import requests
import networkx as nx
from dataclasses import dataclass, asdict
from enum import Enum
import statistics
import numpy as np

# Add chaos directory to path
sys.path.append('/opt/sutazaiapp/chaos')

class ServiceStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

class ResilienceTestType(Enum):
    RECOVERY_TIME = "recovery_time"
    CASCADE_FAILURE = "cascade_failure"
    STRESS_TOLERANCE = "stress_tolerance"
    DEPENDENCY_ISOLATION = "dependency_isolation"
    LOAD_BALANCING = "load_balancing"

@dataclass
class ServiceMetrics:
    """Service health and performance metrics"""
    name: str
    container_id: str
    status: ServiceStatus
    response_time: Optional[float]
    cpu_usage: Optional[float]
    memory_usage: Optional[float]
    error_rate: Optional[float]
    uptime: Optional[float]
    last_health_check: datetime
    dependencies: List[str]

@dataclass
class ResilienceTestResult:
    """Results from resilience testing"""
    test_type: ResilienceTestType
    service_name: str
    start_time: datetime
    end_time: Optional[datetime]
    recovery_time: Optional[float]
    degradation_duration: Optional[float]
    affected_services: List[str]
    cascade_depth: int
    success: bool
    metrics: Dict[str, Any]
    error_message: Optional[str]

class DependencyGraph:
    """Service dependency graph for cascade failure analysis"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.service_metrics: Dict[str, ServiceMetrics] = {}
    
    def add_service(self, service: str, dependencies: List[str] = None):
        """Add a service to the dependency graph"""
        self.graph.add_node(service)
        if dependencies:
            for dep in dependencies:
                self.graph.add_edge(service, dep)
    
    def get_dependents(self, service: str) -> List[str]:
        """Get services that depend on the given service"""
        return list(self.graph.predecessors(service))
    
    def get_dependencies(self, service: str) -> List[str]:
        """Get services that the given service depends on"""
        return list(self.graph.successors(service))
    
    def get_cascade_impact(self, failed_service: str) -> List[str]:
        """Calculate potential cascade impact of a service failure"""
        affected = set()
        queue = [failed_service]
        
        while queue:
            current = queue.pop(0)
            if current in affected:
                continue
                
            affected.add(current)
            dependents = self.get_dependents(current)
            
            for dependent in dependents:
                if dependent not in affected:
                    queue.append(dependent)
        
        return list(affected)
    
    def get_isolation_score(self, service: str) -> float:
        """Calculate isolation score (lower = better isolated)"""
        cascade_impact = len(self.get_cascade_impact(service))
        total_services = len(self.graph.nodes)
        return cascade_impact / total_services if total_services > 0 else 0.0

class HealthChecker:
    """Advanced health checking for resilience testing"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.docker_client = docker.from_env()
        
        # Service endpoint mappings
        self.service_endpoints = {
            'sutazai-backend': 'http://localhost:8000/health',
            'sutazai-frontend': 'http://localhost:8501/healthz',
            'sutazai-postgres': None,  # Use pg_isready
            'sutazai-redis': None,     # Use redis-cli ping
            'sutazai-prometheus': 'http://localhost:9090/-/healthy',
            'sutazai-grafana': 'http://localhost:3000/api/health',
            'sutazai-chromadb': 'http://localhost:8001/api/v1/heartbeat',
            'sutazai-qdrant': 'http://localhost:6333/health',
            'sutazai-langflow': 'http://localhost:8090/health',
            'sutazai-flowise': 'http://localhost:8099/api/v1/ping',
        }
    
    async def get_service_metrics(self, service_name: str) -> ServiceMetrics:
        """Get comprehensive metrics for a service"""
        try:
            container = self.docker_client.containers.get(service_name)
            
            # Basic container info
            status = self._determine_service_status(container)
            
            # Get container stats
            stats = container.stats(stream=False)
            cpu_usage = self._calculate_cpu_usage(stats)
            memory_usage = self._calculate_memory_usage(stats)
            
            # Response time check
            response_time = await self._check_response_time(service_name)
            
            # Error rate (placeholder - would need application metrics)
            error_rate = 0.0
            
            # Uptime calculation
            started_at = container.attrs['State']['StartedAt']
            start_time = datetime.fromisoformat(started_at.replace('Z', '+00:00'))
            uptime = (datetime.now() - start_time.replace(tzinfo=None)).total_seconds()
            
            return ServiceMetrics(
                name=service_name,
                container_id=container.id,
                status=status,
                response_time=response_time,
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                error_rate=error_rate,
                uptime=uptime,
                last_health_check=datetime.now(),
                dependencies=self._get_service_dependencies(service_name)
            )
            
        except docker.errors.NotFound:
            return ServiceMetrics(
                name=service_name,
                container_id="",
                status=ServiceStatus.CRITICAL,
                response_time=None,
                cpu_usage=None,
                memory_usage=None,
                error_rate=None,
                uptime=None,
                last_health_check=datetime.now(),
                dependencies=[]
            )
        except Exception as e:
            self.logger.error(f"Error getting metrics for {service_name}: {e}")
            return ServiceMetrics(
                name=service_name,
                container_id="",
                status=ServiceStatus.UNKNOWN,
                response_time=None,
                cpu_usage=None,
                memory_usage=None,
                error_rate=None,
                uptime=None,
                last_health_check=datetime.now(),
                dependencies=[]
            )
    
    def _determine_service_status(self, container) -> ServiceStatus:
        """Determine service status from container state"""
        if container.status != 'running':
            return ServiceStatus.CRITICAL
        
        # Check health status if available
        if hasattr(container, 'attrs') and 'State' in container.attrs:
            state = container.attrs['State']
            if 'Health' in state:
                health_status = state['Health']['Status']
                if health_status == 'healthy':
                    return ServiceStatus.HEALTHY
                elif health_status == 'starting':
                    return ServiceStatus.DEGRADED
                else:
                    return ServiceStatus.UNHEALTHY
        
        # If no health check, assume healthy if running
        return ServiceStatus.HEALTHY
    
    def _calculate_cpu_usage(self, stats: dict) -> float:
        """Calculate CPU usage percentage from container stats"""
        try:
            cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                       stats['precpu_stats']['cpu_usage']['total_usage']
            system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                          stats['precpu_stats']['system_cpu_usage']
            
            if system_delta > 0:
                return (cpu_delta / system_delta) * len(stats['cpu_stats']['cpu_usage']['percpu_usage']) * 100.0
        except (KeyError, ZeroDivisionError):
            pass
        return 0.0
    
    def _calculate_memory_usage(self, stats: dict) -> float:
        """Calculate memory usage percentage from container stats"""
        try:
            used = stats['memory_stats']['usage']
            limit = stats['memory_stats']['limit']
            return (used / limit) * 100.0
        except (KeyError, ZeroDivisionError):
            return 0.0
    
    async def _check_response_time(self, service_name: str) -> Optional[float]:
        """Check response time for service endpoint"""
        endpoint = self.service_endpoints.get(service_name)
        if not endpoint:
            return None
        
        try:
            start_time = time.time()
            response = requests.get(endpoint, timeout=10)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                return response_time
        except Exception:
            pass
        
        return None
    
    def _get_service_dependencies(self, service_name: str) -> List[str]:
        """Get service dependencies (simplified mapping)"""
        dependency_map = {
            'sutazai-backend': ['sutazai-postgres', 'sutazai-redis', 'sutazai-ollama'],
            'sutazai-frontend': ['sutazai-backend'],
            'sutazai-langflow': ['sutazai-postgres', 'sutazai-redis'],
            'sutazai-flowise': [],
            'sutazai-autogpt': ['sutazai-backend', 'sutazai-ollama'],
            'sutazai-crewai': ['sutazai-backend', 'sutazai-ollama'],
            'sutazai-letta': ['sutazai-postgres', 'sutazai-ollama'],
        }
        return dependency_map.get(service_name, [])

class RecoveryTimeTester:
    """Test and measure service recovery times"""
    
    def __init__(self, health_checker: HealthChecker, logger: logging.Logger):
        self.health_checker = health_checker
        self.logger = logger
        self.docker_client = docker.from_env()
    
    async def test_recovery_time(self, service_name: str, failure_type: str = 'kill') -> ResilienceTestResult:
        """Test recovery time for a specific service"""
        start_time = datetime.now()
        
        try:
            self.logger.info(f"Starting recovery time test for {service_name}")
            
            # Get baseline metrics
            baseline_metrics = await self.health_checker.get_service_metrics(service_name)
            
            # Induce failure
            failure_start = time.time()
            await self._induce_failure(service_name, failure_type)
            
            # Monitor recovery
            recovery_time, degradation_duration = await self._monitor_recovery(
                service_name, timeout=300
            )
            
            # Get final metrics
            final_metrics = await self.health_checker.get_service_metrics(service_name)
            
            success = recovery_time is not None and final_metrics.status == ServiceStatus.HEALTHY
            
            return ResilienceTestResult(
                test_type=ResilienceTestType.RECOVERY_TIME,
                service_name=service_name,
                start_time=start_time,
                end_time=datetime.now(),
                recovery_time=recovery_time,
                degradation_duration=degradation_duration,
                affected_services=[service_name],
                cascade_depth=1,
                success=success,
                metrics={
                    'baseline_metrics': asdict(baseline_metrics),
                    'final_metrics': asdict(final_metrics),
                    'failure_type': failure_type
                },
                error_message=None if success else "Service failed to recover within timeout"
            )
            
        except Exception as e:
            self.logger.error(f"Recovery time test failed for {service_name}: {e}")
            return ResilienceTestResult(
                test_type=ResilienceTestType.RECOVERY_TIME,
                service_name=service_name,
                start_time=start_time,
                end_time=datetime.now(),
                recovery_time=None,
                degradation_duration=None,
                affected_services=[service_name],
                cascade_depth=1,
                success=False,
                metrics={},
                error_message=str(e)
            )
    
    async def _induce_failure(self, service_name: str, failure_type: str):
        """Induce a specific type of failure"""
        container = self.docker_client.containers.get(service_name)
        
        if failure_type == 'kill':
            container.kill()
        elif failure_type == 'stop':
            container.stop()
        elif failure_type == 'pause':
            container.pause()
            # Unpause after a short time
            await asyncio.sleep(30)
            container.unpause()
        else:
            raise ValueError(f"Unknown failure type: {failure_type}")
    
    async def _monitor_recovery(self, service_name: str, timeout: int = 300) -> Tuple[Optional[float], Optional[float]]:
        """Monitor service recovery and return recovery time and degradation duration"""
        start_time = time.time()
        degradation_start = start_time
        recovery_time = None
        degradation_duration = None
        
        while time.time() - start_time < timeout:
            metrics = await self.health_checker.get_service_metrics(service_name)
            
            if metrics.status == ServiceStatus.HEALTHY and metrics.response_time is not None:
                recovery_time = time.time() - start_time
                degradation_duration = time.time() - degradation_start
                break
            elif metrics.status in [ServiceStatus.DEGRADED, ServiceStatus.UNHEALTHY]:
                # Service is recovering but not fully healthy yet
                pass
            
            await asyncio.sleep(5)
        
        return recovery_time, degradation_duration

class CascadeFailureTester:
    """Test for cascade failure patterns and dependencies"""
    
    def __init__(self, health_checker: HealthChecker, dependency_graph: DependencyGraph, logger: logging.Logger):
        self.health_checker = health_checker
        self.dependency_graph = dependency_graph
        self.logger = logger
        self.docker_client = docker.from_env()
    
    async def test_cascade_failure(self, initial_service: str, max_cascade_depth: int = 5) -> ResilienceTestResult:
        """Test cascade failure patterns starting from an initial service"""
        start_time = datetime.now()
        
        try:
            self.logger.info(f"Starting cascade failure test from {initial_service}")
            
            # Get baseline metrics for all services
            baseline_metrics = await self._get_all_service_metrics()
            
            # Induce initial failure
            failure_start = time.time()
            await self._induce_failure(initial_service)
            
            # Monitor cascade effects
            affected_services, cascade_depth = await self._monitor_cascade_effects(
                initial_service, max_cascade_depth, timeout=600
            )
            
            # Measure overall recovery
            recovery_time = await self._monitor_cascade_recovery(affected_services, timeout=900)
            
            # Get final metrics
            final_metrics = await self._get_all_service_metrics()
            
            success = len(affected_services) <= max_cascade_depth and recovery_time is not None
            
            return ResilienceTestResult(
                test_type=ResilienceTestType.CASCADE_FAILURE,
                service_name=initial_service,
                start_time=start_time,
                end_time=datetime.now(),
                recovery_time=recovery_time,
                degradation_duration=time.time() - failure_start if recovery_time else None,
                affected_services=affected_services,
                cascade_depth=cascade_depth,
                success=success,
                metrics={
                    'baseline_metrics': baseline_metrics,
                    'final_metrics': final_metrics,
                    'isolation_score': self.dependency_graph.get_isolation_score(initial_service)
                },
                error_message=None if success else "Cascade failure exceeded acceptable limits"
            )
            
        except Exception as e:
            self.logger.error(f"Cascade failure test failed for {initial_service}: {e}")
            return ResilienceTestResult(
                test_type=ResilienceTestType.CASCADE_FAILURE,
                service_name=initial_service,
                start_time=start_time,
                end_time=datetime.now(),
                recovery_time=None,
                degradation_duration=None,
                affected_services=[initial_service],
                cascade_depth=1,
                success=False,
                metrics={},
                error_message=str(e)
            )
    
    async def _get_all_service_metrics(self) -> Dict[str, Dict]:
        """Get metrics for all monitored services"""
        metrics = {}
        services = [container.name for container in self.docker_client.containers.list()]
        
        for service in services:
            if service.startswith('sutazai-'):
                service_metrics = await self.health_checker.get_service_metrics(service)
                metrics[service] = asdict(service_metrics)
        
        return metrics
    
    async def _induce_failure(self, service_name: str):
        """Induce failure in initial service"""
        container = self.docker_client.containers.get(service_name)
        container.kill()
    
    async def _monitor_cascade_effects(self, initial_service: str, max_depth: int, timeout: int) -> Tuple[List[str], int]:
        """Monitor and track cascade failure effects"""
        affected_services = [initial_service]
        cascade_depth = 1
        start_time = time.time()
        
        while time.time() - start_time < timeout and cascade_depth < max_depth:
            # Check potential dependents
            potential_affected = self.dependency_graph.get_cascade_impact(initial_service)
            
            new_affected = []
            for service in potential_affected:
                if service not in affected_services:
                    metrics = await self.health_checker.get_service_metrics(service)
                    if metrics.status in [ServiceStatus.UNHEALTHY, ServiceStatus.CRITICAL]:
                        new_affected.append(service)
            
            if not new_affected:
                break
            
            affected_services.extend(new_affected)
            cascade_depth += 1
            
            await asyncio.sleep(30)  # Wait for effects to propagate
        
        return affected_services, cascade_depth
    
    async def _monitor_cascade_recovery(self, affected_services: List[str], timeout: int) -> Optional[float]:
        """Monitor recovery of all affected services"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            all_recovered = True
            
            for service in affected_services:
                metrics = await self.health_checker.get_service_metrics(service)
                if metrics.status != ServiceStatus.HEALTHY:
                    all_recovered = False
                    break
            
            if all_recovered:
                return time.time() - start_time
            
            await asyncio.sleep(15)
        
        return None

class ResilienceTestSuite:
    """Complete resilience testing suite"""
    
    def __init__(self, config_path: str = "/opt/sutazaiapp/chaos/config/chaos-config.yaml"):
        self.logger = self._setup_logger()
        self.health_checker = HealthChecker(self.logger)
        self.dependency_graph = self._build_dependency_graph()
        self.recovery_tester = RecoveryTimeTester(self.health_checker, self.logger)
        self.cascade_tester = CascadeFailureTester(self.health_checker, self.dependency_graph, self.logger)
        
        self.test_results: List[ResilienceTestResult] = []
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging for resilience testing"""
        logger = logging.getLogger("resilience_tester")
        logger.setLevel(logging.INFO)
        
        # File handler
        log_file = "/opt/sutazaiapp/logs/resilience_test.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def _build_dependency_graph(self) -> DependencyGraph:
        """Build service dependency graph"""
        graph = DependencyGraph()
        
        # Add services and their dependencies
        services_config = {
            'sutazai-postgres': [],
            'sutazai-redis': [],
            'sutazai-neo4j': [],
            'sutazai-ollama': [],
            'sutazai-chromadb': [],
            'sutazai-qdrant': [],
            'sutazai-backend': ['sutazai-postgres', 'sutazai-redis', 'sutazai-ollama'],
            'sutazai-frontend': ['sutazai-backend'],
            'sutazai-prometheus': [],
            'sutazai-grafana': ['sutazai-prometheus'],
            'sutazai-langflow': ['sutazai-postgres', 'sutazai-redis'],
            'sutazai-flowise': [],
            'sutazai-autogpt': ['sutazai-backend', 'sutazai-ollama'],
            'sutazai-crewai': ['sutazai-backend', 'sutazai-ollama'],
            'sutazai-letta': ['sutazai-postgres', 'sutazai-ollama'],
        }
        
        for service, deps in services_config.items():
            graph.add_service(service, deps)
        
        return graph
    
    async def run_comprehensive_test(self, target_services: List[str] = None) -> Dict[str, Any]:
        """Run comprehensive resilience testing"""
        self.logger.info("Starting comprehensive resilience testing")
        
        if not target_services:
            # Default test targets (non-critical services for testing)
            target_services = [
                'sutazai-autogpt', 'sutazai-crewai', 'sutazai-letta',
                'sutazai-flowise', 'sutazai-chromadb'
            ]
        
        results = {
            'start_time': datetime.now().isoformat(),
            'target_services': target_services,
            'recovery_time_tests': [],
            'cascade_failure_tests': [],
            'summary': {}
        }
        
        # Run recovery time tests
        for service in target_services:
            self.logger.info(f"Running recovery time test for {service}")
            result = await self.recovery_tester.test_recovery_time(service, 'kill')
            results['recovery_time_tests'].append(asdict(result))
            self.test_results.append(result)
            
            # Wait between tests
            await asyncio.sleep(60)
        
        # Run cascade failure tests (limited scope)
        cascade_test_services = ['sutazai-chromadb', 'sutazai-autogpt']
        for service in cascade_test_services:
            self.logger.info(f"Running cascade failure test for {service}")
            result = await self.cascade_tester.test_cascade_failure(service, max_cascade_depth=3)
            results['cascade_failure_tests'].append(asdict(result))
            self.test_results.append(result)
            
            # Wait between tests
            await asyncio.sleep(120)
        
        # Generate summary
        results['summary'] = self._generate_test_summary()
        results['end_time'] = datetime.now().isoformat()
        
        # Save results
        await self._save_test_results(results)
        
        return results
    
    def _generate_test_summary(self) -> Dict[str, Any]:
        """Generate summary of test results"""
        if not self.test_results:
            return {}
        
        # Recovery time statistics
        recovery_times = [r.recovery_time for r in self.test_results 
                         if r.test_type == ResilienceTestType.RECOVERY_TIME and r.recovery_time]
        
        # Cascade failure statistics
        cascade_depths = [r.cascade_depth for r in self.test_results 
                         if r.test_type == ResilienceTestType.CASCADE_FAILURE]
        
        summary = {
            'total_tests': len(self.test_results),
            'successful_tests': sum(1 for r in self.test_results if r.success),
            'success_rate': sum(1 for r in self.test_results if r.success) / len(self.test_results) * 100,
            'recovery_time_stats': {
                'mean': statistics.mean(recovery_times) if recovery_times else None,
                'median': statistics.median(recovery_times) if recovery_times else None,
                'min': min(recovery_times) if recovery_times else None,
                'max': max(recovery_times) if recovery_times else None,
                'std_dev': statistics.stdev(recovery_times) if len(recovery_times) > 1 else None
            },
            'cascade_failure_stats': {
                'max_depth': max(cascade_depths) if cascade_depths else 0,
                'avg_depth': statistics.mean(cascade_depths) if cascade_depths else 0,
                'isolation_effectiveness': sum(1 for d in cascade_depths if d <= 2) / len(cascade_depths) * 100 if cascade_depths else 100
            }
        }
        
        return summary
    
    async def _save_test_results(self, results: Dict[str, Any]):
        """Save test results to file"""
        results_dir = "/opt/sutazaiapp/chaos/reports"
        os.makedirs(results_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{results_dir}/resilience_test_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            self.logger.info(f"Test results saved to {filename}")
        except Exception as e:
            self.logger.error(f"Failed to save test results: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SutazAI Resilience Testing Suite")
    parser.add_argument("--services", nargs='+', help="Target services for testing")
    parser.add_argument("--test-type", choices=['recovery', 'cascade', 'comprehensive'], 
                       default='comprehensive', help="Type of test to run")
    
    args = parser.parse_args()
    
    async def main():
        suite = ResilienceTestSuite()
        
        if args.test_type == 'comprehensive':
            results = await suite.run_comprehensive_test(args.services)
            print(f"Comprehensive testing completed. Results: {json.dumps(results['summary'], indent=2)}")
        elif args.test_type == 'recovery':
            if not args.services:
                print("Error: --services required for recovery testing")
                return
            for service in args.services:
                result = await suite.recovery_tester.test_recovery_time(service)
                print(f"Recovery test for {service}: {result.success}, Recovery time: {result.recovery_time}s")
        elif args.test_type == 'cascade':
            if not args.services:
                print("Error: --services required for cascade testing")
                return
            for service in args.services:
                result = await suite.cascade_tester.test_cascade_failure(service)
                print(f"Cascade test for {service}: {result.success}, Affected: {len(result.affected_services)}")
    
    asyncio.run(main())