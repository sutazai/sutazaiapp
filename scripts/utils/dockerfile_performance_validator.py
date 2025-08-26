#!/usr/bin/env python3
"""
SutazAI Dockerfile Performance Validation Module
Ultra QA Validator - Performance & Load Testing for Consolidation

This module validates that Dockerfile consolidation maintains or improves
performance characteristics of containerized services.

Author: ULTRA QA VALIDATOR  
Date: August 10, 2025
Version: 1.0.0
"""

import asyncio
import aiohttp
import time
import json
import logging
import statistics
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from datetime import datetime
import docker
import psutil
import concurrent.futures

logger = logging.getLogger(__name__)

class DockerfilePerformanceValidator:
    """Performance validation for Dockerfile consolidation."""
    
    def __init__(self):
        """Initialize performance validator."""
        self.docker_client = docker.from_env()
        self.test_results = {
            'timestamp': datetime.now().isoformat(),
            'performance_metrics': {},
            'load_test_results': {},
            'resource_benchmarks': {}
        }
        
    async def test_service_response_time(self, service_name: str, endpoint: str, port: int, 
                                       iterations: int = 10) -> Dict:
        """Test service response time performance."""
        logger.info(f"Testing response time for {service_name} on port {port}")
        
        response_times = []
        successful_requests = 0
        failed_requests = 0
        
        url = f"http://localhost:{port}{endpoint}"
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
            for i in range(iterations):
                try:
                    start_time = time.time()
                    async with session.get(url) as response:
                        await response.read()  # Ensure full response is received
                        end_time = time.time()
                        
                        if response.status in [200, 201, 202]:
                            response_times.append((end_time - start_time) * 1000)  # Convert to ms
                            successful_requests += 1
                        else:
                            failed_requests += 1
                            
                except Exception as e:
                    logger.warning(f"Request {i+1} failed for {service_name}: {str(e)}")
                    failed_requests += 1
                
                # Small delay between requests
                await asyncio.sleep(0.1)
        
        # Calculate statistics
        if response_times:
            metrics = {
                'service': service_name,
                'endpoint': endpoint,
                'port': port,
                'total_requests': iterations,
                'successful_requests': successful_requests,
                'failed_requests': failed_requests,
                'success_rate': (successful_requests / iterations) * 100,
                'avg_response_time_ms': round(statistics.mean(response_times), 2),
                'min_response_time_ms': round(min(response_times), 2),
                'max_response_time_ms': round(max(response_times), 2),
                'median_response_time_ms': round(statistics.median(response_times), 2),
                'p95_response_time_ms': round(
                    sorted(response_times)[int(len(response_times) * 0.95)], 2
                ) if len(response_times) > 1 else round(response_times[0], 2),
                'performance_grade': self._grade_response_time(statistics.mean(response_times))
            }
        else:
            metrics = {
                'service': service_name,
                'endpoint': endpoint,
                'port': port,
                'total_requests': iterations,
                'successful_requests': 0,
                'failed_requests': failed_requests,
                'success_rate': 0,
                'error': 'All requests failed'
            }
        
        return metrics
    
    def _grade_response_time(self, avg_time_ms: float) -> str:
        """Grade response time performance."""
        if avg_time_ms < 100:
            return 'A'  # Excellent
        elif avg_time_ms < 250:
            return 'B'  # Good
        elif avg_time_ms < 500:
            return 'C'  # Acceptable
        elif avg_time_ms < 1000:
            return 'D'  # Poor
        else:
            return 'F'  # Unacceptable
    
    async def run_concurrent_load_test(self, service_name: str, endpoint: str, port: int,
                                     concurrent_users: int = 10, duration_seconds: int = 30) -> Dict:
        """Run concurrent load test on a service."""
        logger.info(f"Running load test on {service_name}: {concurrent_users} users for {duration_seconds}s")
        
        url = f"http://localhost:{port}{endpoint}"
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        results = {
            'service': service_name,
            'concurrent_users': concurrent_users,
            'duration_seconds': duration_seconds,
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'requests_per_second': 0,
            'response_times': []
        }
        
        async def make_requests():
            """Make requests for the duration of the test."""
            request_count = 0
            success_count = 0
            fail_count = 0
            times = []
            
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10)
            ) as session:
                while time.time() < end_time:
                    try:
                        req_start = time.time()
                        async with session.get(url) as response:
                            await response.read()
                            req_end = time.time()
                            
                            request_count += 1
                            if response.status in [200, 201, 202]:
                                success_count += 1
                                times.append((req_end - req_start) * 1000)
                            else:
                                fail_count += 1
                                
                    except Exception:
                        request_count += 1
                        fail_count += 1
                    
                    await asyncio.sleep(0.01)  # Small delay to prevent overwhelming
            
            return request_count, success_count, fail_count, times
        
        # Run concurrent load
        tasks = [make_requests() for _ in range(concurrent_users)]
        task_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Aggregate results
        for task_result in task_results:
            if isinstance(task_result, Exception):
                logger.error(f"Load test task failed: {task_result}")
                continue
                
            req_count, success_count, fail_count, times = task_result
            results['total_requests'] += req_count
            results['successful_requests'] += success_count
            results['failed_requests'] += fail_count
            results['response_times'].extend(times)
        
        # Calculate final metrics
        if results['total_requests'] > 0:
            results['success_rate'] = (results['successful_requests'] / results['total_requests']) * 100
            results['requests_per_second'] = round(results['total_requests'] / duration_seconds, 2)
            
            if results['response_times']:
                results['avg_response_time_ms'] = round(statistics.mean(results['response_times']), 2)
                results['p95_response_time_ms'] = round(
                    sorted(results['response_times'])[int(len(results['response_times']) * 0.95)], 2
                ) if len(results['response_times']) > 1 else round(results['response_times'][0], 2)
                
                # Load test performance grade
                rps_grade = 'A' if results['requests_per_second'] > 50 else 'B' if results['requests_per_second'] > 20 else 'C'
                success_grade = 'A' if results['success_rate'] > 95 else 'B' if results['success_rate'] > 90 else 'C'
                results['performance_grade'] = min(rps_grade, success_grade)  # Take worst grade
        
        return results
    
    def measure_container_resources(self, service_name: str, duration_seconds: int = 60) -> Dict:
        """Measure container resource usage over time."""
        logger.info(f"Measuring resource usage for {service_name} over {duration_seconds}s")
        
        # Find the container
        containers = self.docker_client.containers.list()
        container = None
        
        for c in containers:
            if service_name.lower() in c.name.lower():
                container = c
                break
        
        if not container:
            return {'error': f'Container for {service_name} not found'}
        
        # Collect metrics over time
        measurements = []
        start_time = time.time()
        
        while time.time() - start_time < duration_seconds:
            try:
                stats = container.stats(stream=False)
                
                # CPU calculation
                cpu_stats = stats['cpu_stats']
                precpu_stats = stats['precpu_stats']
                
                cpu_delta = cpu_stats['cpu_usage']['total_usage'] - precpu_stats['cpu_usage']['total_usage']
                system_delta = cpu_stats['system_cpu_usage'] - precpu_stats['system_cpu_usage']
                
                cpu_percent = 0
                if system_delta > 0:
                    cpu_percent = (cpu_delta / system_delta) * len(cpu_stats['cpu_usage']['percpu_usage']) * 100.0
                
                # Memory calculation
                memory_usage = stats['memory_stats']['usage']
                memory_limit = stats['memory_stats']['limit']
                memory_percent = (memory_usage / memory_limit) * 100
                
                # Network I/O
                network_rx = stats['networks'].get('bridge', {}).get('rx_bytes', 0)
                network_tx = stats['networks'].get('bridge', {}).get('tx_bytes', 0)
                
                measurements.append({
                    'timestamp': time.time(),
                    'cpu_percent': cpu_percent,
                    'memory_usage_mb': memory_usage / (1024 * 1024),
                    'memory_percent': memory_percent,
                    'network_rx_bytes': network_rx,
                    'network_tx_bytes': network_tx
                })
                
                time.sleep(2)  # Sample every 2 seconds
                
            except Exception as e:
                logger.warning(f"Failed to collect stats for {service_name}: {e}")
                break
        
        if not measurements:
            return {'error': 'No measurements collected'}
        
        # Calculate aggregated metrics
        cpu_values = [m['cpu_percent'] for m in measurements]
        memory_mb_values = [m['memory_usage_mb'] for m in measurements]
        memory_percent_values = [m['memory_percent'] for m in measurements]
        
        return {
            'service': service_name,
            'duration_seconds': duration_seconds,
            'sample_count': len(measurements),
            'cpu_usage': {
                'avg_percent': round(statistics.mean(cpu_values), 2),
                'max_percent': round(max(cpu_values), 2),
                'min_percent': round(min(cpu_values), 2),
                'p95_percent': round(sorted(cpu_values)[int(len(cpu_values) * 0.95)], 2)
            },
            'memory_usage': {
                'avg_mb': round(statistics.mean(memory_mb_values), 2),
                'max_mb': round(max(memory_mb_values), 2),
                'min_mb': round(min(memory_mb_values), 2),
                'avg_percent': round(statistics.mean(memory_percent_values), 2),
                'max_percent': round(max(memory_percent_values), 2)
            },
            'resource_grade': self._grade_resource_usage(
                statistics.mean(cpu_values), 
                statistics.mean(memory_mb_values)
            )
        }
    
    def _grade_resource_usage(self, avg_cpu_percent: float, avg_memory_mb: float) -> str:
        """Grade resource usage efficiency."""
        # CPU grading
        cpu_grade = 'A' if avg_cpu_percent < 30 else 'B' if avg_cpu_percent < 60 else 'C' if avg_cpu_percent < 80 else 'D'
        
        # Memory grading (in MB)
        memory_grade = 'A' if avg_memory_mb < 200 else 'B' if avg_memory_mb < 500 else 'C' if avg_memory_mb < 1000 else 'D'
        
        # Return worst grade
        grades = ['A', 'B', 'C', 'D', 'F']
        cpu_index = grades.index(cpu_grade)
        memory_index = grades.index(memory_grade)
        
        return grades[max(cpu_index, memory_index)]
    
    async def validate_service_performance(self, services_config: List[Dict]) -> Dict:
        """Validate performance for multiple services."""
        logger.info("Starting comprehensive performance validation")
        
        validation_results = {
            'timestamp': datetime.now().isoformat(),
            'services_tested': len(services_config),
            'performance_summary': {
                'excellent': 0,  # Grade A
                'good': 0,       # Grade B  
                'acceptable': 0, # Grade C
                'poor': 0,       # Grade D
                'failing': 0     # Grade F
            },
            'service_results': {},
            'overall_performance_grade': 'F'
        }
        
        for service_config in services_config:
            service_name = service_config['name']
            port = service_config['port']
            endpoint = service_config.get('endpoint', '/health')
            
            logger.info(f"Testing performance for {service_name}")
            
            service_results = {
                'service': service_name,
                'tests': {}
            }
            
            try:
                # Test 1: Response time performance
                response_metrics = await self.test_service_response_time(
                    service_name, endpoint, port, iterations=20
                )
                service_results['tests']['response_time'] = response_metrics
                
                # Test 2: Load test performance
                load_metrics = await self.run_concurrent_load_test(
                    service_name, endpoint, port, concurrent_users=5, duration_seconds=30
                )
                service_results['tests']['load_test'] = load_metrics
                
                # Test 3: Resource usage monitoring
                resource_metrics = self.measure_container_resources(service_name, duration_seconds=30)
                service_results['tests']['resource_usage'] = resource_metrics
                
                # Calculate overall service grade
                grades = []
                if 'performance_grade' in response_metrics:
                    grades.append(response_metrics['performance_grade'])
                if 'performance_grade' in load_metrics:
                    grades.append(load_metrics['performance_grade'])
                if 'resource_grade' in resource_metrics:
                    grades.append(resource_metrics['resource_grade'])
                
                if grades:
                    grade_values = {'A': 4, 'B': 3, 'C': 2, 'D': 1, 'F': 0}
                    avg_grade_value = sum(grade_values.get(g, 0) for g in grades) / len(grades)
                    
                    if avg_grade_value >= 3.5:
                        service_grade = 'A'
                    elif avg_grade_value >= 2.5:
                        service_grade = 'B'
                    elif avg_grade_value >= 1.5:
                        service_grade = 'C'
                    elif avg_grade_value >= 0.5:
                        service_grade = 'D'
                    else:
                        service_grade = 'F'
                    
                    service_results['overall_grade'] = service_grade
                    
                    # Update summary
                    if service_grade == 'A':
                        validation_results['performance_summary']['excellent'] += 1
                    elif service_grade == 'B':
                        validation_results['performance_summary']['good'] += 1
                    elif service_grade == 'C':
                        validation_results['performance_summary']['acceptable'] += 1
                    elif service_grade == 'D':
                        validation_results['performance_summary']['poor'] += 1
                    else:
                        validation_results['performance_summary']['failing'] += 1
                
            except Exception as e:
                logger.error(f"Performance testing failed for {service_name}: {e}")
                service_results['tests'] = {'error': str(e)}
                service_results['overall_grade'] = 'F'
                validation_results['performance_summary']['failing'] += 1
            
            validation_results['service_results'][service_name] = service_results
        
        # Calculate overall performance grade
        summary = validation_results['performance_summary']
        total_services = validation_results['services_tested']
        
        if total_services > 0:
            excellent_pct = (summary['excellent'] / total_services) * 100
            good_pct = (summary['good'] / total_services) * 100
            acceptable_pct = (summary['acceptable'] / total_services) * 100
            
            if excellent_pct >= 80:
                validation_results['overall_performance_grade'] = 'A'
            elif (excellent_pct + good_pct) >= 70:
                validation_results['overall_performance_grade'] = 'B'
            elif (excellent_pct + good_pct + acceptable_pct) >= 60:
                validation_results['overall_performance_grade'] = 'C'
            elif summary['failing'] < total_services * 0.5:
                validation_results['overall_performance_grade'] = 'D'
            else:
                validation_results['overall_performance_grade'] = 'F'
        
        logger.info(f"Performance validation completed. Overall grade: {validation_results['overall_performance_grade']}")
        return validation_results
    
    def save_performance_results(self, results: Dict, output_file: str):
        """Save performance results to JSON file."""
        project_root = Path(__file__).parent.parent
        output_path = project_root / output_file
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Performance results saved to: {output_path}")

async def main():
    """Main execution function for performance validation."""
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    validator = DockerfilePerformanceValidator()
    
    # Define services to test (key operational services)
    services_config = [
        {'name': 'backend', 'port': 10010, 'endpoint': '/health'},
        {'name': 'frontend', 'port': 10011, 'endpoint': '/'},
        {'name': 'ai-agent-orchestrator', 'port': 8589, 'endpoint': '/health'},
        {'name': 'ollama-integration', 'port': 8090, 'endpoint': '/health'},
        {'name': 'hardware-resource-optimizer', 'port': 11110, 'endpoint': '/health'}
    ]
    
    # Run comprehensive performance validation
    results = await validator.validate_service_performance(services_config)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"dockerfile_performance_validation_{timestamp}.json"
    validator.save_performance_results(results, results_file)
    
    # Print summary
    logger.info("\n" + "="*70)
    logger.info("  DOCKERFILE CONSOLIDATION PERFORMANCE VALIDATION RESULTS")
    logger.info("="*70)
    logger.info(f"Services Tested:        {results['services_tested']}")
    logger.info(f"Excellent (A):          {results['performance_summary']['excellent']}")
    logger.info(f"Good (B):               {results['performance_summary']['good']}")
    logger.info(f"Acceptable (C):         {results['performance_summary']['acceptable']}")
    logger.info(f"Poor (D):               {results['performance_summary']['poor']}")
    logger.info(f"Failing (F):            {results['performance_summary']['failing']}")
    logger.info(f"Overall Grade:          {results['overall_performance_grade']}")
    logger.info("="*70)
    
    # Return appropriate exit code
    return 0 if results['overall_performance_grade'] in ['A', 'B', 'C'] else 1

if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))