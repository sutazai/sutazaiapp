#!/usr/bin/env python3
"""
SutazAI System Baseline Testing Suite
====================================

Comprehensive baseline testing before cleanup operations.
Tests all 28 containers, services, APIs, and integrations.

Created by: QA Team Lead Specialist
Date: August 10, 2025
Purpose: Establish system baseline for cleanup validation
"""

import asyncio
import aiohttp
import json
import time
import psutil
import subprocess
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging
import sys
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SystemBaselineTester:
    """Comprehensive system baseline testing class"""
    
    def __init__(self):
        self.results = {
            "test_start": datetime.now().isoformat(),
            "container_tests": {},
            "service_health_tests": {},
            "api_tests": {},
            "database_tests": {},
            "integration_tests": {},
            "performance_tests": {},
            "summary": {}
        }
        
        # Service endpoints to test
        self.service_endpoints = {
            "backend": "http://localhost:10010/health",
            "frontend": "http://localhost:10011/",
            "ollama": "http://localhost:10104/api/tags",
            "hardware_optimizer": "http://localhost:11110/health",
            "ai_orchestrator": "http://localhost:8589/health",
            "ollama_integration": "http://localhost:8090/health",
            "resource_arbitration": "http://localhost:8588/health",
            "task_coordinator": "http://localhost:8551/health",
            "faiss": "http://localhost:10103/health",
            "jarvis_automation": "http://localhost:11102/health",
            "qdrant": "http://localhost:10101/",
            "chromadb": "http://localhost:10100/api/v1/heartbeat",
            "neo4j": "http://localhost:10002/",
            "rabbitmq": "http://localhost:10008/",
            "prometheus": "http://localhost:10200/-/ready",
            "grafana": "http://localhost:10201/api/health",
            "loki": "http://localhost:10202/ready",
            "consul": "http://localhost:10006/v1/status/leader",
            "kong": "http://localhost:10005/status",
            "alertmanager": "http://localhost:10203/-/ready"
        }
        
        # Backend API endpoints to test
        self.backend_api_endpoints = [
            "/health",
            "/metrics",
            "/api/v1/models/",
            "/docs",
            "/openapi.json"
        ]
        
    async def test_container_status(self):
        """Test all container status and health"""
        logger.info("🔍 Testing container status...")
        
        try:
            # Get container status
            result = subprocess.run(
                ["docker", "ps", "--format", "{{.Names}}\t{{.Status}}\t{{.Ports}}"],
                capture_output=True, text=True, check=True
            )
            
            containers = {}
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        name = parts[0]
                        status = parts[1]
                        ports = parts[2] if len(parts) > 2 else ""
                        
                        containers[name] = {
                            "status": status,
                            "ports": ports,
                            "healthy": "healthy" in status.lower() or "up" in status.lower()
                        }
            
            self.results["container_tests"] = containers
            
            total_containers = len(containers)
            healthy_containers = sum(1 for c in containers.values() if c["healthy"])
            
            logger.info(f"✅ Container Status: {healthy_containers}/{total_containers} containers healthy")
            return healthy_containers == total_containers
            
        except Exception as e:
            logger.error(f"❌ Container status test failed: {e}")
            self.results["container_tests"] = {"error": str(e)}
            return False

    async def test_service_health(self):
        """Test health endpoints of all services"""
        logger.info("🔍 Testing service health endpoints...")
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
            health_results = {}
            
            for service, url in self.service_endpoints.items():
                try:
                    start_time = time.time()
                    async with session.get(url) as response:
                        response_time = time.time() - start_time
                        
                        health_results[service] = {
                            "url": url,
                            "status_code": response.status,
                            "response_time": round(response_time * 1000, 2),  # ms
                            "healthy": response.status in [200, 201, 204],
                            "content_type": response.headers.get('content-type', ''),
                        }
                        
                        # Try to get response content for some services
                        if response.status == 200:
                            try:
                                if 'json' in response.headers.get('content-type', ''):
                                    content = await response.json()
                                    health_results[service]["response_preview"] = str(content)[:200]
                                else:
                                    text = await response.text()
                                    health_results[service]["response_preview"] = text[:200]
                            except (AssertionError, Exception) as e:
                                # Suppressed exception (was bare except)
                                logger.debug(f"Suppressed exception: {e}")
                                pass
                        
                        logger.info(f"  {service}: {response.status} ({response_time:.2f}s)")
                        
                except Exception as e:
                    health_results[service] = {
                        "url": url,
                        "error": str(e),
                        "healthy": False,
                        "response_time": None
                    }
                    logger.warning(f"  {service}: ❌ {e}")
            
            self.results["service_health_tests"] = health_results
            
            total_services = len(health_results)
            healthy_services = sum(1 for r in health_results.values() if r.get("healthy", False))
            
            logger.info(f"✅ Service Health: {healthy_services}/{total_services} services responding")
            return healthy_services >= (total_services * 0.8)  # 80% threshold

    async def test_backend_api(self):
        """Test backend API endpoints comprehensively"""
        logger.info("🔍 Testing backend API endpoints...")
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
            api_results = {}
            base_url = "http://localhost:10010"
            
            for endpoint in self.backend_api_endpoints:
                try:
                    url = f"{base_url}{endpoint}"
                    start_time = time.time()
                    
                    async with session.get(url) as response:
                        response_time = time.time() - start_time
                        
                        api_results[endpoint] = {
                            "url": url,
                            "status_code": response.status,
                            "response_time": round(response_time * 1000, 2),  # ms
                            "healthy": response.status in [200, 201],
                            "content_type": response.headers.get('content-type', '')
                        }
                        
                        # Get response content for analysis
                        try:
                            if 'json' in response.headers.get('content-type', ''):
                                content = await response.json()
                                api_results[endpoint]["response_data"] = content
                            else:
                                text = await response.text()
                                api_results[endpoint]["response_preview"] = text[:200]
                        except (AssertionError, Exception) as e:
                            # Suppressed exception (was bare except)
                            logger.debug(f"Suppressed exception: {e}")
                            pass
                        
                        logger.info(f"  {endpoint}: {response.status} ({response_time:.2f}s)")
                        
                except Exception as e:
                    api_results[endpoint] = {
                        "url": f"{base_url}{endpoint}",
                        "error": str(e),
                        "healthy": False
                    }
                    logger.warning(f"  {endpoint}: ❌ {e}")
            
            self.results["api_tests"] = api_results
            
            total_endpoints = len(api_results)
            healthy_endpoints = sum(1 for r in api_results.values() if r.get("healthy", False))
            
            logger.info(f"✅ API Tests: {healthy_endpoints}/{total_endpoints} endpoints responding")
            return healthy_endpoints >= (total_endpoints * 0.8)

    async def test_databases(self):
        """Test database connectivity and basic operations"""
        logger.info("🔍 Testing database connectivity...")
        
        db_results = {}
        
        # Test PostgreSQL
        try:
            result = subprocess.run([
                "docker", "exec", "sutazai-postgres", 
                "psql", "-U", "sutazai", "-d", "sutazai", 
                "-c", "SELECT version(); SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';"
            ], capture_output=True, text=True, timeout=10)
            
            db_results["postgresql"] = {
                "healthy": result.returncode == 0,
                "output": result.stdout[:500] if result.returncode == 0 else result.stderr[:500],
                "connection": "successful" if result.returncode == 0 else "failed"
            }
            logger.info(f"  PostgreSQL: {'✅' if result.returncode == 0 else '❌'}")
        except Exception as e:
            db_results["postgresql"] = {"healthy": False, "error": str(e)}
            logger.warning(f"  PostgreSQL: ❌ {e}")
        
        # Test Redis
        try:
            result = subprocess.run([
                "docker", "exec", "sutazai-redis", 
                "redis-cli", "ping"
            ], capture_output=True, text=True, timeout=10)
            
            db_results["redis"] = {
                "healthy": result.returncode == 0 and "PONG" in result.stdout,
                "output": result.stdout.strip(),
                "connection": "successful" if result.returncode == 0 else "failed"
            }
            logger.info(f"  Redis: {'✅' if result.returncode == 0 else '❌'}")
        except Exception as e:
            db_results["redis"] = {"healthy": False, "error": str(e)}
            logger.warning(f"  Redis: ❌ {e}")
        
        # Test Neo4j via HTTP
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.get("http://localhost:10002/") as response:
                    db_results["neo4j"] = {
                        "healthy": response.status == 200,
                        "status_code": response.status,
                        "connection": "successful" if response.status == 200 else "failed"
                    }
                    logger.info(f"  Neo4j: {'✅' if response.status == 200 else '❌'}")
        except Exception as e:
            db_results["neo4j"] = {"healthy": False, "error": str(e)}
            logger.warning(f"  Neo4j: ❌ {e}")
        
        self.results["database_tests"] = db_results
        
        total_dbs = len(db_results)
        healthy_dbs = sum(1 for r in db_results.values() if r.get("healthy", False))
        
        logger.info(f"✅ Database Tests: {healthy_dbs}/{total_dbs} databases accessible")
        return healthy_dbs >= 2  # At least 2 of 3 critical databases

    async def test_integration_flows(self):
        """Test key integration flows end-to-end"""
        logger.info("🔍 Testing integration flows...")
        
        integration_results = {}
        
        # Test 1: Backend to Ollama integration
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                # Check if models are available
                async with session.get("http://localhost:10010/api/v1/models/") as response:
                    if response.status == 200:
                        models = await response.json()
                        integration_results["backend_ollama"] = {
                            "healthy": True,
                            "models_available": len(models) > 0,
                            "models": models
                        }
                        logger.info(f"  Backend-Ollama: ✅ ({len(models)} models)")
                    else:
                        integration_results["backend_ollama"] = {
                            "healthy": False,
                            "status_code": response.status
                        }
                        logger.warning(f"  Backend-Ollama: ❌ Status {response.status}")
        except Exception as e:
            integration_results["backend_ollama"] = {"healthy": False, "error": str(e)}
            logger.warning(f"  Backend-Ollama: ❌ {e}")
        
        # Test 2: Agent orchestrator communication
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
                async with session.get("http://localhost:8589/health") as response:
                    if response.status == 200:
                        health_data = await response.json()
                        integration_results["agent_orchestrator"] = {
                            "healthy": True,
                            "health_data": health_data
                        }
                        logger.info("  Agent Orchestrator: ✅")
                    else:
                        integration_results["agent_orchestrator"] = {
                            "healthy": False,
                            "status_code": response.status
                        }
                        logger.warning(f"  Agent Orchestrator: ❌ Status {response.status}")
        except Exception as e:
            integration_results["agent_orchestrator"] = {"healthy": False, "error": str(e)}
            logger.warning(f"  Agent Orchestrator: ❌ {e}")
        
        self.results["integration_tests"] = integration_results
        
        total_integrations = len(integration_results)
        healthy_integrations = sum(1 for r in integration_results.values() if r.get("healthy", False))
        
        logger.info(f"✅ Integration Tests: {healthy_integrations}/{total_integrations} flows working")
        return healthy_integrations > 0

    async def test_performance_baseline(self):
        """Establish performance baselines"""
        logger.info("🔍 Establishing performance baselines...")
        
        performance_results = {}
        
        # System resource usage
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            performance_results["system_resources"] = {
                "cpu_percent": cpu_percent,
                "memory_total_gb": round(memory.total / (1024**3), 2),
                "memory_used_gb": round(memory.used / (1024**3), 2),
                "memory_percent": memory.percent,
                "disk_total_gb": round(disk.total / (1024**3), 2),
                "disk_used_gb": round(disk.used / (1024**3), 2),
                "disk_percent": round((disk.used / disk.total) * 100, 2)
            }
            
            logger.info(f"  System Resources: CPU {cpu_percent}%, Memory {memory.percent}%, Disk {performance_results['system_resources']['disk_percent']}%")
        except Exception as e:
            performance_results["system_resources"] = {"error": str(e)}
            logger.warning(f"  System Resources: ❌ {e}")
        
        # Container resource usage
        try:
            result = subprocess.run([
                "docker", "stats", "--no-stream", "--format", 
                "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}"
            ], capture_output=True, text=True, timeout=15)
            
            if result.returncode == 0:
                container_stats = {}
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                for line in lines:
                    if line.strip():
                        parts = line.split('\t')
                        if len(parts) >= 4:
                            container = parts[0]
                            cpu = parts[1]
                            memory = parts[2]
                            mem_percent = parts[3]
                            
                            container_stats[container] = {
                                "cpu_percent": cpu,
                                "memory_usage": memory,
                                "memory_percent": mem_percent
                            }
                
                performance_results["container_stats"] = container_stats
                logger.info(f"  Container Stats: {len(container_stats)} containers monitored")
            else:
                performance_results["container_stats"] = {"error": result.stderr}
        except Exception as e:
            performance_results["container_stats"] = {"error": str(e)}
            logger.warning(f"  Container Stats: ❌ {e}")
        
        # Response time baselines for key services
        response_times = {}
        test_endpoints = {
            "backend_health": "http://localhost:10010/health",
            "backend_models": "http://localhost:10010/api/v1/models/",
            "ollama_tags": "http://localhost:10104/api/tags",
            "hardware_optimizer": "http://localhost:11110/health",
            "prometheus": "http://localhost:10200/-/ready"
        }
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
            for name, url in test_endpoints.items():
                try:
                    times = []
                    for i in range(3):  # 3 attempts for average
                        start_time = time.time()
                        async with session.get(url) as response:
                            response_time = (time.time() - start_time) * 1000  # ms
                            if response.status == 200:
                                times.append(response_time)
                    
                    if times:
                        response_times[name] = {
                            "avg_response_time_ms": round(sum(times) / len(times), 2),
                            "min_response_time_ms": round(min(times), 2),
                            "max_response_time_ms": round(max(times), 2),
                            "samples": len(times)
                        }
                except Exception as e:
                    response_times[name] = {"error": str(e)}
        
        performance_results["response_times"] = response_times
        
        self.results["performance_tests"] = performance_results
        
        logger.info("✅ Performance baseline established")
        return True

    async def run_all_tests(self):
        """Run all tests and generate comprehensive report"""
        logger.info("🚀 Starting comprehensive system baseline testing...")
        start_time = time.time()
        
        # Run all test suites
        test_results = {}
        
        test_results["container_status"] = await self.test_container_status()
        test_results["service_health"] = await self.test_service_health()
        test_results["backend_api"] = await self.test_backend_api()
        test_results["databases"] = await self.test_databases()
        test_results["integration_flows"] = await self.test_integration_flows()
        test_results["performance_baseline"] = await self.test_performance_baseline()
        
        # Calculate summary
        total_tests = len(test_results)
        passed_tests = sum(1 for result in test_results.values() if result)
        
        self.results["summary"] = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "pass_rate": round((passed_tests / total_tests) * 100, 2),
            "overall_status": "PASS" if passed_tests >= (total_tests * 0.8) else "FAIL",
            "test_duration_seconds": round(time.time() - start_time, 2),
            "test_completion": datetime.now().isoformat(),
            "test_results": test_results
        }
        
        # Save results
        report_file = f"/opt/sutazaiapp/tests/system_baseline_report_{int(time.time())}.json"
        with open(report_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Print summary
        logger.info("=" * 70)
        logger.info("🎯 SYSTEM BASELINE TEST RESULTS")
        logger.info("=" * 70)
        logger.info(f"Total Test Suites: {total_tests}")
        logger.info(f"Passed: {passed_tests}")
        logger.info(f"Failed: {total_tests - passed_tests}")
        logger.info(f"Pass Rate: {self.results['summary']['pass_rate']}%")
        logger.info(f"Overall Status: {self.results['summary']['overall_status']}")
        logger.info(f"Test Duration: {self.results['summary']['test_duration_seconds']}s")
        logger.info(f"Report saved: {report_file}")
        logger.info("=" * 70)
        
        return self.results

async def main():
    """Main test execution"""
    tester = SystemBaselineTester()
    results = await tester.run_all_tests()
    
    # Exit with appropriate code
    if results["summary"]["overall_status"] == "PASS":
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())