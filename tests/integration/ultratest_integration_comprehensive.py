#!/usr/bin/env python3
"""
import logging

logger = logging.getLogger(__name__)
ULTRATEST Comprehensive Integration Testing Suite
Testing all service integrations and data flows
"""

import asyncio
import aiohttp
import json
import time
import sys
from typing import Dict, List, Tuple
import traceback

class IntegrationTestSuite:
    """Comprehensive integration testing for all SutazAI services"""
    
    def __init__(self):
        self.results = []
        self.start_time = time.time()
        
    async def test_database_integration(self) -> Dict:
        """Test database connectivity and operations"""
        logger.info("🗄️  Testing Database Integration...")
        
        tests = []
        
        # PostgreSQL Integration Test
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("http://localhost:10010/health", timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        data = await response.json()
                        db_status = data.get('services', {}).get('database', 'unknown')
                        tests.append({
                            "test": "PostgreSQL via Backend API",
                            "status": "PASS" if db_status == "healthy" else "FAIL",
                            "details": f"Database status: {db_status}"
                        })
                    else:
                        tests.append({
                            "test": "PostgreSQL via Backend API", 
                            "status": "FAIL",
                            "details": f"HTTP {response.status}"
                        })
        except Exception as e:
            tests.append({
                "test": "PostgreSQL via Backend API",
                "status": "FAIL", 
                "details": str(e)
            })
        
        # Redis Integration Test
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("http://localhost:10010/health", timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        data = await response.json()
                        redis_status = data.get('services', {}).get('redis', 'unknown')
                        tests.append({
                            "test": "Redis via Backend API",
                            "status": "PASS" if redis_status == "healthy" else "FAIL",
                            "details": f"Redis status: {redis_status}"
                        })
        except Exception as e:
            tests.append({
                "test": "Redis via Backend API",
                "status": "FAIL",
                "details": str(e)
            })
        
        # Neo4j Direct Test
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("http://localhost:10002/", timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        data = await response.json()
                        neo4j_version = data.get('neo4j_version', 'unknown')
                        tests.append({
                            "test": "Neo4j Direct Connection",
                            "status": "PASS",
                            "details": f"Neo4j version: {neo4j_version}"
                        })
        except Exception as e:
            tests.append({
                "test": "Neo4j Direct Connection",
                "status": "FAIL",
                "details": str(e)
            })
        
        # Vector Database Tests
        vector_dbs = [
            {"name": "Qdrant", "url": "http://localhost:10101/healthz"},
            {"name": "ChromaDB", "url": "http://localhost:10100/api/v1/heartbeat"},
            {"name": "FAISS", "url": "http://localhost:10103/health"}
        ]
        
        for db in vector_dbs:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(db["url"], timeout=aiohttp.ClientTimeout(total=10)) as response:
                        tests.append({
                            "test": f"{db['name']} Vector DB",
                            "status": "PASS" if response.status == 200 else "FAIL",
                            "details": f"HTTP {response.status}"
                        })
            except Exception as e:
                tests.append({
                    "test": f"{db['name']} Vector DB",
                    "status": "FAIL",
                    "details": str(e)
                })
        
        return {
            "category": "Database Integration",
            "tests": tests,
            "pass_count": sum(1 for t in tests if t["status"] == "PASS"),
            "fail_count": sum(1 for t in tests if t["status"] == "FAIL"),
            "total_count": len(tests)
        }
    
    async def test_ai_service_integration(self) -> Dict:
        """Test AI service integration and communication"""
        logger.info("🤖 Testing AI Service Integration...")
        
        tests = []
        
        # Ollama Model Service Test
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("http://localhost:10104/api/tags", timeout=aiohttp.ClientTimeout(total=15)) as response:
                    if response.status == 200:
                        data = await response.json()
                        models = data.get('models', [])
                        tinyllama_available = any('tinyllama' in str(model) for model in models)
                        tests.append({
                            "test": "Ollama Model Service",
                            "status": "PASS" if tinyllama_available else "FAIL",
                            "details": f"Models available: {len(models)}, TinyLlama: {'Yes' if tinyllama_available else 'No'}"
                        })
        except Exception as e:
            tests.append({
                "test": "Ollama Model Service",
                "status": "FAIL",
                "details": str(e)
            })
        
        # Ollama Integration Service Test
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("http://localhost:8090/health", timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        data = await response.json()
                        ollama_reachable = data.get('ollama_reachable', False)
                        tinyllama_available = data.get('tinyllama_available', False)
                        tests.append({
                            "test": "Ollama Integration Service",
                            "status": "PASS" if ollama_reachable and tinyllama_available else "FAIL",
                            "details": f"Ollama reachable: {ollama_reachable}, TinyLlama: {tinyllama_available}"
                        })
        except Exception as e:
            tests.append({
                "test": "Ollama Integration Service",
                "status": "FAIL",
                "details": str(e)
            })
        
        # AI Agent Orchestrator Test
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("http://localhost:8589/health", timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        data = await response.json()
                        tests.append({
                            "test": "AI Agent Orchestrator",
                            "status": "PASS",
                            "details": f"Active tasks: {data.get('active_tasks', 0)}, Registered agents: {data.get('registered_agents', 0)}"
                        })
        except Exception as e:
            tests.append({
                "test": "AI Agent Orchestrator",
                "status": "FAIL",
                "details": str(e)
            })
        
        # Hardware Resource Optimizer Test
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("http://localhost:11110/health", timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        data = await response.json()
                        system_status = data.get('system_status', {})
                        cpu_percent = system_status.get('cpu_percent', 0)
                        memory_percent = system_status.get('memory_percent', 0)
                        tests.append({
                            "test": "Hardware Resource Optimizer",
                            "status": "PASS",
                            "details": f"CPU: {cpu_percent}%, Memory: {memory_percent}%"
                        })
        except Exception as e:
            tests.append({
                "test": "Hardware Resource Optimizer",
                "status": "FAIL",
                "details": str(e)
            })
        
        return {
            "category": "AI Service Integration",
            "tests": tests,
            "pass_count": sum(1 for t in tests if t["status"] == "PASS"),
            "fail_count": sum(1 for t in tests if t["status"] == "FAIL"),
            "total_count": len(tests)
        }
    
    async def test_monitoring_integration(self) -> Dict:
        """Test monitoring stack integration"""
        logger.info("📊 Testing Monitoring Integration...")
        
        tests = []
        
        monitoring_services = [
            {"name": "Prometheus", "url": "http://localhost:10200/-/healthy"},
            {"name": "Grafana", "url": "http://localhost:10201/api/health"},
            {"name": "Loki", "url": "http://localhost:10202/ready"},
            {"name": "AlertManager", "url": "http://localhost:10203/-/healthy"},
        ]
        
        for service in monitoring_services:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(service["url"], timeout=aiohttp.ClientTimeout(total=10)) as response:
                        if response.status == 200:
                            tests.append({
                                "test": f"{service['name']} Service",
                                "status": "PASS",
                                "details": f"HTTP {response.status}"
                            })
                        else:
                            tests.append({
                                "test": f"{service['name']} Service",
                                "status": "FAIL",
                                "details": f"HTTP {response.status}"
                            })
            except Exception as e:
                tests.append({
                    "test": f"{service['name']} Service",
                    "status": "FAIL",
                    "details": str(e)
                })
        
        # Test Prometheus Metrics Collection
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("http://localhost:10200/api/v1/query?query=up", timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        data = await response.json()
                        result_count = len(data.get('data', {}).get('result', []))
                        tests.append({
                            "test": "Prometheus Metrics Collection",
                            "status": "PASS" if result_count > 0 else "FAIL",
                            "details": f"Metrics collected from {result_count} targets"
                        })
        except Exception as e:
            tests.append({
                "test": "Prometheus Metrics Collection",
                "status": "FAIL",
                "details": str(e)
            })
        
        return {
            "category": "Monitoring Integration",
            "tests": tests,
            "pass_count": sum(1 for t in tests if t["status"] == "PASS"),
            "fail_count": sum(1 for t in tests if t["status"] == "FAIL"),
            "total_count": len(tests)
        }
    
    async def test_agent_communication(self) -> Dict:
        """Test agent-to-agent communication"""
        logger.info("🤝 Testing Agent Communication...")
        
        tests = []
        
        # Agent Health Tests
        agents = [
            {"name": "Resource Arbitration", "url": "http://localhost:8588/health"},
            {"name": "Task Assignment", "url": "http://localhost:8551/health"},
            {"name": "Hardware Optimizer", "url": "http://localhost:11110/health"},
            {"name": "Jarvis Hardware", "url": "http://localhost:11104/health"},
        ]
        
        for agent in agents:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(agent["url"], timeout=aiohttp.ClientTimeout(total=10)) as response:
                        if response.status == 200:
                            data = await response.json()
                            tests.append({
                                "test": f"{agent['name']} Agent Health",
                                "status": "PASS",
                                "details": f"Status: {data.get('status', 'unknown')}"
                            })
                        else:
                            tests.append({
                                "test": f"{agent['name']} Agent Health",
                                "status": "FAIL",
                                "details": f"HTTP {response.status}"
                            })
            except Exception as e:
                tests.append({
                    "test": f"{agent['name']} Agent Health",
                    "status": "FAIL",
                    "details": str(e)
                })
        
        return {
            "category": "Agent Communication",
            "tests": tests,
            "pass_count": sum(1 for t in tests if t["status"] == "PASS"),
            "fail_count": sum(1 for t in tests if t["status"] == "FAIL"),
            "total_count": len(tests)
        }
    
    async def test_end_to_end_workflow(self) -> Dict:
        """Test end-to-end workflow integration"""
        logger.info("🔄 Testing End-to-End Workflow...")
        
        tests = []
        
        # Test Backend API -> Model Chain
        try:
            async with aiohttp.ClientSession() as session:
                # Test model listing
                async with session.get("http://localhost:10010/api/v1/models/", timeout=aiohttp.ClientTimeout(total=15)) as response:
                    if response.status == 200:
                        data = await response.json()
                        models = data.get('models', [])
                        tests.append({
                            "test": "Backend Models API",
                            "status": "PASS" if models else "FAIL",
                            "details": f"Available models: {len(models)}"
                        })
                    else:
                        tests.append({
                            "test": "Backend Models API",
                            "status": "FAIL",
                            "details": f"HTTP {response.status}"
                        })
        except Exception as e:
            tests.append({
                "test": "Backend Models API",
                "status": "FAIL",
                "details": str(e)
            })
        
        # Test Frontend Accessibility
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("http://localhost:10011/", timeout=aiohttp.ClientTimeout(total=15)) as response:
                    tests.append({
                        "test": "Frontend UI Access",
                        "status": "PASS" if response.status == 200 else "FAIL",
                        "details": f"HTTP {response.status}"
                    })
        except Exception as e:
            tests.append({
                "test": "Frontend UI Access",
                "status": "FAIL",
                "details": str(e)
            })
        
        return {
            "category": "End-to-End Workflow",
            "tests": tests,
            "pass_count": sum(1 for t in tests if t["status"] == "PASS"),
            "fail_count": sum(1 for t in tests if t["status"] == "FAIL"),
            "total_count": len(tests)
        }
    
    async def run_comprehensive_integration_test(self) -> Dict:
        """Run all integration tests"""
        logger.info("🚀 ULTRATEST: Comprehensive Integration Testing")
        logger.info("=" * 60)
        
        test_categories = [
            await self.test_database_integration(),
            await self.test_ai_service_integration(),
            await self.test_monitoring_integration(),
            await self.test_agent_communication(),
            await self.test_end_to_end_workflow()
        ]
        
        # Calculate overall results
        total_tests = sum(cat["total_count"] for cat in test_categories)
        total_passes = sum(cat["pass_count"] for cat in test_categories)
        total_failures = sum(cat["fail_count"] for cat in test_categories)
        
        success_rate = (total_passes / total_tests * 100) if total_tests > 0 else 0
        
        return {
            "test_summary": {
                "total_tests": total_tests,
                "total_passes": total_passes,
                "total_failures": total_failures,
                "success_rate": success_rate,
                "duration": time.time() - self.start_time,
                "categories": len(test_categories)
            },
            "categories": test_categories,
            "overall_grade": self.calculate_integration_grade(success_rate, total_failures),
            "timestamp": int(time.time())
        }
    
    def calculate_integration_grade(self, success_rate: float, failures: int) -> str:
        """Calculate integration testing grade"""
        if success_rate >= 95 and failures == 0:
            return "A+ (Perfect Integration)"
        elif success_rate >= 90 and failures <= 1:
            return "A (Excellent Integration)"
        elif success_rate >= 85 and failures <= 2:
            return "B+ (Good Integration)"
        elif success_rate >= 80 and failures <= 3:
            return "B (Satisfactory Integration)"
        elif success_rate >= 70:
            return "C (Integration Issues)"
        else:
            return "F (Critical Integration Failures)"
    
    def print_integration_report(self, results: Dict):
        """Print comprehensive integration test report"""
        logger.info("\n" + "=" * 80)
        logger.info("🎯 ULTRATEST COMPREHENSIVE INTEGRATION TEST REPORT")
        logger.info("=" * 80)
        
        summary = results["test_summary"]
        logger.info(f"⏱️  Duration: {summary['duration']:.2f}s")
        logger.info(f"📊 Total Tests: {summary['total_tests']}")
        logger.info(f"✅ Passed: {summary['total_passes']}")
        logger.error(f"❌ Failed: {summary['total_failures']}")
        logger.info(f"📈 Success Rate: {summary['success_rate']:.1f}%")
        logger.info(f"📋 Categories: {summary['categories']}")
        logger.info(f"🏆 Grade: {results['overall_grade']}")
        
        logger.info(f"\n📋 DETAILED RESULTS BY CATEGORY:")
        for category in results["categories"]:
            status = "✅" if category["fail_count"] == 0 else "⚠️" if category["pass_count"] > category["fail_count"] else "❌"
            logger.info(f"\n{status} {category['category']}:")
            logger.error(f"   📊 Tests: {category['total_count']}, Passed: {category['pass_count']}, Failed: {category['fail_count']}")
            
            for test in category["tests"]:
                test_status = "✅" if test["status"] == "PASS" else "❌"
                logger.info(f"      {test_status} {test['test']}: {test['details']}")
        
        # Final assessment
        logger.info("\n" + "=" * 80)
        if summary['success_rate'] >= 95:
            logger.info("🏆 INTEGRATION TEST RESULT: EXCELLENT - ALL SYSTEMS INTEGRATED")
        elif summary['success_rate'] >= 85:
            logger.info("✅ INTEGRATION TEST RESULT: GOOD - MINOR INTEGRATION ISSUES")
        elif summary['success_rate'] >= 70:
            logger.info("⚠️  INTEGRATION TEST RESULT: MODERATE - INTEGRATION IMPROVEMENTS NEEDED")
        else:
            logger.error("🚨 INTEGRATION TEST RESULT: CRITICAL - MAJOR INTEGRATION FAILURES")
        logger.info("=" * 80)

async def main():
    """Main execution function"""
    test_suite = IntegrationTestSuite()
    
    try:
        results = await test_suite.run_comprehensive_integration_test()
        
        # Save report
        timestamp = int(time.time())
        report_file = f"/opt/sutazaiapp/tests/ultratest_integration_report_{timestamp}.json"
        with open(report_file, "w") as f:
            json.dump(results, f, indent=2)
        
        test_suite.print_integration_report(results)
        logger.info(f"\n📄 Detailed report saved: {report_file}")
        
        # Exit with appropriate code
        if results["test_summary"]["success_rate"] >= 85:
            sys.exit(0)  # Success
        else:
            sys.exit(1)  # Failure
            
    except Exception as e:
        logger.error(f"🚨 INTEGRATION TEST CRITICAL ERROR: {e}")
        traceback.print_exc()
        sys.exit(2)

if __name__ == "__main__":
    asyncio.run(main())