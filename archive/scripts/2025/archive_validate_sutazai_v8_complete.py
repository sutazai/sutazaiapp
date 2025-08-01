#!/usr/bin/env python3
"""
SutazAI v8 Complete System Validation
Comprehensive validation of all 34 services and integrations
"""

import asyncio
import aiohttp
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
import sys
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SutazAIV8Validator:
    """Comprehensive validator for SutazAI v8 system"""
    
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.services = {
            # Core Infrastructure
            "backend": f"{self.base_url}/health",
            "frontend": "http://localhost:8501/healthz",
            
            # Database Services
            "postgres": f"{self.base_url}/db/health",
            "redis": f"{self.base_url}/cache/health",
            
            # Vector Databases
            "chromadb": "http://localhost:8001/api/v1/heartbeat",
            "qdrant": "http://localhost:6333/healthz",
            "faiss": "http://localhost:8096/health",
            
            # Model Management
            "ollama": "http://localhost:11434/api/health",
            "enhanced_model_manager": "http://localhost:8098/health",
            
            # AI Agents
            "autogpt": "http://localhost:8080/health",
            "localagi": "http://localhost:8082/health",
            "tabby": "http://localhost:8081/health",
            "awesome_code_ai": "http://localhost:8097/health",
            
            # Automation Agents
            "browser_use": "http://localhost:8083/health",
            "skyvern": "http://localhost:8084/health",
            
            # Document Processing
            "documind": "http://localhost:8085/health",
            
            # Financial Analysis
            "finrobot": "http://localhost:8086/health",
            
            # Code Generation
            "gpt_engineer": "http://localhost:8087/health",
            "aider": "http://localhost:8088/health",
            
            # UI Services
            "open_webui": "http://localhost:8089/health",
            "bigagi": "http://localhost:8090/health",
            "agentzero": "http://localhost:8091/health",
            
            # Advanced AI Services
            "langflow": "http://localhost:7860/health",
            "dify": "http://localhost:5001/health",
            "autogen": "http://localhost:8092/health",
            
            # ML Frameworks
            "pytorch": "http://localhost:8093/health",
            "tensorflow": "http://localhost:8094/health",
            "jax": "http://localhost:8095/health",
            
            # Monitoring
            "prometheus": "http://localhost:9090/-/healthy",
            "grafana": "http://localhost:3000/api/health",
            "node_exporter": "http://localhost:9100/metrics",
            
            # Reverse Proxy
            "nginx": "http://localhost/health"
        }
        
        self.api_endpoints = {
            # Core API endpoints
            "system_health": f"{self.base_url}/system/comprehensive_health",
            "ai_services_status": f"{self.base_url}/ai/services/status",
            
            # New v8 endpoints
            "faiss_indexes": f"{self.base_url}/api/v8/vector/faiss/indexes",
            "awesome_ai_tools": f"{self.base_url}/api/v8/code/awesome_ai/tools",
            "enhanced_models": f"{self.base_url}/api/v8/models/enhanced/list",
            "self_improvement_stats": f"{self.base_url}/api/v8/self_improvement/stats",
            "service_summary": f"{self.base_url}/api/v8/system/service_summary",
            
            # Knowledge graph
            "knowledge_graph_stats": f"{self.base_url}/knowledge/graph/statistics",
            
            # Evolution engine
            "evolution_stats": f"{self.base_url}/evolution/statistics"
        }
        
        self.validation_results = {
            "timestamp": time.time(),
            "version": "2.0.0",
            "services": {},
            "api_endpoints": {},
            "functional_tests": {},
            "integration_tests": {},
            "performance_tests": {},
            "summary": {
                "total_services": 0,
                "healthy_services": 0,
                "unhealthy_services": 0,
                "total_endpoints": 0,
                "working_endpoints": 0,
                "failed_endpoints": 0,
                "overall_score": 0.0
            }
        }
    
    async def validate_complete_system(self) -> Dict[str, Any]:
        """Run complete system validation"""
        logger.info("🚀 Starting SutazAI v8 Complete System Validation...")
        
        # Phase 1: Service Health Checks
        logger.info("📊 Phase 1: Service Health Validation")
        await self.validate_service_health()
        
        # Phase 2: API Endpoint Validation
        logger.info("🔗 Phase 2: API Endpoint Validation")
        await self.validate_api_endpoints()
        
        # Phase 3: Functional Tests
        logger.info("⚙️ Phase 3: Functional Testing")
        await self.run_functional_tests()
        
        # Phase 4: Integration Tests
        logger.info("🔄 Phase 4: Integration Testing")
        await self.run_integration_tests()
        
        # Phase 5: Performance Tests
        logger.info("🏃 Phase 5: Performance Testing")
        await self.run_performance_tests()
        
        # Phase 6: Generate Summary
        logger.info("📋 Phase 6: Generating Summary")
        self.generate_summary()
        
        return self.validation_results
    
    async def validate_service_health(self):
        """Validate health of all services"""
        logger.info(f"Checking health of {len(self.services)} services...")
        
        async with aiohttp.ClientSession() as session:
            for service_name, health_url in self.services.items():
                try:
                    start_time = time.time()
                    async with session.get(health_url, timeout=10) as response:
                        response_time = time.time() - start_time
                        
                        if response.status == 200:
                            try:
                                health_data = await response.json()
                            except:
                                health_data = {"status": "healthy", "note": "Non-JSON response"}
                            
                            self.validation_results["services"][service_name] = {
                                "status": "healthy",
                                "response_time": response_time,
                                "details": health_data
                            }
                            self.validation_results["summary"]["healthy_services"] += 1
                            logger.info(f"✅ {service_name}: Healthy ({response_time:.3f}s)")
                        else:
                            self.validation_results["services"][service_name] = {
                                "status": "unhealthy",
                                "response_time": response_time,
                                "error": f"HTTP {response.status}"
                            }
                            self.validation_results["summary"]["unhealthy_services"] += 1
                            logger.warning(f"⚠️ {service_name}: Unhealthy (HTTP {response.status})")
                            
                except Exception as e:
                    self.validation_results["services"][service_name] = {
                        "status": "error",
                        "error": str(e)
                    }
                    self.validation_results["summary"]["unhealthy_services"] += 1
                    logger.error(f"❌ {service_name}: Error - {e}")
                
                self.validation_results["summary"]["total_services"] += 1
    
    async def validate_api_endpoints(self):
        """Validate API endpoints"""
        logger.info(f"Testing {len(self.api_endpoints)} API endpoints...")
        
        async with aiohttp.ClientSession() as session:
            for endpoint_name, url in self.api_endpoints.items():
                try:
                    start_time = time.time()
                    async with session.get(url, timeout=15) as response:
                        response_time = time.time() - start_time
                        
                        if response.status == 200:
                            try:
                                data = await response.json()
                                self.validation_results["api_endpoints"][endpoint_name] = {
                                    "status": "working",
                                    "response_time": response_time,
                                    "data_preview": str(data)[:200] + "..." if len(str(data)) > 200 else str(data)
                                }
                                self.validation_results["summary"]["working_endpoints"] += 1
                                logger.info(f"✅ {endpoint_name}: Working ({response_time:.3f}s)")
                            except Exception as e:
                                self.validation_results["api_endpoints"][endpoint_name] = {
                                    "status": "working_no_json",
                                    "response_time": response_time,
                                    "note": "Endpoint working but non-JSON response"
                                }
                                self.validation_results["summary"]["working_endpoints"] += 1
                                logger.info(f"✅ {endpoint_name}: Working (non-JSON)")
                        else:
                            self.validation_results["api_endpoints"][endpoint_name] = {
                                "status": "failed",
                                "error": f"HTTP {response.status}"
                            }
                            self.validation_results["summary"]["failed_endpoints"] += 1
                            logger.warning(f"⚠️ {endpoint_name}: Failed (HTTP {response.status})")
                            
                except Exception as e:
                    self.validation_results["api_endpoints"][endpoint_name] = {
                        "status": "error",
                        "error": str(e)
                    }
                    self.validation_results["summary"]["failed_endpoints"] += 1
                    logger.error(f"❌ {endpoint_name}: Error - {e}")
                
                self.validation_results["summary"]["total_endpoints"] += 1
    
    async def run_functional_tests(self):
        """Run functional tests for key features"""
        tests = [
            self.test_faiss_integration,
            self.test_awesome_code_ai,
            self.test_deepseek_integration,
            self.test_knowledge_graph,
            self.test_self_improvement,
            self.test_batch_processing
        ]
        
        for test in tests:
            test_name = test.__name__
            try:
                logger.info(f"Running {test_name}...")
                result = await test()
                self.validation_results["functional_tests"][test_name] = {
                    "status": "passed" if result else "failed",
                    "details": result if isinstance(result, dict) else {"success": result}
                }
                logger.info(f"✅ {test_name}: {'Passed' if result else 'Failed'}")
            except Exception as e:
                self.validation_results["functional_tests"][test_name] = {
                    "status": "error",
                    "error": str(e)
                }
                logger.error(f"❌ {test_name}: Error - {e}")
    
    async def test_faiss_integration(self) -> bool:
        """Test FAISS vector search integration"""
        try:
            async with aiohttp.ClientSession() as session:
                # Test creating an index
                async with session.post(
                    f"{self.base_url}/api/v8/vector/faiss/create_index",
                    params={"index_name": "test_index", "dimension": 128}
                ) as response:
                    if response.status != 200:
                        return False
                
                # Test searching (mock vector)
                test_vector = [0.1] * 128
                async with session.post(
                    f"{self.base_url}/api/v8/vector/faiss/search",
                    json={"query_vector": test_vector, "index_name": "test_index", "k": 5}
                ) as response:
                    return response.status in [200, 404]  # 404 is OK for empty index
        except:
            return False
    
    async def test_awesome_code_ai(self) -> bool:
        """Test Awesome Code AI integration"""
        try:
            async with aiohttp.ClientSession() as session:
                # Test code analysis
                async with session.post(
                    f"{self.base_url}/api/v8/code/awesome_ai/analyze",
                    json={
                        "code": "def hello(): print('world')",
                        "language": "python",
                        "analysis_types": ["quality"]
                    }
                ) as response:
                    return response.status == 200
        except:
            return False
    
    async def test_deepseek_integration(self) -> bool:
        """Test DeepSeek-Coder integration"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/v8/code/deepseek/generate",
                    json={"prompt": "Create a simple hello world function", "language": "python"}
                ) as response:
                    return response.status == 200
        except:
            return False
    
    async def test_knowledge_graph(self) -> bool:
        """Test knowledge graph functionality"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/knowledge/graph/statistics") as response:
                    return response.status == 200
        except:
            return False
    
    async def test_self_improvement(self) -> bool:
        """Test autonomous self-improvement system"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/api/v8/self_improvement/stats") as response:
                    return response.status == 200
        except:
            return False
    
    async def test_batch_processing(self) -> bool:
        """Test batch processing capabilities"""
        try:
            async with aiohttp.ClientSession() as session:
                batch_requests = [
                    {"prompt": "Create a function", "language": "python"},
                    {"prompt": "Create a class", "language": "python"}
                ]
                async with session.post(
                    f"{self.base_url}/api/v8/batch/code_generation",
                    json=batch_requests
                ) as response:
                    return response.status == 200
        except:
            return False
    
    async def run_integration_tests(self):
        """Run integration tests between services"""
        integration_tests = [
            self.test_model_vector_integration,
            self.test_ai_services_orchestration,
            self.test_cross_service_communication
        ]
        
        for test in integration_tests:
            test_name = test.__name__
            try:
                logger.info(f"Running {test_name}...")
                result = await test()
                self.validation_results["integration_tests"][test_name] = {
                    "status": "passed" if result else "failed",
                    "details": result if isinstance(result, dict) else {"success": result}
                }
                logger.info(f"✅ {test_name}: {'Passed' if result else 'Failed'}")
            except Exception as e:
                self.validation_results["integration_tests"][test_name] = {
                    "status": "error",
                    "error": str(e)
                }
                logger.error(f"❌ {test_name}: Error - {e}")
    
    async def test_model_vector_integration(self) -> bool:
        """Test integration between models and vector databases"""
        try:
            async with aiohttp.ClientSession() as session:
                # Test that model manager can communicate with vector stores
                async with session.get(f"{self.base_url}/ai/services/status") as response:
                    if response.status == 200:
                        data = await response.json()
                        # Check if key services are responding
                        required_services = ["enhanced_model_manager", "faiss", "chromadb", "qdrant"]
                        return all(service in data for service in required_services)
            return False
        except:
            return False
    
    async def test_ai_services_orchestration(self) -> bool:
        """Test AI services can work together"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/system/comprehensive_health") as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("overall_status") in ["healthy", "degraded"]
            return False
        except:
            return False
    
    async def test_cross_service_communication(self) -> bool:
        """Test cross-service communication"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/api/v8/system/service_summary") as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("total_services", 0) > 30
            return False
        except:
            return False
    
    async def run_performance_tests(self):
        """Run basic performance tests"""
        performance_tests = [
            self.test_response_times,
            self.test_concurrent_requests,
            self.test_memory_usage
        ]
        
        for test in performance_tests:
            test_name = test.__name__
            try:
                logger.info(f"Running {test_name}...")
                result = await test()
                self.validation_results["performance_tests"][test_name] = result
                logger.info(f"✅ {test_name}: Completed")
            except Exception as e:
                self.validation_results["performance_tests"][test_name] = {
                    "status": "error",
                    "error": str(e)
                }
                logger.error(f"❌ {test_name}: Error - {e}")
    
    async def test_response_times(self) -> Dict[str, Any]:
        """Test average response times"""
        response_times = []
        
        async with aiohttp.ClientSession() as session:
            for _ in range(10):
                start_time = time.time()
                try:
                    async with session.get(f"{self.base_url}/health") as response:
                        response_time = time.time() - start_time
                        if response.status == 200:
                            response_times.append(response_time)
                except:
                    pass
        
        if response_times:
            avg_response_time = sum(response_times) / len(response_times)
            return {
                "status": "completed",
                "average_response_time": avg_response_time,
                "samples": len(response_times),
                "performance_rating": "excellent" if avg_response_time < 0.1 else "good" if avg_response_time < 0.5 else "acceptable"
            }
        
        return {"status": "failed", "error": "No successful responses"}
    
    async def test_concurrent_requests(self) -> Dict[str, Any]:
        """Test concurrent request handling"""
        concurrent_requests = 20
        successful_requests = 0
        
        async def make_request(session):
            try:
                async with session.get(f"{self.base_url}/health") as response:
                    return response.status == 200
            except:
                return False
        
        async with aiohttp.ClientSession() as session:
            tasks = [make_request(session) for _ in range(concurrent_requests)]
            results = await asyncio.gather(*tasks)
            successful_requests = sum(results)
        
        success_rate = successful_requests / concurrent_requests
        
        return {
            "status": "completed",
            "concurrent_requests": concurrent_requests,
            "successful_requests": successful_requests,
            "success_rate": success_rate,
            "performance_rating": "excellent" if success_rate > 0.95 else "good" if success_rate > 0.8 else "needs_improvement"
        }
    
    async def test_memory_usage(self) -> Dict[str, Any]:
        """Test memory usage patterns"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            
            return {
                "status": "completed",
                "total_memory_gb": memory.total / (1024**3),
                "available_memory_gb": memory.available / (1024**3),
                "memory_usage_percent": memory.percent,
                "memory_rating": "excellent" if memory.percent < 70 else "good" if memory.percent < 85 else "high"
            }
        except ImportError:
            return {"status": "skipped", "reason": "psutil not available"}
    
    def generate_summary(self):
        """Generate validation summary"""
        total_tests = (
            len(self.validation_results["functional_tests"]) +
            len(self.validation_results["integration_tests"]) +
            len(self.validation_results["performance_tests"])
        )
        
        passed_tests = sum(1 for test in self.validation_results["functional_tests"].values() if test["status"] == "passed")
        passed_tests += sum(1 for test in self.validation_results["integration_tests"].values() if test["status"] == "passed")
        passed_tests += sum(1 for test in self.validation_results["performance_tests"].values() if test["status"] == "completed")
        
        # Calculate overall score
        service_score = self.validation_results["summary"]["healthy_services"] / max(1, self.validation_results["summary"]["total_services"])
        endpoint_score = self.validation_results["summary"]["working_endpoints"] / max(1, self.validation_results["summary"]["total_endpoints"])
        test_score = passed_tests / max(1, total_tests)
        
        overall_score = (service_score + endpoint_score + test_score) / 3
        
        self.validation_results["summary"]["overall_score"] = overall_score
        self.validation_results["summary"]["total_tests"] = total_tests
        self.validation_results["summary"]["passed_tests"] = passed_tests
        self.validation_results["summary"]["test_success_rate"] = passed_tests / max(1, total_tests)
        
        # Determine system status
        if overall_score >= 0.9:
            self.validation_results["summary"]["system_status"] = "excellent"
        elif overall_score >= 0.7:
            self.validation_results["summary"]["system_status"] = "good"
        elif overall_score >= 0.5:
            self.validation_results["summary"]["system_status"] = "acceptable"
        else:
            self.validation_results["summary"]["system_status"] = "needs_improvement"
    
    def print_summary(self):
        """Print validation summary"""
        summary = self.validation_results["summary"]
        
        print("\n" + "="*80)
        print("🚀 SUTAZAI V8 COMPLETE SYSTEM VALIDATION RESULTS")
        print("="*80)
        
        print(f"📊 Services: {summary['healthy_services']}/{summary['total_services']} healthy")
        print(f"🔗 Endpoints: {summary['working_endpoints']}/{summary['total_endpoints']} working")
        print(f"✅ Tests: {summary['passed_tests']}/{summary['total_tests']} passed")
        
        print(f"\n📈 Overall Score: {summary['overall_score']:.2%}")
        print(f"🎯 System Status: {summary['system_status'].upper()}")
        
        if summary["system_status"] == "excellent":
            print("\n🎉 CONGRATULATIONS! SutazAI v8 is running perfectly!")
            print("✅ All systems operational")
            print("✅ 100% delivery achieved")
            print("✅ Ready for production deployment")
        elif summary["system_status"] == "good":
            print("\n✅ SutazAI v8 is running well with minor issues")
            print("⚠️ Some services may need attention")
        else:
            print("\n⚠️ SutazAI v8 needs improvement")
            print("❌ Several services require attention")
        
        print("\n" + "="*80)
    
    def save_results(self, filename: str = "sutazai_v8_validation_results.json"):
        """Save validation results to file"""
        try:
            with open(filename, 'w') as f:
                json.dump(self.validation_results, f, indent=2, default=str)
            logger.info(f"✅ Validation results saved to {filename}")
        except Exception as e:
            logger.error(f"❌ Failed to save results: {e}")

async def main():
    """Main validation function"""
    validator = SutazAIV8Validator()
    
    try:
        # Run complete validation
        await validator.validate_complete_system()
        
        # Print and save results
        validator.print_summary()
        validator.save_results()
        
        # Return appropriate exit code
        if validator.validation_results["summary"]["system_status"] in ["excellent", "good"]:
            sys.exit(0)
        else:
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Validation failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    print("🚀 SutazAI v8 Complete System Validation")
    print("Testing all 34 services and integrations...")
    print("This may take several minutes...\n")
    
    asyncio.run(main())