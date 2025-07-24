#!/usr/bin/env python3
"""
SutazAI Frontend Integration Test Suite
Tests all API endpoints and service integrations
"""

import asyncio
import httpx
import json
import time
from typing import Dict, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sutazai_test")

# Test configuration
BACKEND_URL = "http://localhost:8000"
FRONTEND_URL = "http://localhost:8501"
SERVICE_ENDPOINTS = [
    {"name": "Backend API", "url": f"{BACKEND_URL}/health"},
    {"name": "Frontend App", "url": f"{FRONTEND_URL}"},
    {"name": "LangFlow", "url": "http://localhost:8090"},
    {"name": "FlowiseAI", "url": "http://localhost:8099"},
    {"name": "BigAGI", "url": "http://localhost:8106"},
    {"name": "Dify", "url": "http://localhost:8107"},
    {"name": "n8n", "url": "http://localhost:5678"},
    {"name": "Ollama", "url": "http://localhost:11434/api/tags"},
    {"name": "ChromaDB", "url": "http://localhost:8001/api/v1/heartbeat"},
    {"name": "Qdrant", "url": "http://localhost:6333/health"},
    {"name": "Neo4j", "url": "http://localhost:7474"}
]

API_ENDPOINTS = [
    {"endpoint": "/health", "method": "GET", "expected_status": 200},
    {"endpoint": "/agents", "method": "GET", "expected_status": 200},
    {"endpoint": "/models", "method": "GET", "expected_status": 200},
    {"endpoint": "/metrics", "method": "GET", "expected_status": 200},
    {"endpoint": "/api/v1/system/status", "method": "GET", "expected_status": 200},
    {"endpoint": "/simple-chat", "method": "POST", "expected_status": 200, 
     "data": {"message": "Hello, test message"}},
    {"endpoint": "/think", "method": "POST", "expected_status": 200,
     "data": {"query": "Test neural processing"}},
    {"endpoint": "/api/v1/neural/process", "method": "POST", "expected_status": 200,
     "data": {"query": "Test AGI processing", "processing_mode": "standard"}},
]

class SutazAITester:
    def __init__(self):
        self.results = {
            "services": {},
            "api_endpoints": {},
            "integration_tests": {},
            "summary": {}
        }
    
    async def test_service_health(self, service: Dict) -> bool:
        """Test if a service is healthy"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(service["url"])
                is_healthy = response.status_code in [200, 201, 202]
                
                self.results["services"][service["name"]] = {
                    "status": "healthy" if is_healthy else "unhealthy",
                    "status_code": response.status_code,
                    "response_time_ms": response.elapsed.total_seconds() * 1000
                }
                
                logger.info(f"✅ {service['name']}: {response.status_code}" if is_healthy 
                           else f"❌ {service['name']}: {response.status_code}")
                return is_healthy
                
        except Exception as e:
            self.results["services"][service["name"]] = {
                "status": "error",
                "error": str(e)
            }
            logger.error(f"❌ {service['name']}: {str(e)}")
            return False
    
    async def test_api_endpoint(self, endpoint_config: Dict) -> bool:
        """Test a specific API endpoint"""
        endpoint = endpoint_config["endpoint"]
        method = endpoint_config["method"]
        expected_status = endpoint_config["expected_status"]
        data = endpoint_config.get("data")
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                url = f"{BACKEND_URL}{endpoint}"
                
                if method == "GET":
                    response = await client.get(url)
                elif method == "POST":
                    response = await client.post(url, json=data)
                else:
                    raise ValueError(f"Unsupported method: {method}")
                
                is_success = response.status_code == expected_status
                response_data = None
                
                try:
                    response_data = response.json()
                except:
                    response_data = response.text[:200]
                
                self.results["api_endpoints"][endpoint] = {
                    "status": "success" if is_success else "failed",
                    "status_code": response.status_code,
                    "expected_status": expected_status,
                    "response_time_ms": response.elapsed.total_seconds() * 1000,
                    "response_data": response_data
                }
                
                logger.info(f"✅ {method} {endpoint}: {response.status_code}" if is_success 
                           else f"❌ {method} {endpoint}: {response.status_code} (expected {expected_status})")
                return is_success
                
        except Exception as e:
            self.results["api_endpoints"][endpoint] = {
                "status": "error",
                "error": str(e)
            }
            logger.error(f"❌ {method} {endpoint}: {str(e)}")
            return False
    
    async def test_integration_scenarios(self):
        """Test complex integration scenarios"""
        scenarios = [
            self.test_chat_flow(),
            self.test_agent_management(),
            self.test_neural_processing(),
            self.test_system_monitoring()
        ]
        
        for scenario in scenarios:
            try:
                await scenario
            except Exception as e:
                logger.error(f"Integration test failed: {str(e)}")
    
    async def test_chat_flow(self):
        """Test complete chat flow"""
        logger.info("🧪 Testing chat flow...")
        
        # Test models endpoint
        models_response = await self.call_api("/models")
        
        # Test simple chat
        chat_response = await self.call_api("/simple-chat", "POST", {
            "message": "Hello, this is a test message"
        })
        
        # Test neural processing
        neural_response = await self.call_api("/api/v1/neural/process", "POST", {
            "query": "Test neural processing capabilities",
            "processing_mode": "standard"
        })
        
        self.results["integration_tests"]["chat_flow"] = {
            "models_available": len(models_response.get("models", [])) if models_response else 0,
            "simple_chat_works": bool(chat_response and "response" in chat_response),
            "neural_processing_works": bool(neural_response and "response" in neural_response)
        }
    
    async def test_agent_management(self):
        """Test agent management functionality"""
        logger.info("🧪 Testing agent management...")
        
        agents_response = await self.call_api("/agents")
        
        self.results["integration_tests"]["agent_management"] = {
            "agents_endpoint_works": bool(agents_response),
            "agent_count": len(agents_response.get("agents", [])) if agents_response else 0
        }
    
    async def test_neural_processing(self):
        """Test neural processing capabilities"""
        logger.info("🧪 Testing neural processing...")
        
        think_response = await self.call_api("/think", "POST", {
            "query": "Test reasoning capabilities",
            "trace_enabled": True
        })
        
        self.results["integration_tests"]["neural_processing"] = {
            "think_endpoint_works": bool(think_response),
            "has_cognitive_trace": bool(think_response and think_response.get("cognitive_trace"))
        }
    
    async def test_system_monitoring(self):
        """Test system monitoring features"""
        logger.info("🧪 Testing system monitoring...")
        
        health_response = await self.call_api("/health")
        metrics_response = await self.call_api("/metrics")
        system_status = await self.call_api("/api/v1/system/status")
        
        self.results["integration_tests"]["system_monitoring"] = {
            "health_check_works": bool(health_response),
            "metrics_available": bool(metrics_response),
            "system_status_works": bool(system_status)
        }
    
    async def call_api(self, endpoint: str, method: str = "GET", data: Dict = None):
        """Helper method to call API endpoints"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                url = f"{BACKEND_URL}{endpoint}"
                
                if method == "GET":
                    response = await client.get(url)
                elif method == "POST":
                    response = await client.post(url, json=data)
                
                if response.status_code == 200:
                    return response.json()
                else:
                    return None
        except:
            return None
    
    async def run_all_tests(self):
        """Run complete test suite"""
        logger.info("🚀 Starting SutazAI Frontend Integration Tests...")
        start_time = time.time()
        
        # Test service health
        logger.info("📡 Testing service health...")
        service_results = []
        for service in SERVICE_ENDPOINTS:
            result = await self.test_service_health(service)
            service_results.append(result)
        
        # Test API endpoints
        logger.info("🔗 Testing API endpoints...")
        api_results = []
        for endpoint_config in API_ENDPOINTS:
            result = await self.test_api_endpoint(endpoint_config)
            api_results.append(result)
        
        # Test integration scenarios
        logger.info("🧪 Testing integration scenarios...")
        await self.test_integration_scenarios()
        
        # Generate summary
        total_time = time.time() - start_time
        
        self.results["summary"] = {
            "test_duration_seconds": total_time,
            "services_healthy": sum(service_results),
            "total_services": len(service_results),
            "api_endpoints_working": sum(api_results),
            "total_api_endpoints": len(api_results),
            "overall_health_score": (sum(service_results) + sum(api_results)) / (len(service_results) + len(api_results)) * 100
        }
        
        # Print results
        self.print_results()
        
        return self.results
    
    def print_results(self):
        """Print test results summary"""
        summary = self.results["summary"]
        
        print("\n" + "="*60)
        print("🏁 SUTAZAI FRONTEND INTEGRATION TEST RESULTS")
        print("="*60)
        
        print(f"⏱️  Total Test Time: {summary['test_duration_seconds']:.2f}s")
        print(f"🏥 Services Health: {summary['services_healthy']}/{summary['total_services']} ({summary['services_healthy']/summary['total_services']*100:.1f}%)")
        print(f"🔗 API Endpoints: {summary['api_endpoints_working']}/{summary['total_api_endpoints']} ({summary['api_endpoints_working']/summary['total_api_endpoints']*100:.1f}%)")
        print(f"📊 Overall Health Score: {summary['overall_health_score']:.1f}%")
        
        # Service status details
        print("\n📡 SERVICE STATUS:")
        for service_name, service_data in self.results["services"].items():
            status_icon = "✅" if service_data["status"] == "healthy" else "❌"
            print(f"  {status_icon} {service_name}: {service_data['status']}")
        
        # API endpoint details
        print("\n🔗 API ENDPOINT STATUS:")
        for endpoint, endpoint_data in self.results["api_endpoints"].items():
            status_icon = "✅" if endpoint_data["status"] == "success" else "❌"
            print(f"  {status_icon} {endpoint}: {endpoint_data['status']}")
        
        # Integration test results
        print("\n🧪 INTEGRATION TEST RESULTS:")
        for test_name, test_data in self.results["integration_tests"].items():
            print(f"  🔬 {test_name}:")
            for key, value in test_data.items():
                icon = "✅" if value else "❌"
                print(f"    {icon} {key}: {value}")
        
        print("\n" + "="*60)
        
        if summary["overall_health_score"] >= 80:
            print("🎉 EXCELLENT! SutazAI system is running optimally!")
        elif summary["overall_health_score"] >= 60:
            print("⚠️  GOOD! Most components are working, some may need attention.")
        else:
            print("🚨 CRITICAL! Several components need immediate attention.")
        
        print("="*60)

async def main():
    """Main test function"""
    tester = SutazAITester()
    results = await tester.run_all_tests()
    
    # Save results to file
    with open("/opt/sutazaiapp/logs/frontend_test_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info("📄 Test results saved to /opt/sutazaiapp/logs/frontend_test_results.json")

if __name__ == "__main__":
    asyncio.run(main())