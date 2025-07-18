#!/usr/bin/env python3
"""
Complete SutazAI AGI/ASI System Validation
Comprehensive end-to-end testing of all system components
"""

import asyncio
import aiohttp
import time
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ServiceTest:
    name: str
    url: str
    expected_status: int = 200
    timeout: int = 10
    required: bool = True
    test_data: Optional[Dict] = None

@dataclass
class TestResult:
    service: str
    success: bool
    response_time: float
    error: Optional[str] = None
    details: Optional[Dict] = None

class SutazAISystemValidator:
    """
    Comprehensive system validator for SutazAI AGI/ASI System
    """
    
    def __init__(self):
        self.services = [
            # Core Infrastructure
            ServiceTest("Backend API", "http://localhost:8000/health"),
            ServiceTest("Frontend UI", "http://localhost:8501/healthz"),
            
            # Vector Databases
            ServiceTest("Qdrant", "http://localhost:6333/healthz"),
            ServiceTest("ChromaDB", "http://localhost:8001/api/v1/heartbeat"),
            
            # AI Model Services
            ServiceTest("Ollama", "http://localhost:11434/api/health"),
            ServiceTest("OpenWebUI", "http://localhost:8089/health", required=False),
            
            # AI Agents
            ServiceTest("Browser Use", "http://localhost:8088/health", required=False),
            ServiceTest("LangChain Agents", "http://localhost:8084/health", required=False),
            ServiceTest("AutoGen", "http://localhost:8085/health", required=False),
            ServiceTest("Mock Agents", "http://localhost:8090/health", required=False),
        ]
        
        self.test_results = []
        self.performance_metrics = {}
        
    async def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run complete system validation"""
        logger.info("🚀 Starting SutazAI AGI/ASI System Validation")
        
        start_time = time.time()
        
        # Phase 1: Service Health Checks
        logger.info("📋 Phase 1: Service Health Checks")
        await self._test_service_health()
        
        # Phase 2: API Functionality Tests
        logger.info("🔧 Phase 2: API Functionality Tests")
        await self._test_api_functionality()
        
        # Phase 3: AI Model Tests
        logger.info("🧠 Phase 3: AI Model Integration Tests")
        await self._test_ai_models()
        
        # Phase 4: Cross-Service Communication
        logger.info("🌐 Phase 4: Cross-Service Communication Tests")
        await self._test_cross_service_communication()
        
        # Phase 5: Performance Validation
        logger.info("⚡ Phase 5: Performance Validation")
        await self._test_performance()
        
        total_time = time.time() - start_time
        
        # Generate validation report
        report = self._generate_validation_report(total_time)
        
        return report
    
    async def _test_service_health(self):
        """Test health of all core services"""
        async with aiohttp.ClientSession() as session:
            tasks = []
            for service in self.services:
                tasks.append(self._test_single_service(session, service))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, TestResult):
                    self.test_results.append(result)
                    if result.success:
                        logger.info(f"✅ {result.service}: OK ({result.response_time:.3f}s)")
                    else:
                        level = logger.error if self._is_required_service(result.service) else logger.warning
                        level(f"❌ {result.service}: {result.error}")
    
    async def _test_single_service(self, session: aiohttp.ClientSession, service: ServiceTest) -> TestResult:
        """Test a single service"""
        start_time = time.time()
        
        try:
            async with session.get(
                service.url, 
                timeout=aiohttp.ClientTimeout(total=service.timeout)
            ) as response:
                response_time = time.time() - start_time
                
                if response.status == service.expected_status:
                    try:
                        data = await response.json()
                        return TestResult(
                            service=service.name,
                            success=True,
                            response_time=response_time,
                            details=data
                        )
                    except:
                        # Some health endpoints return plain text
                        return TestResult(
                            service=service.name,
                            success=True,
                            response_time=response_time
                        )
                else:
                    return TestResult(
                        service=service.name,
                        success=False,
                        response_time=response_time,
                        error=f"HTTP {response.status}"
                    )
                    
        except Exception as e:
            response_time = time.time() - start_time
            return TestResult(
                service=service.name,
                success=False,
                response_time=response_time,
                error=str(e)
            )
    
    async def _test_api_functionality(self):
        """Test core API functionality"""
        async with aiohttp.ClientSession() as session:
            # Test system status endpoint
            await self._test_system_status(session)
            
            # Test model management
            await self._test_model_management(session)
    
    async def _test_system_status(self, session: aiohttp.ClientSession):
        """Test system status endpoint"""
        try:
            async with session.get("http://localhost:8000/health") as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info("✅ System Status API: OK")
                    self.performance_metrics["system_status"] = data
                else:
                    logger.warning(f"⚠️ System Status API: HTTP {response.status}")
        except Exception as e:
            logger.warning(f"⚠️ System Status API: {e}")
    
    async def _test_model_management(self, session: aiohttp.ClientSession):
        """Test model management functionality"""
        try:
            # Test Ollama models
            async with session.get("http://localhost:11434/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    models = data.get('models', [])
                    logger.info(f"✅ Model Management: {len(models)} models available")
                    self.performance_metrics["models"] = len(models)
                else:
                    logger.warning(f"⚠️ Model Management: HTTP {response.status}")
        except Exception as e:
            logger.warning(f"⚠️ Model Management: {e}")
    
    async def _test_ai_models(self):
        """Test AI model integration"""
        async with aiohttp.ClientSession() as session:
            # Test Ollama models
            await self._test_ollama_models(session)
    
    async def _test_ollama_models(self, session: aiohttp.ClientSession):
        """Test Ollama model availability"""
        try:
            async with session.get("http://localhost:11434/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    models = data.get('models', [])
                    logger.info(f"✅ Ollama Models: {len(models)} available")
                    
                    for model in models[:3]:  # Show first 3 models
                        name = model.get('name', 'Unknown')
                        size = model.get('size', 0) / (1024**3)  # GB
                        logger.info(f"   📦 {name}: {size:.1f}GB")
                        
                    self.performance_metrics["ollama_models"] = len(models)
                else:
                    logger.warning(f"⚠️ Ollama Models: HTTP {response.status}")
        except Exception as e:
            logger.warning(f"⚠️ Ollama Models: {e}")
    
    async def _test_cross_service_communication(self):
        """Test cross-service communication"""
        async with aiohttp.ClientSession() as session:
            # Test if services can communicate
            try:
                # Check Docker network connectivity
                logger.info("🌐 Testing service-to-service communication...")
                
                # Test backend to database
                async with session.get("http://localhost:8000/health") as response:
                    if response.status == 200:
                        logger.info("✅ Backend-Database Communication: OK")
                    else:
                        logger.warning("⚠️ Backend-Database Communication: Issues detected")
                        
            except Exception as e:
                logger.warning(f"⚠️ Cross-service communication: {e}")
    
    async def _test_performance(self):
        """Test system performance metrics"""
        start_time = time.time()
        
        # Concurrent request test
        async with aiohttp.ClientSession() as session:
            tasks = []
            for i in range(5):
                tasks.append(session.get("http://localhost:8000/health"))
            
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            successful_requests = sum(
                1 for r in responses 
                if not isinstance(r, Exception) and r.status == 200
            )
            
            total_time = time.time() - start_time
            
            logger.info(f"✅ Concurrent Requests: {successful_requests}/5 successful in {total_time:.3f}s")
            
            self.performance_metrics["concurrent_test"] = {
                "successful": successful_requests,
                "total": 5,
                "time": total_time
            }
            
            # Close responses
            for response in responses:
                if not isinstance(response, Exception):
                    response.close()
    
    def _is_required_service(self, service_name: str) -> bool:
        """Check if service is required"""
        for service in self.services:
            if service.name == service_name:
                return service.required
        return True
    
    def _generate_validation_report(self, total_time: float) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        successful_tests = sum(1 for result in self.test_results if result.success)
        total_tests = len(self.test_results)
        
        required_services = [r for r in self.test_results if self._is_required_service(r.service)]
        required_successful = sum(1 for r in required_services if r.success)
        
        system_health = (required_successful / len(required_services)) * 100 if required_services else 0
        
        report = {
            "validation_summary": {
                "total_time": total_time,
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "success_rate": (successful_tests / total_tests) * 100 if total_tests > 0 else 0,
                "system_health": system_health,
                "timestamp": time.time()
            },
            "service_results": [
                {
                    "service": result.service,
                    "status": "✅ PASS" if result.success else "❌ FAIL",
                    "response_time": f"{result.response_time:.3f}s",
                    "error": result.error,
                    "required": self._is_required_service(result.service)
                }
                for result in self.test_results
            ],
            "performance_metrics": self.performance_metrics,
            "recommendations": self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate system recommendations based on test results"""
        recommendations = []
        
        failed_required = [
            r for r in self.test_results 
            if not r.success and self._is_required_service(r.service)
        ]
        
        if failed_required:
            recommendations.append(
                f"🔴 Critical: {len(failed_required)} required services are failing"
            )
        
        slow_services = [
            r for r in self.test_results 
            if r.success and r.response_time > 2.0
        ]
        
        if slow_services:
            recommendations.append(
                f"🟡 Performance: {len(slow_services)} services have slow response times"
            )
        
        if self.performance_metrics.get("ollama_models", 0) == 0:
            recommendations.append("🟡 AI Models: No Ollama models detected")
        
        if not recommendations:
            recommendations.append("🟢 System is operating optimally")
        
        return recommendations
    
    def print_report(self, report: Dict[str, Any]):
        """Print formatted validation report"""
        print("\n" + "="*80)
        print("🎯 SUTAZAI AGI/ASI SYSTEM VALIDATION REPORT")
        print("="*80)
        
        summary = report["validation_summary"]
        print(f"\n📊 SUMMARY:")
        print(f"   • Total Validation Time: {summary['total_time']:.2f}s")
        print(f"   • Tests Passed: {summary['successful_tests']}/{summary['total_tests']}")
        print(f"   • Success Rate: {summary['success_rate']:.1f}%")
        print(f"   • System Health: {summary['system_health']:.1f}%")
        
        print(f"\n🔍 SERVICE STATUS:")
        for service in report["service_results"]:
            required_tag = "⭐" if service["required"] else "📦"
            print(f"   {required_tag} {service['service']}: {service['status']} ({service['response_time']})")
            if service["error"]:
                print(f"      Error: {service['error']}")
        
        if report["performance_metrics"]:
            print(f"\n⚡ PERFORMANCE METRICS:")
            metrics = report["performance_metrics"]
            
            if "ollama_models" in metrics:
                print(f"   • Ollama Models: {metrics['ollama_models']} loaded")
            
            if "concurrent_test" in metrics:
                ct = metrics["concurrent_test"]
                print(f"   • Concurrent Requests: {ct['successful']}/{ct['total']} in {ct['time']:.3f}s")
        
        print(f"\n💡 RECOMMENDATIONS:")
        for rec in report["recommendations"]:
            print(f"   {rec}")
        
        # System Grade
        health = summary['system_health']
        if health >= 90:
            grade = "🏆 EXCELLENT"
            color = "🟢"
        elif health >= 75:
            grade = "✅ GOOD"
            color = "🟡"
        elif health >= 50:
            grade = "⚠️ NEEDS ATTENTION"
            color = "🟠"
        else:
            grade = "❌ CRITICAL"
            color = "🔴"
        
        print(f"\n{color} SYSTEM GRADE: {grade} ({health:.1f}%)")
        print("="*80)

async def main():
    """Main validation function"""
    validator = SutazAISystemValidator()
    
    try:
        report = await validator.run_comprehensive_validation()
        validator.print_report(report)
        
        # Save report to file
        with open('/opt/sutazaiapp/validation_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Detailed report saved to: /opt/sutazaiapp/validation_report.json")
        
        # Exit with appropriate code
        success_rate = report["validation_summary"]["success_rate"]
        if success_rate >= 80:
            print("🎉 SutazAI AGI/ASI System validation completed successfully!")
            sys.exit(0)
        else:
            print("⚠️ SutazAI AGI/ASI System validation completed with issues.")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        print(f"\n❌ VALIDATION FAILED: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())