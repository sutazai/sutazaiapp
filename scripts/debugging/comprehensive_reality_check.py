#!/usr/bin/env python3
"""
COMPREHENSIVE REALITY CHECK - Elite Debugging Validation Suite
Validates real functionality vs fake implementations across SutazAI system

This script systematically tests all components to identify Rule 1 violations
and fantasy implementations that appear functional but don't work.

Usage:
    python scripts/debugging/comprehensive_reality_check.py
    python scripts/debugging/comprehensive_reality_check.py --component frontend
    python scripts/debugging/comprehensive_reality_check.py --component backend
    python scripts/debugging/comprehensive_reality_check.py --component service-mesh
"""

import asyncio
import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import argparse

import httpx
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    """Test result with detailed information"""
    name: str
    component: str
    status: str  # PASS, FAIL, WARN
    message: str
    details: Optional[Dict[str, Any]] = None
    response_time: Optional[float] = None
    evidence: Optional[str] = None

class RealityChecker:
    """Comprehensive system reality checker"""
    
    def __init__(self):
        self.results: List[TestResult] = []
        self.backend_base = "http://localhost:10010"
        self.frontend_base = "http://localhost:10011"
        self.kong_base = "http://localhost:10005"
        self.consul_base = "http://localhost:10006"
        self.rabbitmq_base = "http://localhost:10008"
        
    def log_result(self, result: TestResult):
        """Log and store test result"""
        self.results.append(result)
        status_emoji = "✅" if result.status == "PASS" else "❌" if result.status == "FAIL" else "⚠️"
        logger.info(f"{status_emoji} [{result.component}] {result.name}: {result.message}")
        if result.response_time:
            logger.info(f"   └─ Response time: {result.response_time:.3f}s")
        if result.evidence:
            logger.info(f"   └─ Evidence: {result.evidence}")

    async def test_frontend_api_s(self) -> List[TestResult]:
        """Test frontend for  API implementations"""
        results = []
        
        # Read the frontend API client
        api_client_path = Path("/opt/sutazaiapp/frontend/utils/resilient_api_client.py")
        if not api_client_path.exists():
            results.append(TestResult(
                name="Frontend API Client File Check",
                component="frontend",
                status="FAIL",
                message="API client file not found",
                evidence=str(api_client_path)
            ))
            return results
            
        content = api_client_path.read_text()
        
        # Check for  implementations
        _indicators = [
            " health check response",
            " API response",
            "#  s",
            "hardcoded",
            "fake",
            "return {\"status\": \"healthy\""
        ]
        
        found_s = []
        for indicator in _indicators:
            if indicator in content:
                found_s.append(indicator)
                
        if found_s:
            results.append(TestResult(
                name="Frontend  API Detection",
                component="frontend",
                status="FAIL",
                message=f"RULE 1 VIOLATION: Found {len(found_s)}  implementations",
                evidence=f" patterns: {found_s}",
                details={"_count": len(found_s), "patterns": found_s}
            ))
        else:
            results.append(TestResult(
                name="Frontend  API Detection",
                component="frontend",
                status="PASS",
                message="No obvious  patterns detected"
            ))
            
        # Test if frontend actually calls backend
        try:
            response = requests.get(f"{self.frontend_base}/", timeout=5)
            if response.status_code == 200:
                results.append(TestResult(
                    name="Frontend Service Availability",
                    component="frontend",
                    status="PASS",
                    message="Frontend service is running"
                ))
            else:
                results.append(TestResult(
                    name="Frontend Service Availability",
                    component="frontend",
                    status="FAIL",
                    message=f"Frontend returned status {response.status_code}"
                ))
        except Exception as e:
            results.append(TestResult(
                name="Frontend Service Availability",
                component="frontend",
                status="FAIL",
                message=f"Frontend service unreachable: {str(e)}"
            ))
            
        return results

    async def test_backend_real_endpoints(self) -> List[TestResult]:
        """Test backend real endpoint functionality"""
        results = []
        
        # Test health endpoint
        start_time = time.time()
        try:
            response = requests.get(f"{self.backend_base}/health", timeout=10)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                if "status" in data and data["status"] == "healthy":
                    results.append(TestResult(
                        name="Backend Health Endpoint",
                        component="backend",
                        status="PASS",
                        message="Real health endpoint working",
                        response_time=response_time,
                        details=data
                    ))
                else:
                    results.append(TestResult(
                        name="Backend Health Endpoint",
                        component="backend",
                        status="FAIL",
                        message="Health endpoint returned invalid data",
                        response_time=response_time,
                        details=data
                    ))
            else:
                results.append(TestResult(
                    name="Backend Health Endpoint",
                    component="backend",
                    status="FAIL",
                    message=f"Health endpoint failed with status {response.status_code}",
                    response_time=response_time
                ))
        except Exception as e:
            results.append(TestResult(
                name="Backend Health Endpoint",
                component="backend",
                status="FAIL",
                message=f"Health endpoint unreachable: {str(e)}"
            ))

        # Test chat endpoint performance
        start_time = time.time()
        try:
            chat_data = {
                "message": "Test message for performance check",
                "model": "tinyllama",
                "use_cache": False
            }
            response = requests.post(
                f"{self.backend_base}/api/v1/chat",
                json=chat_data,
                timeout=15
            )
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                if "response" in data:
                    if response_time > 8.0:
                        results.append(TestResult(
                            name="Backend Chat Performance",
                            component="backend",
                            status="FAIL",
                            message=f"PERFORMANCE ISSUE: Chat took {response_time:.1f}s (target <2s)",
                            response_time=response_time,
                            evidence="Performance degradation confirmed"
                        ))
                    elif response_time > 3.0:
                        results.append(TestResult(
                            name="Backend Chat Performance",
                            component="backend",
                            status="WARN",
                            message=f"Chat performance slow: {response_time:.1f}s",
                            response_time=response_time
                        ))
                    else:
                        results.append(TestResult(
                            name="Backend Chat Performance",
                            component="backend",
                            status="PASS",
                            message=f"Chat performance acceptable: {response_time:.1f}s",
                            response_time=response_time
                        ))
                else:
                    results.append(TestResult(
                        name="Backend Chat Functionality",
                        component="backend",
                        status="FAIL",
                        message="Chat endpoint returned invalid response format",
                        response_time=response_time
                    ))
            else:
                results.append(TestResult(
                    name="Backend Chat Functionality",
                    component="backend",
                    status="FAIL",
                    message=f"Chat endpoint failed with status {response.status_code}",
                    response_time=response_time
                ))
        except Exception as e:
            results.append(TestResult(
                name="Backend Chat Functionality",
                component="backend",
                status="FAIL",
                message=f"Chat endpoint error: {str(e)}"
            ))

        # Test agents endpoint
        try:
            response = requests.get(f"{self.backend_base}/api/v1/agents", timeout=10)
            if response.status_code == 200:
                agents = response.json()
                if isinstance(agents, list) and len(agents) > 0:
                    results.append(TestResult(
                        name="Backend Agents Endpoint",
                        component="backend",
                        status="PASS",
                        message=f"Agents endpoint working with {len(agents)} agents",
                        details={"agent_count": len(agents)}
                    ))
                else:
                    results.append(TestResult(
                        name="Backend Agents Endpoint",
                        component="backend",
                        status="WARN",
                        message="Agents endpoint returned empty list"
                    ))
            else:
                results.append(TestResult(
                    name="Backend Agents Endpoint",
                    component="backend",
                    status="FAIL",
                    message=f"Agents endpoint failed with status {response.status_code}"
                ))
        except Exception as e:
            results.append(TestResult(
                name="Backend Agents Endpoint",
                component="backend",
                status="FAIL",
                message=f"Agents endpoint error: {str(e)}"
            ))
            
        return results

    async def test_service_mesh_integration(self) -> List[TestResult]:
        """Test service mesh integration reality"""
        results = []
        
        # Test Kong API Gateway
        try:
            response = requests.get(f"{self.kong_base}/", timeout=5)
            if "no Route matched" in response.text:
                results.append(TestResult(
                    name="Kong Gateway Integration",
                    component="service-mesh",
                    status="FAIL",
                    message="FACADE CONFIRMED: Kong running but no routes configured",
                    evidence="Kong returns 'no Route matched' - not integrated"
                ))
            else:
                results.append(TestResult(
                    name="Kong Gateway Integration",
                    component="service-mesh",
                    status="PASS",
                    message="Kong appears to have routes configured"
                ))
        except Exception as e:
            results.append(TestResult(
                name="Kong Gateway Integration",
                component="service-mesh",
                status="FAIL",
                message=f"Kong unreachable: {str(e)}"
            ))

        # Test Consul Service Discovery
        try:
            response = requests.get(f"{self.consul_base}/v1/agent/services", timeout=5)
            if response.status_code == 200:
                services = response.json()
                if len(services) < 5:  # Should have many services registered
                    results.append(TestResult(
                        name="Consul Service Discovery",
                        component="service-mesh",
                        status="FAIL",
                        message=f"FACADE CONFIRMED: Only {len(services)} services registered (expected 10+)",
                        evidence=f"Services: {list(services.keys())}"
                    ))
                else:
                    results.append(TestResult(
                        name="Consul Service Discovery",
                        component="service-mesh",
                        status="PASS",
                        message=f"Consul has {len(services)} services registered"
                    ))
            else:
                results.append(TestResult(
                    name="Consul Service Discovery",
                    component="service-mesh",
                    status="FAIL",
                    message=f"Consul API failed with status {response.status_code}"
                ))
        except Exception as e:
            results.append(TestResult(
                name="Consul Service Discovery",
                component="service-mesh",
                status="FAIL",
                message=f"Consul unreachable: {str(e)}"
            ))

        # Test RabbitMQ Message Broker
        try:
            response = requests.get(
                f"{self.rabbitmq_base}/api/queues",
                auth=("guest", "guest"),
                timeout=5
            )
            if response.status_code == 200:
                queues = response.json()
                if len(queues) == 0:
                    results.append(TestResult(
                        name="RabbitMQ Message Broker",
                        component="service-mesh",
                        status="FAIL",
                        message="FACADE CONFIRMED: RabbitMQ running but no queues (unused)",
                        evidence="Zero queues despite agent connections configured"
                    ))
                else:
                    results.append(TestResult(
                        name="RabbitMQ Message Broker",
                        component="service-mesh",
                        status="PASS",
                        message=f"RabbitMQ has {len(queues)} active queues"
                    ))
            else:
                results.append(TestResult(
                    name="RabbitMQ Message Broker",
                    component="service-mesh",
                    status="FAIL",
                    message=f"RabbitMQ API failed with status {response.status_code}"
                ))
        except Exception as e:
            results.append(TestResult(
                name="RabbitMQ Message Broker",
                component="service-mesh",
                status="FAIL",
                message=f"RabbitMQ unreachable: {str(e)}"
            ))
            
        return results

    async def test_mcp_integration(self) -> List[TestResult]:
        """Test MCP integration reality"""
        results = []
        
        # Check MCP adapter architecture
        mcp_adapter_path = Path("/opt/sutazaiapp/backend/app/mesh/mcp_adapter.py")
        if mcp_adapter_path.exists():
            content = mcp_adapter_path.read_text()
            
            # Look for STDIO to HTTP bridge attempts
            if "subprocess.Popen" in content and "stdin=subprocess.PIPE" in content:
                results.append(TestResult(
                    name="MCP Architecture Analysis",
                    component="mcp",
                    status="FAIL",
                    message="RULE 1 VIOLATION: MCP adapter tries to bridge STDIO to HTTP (impossible)",
                    evidence="subprocess.Popen with STDIO pipes cannot become HTTP endpoints"
                ))
            else:
                results.append(TestResult(
                    name="MCP Architecture Analysis",
                    component="mcp",
                    status="PASS",
                    message="MCP adapter appears to use proper HTTP architecture"
                ))
        else:
            results.append(TestResult(
                name="MCP Architecture Analysis",
                component="mcp",
                status="WARN",
                message="MCP adapter file not found"
            ))

        # Test actual MCP wrapper
        postgres_wrapper = Path("/opt/sutazaiapp/scripts/mcp/wrappers/postgres.sh")
        if postgres_wrapper.exists():
            content = postgres_wrapper.read_text()
            if "docker run" in content and "-i" in content:
                results.append(TestResult(
                    name="MCP Wrapper Architecture",
                    component="mcp",
                    status="FAIL",
                    message="CONFIRMED: MCP wrappers use STDIO (docker run -i), not HTTP",
                    evidence="MCP processes are command-line tools, incompatible with HTTP service mesh"
                ))
            else:
                results.append(TestResult(
                    name="MCP Wrapper Architecture",
                    component="mcp",
                    status="PASS",
                    message="MCP wrapper appears HTTP-compatible"
                ))
        else:
            results.append(TestResult(
                name="MCP Wrapper Architecture",
                component="mcp",
                status="WARN",
                message="MCP wrapper not found"
            ))
            
        return results

    async def test_frontend_backend_integration(self) -> List[TestResult]:
        """Test if frontend actually calls backend"""
        results = []
        
        # This is hard to test directly, but we can check if frontend health
        # matches backend health (if ed, they won't match reality)
        
        try:
            # Get backend health
            backend_response = requests.get(f"{self.backend_base}/health", timeout=5)
            backend_health = backend_response.json() if backend_response.status_code == 200 else {}
            
            # Try to access frontend health info if available
            # (This would require knowing frontend's health check mechanism)
            
            # For now, we check if frontend API client has s
            api_client_path = Path("/opt/sutazaiapp/frontend/utils/resilient_api_client.py")
            if api_client_path.exists():
                content = api_client_path.read_text()
                if "return {" in content and "\"status\": \"healthy\"" in content:
                    results.append(TestResult(
                        name="Frontend-Backend Integration",
                        component="integration",
                        status="FAIL",
                        message="RULE 1 VIOLATION: Frontend uses hardcoded responses instead of backend calls",
                        evidence="Found hardcoded health responses in frontend API client"
                    ))
                else:
                    results.append(TestResult(
                        name="Frontend-Backend Integration",
                        component="integration",
                        status="PASS",
                        message="Frontend appears to make real backend calls"
                    ))
            else:
                results.append(TestResult(
                    name="Frontend-Backend Integration",
                    component="integration",
                    status="WARN",
                    message="Cannot locate frontend API client"
                ))
                
        except Exception as e:
            results.append(TestResult(
                name="Frontend-Backend Integration",
                component="integration",
                status="FAIL",
                message=f"Integration test failed: {str(e)}"
            ))
            
        return results

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive report"""
        
        total_tests = len(self.results)
        passed = len([r for r in self.results if r.status == "PASS"])
        failed = len([r for r in self.results if r.status == "FAIL"])
        warnings = len([r for r in self.results if r.status == "WARN"])
        
        # Group by component
        by_component = {}
        for result in self.results:
            if result.component not in by_component:
                by_component[result.component] = []
            by_component[result.component].append(result)
        
        # Calculate functionality scores
        component_scores = {}
        for component, results in by_component.items():
            comp_total = len(results)
            comp_passed = len([r for r in results if r.status == "PASS"])
            comp_score = (comp_passed / comp_total * 100) if comp_total > 0 else 0
            component_scores[component] = comp_score
        
        # Identify critical violations
        critical_violations = [
            r for r in self.results 
            if r.status == "FAIL" and any(keyword in r.message.upper() for keyword in 
                ["RULE 1", "FACADE", "", "VIOLATION", "IMPOSSIBLE"])
        ]
        
        return {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_tests": total_tests,
                "passed": passed,
                "failed": failed,
                "warnings": warnings,
                "success_rate": (passed / total_tests * 100) if total_tests > 0 else 0
            },
            "component_scores": component_scores,
            "critical_violations": len(critical_violations),
            "violations": [
                {
                    "component": v.component,
                    "name": v.name,
                    "message": v.message,
                    "evidence": v.evidence
                }
                for v in critical_violations
            ],
            "by_component": {
                comp: [
                    {
                        "name": r.name,
                        "status": r.status,
                        "message": r.message,
                        "response_time": r.response_time,
                        "evidence": r.evidence
                    }
                    for r in results
                ]
                for comp, results in by_component.items()
            },
            "recommendations": self.generate_recommendations(critical_violations, component_scores)
        }
    
    def generate_recommendations(self, violations, scores) -> List[str]:
        """Generate fix recommendations based on findings"""
        recommendations = []
        
        if any("frontend" in v.component for v in violations):
            recommendations.append(
                "CRITICAL: Rewrite frontend API client to make real HTTP calls instead of  responses"
            )
        
        if any("service-mesh" in v.component for v in violations):
            recommendations.append(
                "CRITICAL: Integrate Kong, Consul, RabbitMQ properly or remove facade infrastructure"
            )
            
        if any("mcp" in v.component for v in violations):
            recommendations.append(
                "ARCHITECTURAL: Redesign MCP integration - STDIO processes cannot become HTTP services"
            )
            
        if scores.get("backend", 100) < 80:
            recommendations.append(
                "HIGH: Optimize backend performance - implement caching, connection pooling, async processing"
            )
            
        if len(violations) > 5:
            recommendations.append(
                "SYSTEMATIC: Implement Rule 1 compliance checking in CI/CD to prevent fantasy code"
            )
            
        return recommendations

    async def run_all_tests(self, component_filter: Optional[str] = None) -> Dict[str, Any]:
        """Run comprehensive reality check"""
        logger.info("🔍 Starting Comprehensive Reality Check...")
        
        if not component_filter or component_filter == "frontend":
            logger.info("Testing frontend  implementations...")
            frontend_results = await self.test_frontend_api_s()
            for result in frontend_results:
                self.log_result(result)
        
        if not component_filter or component_filter == "backend":
            logger.info("Testing backend real endpoints...")
            backend_results = await self.test_backend_real_endpoints()
            for result in backend_results:
                self.log_result(result)
        
        if not component_filter or component_filter == "service-mesh":
            logger.info("Testing service mesh integration...")
            mesh_results = await self.test_service_mesh_integration()
            for result in mesh_results:
                self.log_result(result)
        
        if not component_filter or component_filter == "mcp":
            logger.info("Testing MCP integration...")
            mcp_results = await self.test_mcp_integration()
            for result in mcp_results:
                self.log_result(result)
        
        if not component_filter or component_filter == "integration":
            logger.info("Testing frontend-backend integration...")
            integration_results = await self.test_frontend_backend_integration()
            for result in integration_results:
                self.log_result(result)
        
        return self.generate_report()

async def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Comprehensive Reality Check for SutazAI")
    parser.add_argument(
        "--component",
        choices=["frontend", "backend", "service-mesh", "mcp", "integration"],
        help="Test specific component only"
    )
    parser.add_argument(
        "--output",
        default="/opt/sutazaiapp/reports/reality_check_report.json",
        help="Output file for detailed JSON report"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    checker = RealityChecker()
    
    try:
        report = await checker.run_all_tests(args.component)
        
        # Save detailed report
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print("\n" + "="*80)
        print("🚨 COMPREHENSIVE REALITY CHECK SUMMARY")
        print("="*80)
        
        summary = report["summary"]
        print(f"Total Tests: {summary['total_tests']}")
        print(f"✅ Passed: {summary['passed']}")
        print(f"❌ Failed: {summary['failed']}")
        print(f"⚠️  Warnings: {summary['warnings']}")
        print(f"Success Rate: {summary['success_rate']:.1f}%")
        
        print(f"\n🔥 Critical Rule 1 Violations: {report['critical_violations']}")
        
        print("\n📊 Component Functionality Scores:")
        for component, score in report["component_scores"].items():
            status = "🔴" if score < 50 else "🟡" if score < 80 else "🟢"
            print(f"  {status} {component}: {score:.1f}%")
        
        if report["violations"]:
            print(f"\n🚨 CRITICAL VIOLATIONS FOUND:")
            for violation in report["violations"]:
                print(f"  ❌ [{violation['component']}] {violation['name']}")
                print(f"     {violation['message']}")
                if violation.get('evidence'):
                    print(f"     Evidence: {violation['evidence']}")
        
        print(f"\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(report["recommendations"], 1):
            print(f"  {i}. {rec}")
        
        print(f"\n📄 Detailed report saved to: {output_path}")
        
        # Exit with error code if critical violations found
        if report["critical_violations"] > 0:
            print(f"\n🚨 REALITY CHECK FAILED: {report['critical_violations']} critical violations found")
            sys.exit(1)
        else:
            print(f"\n✅ REALITY CHECK PASSED: No critical violations detected")
            sys.exit(0)
            
    except KeyboardInterrupt:
        print("\n⚠️  Reality check interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Reality check failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())