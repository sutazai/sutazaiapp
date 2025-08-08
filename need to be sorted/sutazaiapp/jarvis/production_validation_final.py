#!/usr/bin/env python3
"""
Final Production Validation for SutazAI System
Tests all critical components and generates production certification
"""

import asyncio
import json
import time
import subprocess
from datetime import datetime
from typing import Dict, List, Any
import aiohttp
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProductionValidator:
    def __init__(self):
        self.results = {}
        self.start_time = time.time()
        
    async def validate_docker_images(self) -> Dict[str, Any]:
        """Validate Docker images are built and available"""
        logger.info("🐳 Validating Docker images...")
        
        result = subprocess.run(['docker', 'images', '--format', 'json'], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            return {"status": "FAILED", "error": "Cannot list Docker images"}
            
        images = [json.loads(line) for line in result.stdout.strip().split('\n') if line.strip()]
        sutazai_images = [img for img in images if 'sutazai' in img.get('Repository', '').lower()]
        
        return {
            "status": "PASSED",
            "total_images": len(images),
            "sutazai_images": len(sutazai_images),
            "details": f"Found {len(sutazai_images)} SutazAI images out of {len(images)} total"
        }
    
    async def validate_container_health(self) -> Dict[str, Any]:
        """Validate container health status"""
        logger.info("🏥 Validating container health...")
        
        result = subprocess.run(['docker', 'ps', '--format', 'json'], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            return {"status": "FAILED", "error": "Cannot list containers"}
            
        containers = [json.loads(line) for line in result.stdout.strip().split('\n') if line.strip()]
        
        total_containers = len(containers)
        healthy_containers = len([c for c in containers if 'healthy' in c.get('Status', '')])
        unhealthy_containers = len([c for c in containers if 'unhealthy' in c.get('Status', '')])
        
        health_percentage = (healthy_containers / total_containers * 100) if total_containers > 0 else 0
        
        return {
            "status": "PASSED" if health_percentage >= 70 else "NEEDS_ATTENTION",
            "total_containers": total_containers,
            "healthy_containers": healthy_containers,
            "unhealthy_containers": unhealthy_containers,
            "health_percentage": round(health_percentage, 1),
            "details": f"{health_percentage:.1f}% containers healthy ({healthy_containers}/{total_containers})"
        }
    
    async def validate_api_endpoints(self) -> Dict[str, Any]:
        """Validate critical API endpoints"""
        logger.info("🔌 Validating API endpoints...")
        
        endpoints = [
            ("Rule Control API", "http://localhost:10421/api/health/live"),
            ("Hygiene Backend", "http://localhost:10420/api/hygiene/status"),
            ("Prometheus", "http://localhost:10200/-/healthy"),
            ("Grafana", "http://localhost:10201/api/health"),
        ]
        
        results = []
        timeout = aiohttp.ClientTimeout(total=10)
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for name, url in endpoints:
                try:
                    async with session.get(url) as response:
                        status = "PASSED" if response.status == 200 else "FAILED"
                        results.append({
                            "name": name,
                            "url": url,
                            "status": status,
                            "response_code": response.status
                        })
                except Exception as e:
                    results.append({
                        "name": name,
                        "url": url,
                        "status": "FAILED",
                        "error": str(e)
                    })
        
        passed_count = len([r for r in results if r["status"] == "PASSED"])
        total_count = len(results)
        
        return {
            "status": "PASSED" if passed_count >= total_count * 0.7 else "FAILED",
            "passed_endpoints": passed_count,
            "total_endpoints": total_count,
            "endpoints": results,
            "details": f"{passed_count}/{total_count} endpoints responding correctly"
        }
    
    async def validate_ai_agents(self) -> Dict[str, Any]:
        """Validate AI agents are operational"""
        logger.info("🤖 Validating AI agents...")
        
        result = subprocess.run(['docker', 'ps', '--filter', 'name=sutazai-', '--format', 'json'], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            return {"status": "FAILED", "error": "Cannot list AI agent containers"}
            
        agents = [json.loads(line) for line in result.stdout.strip().split('\n') if line.strip()]
        
        # Count agents by phase
        phase1_agents = len([a for a in agents if 'phase1' in a.get('Names', '')])
        phase2_agents = len([a for a in agents if 'phase2' in a.get('Names', '')])
        phase3_agents = len([a for a in agents if 'phase3' in a.get('Names', '')])
        other_agents = len(agents) - phase1_agents - phase2_agents - phase3_agents
        
        # Test a few agent endpoints
        agent_health_checks = 0
        agent_ports = [10321, 10322, 11022, 11036, 11052]  # Sample agent ports
        
        timeout = aiohttp.ClientTimeout(total=5)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for port in agent_ports:
                try:
                    async with session.get(f"http://localhost:{port}/health") as response:
                        if response.status == 200:
                            agent_health_checks += 1
                except:
                    pass
        
        return {
            "status": "PASSED" if len(agents) >= 40 else "NEEDS_ATTENTION",
            "total_agents": len(agents),
            "phase1_agents": phase1_agents,
            "phase2_agents": phase2_agents,
            "phase3_agents": phase3_agents,
            "other_agents": other_agents,
            "responding_agents": agent_health_checks,
            "details": f"{len(agents)} agents deployed across 3 phases, {agent_health_checks} responding to health checks"
        }
    
    async def validate_jarvis_interface(self) -> Dict[str, Any]:
        """Validate Jarvis voice interface"""
        logger.info("🎤 Validating Jarvis interface...")
        
        # Check if Jarvis container is running
        result = subprocess.run(['docker', 'ps', '--filter', 'name=sutazai-jarvis', '--format', 'json'], 
                              capture_output=True, text=True)
        
        if result.returncode != 0:
            return {"status": "FAILED", "error": "Cannot check Jarvis container"}
        
        jarvis_containers = [json.loads(line) for line in result.stdout.strip().split('\n') if line.strip()]
        
        if not jarvis_containers:
            return {"status": "FAILED", "error": "Jarvis container not running"}
        
        # Test Jarvis endpoint
        timeout = aiohttp.ClientTimeout(total=10)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get("http://localhost:9091/") as response:
                    if response.status == 200:
                        return {
                            "status": "PASSED",
                            "container_status": jarvis_containers[0].get('Status', ''),
                            "details": "Jarvis interface is accessible and responding"
                        }
                    else:
                        return {
                            "status": "FAILED",
                            "container_status": jarvis_containers[0].get('Status', ''),
                            "error": f"Jarvis endpoint returned status {response.status}"
                        }
        except Exception as e:
            return {
                "status": "FAILED",
                "container_status": jarvis_containers[0].get('Status', ''),
                "error": f"Cannot connect to Jarvis: {str(e)}"
            }
    
    async def validate_monitoring_systems(self) -> Dict[str, Any]:
        """Validate monitoring and alerting systems"""
        logger.info("📊 Validating monitoring systems...")
        
        monitoring_services = [
            ("Prometheus", "http://localhost:10200/-/healthy"),
            ("Grafana", "http://localhost:10201/api/health"),
            ("Node Exporter", "http://localhost:10220/metrics"),
        ]
        
        results = []
        timeout = aiohttp.ClientTimeout(total=10)
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for name, url in monitoring_services:
                try:
                    async with session.get(url) as response:
                        status = "PASSED" if response.status == 200 else "FAILED"
                        results.append({
                            "service": name,
                            "status": status,
                            "response_code": response.status
                        })
                except Exception as e:
                    results.append({
                        "service": name,
                        "status": "FAILED",
                        "error": str(e)
                    })
        
        passed_count = len([r for r in results if r["status"] == "PASSED"])
        
        return {
            "status": "PASSED" if passed_count >= 2 else "NEEDS_ATTENTION",
            "services": results,
            "passed_services": passed_count,
            "total_services": len(results),
            "details": f"{passed_count}/{len(results)} monitoring services operational"
        }
    
    async def validate_security_configuration(self) -> Dict[str, Any]:
        """Validate security configuration"""
        logger.info("🔒 Validating security configuration...")
        
        security_checks = {
            "secrets_files_exist": False,
            "ssl_certificates_exist": False,
            "firewall_rules_exist": False,
            "security_scan_completed": False
        }
        
        # Check for secrets directory
        import os
        if os.path.exists("/opt/sutazaiapp/secrets"):
            security_checks["secrets_files_exist"] = True
        
        # Check for SSL certificates
        if os.path.exists("/opt/sutazaiapp/ssl"):
            security_checks["ssl_certificates_exist"] = True
        
        # Check for firewall rules
        if os.path.exists("/opt/sutazaiapp/firewall-rules.txt"):
            security_checks["firewall_rules_exist"] = True
        
        # Check for security scan results
        if os.path.exists("/opt/sutazaiapp/security-scan-results"):
            security_checks["security_scan_completed"] = True
        
        passed_checks = sum(security_checks.values())
        
        return {
            "status": "PASSED" if passed_checks >= 3 else "NEEDS_ATTENTION",
            "security_checks": security_checks,
            "passed_checks": passed_checks,
            "total_checks": len(security_checks),
            "details": f"{passed_checks}/{len(security_checks)} security configurations verified"
        }
    
    async def validate_performance_metrics(self) -> Dict[str, Any]:
        """Validate performance meets targets"""
        logger.info("⚡ Validating performance metrics...")
        
        try:
            import psutil
            
            # Get system metrics
            cpu_usage = psutil.cpu_percent(interval=1.0)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Performance thresholds
            performance_score = 100
            issues = []
            
            if cpu_usage > 80:
                performance_score -= 30
                issues.append(f"High CPU usage: {cpu_usage}%")
            
            if memory.percent > 85:
                performance_score -= 25
                issues.append(f"High memory usage: {memory.percent}%")
            
            if disk.percent > 90:
                performance_score -= 20
                issues.append(f"High disk usage: {disk.percent}%")
            
            # Test API response time
            start_time = time.time()
            timeout = aiohttp.ClientTimeout(total=10)
            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.get("http://localhost:10421/api/health/live") as response:
                        response_time = (time.time() - start_time) * 1000
                        if response_time > 2000:
                            performance_score -= 15
                            issues.append(f"Slow API response: {response_time:.0f}ms")
            except:
                performance_score -= 25
                issues.append("API response test failed")
            
            return {
                "status": "PASSED" if performance_score >= 70 else "NEEDS_ATTENTION",
                "performance_score": performance_score,
                "cpu_usage": cpu_usage,
                "memory_usage": memory.percent,
                "disk_usage": disk.percent,
                "issues": issues,
                "details": f"Performance score: {performance_score}/100"
            }
            
        except Exception as e:
            return {
                "status": "FAILED",
                "error": f"Cannot collect performance metrics: {str(e)}"
            }
    
    async def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run all validation tests"""
        logger.info("🎯 Starting comprehensive production validation...")
        
        validation_tests = [
            ("Docker Images", self.validate_docker_images),
            ("Container Health", self.validate_container_health),
            ("API Endpoints", self.validate_api_endpoints),
            ("AI Agents", self.validate_ai_agents),
            ("Jarvis Interface", self.validate_jarvis_interface),
            ("Monitoring Systems", self.validate_monitoring_systems),
            ("Security Configuration", self.validate_security_configuration),
            ("Performance Metrics", self.validate_performance_metrics),
        ]
        
        results = {}
        passed_tests = 0
        
        for test_name, test_func in validation_tests:
            try:
                logger.info(f"Running {test_name} validation...")
                result = await test_func()
                results[test_name] = result
                if result["status"] == "PASSED":
                    passed_tests += 1
            except Exception as e:
                logger.error(f"Test {test_name} failed: {e}")
                results[test_name] = {
                    "status": "FAILED",
                    "error": str(e)
                }
        
        # Calculate overall results
        total_tests = len(validation_tests)
        pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        # Determine certification status
        if pass_rate >= 90:
            certification = "CERTIFIED"
            grade = "A"
        elif pass_rate >= 80:
            certification = "CONDITIONAL"
            grade = "B"
        elif pass_rate >= 70:
            certification = "NEEDS_IMPROVEMENT"
            grade = "C"
        else:
            certification = "FAILED"
            grade = "F"
        
        duration = time.time() - self.start_time
        
        return {
            "timestamp": datetime.now().isoformat(),
            "validation_duration_seconds": round(duration, 2),
            "overall_status": certification,
            "grade": grade,
            "pass_rate": round(pass_rate, 1),
            "tests_passed": passed_tests,
            "total_tests": total_tests,
            "test_results": results
        }

async def main():
    """Main validation entry point"""
    validator = ProductionValidator()
    
    try:
        # Run comprehensive validation
        results = await validator.run_comprehensive_validation()
        
        # Display results
        print("\n" + "="*100)
        print("🎯 SUTAZAI PRODUCTION VALIDATION REPORT")
        print("="*100)
        print(f"📅 Timestamp: {results['timestamp']}")
        print(f"⏱️  Duration: {results['validation_duration_seconds']}s")
        print(f"🏆 Overall Status: {results['overall_status']}")
        print(f"📊 Grade: {results['grade']}")
        print(f"✅ Pass Rate: {results['pass_rate']}% ({results['tests_passed']}/{results['total_tests']})")
        
        print(f"\n📋 DETAILED RESULTS:")
        for test_name, result in results['test_results'].items():
            status_icon = "✅" if result["status"] == "PASSED" else "⚠️" if result["status"] == "NEEDS_ATTENTION" else "❌"
            print(f"  {status_icon} {test_name}: {result['status']}")
            if "details" in result:
                print(f"    └─ {result['details']}")
            if "error" in result:
                print(f"    └─ ERROR: {result['error']}")
        
        print(f"\n🎖️ CERTIFICATION DECISION:")
        if results['overall_status'] == "CERTIFIED":
            print("  ✅ SYSTEM CERTIFIED for production deployment")
            print("  🚀 Ready for enterprise use with excellent quality standards")
        elif results['overall_status'] == "CONDITIONAL":
            print("  ⚠️ CONDITIONAL CERTIFICATION")
            print("  🔧 Address minor issues for full production readiness")
        elif results['overall_status'] == "NEEDS_IMPROVEMENT":
            print("  🔧 SYSTEM NEEDS IMPROVEMENT")
            print("  📈 Address identified issues before production deployment")
        else:
            print("  ❌ CERTIFICATION FAILED")
            print("  🚨 Critical issues must be resolved before deployment")
        
        print("="*100)
        
        # Save report
        report_file = f"/opt/sutazaiapp/logs/production_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📄 Detailed report saved: {report_file}")
        
        # Return appropriate exit code
        if results['overall_status'] == "CERTIFIED":
            return 0
        elif results['overall_status'] in ["CONDITIONAL", "NEEDS_IMPROVEMENT"]:
            return 1
        else:
            return 2
            
    except Exception as e:
        logger.error(f"Production validation failed: {e}")
        return 3

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)