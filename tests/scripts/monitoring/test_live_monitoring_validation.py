#!/usr/bin/env python3
"""
import logging

logger = logging.getLogger(__name__)
Live Monitoring System Validation
=================================

Tests the monitoring system against the actual running SutazAI system.
Validates that the monitoring system correctly detects and reports on live agents.
"""

import subprocess
import json
import requests
import time
import sys
import os
from datetime import datetime
import unittest
from unittest.Mock import patch


class LiveMonitoringValidator:
    """Validates the monitoring system against live system"""
    
    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "tests": [],
            "summary": {}
        }
    
    def test_docker_container_detection(self):
        """Test that Docker containers are correctly detected"""
        logger.info("🐳 Testing Docker Container Detection...")
        
        try:
            # Get Docker containers
            result = subprocess.run(
                ['docker', 'ps', '--filter', 'name=sutazai-', 
                 '--format', '{{.Names}}\t{{.Status}}\t{{.Ports}}'],
                capture_output=True, text=True, timeout=10
            )
            
            if result.returncode != 0:
                self.results["tests"].append({
                    "name": "docker_container_detection",
                    "status": "failed",
                    "error": f"Docker command failed: {result.stderr}"
                })
                return False
            
            containers = []
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if line.strip():
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        name = parts[0].replace('sutazai-', '')
                        status = parts[1]
                        ports = parts[2] if len(parts) > 2 else ''
                        
                        containers.append({
                            "name": name,
                            "status": status,
                            "ports": ports,
                            "docker_status": self._parse_docker_status(status)
                        })
            
            self.results["tests"].append({
                "name": "docker_container_detection",
                "status": "passed",
                "containers_found": len(containers),
                "containers": containers[:10]  # Limit for readability
            })
            
            logger.info(f"  ✅ Found {len(containers)} SutazAI containers")
            return True
            
        except Exception as e:
            self.results["tests"].append({
                "name": "docker_container_detection",
                "status": "failed",
                "error": str(e)
            })
            logger.error(f"  ❌ Error: {e}")
            return False
    
    def test_agent_health_endpoints(self):
        """Test health endpoints of running agents"""
        logger.info("🏥 Testing Agent Health Endpoints...")
        
        try:
            # Get running containers with ports
            result = subprocess.run(
                ['docker', 'ps', '--filter', 'name=sutazai-', '--filter', 'status=running',
                 '--format', '{{.Names}}\t{{.Ports}}'],
                capture_output=True, text=True, timeout=10
            )
            
            if result.returncode != 0:
                logger.error(f"  ❌ Docker command failed: {result.stderr}")
                return False
            
            health_results = []
            lines = result.stdout.strip().split('\n')
            
            for line in lines:
                if line.strip():
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        name = parts[0].replace('sutazai-', '')
                        ports = parts[1]
                        
                        # Extract port number
                        port = self._extract_port(ports)
                        if port:
                            health_status = self._check_health_endpoint(port, timeout=3)
                            health_results.append({
                                "agent": name,
                                "port": port,
                                "health_status": health_status
                            })
            
            healthy_count = len([r for r in health_results if r["health_status"] == "healthy"])
            total_count = len(health_results)
            
            self.results["tests"].append({
                "name": "agent_health_endpoints",
                "status": "passed",
                "healthy_agents": healthy_count,
                "total_agents": total_count,
                "health_rate": (healthy_count / total_count * 100) if total_count > 0 else 0,
                "results": health_results[:10]
            })
            
            logger.info(f"  ✅ Tested {total_count} agents, {healthy_count} healthy ({healthy_count/total_count*100:.1f}%)")
            return True
            
        except Exception as e:
            self.results["tests"].append({
                "name": "agent_health_endpoints",
                "status": "failed",
                "error": str(e)
            })
            logger.error(f"  ❌ Error: {e}")
            return False
    
    def test_monitoring_system_integration(self):
        """Test the monitoring system components work together"""
        logger.info("🔄 Testing Monitoring System Integration...")
        
        try:
            # Test that agent registry exists and is valid
            registry_path = '/opt/sutazaiapp/agents/communication_config.json'
            if os.path.exists(registry_path):
                with open(registry_path, 'r') as f:
                    config = json.load(f)
                
                required_sections = ['health_monitoring', 'agent_endpoints']
                missing_sections = [s for s in required_sections if s not in config]
                
                if missing_sections:
                    self.results["tests"].append({
                        "name": "monitoring_system_integration",
                        "status": "failed",
                        "error": f"Missing config sections: {missing_sections}"
                    })
                    return False
                
                health_config = config['health_monitoring']
                is_enabled = health_config.get('enabled', False)
                check_interval = health_config.get('check_interval', 0)
                
                self.results["tests"].append({
                    "name": "monitoring_system_integration",
                    "status": "passed",
                    "health_monitoring_enabled": is_enabled,
                    "check_interval": check_interval,
                    "config_sections": list(config.keys())
                })
                
                logger.info(f"  ✅ Configuration valid, health monitoring: {'enabled' if is_enabled else 'disabled'}")
                return True
            else:
                self.results["tests"].append({
                    "name": "monitoring_system_integration",
                    "status": "failed",
                    "error": "Agent registry configuration not found"
                })
                logger.info("  ❌ Agent registry not found")
                return False
                
        except Exception as e:
            self.results["tests"].append({
                "name": "monitoring_system_integration",
                "status": "failed",
                "error": str(e)
            })
            logger.error(f"  ❌ Error: {e}")
            return False
    
    def test_status_consistency(self):
        """Test consistency between different status sources"""
        logger.info("🔍 Testing Status Consistency...")
        
        try:
            # Get Docker status
            docker_result = subprocess.run(
                ['docker', 'ps', '-a', '--filter', 'name=sutazai-',
                 '--format', '{{.Names}}\t{{.Status}}'],
                capture_output=True, text=True, timeout=10
            )
            
            if docker_result.returncode != 0:
                logger.error(f"  ❌ Docker command failed")
                return False
            
            docker_agents = {}
            for line in docker_result.stdout.strip().split('\n'):
                if line.strip():
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        name = parts[0].replace('sutazai-', '')
                        status = self._parse_docker_status(parts[1])
                        docker_agents[name] = status
            
            # Count status types
            status_counts = {}
            for status in docker_agents.values():
                status_counts[status] = status_counts.get(status, 0) + 1
            
            # Validate status distribution
            total_agents = len(docker_agents)
            running_agents = status_counts.get('running', 0) + status_counts.get('healthy', 0)
            problem_agents = status_counts.get('restarting', 0) + status_counts.get('exited', 0)
            
            self.results["tests"].append({
                "name": "status_consistency",
                "status": "passed",
                "total_agents": total_agents,
                "running_agents": running_agents,
                "problem_agents": problem_agents,
                "status_distribution": status_counts
            })
            
            logger.info(f"  ✅ Status consistency validated: {running_agents}/{total_agents} agents running")
            return True
            
        except Exception as e:
            self.results["tests"].append({
                "name": "status_consistency",
                "status": "failed",
                "error": str(e)
            })
            logger.error(f"  ❌ Error: {e}")
            return False
    
    def _parse_docker_status(self, status_string):
        """Parse Docker status string to standard status"""
        if 'Up' in status_string:
            if 'unhealthy' in status_string:
                return 'unhealthy'
            elif 'healthy' in status_string:
                return 'healthy'
            else:
                return 'running'
        elif 'Restarting' in status_string:
            return 'restarting'
        elif 'Exited' in status_string:
            return 'exited'
        else:
            return 'unknown'
    
    def _extract_port(self, ports_string):
        """Extract port number from Docker ports string"""
        if not ports_string:
            return None
        
        # Look for patterns like "0.0.0.0:8080->8080/tcp"
        import re
        match = re.search(r'0\.0\.0\.0:(\d+)->', ports_string)
        if match:
            return int(match.group(1))
        
        return None
    
    def _check_health_endpoint(self, port, timeout=3):
        """Check health endpoint of an agent"""
        try:
            response = requests.get(f"http://localhost:{port}/health", timeout=timeout)
            if response.status_code == 200:
                return "healthy"
            else:
                return "unhealthy"
        except requests.exceptions.ConnectionError:
            return "unreachable"
        except requests.exceptions.Timeout:
            return "timeout"
        except Exception:
            return "error"
    
    def generate_report(self):
        """Generate comprehensive test report"""
        passed_tests = len([t for t in self.results["tests"] if t["status"] == "passed"])
        total_tests = len(self.results["tests"])
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        self.results["summary"] = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "success_rate": success_rate
        }
        
        return self.results


def run_live_validation():
    """Run live validation tests"""
    logger.info("🧪 SutazAI Monitoring System Live Validation")
    logger.info("=" * 50)
    
    validator = LiveMonitoringValidator()
    
    # Run tests
    tests = [
        validator.test_docker_container_detection,
        validator.test_agent_health_endpoints,
        validator.test_monitoring_system_integration,
        validator.test_status_consistency
    ]
    
    for test in tests:
        test()
        time.sleep(1)  # Brief pause between tests
    
    # Generate report
    report = validator.generate_report()
    
    logger.info("\n" + "=" * 50)
    logger.info("📊 LIVE VALIDATION RESULTS")
    logger.info("=" * 50)
    
    summary = report["summary"]
    logger.info(f"Tests Run: {summary['total_tests']}")
    logger.info(f"Passed: {summary['passed_tests']}")
    logger.error(f"Failed: {summary['failed_tests']}")
    logger.info(f"Success Rate: {summary['success_rate']:.1f}%")
    
    # Show detailed results
    for test in report["tests"]:
        status_icon = "✅" if test["status"] == "passed" else "❌"
        logger.info(f"\n{status_icon} {test['name']}")
        
        if test["status"] == "failed":
            logger.error(f"  Error: {test.get('error', 'Unknown error')}")
        else:
            # Show relevant metrics
            if "containers_found" in test:
                logger.info(f"  Containers found: {test['containers_found']}")
            if "healthy_agents" in test:
                logger.info(f"  Healthy agents: {test['healthy_agents']}/{test['total_agents']}")
            if "health_monitoring_enabled" in test:
                logger.info(f"  Health monitoring: {'enabled' if test['health_monitoring_enabled'] else 'disabled'}")
            if "running_agents" in test:
                logger.info(f"  Running agents: {test['running_agents']}/{test['total_agents']}")
    
    # Overall assessment
    logger.info(f"\n🎯 OVERALL ASSESSMENT")
    if summary["success_rate"] >= 90:
        logger.info("🟢 EXCELLENT - Monitoring system is working perfectly!")
    elif summary["success_rate"] >= 75:
        logger.info("🟡 GOOD - Monitoring system is working well with minor issues")
    elif summary["success_rate"] >= 50:
        logger.info("🟠 FAIR - Monitoring system has significant issues that need attention")
    else:
        logger.info("🔴 POOR - Monitoring system requires immediate fixes")
    
    # Save report
    report_file = f"/opt/sutazaiapp/tests/live_validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"\n📄 Detailed report saved to: {report_file}")
    
    return summary["success_rate"] >= 75


if __name__ == "__main__":
    success = run_live_validation()
    sys.exit(0 if success else 1)