#!/usr/bin/env python3
"""
Comprehensive MCP Automation System Validation

This script performs extensive validation of the MCP automation system including:
- MCP server infrastructure validation
- Component health checks
- Performance testing
- Security validation
- Compliance verification
- Integration testing

Author: AI Testing and Validation Specialist (Claude Code)
Created: 2025-08-15
Version: 1.0.0

Rule 8 Compliance: Replaced all logger.info() statements with proper logging
"""

import asyncio
import json
import subprocess
import time
import requests
import os
import psutil
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import tempfile

# Add path for logging configuration
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend', 'app', 'core'))
from logging_config import get_logger
from service_config import get_service_url

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    """Validation test result"""
    test_name: str
    success: bool
    message: str
    details: Dict[str, Any] = None
    duration_seconds: float = 0.0
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}


@dataclass
class ComponentStatus:
    """Component status information"""
    name: str
    type: str
    healthy: bool
    version: Optional[str] = None
    uptime: Optional[float] = None
    memory_mb: Optional[float] = None
    response_time_ms: Optional[float] = None
    error: Optional[str] = None


class MCPValidationSuite:
    """Comprehensive MCP validation test suite"""
    
    def __init__(self):
        self.results: List[ValidationResult] = []
        self.sutazai_root = Path("/opt/sutazaiapp")
        self.mcp_config_path = self.sutazai_root / ".mcp.json"
        self.mcp_wrappers_dir = self.sutazai_root / "scripts" / "mcp" / "wrappers"
        self.automation_dir = self.sutazai_root / "scripts" / "mcp" / "automation"
        
    def log_result(self, test_name: str, success: bool, message: str, 
                   details: Dict[str, Any] = None, duration: float = 0.0):
        """Log a test result"""
        result = ValidationResult(test_name, success, message, details, duration)
        self.results.append(result)
        
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{status} {test_name}: {message}")
        
        if details and not success:
            logger.info(f"   Details: {details}")
    
    async def validate_mcp_infrastructure(self) -> bool:
        """Validate MCP infrastructure components"""
        logger.info("\n🔍 Validating MCP Infrastructure")
        logger.info("-" * 40)
        
        start_time = time.time()
        
        try:
            # Check .mcp.json configuration
            if not self.mcp_config_path.exists():
                self.log_result(
                    "MCP Configuration",
                    False,
                    "MCP configuration file .mcp.json not found",
                    {"path": str(self.mcp_config_path)}
                )
                return False
            
            # Load and validate MCP configuration
            with open(self.mcp_config_path) as f:
                mcp_config = json.load(f)
            
            servers = mcp_config.get("mcpServers", {})
            self.log_result(
                "MCP Configuration",
                True,
                f"Found {len(servers)} MCP servers configured",
                {"servers": list(servers.keys())}
            )
            
            # Check wrapper scripts
            if not self.mcp_wrappers_dir.exists():
                self.log_result(
                    "MCP Wrappers Directory",
                    False,
                    "MCP wrappers directory not found",
                    {"path": str(self.mcp_wrappers_dir)}
                )
                return False
            
            wrapper_scripts = list(self.mcp_wrappers_dir.glob("*.sh"))
            self.log_result(
                "MCP Wrapper Scripts",
                len(wrapper_scripts) > 0,
                f"Found {len(wrapper_scripts)} wrapper scripts",
                {"scripts": [s.name for s in wrapper_scripts]}
            )
            
            # Test individual MCP servers
            healthy_servers = 0
            total_servers = 0
            
            for server_name in list(servers.keys())[:5]:  # Test first 5 servers
                total_servers += 1
                wrapper_path = self.mcp_wrappers_dir / f"{server_name}.sh"
                
                if not wrapper_path.exists():
                    # Try alternative naming
                    possible_wrappers = [
                        self.mcp_wrappers_dir / f"{server_name}.sh",
                        self.mcp_wrappers_dir / f"{server_name}_mcp.sh",
                        self.mcp_wrappers_dir / f"{server_name}-mcp.sh"
                    ]
                    
                    wrapper_path = None
                    for wp in possible_wrappers:
                        if wp.exists():
                            wrapper_path = wp
                            break
                
                if wrapper_path and wrapper_path.exists():
                    try:
                        # Test server health
                        result = subprocess.run(
                            [str(wrapper_path), "--selfcheck"],
                            capture_output=True,
                            text=True,
                            timeout=30
                        )
                        
                        if result.returncode == 0:
                            healthy_servers += 1
                            self.log_result(
                                f"MCP Server: {server_name}",
                                True,
                                "Health check passed",
                                {"output": result.stdout[:200]}
                            )
                        else:
                            self.log_result(
                                f"MCP Server: {server_name}",
                                False,
                                "Health check failed",
                                {"stderr": result.stderr[:200]}
                            )
                    
                    except subprocess.TimeoutExpired:
                        self.log_result(
                            f"MCP Server: {server_name}",
                            False,
                            "Health check timed out",
                            {"timeout": 30}
                        )
                    except Exception as e:
                        self.log_result(
                            f"MCP Server: {server_name}",
                            False,
                            f"Health check error: {e}",
                            {"error": str(e)}
                        )
                else:
                    self.log_result(
                        f"MCP Server: {server_name}",
                        False,
                        "Wrapper script not found",
                        {"expected_path": str(wrapper_path)}
                    )
            
            # Overall MCP infrastructure health
            success_rate = (healthy_servers / total_servers) * 100 if total_servers > 0 else 0
            infrastructure_healthy = success_rate >= 60  # 60% threshold
            
            duration = time.time() - start_time
            self.log_result(
                "MCP Infrastructure Overall",
                infrastructure_healthy,
                f"Infrastructure health: {success_rate:.1f}% ({healthy_servers}/{total_servers})",
                {
                    "healthy_servers": healthy_servers,
                    "total_servers": total_servers,
                    "success_rate": success_rate
                },
                duration
            )
            
            return infrastructure_healthy
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_result(
                "MCP Infrastructure Overall",
                False,
                f"Infrastructure validation failed: {e}",
                {"error": str(e)},
                duration
            )
            return False
    
    async def validate_automation_components(self) -> bool:
        """Validate MCP automation system components"""
        logger.info("\n🔍 Validating Automation Components")
        logger.info("-" * 40)
        
        start_time = time.time()
        
        try:
            # Check automation directory structure
            if not self.automation_dir.exists():
                self.log_result(
                    "Automation Directory",
                    False,
                    "Automation directory not found",
                    {"path": str(self.automation_dir)}
                )
                return False
            
            # Check for automation components
            components = {
                "config.py": "Configuration Management",
                "mcp_update_manager.py": "Update Manager",
                "version_manager.py": "Version Manager", 
                "download_manager.py": "Download Manager",
                "error_handling.py": "Error Handling",
                "cleanup/cleanup_manager.py": "Cleanup Manager",
                "monitoring/health_monitor.py": "Health Monitor",
                "orchestration/orchestrator.py": "Orchestrator",
                "tests/conftest.py": "Test Configuration"
            }
            
            component_results = {}
            for component_file, component_name in components.items():
                component_path = self.automation_dir / component_file
                exists = component_path.exists()
                component_results[component_name] = exists
                
                self.log_result(
                    f"Component: {component_name}",
                    exists,
                    "Found" if exists else "Missing",
                    {"path": str(component_path)}
                )
            
            # Check for CHANGELOG files (Rule 18 compliance)
            changelog_files = list(self.automation_dir.glob("**/CHANGELOG.md"))
            self.log_result(
                "CHANGELOG Files",
                len(changelog_files) > 0,
                f"Found {len(changelog_files)} CHANGELOG files",
                {"files": [str(f) for f in changelog_files]}
            )
            
            # Overall component availability
            available_components = sum(component_results.values())
            total_components = len(components)
            availability_rate = (available_components / total_components) * 100
            
            components_healthy = availability_rate >= 80  # 80% threshold
            
            duration = time.time() - start_time
            self.log_result(
                "Automation Components Overall",
                components_healthy,
                f"Component availability: {availability_rate:.1f}% ({available_components}/{total_components})",
                {
                    "available": available_components,
                    "total": total_components,
                    "availability_rate": availability_rate
                },
                duration
            )
            
            return components_healthy
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_result(
                "Automation Components Overall",
                False,
                f"Component validation failed: {e}",
                {"error": str(e)},
                duration
            )
            return False
    
    async def validate_system_integration(self) -> bool:
        """Validate system integration and health"""
        logger.info("\n🔍 Validating System Integration")
        logger.info("-" * 40)
        
        start_time = time.time()
        
        try:
            # Check backend API health
            backend_healthy = False
            try:
                response = requests.get(f"{get_service_url('backend')}/health", timeout=10)
                backend_healthy = response.status_code == 200
                self.log_result(
                    "Backend API Health",
                    backend_healthy,
                    f"Status: {response.status_code}" if backend_healthy else "Failed to connect",
                    {"status_code": response.status_code if backend_healthy else None}
                )
            except Exception as e:
                self.log_result(
                    "Backend API Health",
                    False,
                    f"Backend API not accessible: {e}",
                    {"error": str(e)}
                )
            
            # Check database connectivity
            postgres_healthy = False
            try:
                postgres_result = subprocess.run(
                    ["docker", "exec", "sutazai-postgres", "pg_isready", "-U", "sutazai"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                postgres_healthy = postgres_result.returncode == 0
                self.log_result(
                    "PostgreSQL Database",
                    postgres_healthy,
                    "Database ready" if postgres_healthy else "Database not ready",
                    {"output": postgres_result.stdout if postgres_healthy else postgres_result.stderr}
                )
            except Exception as e:
                self.log_result(
                    "PostgreSQL Database",
                    False,
                    f"Database check failed: {e}",
                    {"error": str(e)}
                )
            
            # Check Redis connectivity
            redis_healthy = False
            try:
                redis_result = subprocess.run(
                    ["docker", "exec", "sutazai-redis", "redis-cli", "ping"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                redis_healthy = redis_result.returncode == 0 and "PONG" in redis_result.stdout
                self.log_result(
                    "Redis Cache",
                    redis_healthy,
                    "Cache responsive" if redis_healthy else "Cache not responding",
                    {"output": redis_result.stdout if redis_healthy else redis_result.stderr}
                )
            except Exception as e:
                self.log_result(
                    "Redis Cache",
                    False,
                    f"Cache check failed: {e}",
                    {"error": str(e)}
                )
            
            # Check Ollama AI service
            ollama_healthy = False
            try:
                response = requests.get(f"{get_service_url('ollama')}/api/version", timeout=10)
                ollama_healthy = response.status_code == 200
                self.log_result(
                    "Ollama AI Service",
                    ollama_healthy,
                    "AI service responsive" if ollama_healthy else "AI service not responding",
                    {"status_code": response.status_code if ollama_healthy else None}
                )
            except Exception as e:
                self.log_result(
                    "Ollama AI Service",
                    False,
                    f"AI service check failed: {e}",
                    {"error": str(e)}
                )
            
            # Overall integration health
            integration_components = [backend_healthy, postgres_healthy, redis_healthy, ollama_healthy]
            healthy_integrations = sum(integration_components)
            total_integrations = len(integration_components)
            integration_rate = (healthy_integrations / total_integrations) * 100
            
            integration_healthy = integration_rate >= 75  # 75% threshold
            
            duration = time.time() - start_time
            self.log_result(
                "System Integration Overall",
                integration_healthy,
                f"Integration health: {integration_rate:.1f}% ({healthy_integrations}/{total_integrations})",
                {
                    "healthy": healthy_integrations,
                    "total": total_integrations,
                    "integration_rate": integration_rate
                },
                duration
            )
            
            return integration_healthy
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_result(
                "System Integration Overall",
                False,
                f"Integration validation failed: {e}",
                {"error": str(e)},
                duration
            )
            return False
    
    async def validate_performance_characteristics(self) -> bool:
        """Validate performance characteristics"""
        logger.info("\n🔍 Validating Performance Characteristics")
        logger.info("-" * 40)
        
        start_time = time.time()
        
        try:
            # System resource utilization
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Performance thresholds
            cpu_healthy = cpu_percent < 80
            memory_healthy = memory.percent < 85
            disk_healthy = disk.percent < 90
            
            self.log_result(
                "CPU Utilization",
                cpu_healthy,
                f"CPU usage: {cpu_percent:.1f}%",
                {"cpu_percent": cpu_percent, "threshold": 80}
            )
            
            self.log_result(
                "Memory Utilization",
                memory_healthy,
                f"Memory usage: {memory.percent:.1f}%",
                {"memory_percent": memory.percent, "threshold": 85}
            )
            
            self.log_result(
                "Disk Utilization",
                disk_healthy,
                f"Disk usage: {disk.percent:.1f}%",
                {"disk_percent": disk.percent, "threshold": 90}
            )
            
            # API response time test
            api_response_healthy = False
            api_response_time = None
            try:
                api_start = time.time()
                response = requests.get(f"{get_service_url('backend')}/health", timeout=10)
                api_response_time = (time.time() - api_start) * 1000  # ms
                api_response_healthy = response.status_code == 200 and api_response_time < 1000
                
                self.log_result(
                    "API Response Time",
                    api_response_healthy,
                    f"Response time: {api_response_time:.1f}ms",
                    {"response_time_ms": api_response_time, "threshold_ms": 1000}
                )
            except Exception as e:
                self.log_result(
                    "API Response Time",
                    False,
                    f"API performance test failed: {e}",
                    {"error": str(e)}
                )
            
            # Overall performance health
            performance_components = [cpu_healthy, memory_healthy, disk_healthy, api_response_healthy]
            healthy_performance = sum(performance_components)
            total_performance = len(performance_components)
            performance_rate = (healthy_performance / total_performance) * 100
            
            performance_healthy = performance_rate >= 75  # 75% threshold
            
            duration = time.time() - start_time
            self.log_result(
                "Performance Overall",
                performance_healthy,
                f"Performance health: {performance_rate:.1f}% ({healthy_performance}/{total_performance})",
                {
                    "healthy": healthy_performance,
                    "total": total_performance,
                    "performance_rate": performance_rate
                },
                duration
            )
            
            return performance_healthy
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_result(
                "Performance Overall",
                False,
                f"Performance validation failed: {e}",
                {"error": str(e)},
                duration
            )
            return False
    
    async def validate_security_compliance(self) -> bool:
        """Validate security compliance and protection"""
        logger.info("\n🔍 Validating Security Compliance")
        logger.info("-" * 40)
        
        start_time = time.time()
        
        try:
            # Check MCP infrastructure protection (Rule 20)
            mcp_protected = True
            
            # Verify .mcp.json hasn't been modified recently (should be stable)
            if self.mcp_config_path.exists():
                mcp_stat = self.mcp_config_path.stat()
                mcp_age_hours = (time.time() - mcp_stat.st_mtime) / 3600
                mcp_protected = mcp_age_hours > 1  # Stable for at least 1 hour
                
                self.log_result(
                    "MCP Configuration Protection",
                    mcp_protected,
                    f"MCP config stable for {mcp_age_hours:.1f} hours",
                    {"age_hours": mcp_age_hours, "threshold_hours": 1}
                )
            
            # Check wrapper script permissions
            wrapper_security = True
            if self.mcp_wrappers_dir.exists():
                for wrapper in self.mcp_wrappers_dir.glob("*.sh"):
                    stat_info = wrapper.stat()
                    # Check if executable but not world-writable
                    if not (stat_info.st_mode & 0o111) or (stat_info.st_mode & 0o002):
                        wrapper_security = False
                        break
                
                self.log_result(
                    "Wrapper Script Security",
                    wrapper_security,
                    "Wrapper scripts have secure permissions" if wrapper_security else "Insecure wrapper permissions",
                    {"scripts_checked": len(list(self.mcp_wrappers_dir.glob("*.sh")))}
                )
            
            # Check for hardcoded secrets (basic scan)
            secret_scan_clean = True
            if self.automation_dir.exists():
                for py_file in self.automation_dir.rglob("*.py"):
                    try:
                        content = py_file.read_text()
                        # Basic patterns for hardcoded secrets
                        secret_patterns = ["password=", "token=", "api_key=", "secret="]
                        for pattern in secret_patterns:
                            if pattern in content.lower():
                                secret_scan_clean = False
                                break
                    except Exception:
                        continue  # Skip files that can't be read
                
                self.log_result(
                    "Secret Scan",
                    secret_scan_clean,
                    "No hardcoded secrets detected" if secret_scan_clean else "Potential hardcoded secrets found",
                    {"patterns_checked": ["password=", "token=", "api_key=", "secret="]}
                )
            
            # Overall security compliance
            security_components = [mcp_protected, wrapper_security, secret_scan_clean]
            healthy_security = sum(security_components)
            total_security = len(security_components)
            security_rate = (healthy_security / total_security) * 100
            
            security_compliant = security_rate >= 90  # 90% threshold for security
            
            duration = time.time() - start_time
            self.log_result(
                "Security Compliance Overall",
                security_compliant,
                f"Security compliance: {security_rate:.1f}% ({healthy_security}/{total_security})",
                {
                    "compliant": healthy_security,
                    "total": total_security,
                    "compliance_rate": security_rate
                },
                duration
            )
            
            return security_compliant
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_result(
                "Security Compliance Overall",
                False,
                f"Security validation failed: {e}",
                {"error": str(e)},
                duration
            )
            return False
    
    def generate_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.success)
        failed_tests = total_tests - passed_tests
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        # Categorize results
        categories = {
            "MCP Infrastructure": [r for r in self.results if "MCP" in r.test_name or "Server" in r.test_name],
            "Automation Components": [r for r in self.results if "Component" in r.test_name or "CHANGELOG" in r.test_name],
            "System Integration": [r for r in self.results if any(x in r.test_name for x in ["Backend", "Database", "Cache", "AI Service", "Integration"])],
            "Performance": [r for r in self.results if any(x in r.test_name for x in ["CPU", "Memory", "Disk", "Response", "Performance"])],
            "Security": [r for r in self.results if any(x in r.test_name for x in ["Security", "Protection", "Secret", "Wrapper"])]
        }
        
        category_summaries = {}
        for category, category_results in categories.items():
            if category_results:
                category_passed = sum(1 for r in category_results if r.success)
                category_total = len(category_results)
                category_rate = (category_passed / category_total) * 100
                category_summaries[category] = {
                    "passed": category_passed,
                    "total": category_total,
                    "success_rate": category_rate,
                    "status": "PASS" if category_rate >= 75 else "FAIL"
                }
        
        # Overall system health assessment
        overall_health = "HEALTHY" if success_rate >= 85 else "DEGRADED" if success_rate >= 70 else "UNHEALTHY"
        
        # Production readiness assessment
        critical_categories = ["MCP Infrastructure", "System Integration", "Security"]
        critical_passing = all(
            category_summaries.get(cat, {}).get("success_rate", 0) >= 75 
            for cat in critical_categories
        )
        production_ready = critical_passing and success_rate >= 80
        
        report = {
            "validation_summary": {
                "timestamp": datetime.now().isoformat(),
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "success_rate": success_rate,
                "overall_health": overall_health,
                "production_ready": production_ready
            },
            "category_summaries": category_summaries,
            "detailed_results": [asdict(r) for r in self.results],
            "recommendations": self._generate_recommendations(),
            "next_steps": self._generate_next_steps()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        failed_results = [r for r in self.results if not r.success]
        
        if any("MCP" in r.test_name for r in failed_results):
            recommendations.append("Review MCP server health and wrapper script configurations")
        
        if any("Component" in r.test_name for r in failed_results):
            recommendations.append("Complete automation component implementation and testing")
        
        if any("Performance" in r.test_name for r in failed_results):
            recommendations.append("Optimize system performance and resource utilization")
        
        if any("Security" in r.test_name for r in failed_results):
            recommendations.append("Address security compliance issues and hardening requirements")
        
        if any("Integration" in r.test_name for r in failed_results):
            recommendations.append("Improve system integration and service connectivity")
        
        return recommendations
    
    def _generate_next_steps(self) -> List[str]:
        """Generate next steps for improvement"""
        next_steps = []
        
        # Calculate success rates by category
        results_by_category = {}
        for result in self.results:
            if "MCP" in result.test_name:
                category = "mcp_infrastructure"
            elif "Component" in result.test_name:
                category = "automation_components"
            elif any(x in result.test_name for x in ["Backend", "Database", "Integration"]):
                category = "system_integration"
            elif "Performance" in result.test_name:
                category = "performance"
            elif "Security" in result.test_name:
                category = "security"
            else:
                category = "other"
            
            if category not in results_by_category:
                results_by_category[category] = {"passed": 0, "total": 0}
            
            results_by_category[category]["total"] += 1
            if result.success:
                results_by_category[category]["passed"] += 1
        
        # Generate specific next steps
        for category, stats in results_by_category.items():
            success_rate = (stats["passed"] / stats["total"]) * 100 if stats["total"] > 0 else 0
            if success_rate < 80:
                if category == "mcp_infrastructure":
                    next_steps.append("1. Fix failing MCP server health checks and wrapper scripts")
                elif category == "automation_components":
                    next_steps.append("2. Complete missing automation component implementations")
                elif category == "system_integration":
                    next_steps.append("3. Resolve service connectivity and integration issues")
                elif category == "performance":
                    next_steps.append("4. Optimize system performance and resource usage")
                elif category == "security":
                    next_steps.append("5. Address security compliance and protection requirements")
        
        if not next_steps:
            next_steps.append("System validation successful - monitor and maintain current health status")
        
        return next_steps


async def main():
    """Run comprehensive MCP validation suite"""
    logger.info("🔬 MCP AUTOMATION SYSTEM COMPREHENSIVE VALIDATION")
    logger.info("=" * 60)
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)
    
    validator = MCPValidationSuite()
    
    # Run validation phases
    validation_phases = [
        ("MCP Infrastructure", validator.validate_mcp_infrastructure),
        ("Automation Components", validator.validate_automation_components),
        ("System Integration", validator.validate_system_integration),
        ("Performance Characteristics", validator.validate_performance_characteristics),
        ("Security Compliance", validator.validate_security_compliance)
    ]
    
    phase_results = []
    for phase_name, phase_func in validation_phases:
        logger.info(f"\n🚀 Starting Phase: {phase_name}")
        try:
            result = await phase_func()
            phase_results.append((phase_name, result))
        except Exception as e:
            logger.info(f"❌ Phase {phase_name} failed with exception: {e}")
            phase_results.append((phase_name, False))
    
    # Generate and display report
    logger.info("\n" + "=" * 60)
    logger.info("📊 VALIDATION REPORT")
    logger.info("=" * 60)
    
    report = validator.generate_validation_report()
    
    # Display summary
    summary = report["validation_summary"]
    logger.info(f"Total Tests: {summary['total_tests']}")
    logger.info(f"Passed: {summary['passed_tests']}")
    logger.info(f"Failed: {summary['failed_tests']}")
    logger.info(f"Success Rate: {summary['success_rate']:.1f}%")
    logger.info(f"Overall Health: {summary['overall_health']}")
    logger.info(f"Production Ready: {'YES' if summary['production_ready'] else 'NO'}")
    
    # Display category summaries
    logger.info("\n📋 Category Summaries:")
    for category, category_summary in report["category_summaries"].items():
        status_icon = "✅" if category_summary["status"] == "PASS" else "❌"
        logger.info(f"{status_icon} {category}: {category_summary['success_rate']:.1f}% "
              f"({category_summary['passed']}/{category_summary['total']})")
    
    # Display recommendations
    if report["recommendations"]:
        logger.info("\n💡 Recommendations:")
        for i, rec in enumerate(report["recommendations"], 1):
            logger.info(f"{i}. {rec}")
    
    # Display next steps
    if report["next_steps"]:
        logger.info("\n🎯 Next Steps:")
        for step in report["next_steps"]:
            logger.info(f"   {step}")
    
    # Save detailed report
    report_file = Path("/opt/sutazaiapp/mcp_validation_report.json")
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"\n📄 Detailed report saved to: {report_file}")
    
    # Final assessment
    if summary["success_rate"] >= 85:
        logger.info("\n🎉 MCP AUTOMATION SYSTEM VALIDATION: EXCELLENT")
    elif summary["success_rate"] >= 70:
        logger.info("\n⚠️  MCP AUTOMATION SYSTEM VALIDATION: GOOD (NEEDS IMPROVEMENT)")
    else:
        logger.info("\n❌ MCP AUTOMATION SYSTEM VALIDATION: REQUIRES ATTENTION")
    
    return summary["production_ready"]


if __name__ == "__main__":
    try:
        production_ready = asyncio.run(main())
        exit(0 if production_ready else 1)
    except KeyboardInterrupt:
        logger.info("\n\n⏹️  Validation interrupted by user")
        exit(130)
    except Exception as e:
        logger.info(f"\n\n💥 Validation failed with error: {e}")
        exit(1)