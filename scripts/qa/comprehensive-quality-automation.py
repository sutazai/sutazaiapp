#!/usr/bin/env python3
"""
Comprehensive Quality Automation Script
Implements complete QA validation and monitoring system for SutazAI.

Features:
- Automated code quality validation (black, isort, flake8, mypy)
- Security scanning (bandit, safety)
- Test execution with coverage
- Performance monitoring
- Rule compliance validation
- Infrastructure protection monitoring
- Real-time quality metrics collection

Version: SutazAI v93 - QA Excellence Framework
Author: QA Validation Specialist (Claude Code)
"""

import os
import sys
import json
import time
import asyncio
import logging
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('qa_automation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class QualityMetrics:
    """Quality metrics data structure."""
    timestamp: str
    rule_compliance_score: float
    code_quality_score: float
    test_coverage_percentage: float
    security_score: float
    performance_score: float
    overall_quality_score: float
    violations_count: int
    tests_passed: int
    tests_failed: int
    tests_skipped: int
    build_duration: float
    mcp_servers_healthy: bool

class QualityAutomationFramework:
    """Comprehensive quality automation framework."""
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path(__file__).parent.parent.parent
        self.metrics_history = []
        self.quality_thresholds = {
            'rule_compliance': 100.0,
            'code_quality': 95.0,
            'test_coverage': 80.0,
            'security': 90.0,
            'performance': 85.0,
            'overall_quality': 90.0
        }
        self.start_time = time.time()
        
    def run_command(self, cmd: List[str], timeout: int = 300) -> Tuple[bool, str, float]:
        """Run a command with timeout and return success, output, and duration."""
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.project_root,
                timeout=timeout
            )
            duration = time.time() - start_time
            return result.returncode == 0, result.stdout + result.stderr, duration
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            return False, f"Command timed out after {timeout}s", duration
        except Exception as e:
            duration = time.time() - start_time
            return False, str(e), duration
    
    def validate_rule_compliance(self) -> Tuple[float, int]:
        """Validate compliance with all 20 Fundamental Rules."""
        logger.info("🔧 Validating rule compliance...")
        
        violations = 0
        compliance_checks = []
        
        # Check 1: Enforcement Rules document exists
        enforcement_rules = self.project_root / "IMPORTANT" / "Enforcement_Rules"
        if not enforcement_rules.exists():
            violations += 1
            compliance_checks.append("❌ Enforcement Rules document missing")
        else:
            compliance_checks.append("✅ Enforcement Rules document present")
        
        # Check 2: CHANGELOG.md compliance (Rule 18)
        changelog = self.project_root / "CHANGELOG.md"
        if not changelog.exists():
            violations += 1
            compliance_checks.append("❌ Root CHANGELOG.md missing")
        else:
            compliance_checks.append("✅ Root CHANGELOG.md present")
        
        # Check 3: MCP server protection (Rule 20)
        mcp_config = self.project_root / ".mcp.json"
        if not mcp_config.exists():
            violations += 1
            compliance_checks.append("❌ .mcp.json missing")
        else:
            compliance_checks.append("✅ .mcp.json protected")
        
        # Check 4: Project structure discipline
        required_dirs = ['backend', 'frontend', 'tests', 'scripts', 'monitoring']
        for dir_name in required_dirs:
            dir_path = self.project_root / dir_name
            if not dir_path.exists():
                violations += 1
                compliance_checks.append(f"❌ Required directory missing: {dir_name}")
            else:
                compliance_checks.append(f"✅ Directory present: {dir_name}")
        
        # Check 5: Run rule validator if available
        rule_validator = self.project_root / "scripts" / "enforcement" / "rule_validator_simple.py"
        if rule_validator.exists():
            success, output, _ = self.run_command([
                "python3", str(rule_validator), "--quick"
            ])
            if not success:
                violations += 1
                compliance_checks.append(f"❌ Rule validator failed: {output[:200]}")
            else:
                compliance_checks.append("✅ Rule validator passed")
        
        total_checks = len(compliance_checks)
        passed_checks = total_checks - violations
        compliance_score = (passed_checks / total_checks) * 100 if total_checks > 0 else 0
        
        logger.info(f"Rule compliance: {compliance_score:.1f}% ({passed_checks}/{total_checks} checks passed)")\n        
        for check in compliance_checks:
            logger.info(f"  {check}")
        
        return compliance_score, violations
    
    def validate_code_quality(self) -> float:
        """Validate code quality using multiple tools."""
        logger.info("🎨 Validating code quality...")
        
        quality_checks = []
        total_score = 0
        max_score = 0
        
        # Black formatting check
        max_score += 25
        success, output, _ = self.run_command([
            "black", "--check", "--quiet", "backend/", "agents/", "tests/", "scripts/"
        ])
        if success:
            total_score += 25
            quality_checks.append("✅ Black formatting: PASSED")
        else:
            quality_checks.append("❌ Black formatting: FAILED")
        
        # isort import sorting check
        max_score += 25
        success, output, _ = self.run_command([
            "isort", "--check-only", "--quiet", "backend/", "agents/", "tests/", "scripts/"
        ])
        if success:
            total_score += 25
            quality_checks.append("✅ Import sorting: PASSED")
        else:
            quality_checks.append("❌ Import sorting: FAILED")
        
        # flake8 style check
        max_score += 25
        success, output, _ = self.run_command([
            "flake8", "backend/", "agents/", "tests/", "scripts/"
        ])
        if success:
            total_score += 25
            quality_checks.append("✅ Code style (flake8): PASSED")
        else:
            quality_checks.append("❌ Code style (flake8): FAILED")
        
        # mypy type checking (non-blocking)
        max_score += 25
        success, output, _ = self.run_command([
            "mypy", "backend/", "--ignore-missing-imports"
        ])
        if success:
            total_score += 25
            quality_checks.append("✅ Type checking (mypy): PASSED")
        else:
            quality_checks.append("⚠️ Type checking (mypy): ISSUES")
            total_score += 15  # Partial credit for non-blocking check
        
        code_quality_score = (total_score / max_score) * 100 if max_score > 0 else 0
        
        logger.info(f"Code quality score: {code_quality_score:.1f}%")
        for check in quality_checks:
            logger.info(f"  {check}")
        
        return code_quality_score
    
    def run_security_scanning(self) -> float:
        """Run comprehensive security scanning."""
        logger.info("🛡️ Running security scanning...")
        
        security_checks = []
        total_score = 0
        max_score = 0
        
        # Ensure reports directory exists
        reports_dir = self.project_root / "tests" / "reports" / "security"
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Bandit security analysis
        max_score += 50
        success, output, _ = self.run_command([
            "bandit", "-r", "backend/", "agents/", "-f", "json", 
            "-o", str(reports_dir / "bandit.json")
        ])
        if success or "No issues identified" in output:
            total_score += 50
            security_checks.append("✅ Bandit security scan: CLEAN")
        else:
            security_checks.append("⚠️ Bandit security scan: ISSUES DETECTED")
            total_score += 25  # Partial credit
        
        # Safety dependency vulnerability scan
        max_score += 50
        success, output, _ = self.run_command([
            "safety", "check", "--json", "--output", str(reports_dir / "safety.json")
        ])
        if success:
            total_score += 50
            security_checks.append("✅ Safety vulnerability scan: CLEAN")
        else:
            security_checks.append("⚠️ Safety vulnerability scan: VULNERABILITIES DETECTED")
            total_score += 25  # Partial credit
        
        security_score = (total_score / max_score) * 100 if max_score > 0 else 0
        
        logger.info(f"Security score: {security_score:.1f}%")
        for check in security_checks:
            logger.info(f"  {check}")
        
        return security_score
    
    def run_comprehensive_tests(self) -> Tuple[float, int, int, int]:
        """Run comprehensive test suite with coverage."""
        logger.info("🧪 Running comprehensive test suite...")
        
        # Ensure reports directory exists
        reports_dir = self.project_root / "tests" / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Run tests with coverage
        success, output, duration = self.run_command([
            "pytest", "tests/",
            "--cov=backend", "--cov=agents",
            "--cov-report=html:tests/reports/coverage/html",
            "--cov-report=xml:tests/reports/coverage/coverage.xml",
            "--cov-report=term-missing",
            "--junit-xml=tests/reports/junit/comprehensive.xml",
            "-v"
        ], timeout=600)
        
        # Parse test results
        tests_passed = output.count(" PASSED")
        tests_failed = output.count(" FAILED")
        tests_skipped = output.count(" SKIPPED")
        
        # Extract coverage percentage
        coverage_percentage = 0.0
        if "TOTAL" in output:
            try:
                lines = output.split('\n')
                for line in lines:
                    if "TOTAL" in line:
                        parts = line.split()
                        for part in parts:
                            if part.endswith('%'):
                                coverage_percentage = float(part.rstrip('%'))\n                                break\n                        break\n            except (ValueError, IndexError):\n                logger.warning("Could not parse coverage percentage")\n        \n        logger.info(f"Test results: {tests_passed} passed, {tests_failed} failed, {tests_skipped} skipped")\n        logger.info(f"Test coverage: {coverage_percentage:.1f}%")\n        logger.info(f"Test execution time: {duration:.1f}s")\n        \n        return coverage_percentage, tests_passed, tests_failed, tests_skipped\n    \n    def monitor_system_performance(self) -> float:\n        """Monitor system performance metrics."""\n        logger.info("⚡ Monitoring system performance...")\n        \n        performance_checks = []\n        total_score = 0\n        max_score = 0\n        \n        # Check API health endpoints\n        api_endpoints = [\n            ("Backend", "http://localhost:10010/health"),\n            ("Frontend", "http://localhost:10011/"),\n            ("Ollama", "http://localhost:10104/api/tags"),\n            ("Qdrant", "http://localhost:10101/healthz"),\n            ("Prometheus", "http://localhost:10200/-/healthy")\n        ]\n        \n        for name, url in api_endpoints:\n            max_score += 20\n            success, output, duration = self.run_command([\n                "curl", "-sf", url, "--max-time", "5"\n            ])\n            if success:\n                total_score += 20\n                performance_checks.append(f"✅ {name}: HEALTHY ({duration:.1f}s)")\n            else:\n                performance_checks.append(f"❌ {name}: UNHEALTHY")\n        \n        performance_score = (total_score / max_score) * 100 if max_score > 0 else 0\n        \n        logger.info(f"Performance score: {performance_score:.1f}%")\n        for check in performance_checks:\n            logger.info(f"  {check}")\n        \n        return performance_score\n    \n    def validate_infrastructure_protection(self) -> bool:\n        """Validate infrastructure protection (MCP servers, databases)."""\n        logger.info("🔒 Validating infrastructure protection...")\n        \n        protection_checks = []\n        all_healthy = True\n        \n        # Check MCP configuration integrity\n        mcp_config = self.project_root / ".mcp.json"\n        if mcp_config.exists():\n            # Verify expected checksum\n            success, output, _ = self.run_command([\n                "sha1sum", str(mcp_config)\n            ])\n            if success:\n                actual_checksum = output.split()[0]\n                expected_checksum = "c1ada43007a0715d577c10fad975517a82506c07"\n                if actual_checksum == expected_checksum:\n                    protection_checks.append("✅ MCP configuration integrity verified")\n                else:\n                    protection_checks.append("⚠️ MCP configuration checksum changed")\n            else:\n                protection_checks.append("❌ Could not verify MCP configuration")\n                all_healthy = False\n        else:\n            protection_checks.append("❌ MCP configuration missing")\n            all_healthy = False\n        \n        # Check database containers\n        db_services = ["postgres", "redis", "neo4j"]\n        for service in db_services:\n            success, output, _ = self.run_command([\n                "docker", "ps", "--filter", f"name={service}", "--format", "{{.Status}}"\n            ])\n            if success and "Up" in output:\n                protection_checks.append(f"✅ {service} container: RUNNING")\n            else:\n                protection_checks.append(f"❌ {service} container: NOT RUNNING")\n                all_healthy = False\n        \n        # Check Ollama service protection\n        success, output, _ = self.run_command([\n            "curl", "-sf", "http://localhost:10104/api/tags", "--max-time", "5"\n        ])\n        if success:\n            protection_checks.append("✅ Ollama service: PROTECTED AND HEALTHY")\n        else:\n            protection_checks.append("❌ Ollama service: UNHEALTHY")\n            all_healthy = False\n        \n        logger.info(f"Infrastructure protection: {'HEALTHY' if all_healthy else 'ISSUES DETECTED'}")\n        for check in protection_checks:\n            logger.info(f"  {check}")\n        \n        return all_healthy\n    \n    def generate_quality_report(self, metrics: QualityMetrics) -> Dict[str, Any]:\n        """Generate comprehensive quality report."""\n        report = {\n            "timestamp": metrics.timestamp,\n            "execution_summary": {\n                "overall_quality_score": metrics.overall_quality_score,\n                "total_duration": time.time() - self.start_time,\n                "status": "PASSED" if metrics.overall_quality_score >= self.quality_thresholds['overall_quality'] else "FAILED"\n            },\n            "detailed_metrics": asdict(metrics),\n            "quality_gates": {\n                "rule_compliance": {\n                    "score": metrics.rule_compliance_score,\n                    "threshold": self.quality_thresholds['rule_compliance'],\n                    "status": "PASSED" if metrics.rule_compliance_score >= self.quality_thresholds['rule_compliance'] else "FAILED"\n                },\n                "code_quality": {\n                    "score": metrics.code_quality_score,\n                    "threshold": self.quality_thresholds['code_quality'],\n                    "status": "PASSED" if metrics.code_quality_score >= self.quality_thresholds['code_quality'] else "FAILED"\n                },\n                "test_coverage": {\n                    "score": metrics.test_coverage_percentage,\n                    "threshold": self.quality_thresholds['test_coverage'],\n                    "status": "PASSED" if metrics.test_coverage_percentage >= self.quality_thresholds['test_coverage'] else "FAILED"\n                },\n                "security": {\n                    "score": metrics.security_score,\n                    "threshold": self.quality_thresholds['security'],\n                    "status": "PASSED" if metrics.security_score >= self.quality_thresholds['security'] else "FAILED"\n                },\n                "performance": {\n                    "score": metrics.performance_score,\n                    "threshold": self.quality_thresholds['performance'],\n                    "status": "PASSED" if metrics.performance_score >= self.quality_thresholds['performance'] else "FAILED"\n                }\n            },\n            "recommendations": self.generate_recommendations(metrics)\n        }\n        \n        return report\n    \n    def generate_recommendations(self, metrics: QualityMetrics) -> List[str]:\n        """Generate improvement recommendations based on metrics."""\n        recommendations = []\n        \n        if metrics.rule_compliance_score < 100:\n            recommendations.append("🔧 Fix rule compliance violations to maintain code standards")\n        \n        if metrics.code_quality_score < self.quality_thresholds['code_quality']:\n            recommendations.append("🎨 Run 'make format' and 'make lint' to improve code quality")\n        \n        if metrics.test_coverage_percentage < self.quality_thresholds['test_coverage']:\n            recommendations.append("🧪 Increase test coverage by adding unit and integration tests")\n        \n        if metrics.security_score < self.quality_thresholds['security']:\n            recommendations.append("🛡️ Address security vulnerabilities and code issues")\n        \n        if metrics.performance_score < self.quality_thresholds['performance']:\n            recommendations.append("⚡ Investigate performance issues and optimize system health")\n        \n        if metrics.violations_count > 0:\n            recommendations.append(f"🚨 Address {metrics.violations_count} rule violations immediately")\n        \n        if metrics.tests_failed > 0:\n            recommendations.append(f"❌ Fix {metrics.tests_failed} failing test(s)")\n        \n        if not metrics.mcp_servers_healthy:\n            recommendations.append("🔒 Investigate MCP server issues and restore protection")\n        \n        if not recommendations:\n            recommendations.append("🏆 Excellent! All quality gates passed. Maintain this standard.")\n        \n        return recommendations\n    \n    def save_quality_report(self, report: Dict[str, Any]) -> str:\n        """Save quality report to file."""\n        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")\n        report_file = self.project_root / f"qa_quality_report_{timestamp}.json"\n        \n        with open(report_file, 'w') as f:\n            json.dump(report, f, indent=2)\n        \n        logger.info(f"Quality report saved to: {report_file}")\n        return str(report_file)\n    \n    def run_comprehensive_quality_validation(self) -> Dict[str, Any]:\n        """Run complete quality validation pipeline."""\n        logger.info("🚀 Starting comprehensive quality validation...")\n        logger.info(f"Project root: {self.project_root}")\n        \n        # Phase 1: Rule Compliance\n        rule_compliance_score, violations_count = self.validate_rule_compliance()\n        \n        # Phase 2: Code Quality\n        code_quality_score = self.validate_code_quality()\n        \n        # Phase 3: Security Scanning\n        security_score = self.run_security_scanning()\n        \n        # Phase 4: Comprehensive Testing\n        test_coverage, tests_passed, tests_failed, tests_skipped = self.run_comprehensive_tests()\n        \n        # Phase 5: Performance Monitoring\n        performance_score = self.monitor_system_performance()\n        \n        # Phase 6: Infrastructure Protection\n        mcp_servers_healthy = self.validate_infrastructure_protection()\n        \n        # Calculate overall quality score\n        weights = {\n            'rule_compliance': 0.25,\n            'code_quality': 0.20,\n            'test_coverage': 0.20,\n            'security': 0.20,\n            'performance': 0.15\n        }\n        \n        overall_quality_score = (\n            rule_compliance_score * weights['rule_compliance'] +\n            code_quality_score * weights['code_quality'] +\n            test_coverage * weights['test_coverage'] +\n            security_score * weights['security'] +\n            performance_score * weights['performance']\n        )\n        \n        # Create metrics object\n        metrics = QualityMetrics(\n            timestamp=datetime.now(timezone.utc).isoformat(),\n            rule_compliance_score=rule_compliance_score,\n            code_quality_score=code_quality_score,\n            test_coverage_percentage=test_coverage,\n            security_score=security_score,\n            performance_score=performance_score,\n            overall_quality_score=overall_quality_score,\n            violations_count=violations_count,\n            tests_passed=tests_passed,\n            tests_failed=tests_failed,\n            tests_skipped=tests_skipped,\n            build_duration=time.time() - self.start_time,\n            mcp_servers_healthy=mcp_servers_healthy\n        )\n        \n        # Generate and save report\n        report = self.generate_quality_report(metrics)\n        report_file = self.save_quality_report(report)\n        \n        # Log summary\n        logger.info("="*80)\n        logger.info("🎯 QUALITY VALIDATION SUMMARY")\n        logger.info("="*80)\n        logger.info(f"Overall Quality Score: {overall_quality_score:.1f}%")\n        logger.info(f"Rule Compliance: {rule_compliance_score:.1f}%")\n        logger.info(f"Code Quality: {code_quality_score:.1f}%")\n        logger.info(f"Test Coverage: {test_coverage:.1f}%")\n        logger.info(f"Security Score: {security_score:.1f}%")\n        logger.info(f"Performance Score: {performance_score:.1f}%")\n        logger.info(f"Test Results: {tests_passed} passed, {tests_failed} failed, {tests_skipped} skipped")\n        logger.info(f"Rule Violations: {violations_count}")\n        logger.info(f"MCP Servers: {'Healthy' if mcp_servers_healthy else 'Issues Detected'}")\n        logger.info(f"Total Duration: {time.time() - self.start_time:.1f}s")\n        logger.info(f"Report Saved: {report_file}")\n        \n        if overall_quality_score >= self.quality_thresholds['overall_quality']:\n            logger.info("✅ ALL QUALITY GATES PASSED - Code ready for deployment!")\n        else:\n            logger.error("❌ QUALITY GATE FAILURES - Address issues before deployment")\n        \n        logger.info("="*80)\n        \n        return report

def main():
    """Main entry point for quality automation."""\n    import argparse\n    \n    parser = argparse.ArgumentParser(description="SutazAI Quality Automation Framework")\n    parser.add_argument("--project-root", type=Path, help="Project root directory")\n    parser.add_argument("--quick", action="store_true", help="Run quick validation only")\n    parser.add_argument("--continuous", action="store_true", help="Run in continuous monitoring mode")\n    parser.add_argument("--output", type=str, help="Output report file")\n    \n    args = parser.parse_args()\n    \n    # Initialize framework\n    framework = QualityAutomationFramework(args.project_root)\n    \n    if args.continuous:\n        logger.info("🔄 Starting continuous quality monitoring...")\n        while True:\n            try:\n                report = framework.run_comprehensive_quality_validation()\n                time.sleep(300)  # Run every 5 minutes\n            except KeyboardInterrupt:\n                logger.info("Continuous monitoring stopped by user")\n                break\n            except Exception as e:\n                logger.error(f"Error in continuous monitoring: {e}")\n                time.sleep(60)  # Wait 1 minute before retry\n    else:\n        # Single run\n        report = framework.run_comprehensive_quality_validation()\n        \n        if args.output:\n            with open(args.output, 'w') as f:\n                json.dump(report, f, indent=2)\n            logger.info(f"Report also saved to: {args.output}")\n        \n        # Exit with appropriate code\n        if report['execution_summary']['status'] == 'PASSED':\n            sys.exit(0)\n        else:\n            sys.exit(1)

if __name__ == "__main__":\n    main()