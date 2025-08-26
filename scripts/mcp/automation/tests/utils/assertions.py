#!/usr/bin/env python3
"""
Test Assertion Utilities

Comprehensive assertion utilities for MCP automation testing.
Provides specialized assertions for MCP operations, performance validation,
security checks, and system state verification.

Author: Claude AI Assistant (senior-automated-tester)
Created: 2025-08-15 UTC
Version: 1.0.0
"""

import time
import statistics
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass


@dataclass
class AssertionResult:
    """Assertion result with detailed information."""
    passed: bool
    message: str
    expected: Any = None
    actual: Any = None
    context: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.context is None:
            self.context = {}


class MCPAssertions:
    """Specialized assertions for MCP server operations."""
    
    @staticmethod
    def assert_server_healthy(
        health_result: Dict[str, Any],
        server_name: str,
        timeout_tolerance: float = 5.0
    ) -> AssertionResult:
        """Assert that server health check indicates healthy status."""
        if not isinstance(health_result, dict):
            return AssertionResult(
                passed=False,
                message=f"Health result for {server_name} is not a dictionary",
                expected="dict",
                actual=type(health_result).__name__
            )
        
        required_fields = ["server_name", "healthy", "response_time"]
        missing_fields = [field for field in required_fields if field not in health_result]
        
        if missing_fields:
            return AssertionResult(
                passed=False,
                message=f"Health result for {server_name} missing required fields: {missing_fields}",
                expected=required_fields,
                actual=list(health_result.keys())
            )
        
        if health_result["server_name"] != server_name:
            return AssertionResult(
                passed=False,
                message=f"Health result server name mismatch",
                expected=server_name,
                actual=health_result["server_name"]
            )
        
        if not health_result["healthy"]:
            error_msg = health_result.get("error", "Unknown health check failure")
            return AssertionResult(
                passed=False,
                message=f"Server {server_name} is not healthy: {error_msg}",
                expected=True,
                actual=False,
                context={"error": error_msg}
            )
        
        response_time = health_result["response_time"]
        if response_time > timeout_tolerance:
            return AssertionResult(
                passed=False,
                message=f"Server {server_name} response time {response_time}s exceeds tolerance {timeout_tolerance}s",
                expected=f"<= {timeout_tolerance}",
                actual=response_time
            )
        
        return AssertionResult(
            passed=True,
            message=f"Server {server_name} is healthy with response time {response_time}s",
            context={"response_time": response_time}
        )
    
    @staticmethod
    def assert_version_correct(
        version_status: Dict[str, Any],
        server_name: str,
        expected_version: str
    ) -> AssertionResult:
        """Assert that server is running the expected version."""
        if "current_version" not in version_status:
            return AssertionResult(
                passed=False,
                message=f"Version status for {server_name} missing current_version field",
                expected="current_version field",
                actual=list(version_status.keys())
            )
        
        actual_version = version_status["current_version"]
        if actual_version != expected_version:
            return AssertionResult(
                passed=False,
                message=f"Server {server_name} version mismatch",
                expected=expected_version,
                actual=actual_version
            )
        
        return AssertionResult(
            passed=True,
            message=f"Server {server_name} running correct version {expected_version}",
            context={"version": actual_version}
        )
    
    @staticmethod
    def assert_download_successful(
        download_result: Dict[str, Any],
        package_name: str,
        expected_version: str
    ) -> AssertionResult:
        """Assert that package download completed successfully."""
        required_fields = ["package", "version", "checksum_verified", "install_path"]
        missing_fields = [field for field in required_fields if field not in download_result]
        
        if missing_fields:
            return AssertionResult(
                passed=False,
                message=f"Download result for {package_name} missing fields: {missing_fields}",
                expected=required_fields,
                actual=list(download_result.keys())
            )
        
        if download_result["package"] != package_name:
            return AssertionResult(
                passed=False,
                message=f"Downloaded package name mismatch",
                expected=package_name,
                actual=download_result["package"]
            )
        
        if download_result["version"] != expected_version:
            return AssertionResult(
                passed=False,
                message=f"Downloaded package version mismatch",
                expected=expected_version,
                actual=download_result["version"]
            )
        
        if not download_result["checksum_verified"]:
            return AssertionResult(
                passed=False,
                message=f"Checksum verification failed for {package_name}",
                expected=True,
                actual=False
            )
        
        install_path = Path(download_result["install_path"])
        if not install_path.exists():
            return AssertionResult(
                passed=False,
                message=f"Install path does not exist: {install_path}",
                expected="existing path",
                actual=str(install_path)
            )
        
        return AssertionResult(
            passed=True,
            message=f"Package {package_name} v{expected_version} downloaded successfully",
            context={
                "install_path": str(install_path),
                "size_bytes": download_result.get("size_bytes", 0)
            }
        )
    
    @staticmethod
    def assert_rollback_successful(
        rollback_result: Dict[str, Any],
        server_name: str,
        expected_rollback_version: str
    ) -> AssertionResult:
        """Assert that rollback completed successfully."""
        if not rollback_result.get("success", False):
            error_msg = rollback_result.get("error", "Unknown rollback failure")
            return AssertionResult(
                passed=False,
                message=f"Rollback failed for {server_name}: {error_msg}",
                expected=True,
                actual=False,
                context={"error": error_msg}
            )
        
        rolled_back_to = rollback_result.get("rolled_back_to")
        if rolled_back_to != expected_rollback_version:
            return AssertionResult(
                passed=False,
                message=f"Rollback version mismatch for {server_name}",
                expected=expected_rollback_version,
                actual=rolled_back_to
            )
        
        return AssertionResult(
            passed=True,
            message=f"Rollback successful for {server_name} to version {expected_rollback_version}",
            context={"rolled_back_to": rolled_back_to}
        )
    
    @staticmethod
    def assert_system_status_healthy(
        system_status: Dict[str, Any],
        min_healthy_ratio: float = 0.8
    ) -> AssertionResult:
        """Assert that overall system status is healthy."""
        required_fields = ["total_servers", "healthy_servers", "unhealthy_servers"]
        missing_fields = [field for field in required_fields if field not in system_status]
        
        if missing_fields:
            return AssertionResult(
                passed=False,
                message=f"System status missing fields: {missing_fields}",
                expected=required_fields,
                actual=list(system_status.keys())
            )
        
        total_servers = system_status["total_servers"]
        healthy_servers = system_status["healthy_servers"]
        
        if total_servers == 0:
            return AssertionResult(
                passed=False,
                message="No servers found in system status",
                expected="> 0",
                actual=0
            )
        
        healthy_ratio = healthy_servers / total_servers
        if healthy_ratio < min_healthy_ratio:
            return AssertionResult(
                passed=False,
                message=f"System health ratio {healthy_ratio:.2f} below minimum {min_healthy_ratio}",
                expected=f">= {min_healthy_ratio}",
                actual=healthy_ratio,
                context={
                    "total_servers": total_servers,
                    "healthy_servers": healthy_servers,
                    "unhealthy_servers": system_status["unhealthy_servers"]
                }
            )
        
        return AssertionResult(
            passed=True,
            message=f"System healthy: {healthy_servers}/{total_servers} servers healthy ({healthy_ratio:.2f})",
            context={
                "healthy_ratio": healthy_ratio,
                "total_servers": total_servers,
                "healthy_servers": healthy_servers
            }
        )


class PerformanceAssertions:
    """Specialized assertions for performance validation."""
    
    @staticmethod
    def assert_response_time_within_threshold(
        response_time: float,
        threshold: float,
        operation: str = "operation"
    ) -> AssertionResult:
        """Assert that response time is within acceptable threshold."""
        if response_time <= threshold:
            return AssertionResult(
                passed=True,
                message=f"{operation} completed in {response_time:.2f}s (within {threshold}s threshold)",
                context={"response_time": response_time, "threshold": threshold}
            )
        else:
            return AssertionResult(
                passed=False,
                message=f"{operation} took {response_time:.2f}s, exceeds {threshold}s threshold",
                expected=f"<= {threshold}",
                actual=response_time
            )
    
    @staticmethod
    def assert_memory_usage_acceptable(
        memory_usage_mb: float,
        max_memory_mb: float,
        server_name: str = "server"
    ) -> AssertionResult:
        """Assert that memory usage is within acceptable limits."""
        if memory_usage_mb <= max_memory_mb:
            return AssertionResult(
                passed=True,
                message=f"{server_name} memory usage {memory_usage_mb:.1f}MB within {max_memory_mb}MB limit",
                context={"memory_usage_mb": memory_usage_mb, "max_memory_mb": max_memory_mb}
            )
        else:
            return AssertionResult(
                passed=False,
                message=f"{server_name} memory usage {memory_usage_mb:.1f}MB exceeds {max_memory_mb}MB limit",
                expected=f"<= {max_memory_mb}",
                actual=memory_usage_mb
            )
    
    @staticmethod
    def assert_throughput_meets_target(
        actual_throughput: float,
        target_throughput: float,
        tolerance: float = 0.1,
        operation: str = "operation"
    ) -> AssertionResult:
        """Assert that throughput meets target within tolerance."""
        min_acceptable = target_throughput * (1 - tolerance)
        
        if actual_throughput >= min_acceptable:
            return AssertionResult(
                passed=True,
                message=f"{operation} throughput {actual_throughput:.2f} meets target {target_throughput:.2f} (±{tolerance*100}%)",
                context={
                    "actual_throughput": actual_throughput,
                    "target_throughput": target_throughput,
                    "tolerance": tolerance
                }
            )
        else:
            return AssertionResult(
                passed=False,
                message=f"{operation} throughput {actual_throughput:.2f} below target {target_throughput:.2f} (±{tolerance*100}%)",
                expected=f">= {min_acceptable:.2f}",
                actual=actual_throughput
            )
    
    @staticmethod
    def assert_performance_regression(
        current_metrics: List[float],
        baseline_metrics: List[float],
        max_regression: float = 0.2,
        metric_name: str = "metric"
    ) -> AssertionResult:
        """Assert that performance has not regressed beyond acceptable threshold."""
        if not current_metrics or not baseline_metrics:
            return AssertionResult(
                passed=False,
                message=f"Insufficient data for {metric_name} regression analysis",
                expected="non-empty metric lists",
                actual=f"current: {len(current_metrics)}, baseline: {len(baseline_metrics)}"
            )
        
        current_avg = statistics.mean(current_metrics)
        baseline_avg = statistics.mean(baseline_metrics)
        
        # Calculate regression (higher values indicate worse performance)
        regression_ratio = (current_avg - baseline_avg) / baseline_avg if baseline_avg > 0 else 0
        
        if regression_ratio <= max_regression:
            improvement = -regression_ratio * 100 if regression_ratio < 0 else 0
            message = f"{metric_name} performance acceptable"
            if improvement > 0:
                message += f" (improved by {improvement:.1f}%)"
            
            return AssertionResult(
                passed=True,
                message=message,
                context={
                    "current_avg": current_avg,
                    "baseline_avg": baseline_avg,
                    "regression_ratio": regression_ratio
                }
            )
        else:
            return AssertionResult(
                passed=False,
                message=f"{metric_name} regressed by {regression_ratio*100:.1f}%, exceeds {max_regression*100}% threshold",
                expected=f"<= {max_regression*100}%",
                actual=f"{regression_ratio*100:.1f}%",
                context={
                    "current_avg": current_avg,
                    "baseline_avg": baseline_avg,
                    "regression_ratio": regression_ratio
                }
            )
    
    @staticmethod
    def assert_concurrent_performance_scaling(
        single_thread_time: float,
        multi_thread_time: float,
        thread_count: int,
        min_efficiency: float = 0.5
    ) -> AssertionResult:
        """Assert that concurrent operations scale efficiently."""
        theoretical_speedup = thread_count
        actual_speedup = single_thread_time / multi_thread_time if multi_thread_time > 0 else 0
        efficiency = actual_speedup / theoretical_speedup if theoretical_speedup > 0 else 0
        
        if efficiency >= min_efficiency:
            return AssertionResult(
                passed=True,
                message=f"Concurrent scaling efficiency {efficiency:.2f} meets {min_efficiency} threshold",
                context={
                    "single_thread_time": single_thread_time,
                    "multi_thread_time": multi_thread_time,
                    "thread_count": thread_count,
                    "actual_speedup": actual_speedup,
                    "efficiency": efficiency
                }
            )
        else:
            return AssertionResult(
                passed=False,
                message=f"Concurrent scaling efficiency {efficiency:.2f} below {min_efficiency} threshold",
                expected=f">= {min_efficiency}",
                actual=efficiency,
                context={
                    "single_thread_time": single_thread_time,
                    "multi_thread_time": multi_thread_time,
                    "thread_count": thread_count,
                    "actual_speedup": actual_speedup
                }
            )


class SecurityAssertions:
    """Specialized assertions for security validation."""
    
    @staticmethod
    def assert_no_vulnerabilities(
        scan_result: Dict[str, Any],
        package_name: str,
        max_severity: str = "medium"
    ) -> AssertionResult:
        """Assert that security scan found no vulnerabilities above specified severity."""
        severity_levels = {"low": 1, "medium": 2, "high": 3, "critical": 4}
        max_severity_level = severity_levels.get(max_severity, 2)
        
        vulnerabilities = scan_result.get("vulnerabilities", [])
        high_severity_vulns = [
            vuln for vuln in vulnerabilities
            if severity_levels.get(vuln.get("severity", "low"), 1) > max_severity_level
        ]
        
        if not high_severity_vulns:
            vuln_count = len(vulnerabilities)
            return AssertionResult(
                passed=True,
                message=f"Package {package_name} has {vuln_count} vulnerabilities, none above {max_severity} severity",
                context={
                    "total_vulnerabilities": vuln_count,
                    "max_allowed_severity": max_severity
                }
            )
        else:
            return AssertionResult(
                passed=False,
                message=f"Package {package_name} has {len(high_severity_vulns)} vulnerabilities above {max_severity} severity",
                expected=f"0 vulnerabilities > {max_severity}",
                actual=f"{len(high_severity_vulns)} vulnerabilities",
                context={
                    "high_severity_vulnerabilities": high_severity_vulns,
                    "total_vulnerabilities": len(vulnerabilities)
                }
            )
    
    @staticmethod
    def assert_checksum_verified(
        download_result: Dict[str, Any],
        package_name: str
    ) -> AssertionResult:
        """Assert that package checksum was successfully verified."""
        checksum_verified = download_result.get("checksum_verified", False)
        
        if checksum_verified:
            return AssertionResult(
                passed=True,
                message=f"Package {package_name} checksum verified successfully",
                context={
                    "expected_checksum": download_result.get("expected_checksum"),
                    "actual_checksum": download_result.get("actual_checksum")
                }
            )
        else:
            return AssertionResult(
                passed=False,
                message=f"Package {package_name} checksum verification failed",
                expected=True,
                actual=False,
                context={
                    "expected_checksum": download_result.get("expected_checksum"),
                    "actual_checksum": download_result.get("actual_checksum")
                }
            )
    
    @staticmethod
    def assert_secure_permissions(
        file_path: Path,
        max_permissions: int = 0o644
    ) -> AssertionResult:
        """Assert that file has secure permissions."""
        if not file_path.exists():
            return AssertionResult(
                passed=False,
                message=f"File does not exist: {file_path}",
                expected="existing file",
                actual="non-existent"
            )
        
        actual_permissions = file_path.stat().st_mode & 0o777
        
        if actual_permissions <= max_permissions:
            return AssertionResult(
                passed=True,
                message=f"File {file_path} has secure permissions {oct(actual_permissions)}",
                context={
                    "file_path": str(file_path),
                    "permissions": oct(actual_permissions),
                    "max_allowed": oct(max_permissions)
                }
            )
        else:
            return AssertionResult(
                passed=False,
                message=f"File {file_path} has insecure permissions {oct(actual_permissions)}",
                expected=f"<= {oct(max_permissions)}",
                actual=oct(actual_permissions),
                context={"file_path": str(file_path)}
            )
    
    @staticmethod
    def assert_no_sensitive_data_in_logs(
        log_content: str,
        sensitive_patterns: List[str] = None
    ) -> AssertionResult:
        """Assert that logs do not contain sensitive data."""
        if sensitive_patterns is None:
            sensitive_patterns = [
                r"password",
                r"secret",
                r"token",
                r"key",
                r"credential",
                r"[A-Za-z0-9+/]{40,}",  # Base64-like strings
                r"sk_[a-zA-Z0-9]{32,}",  # API keys
                r"\d{4}-\d{4}-\d{4}-\d{4}"  # Credit card patterns
            ]
        
        import re
        found_patterns = []
        
        for pattern in sensitive_patterns:
            matches = re.finditer(pattern, log_content, re.IGNORECASE)
            for match in matches:
                found_patterns.append({
                    "pattern": pattern,
                    "match": match.group(0),
                    "position": match.start()
                })
        
        if not found_patterns:
            return AssertionResult(
                passed=True,
                message="No sensitive data found in logs",
                context={"log_length": len(log_content)}
            )
        else:
            return AssertionResult(
                passed=False,
                message=f"Found {len(found_patterns)} potential sensitive data matches in logs",
                expected="0 sensitive data matches",
                actual=f"{len(found_patterns)} matches",
                context={"found_patterns": found_patterns}
            )


def assert_all(assertions: List[AssertionResult]) -> AssertionResult:
    """Combine multiple assertions with AND logic."""
    failed_assertions = [a for a in assertions if not a.passed]
    
    if not failed_assertions:
        return AssertionResult(
            passed=True,
            message=f"All {len(assertions)} assertions passed",
            context={"total_assertions": len(assertions)}
        )
    else:
        failed_messages = [a.message for a in failed_assertions]
        return AssertionResult(
            passed=False,
            message=f"{len(failed_assertions)}/{len(assertions)} assertions failed",
            context={
                "failed_count": len(failed_assertions),
                "total_count": len(assertions),
                "failed_messages": failed_messages
            }
        )


def assert_any(assertions: List[AssertionResult]) -> AssertionResult:
    """Combine multiple assertions with OR logic."""
    passed_assertions = [a for a in assertions if a.passed]
    
    if passed_assertions:
        return AssertionResult(
            passed=True,
            message=f"{len(passed_assertions)}/{len(assertions)} assertions passed (OR logic)",
            context={
                "passed_count": len(passed_assertions),
                "total_count": len(assertions)
            }
        )
    else:
        failed_messages = [a.message for a in assertions]
        return AssertionResult(
            passed=False,
            message=f"All {len(assertions)} assertions failed",
            context={
                "total_count": len(assertions),
                "failed_messages": failed_messages
            }
        )