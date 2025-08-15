#!/usr/bin/env python3
"""
Test Reporting Utilities

Comprehensive reporting utilities for MCP automation testing.
Provides structured test reporting, performance metrics collection,
compliance validation, and automated report generation.

Author: Claude AI Assistant (senior-automated-tester)
Created: 2025-08-15 UTC
Version: 1.0.0
"""

import json
import time
import statistics
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import csv


class TestStatus(Enum):
    """Test execution status enumeration."""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"
    TIMEOUT = "timeout"


class ReportFormat(Enum):
    """Report output format enumeration."""
    JSON = "json"
    HTML = "html"
    CSV = "csv"
    XML = "xml"
    MARKDOWN = "markdown"


@dataclass
class TestResult:
    """Individual test result structure."""
    test_name: str
    test_category: str
    status: TestStatus
    duration: float
    start_time: float
    end_time: float
    error_message: Optional[str] = None
    assertions: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)


@dataclass
class TestSuiteResult:
    """Test suite execution result."""
    suite_name: str
    start_time: float
    end_time: float
    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    error_tests: int
    tests: List[TestResult] = field(default_factory=list)
    setup_duration: float = 0.0
    teardown_duration: float = 0.0


@dataclass
class PerformanceMetric:
    """Performance metric data structure."""
    metric_name: str
    value: Union[int, float]
    unit: str
    timestamp: float
    context: Dict[str, Any] = field(default_factory=dict)
    threshold: Optional[Union[int, float]] = None
    passed_threshold: Optional[bool] = None


@dataclass
class ComplianceCheck:
    """Compliance validation result."""
    check_name: str
    rule_reference: str
    status: bool
    details: str
    severity: str = "medium"
    remediation: Optional[str] = None


class TestReporter:
    """Comprehensive test reporting and metrics collection."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.test_suites: List[TestSuiteResult] = []
        self.current_suite: Optional[TestSuiteResult] = None
        self.session_start_time = time.time()
        
        # Report metadata
        self.metadata = {
            "session_id": f"mcp_test_{int(self.session_start_time)}",
            "framework_version": "1.0.0",
            "platform": self._get_platform_info(),
            "configuration": {}
        }
    
    def _get_platform_info(self) -> Dict[str, str]:
        """Get platform information for reports."""
        import platform
        return {
            "system": platform.system(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "architecture": platform.architecture()[0]
        }
    
    def start_suite(self, suite_name: str) -> None:
        """Start a new test suite."""
        if self.current_suite:
            self.end_suite()
        
        self.current_suite = TestSuiteResult(
            suite_name=suite_name,
            start_time=time.time(),
            end_time=0.0,
            total_tests=0,
            passed_tests=0,
            failed_tests=0,
            skipped_tests=0,
            error_tests=0
        )
    
    def end_suite(self) -> None:
        """End the current test suite."""
        if self.current_suite:
            self.current_suite.end_time = time.time()
            self.test_suites.append(self.current_suite)
            self.current_suite = None
    
    def add_test_result(self, test_result: TestResult) -> None:
        """Add a test result to the current suite."""
        if not self.current_suite:
            raise ValueError("No active test suite. Call start_suite() first.")
        
        self.current_suite.tests.append(test_result)
        self.current_suite.total_tests += 1
        
        # Update counters
        if test_result.status == TestStatus.PASSED:
            self.current_suite.passed_tests += 1
        elif test_result.status == TestStatus.FAILED:
            self.current_suite.failed_tests += 1
        elif test_result.status == TestStatus.SKIPPED:
            self.current_suite.skipped_tests += 1
        elif test_result.status == TestStatus.ERROR:
            self.current_suite.error_tests += 1
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate comprehensive test summary report."""
        session_end_time = time.time()
        session_duration = session_end_time - self.session_start_time
        
        # Aggregate statistics
        total_tests = sum(suite.total_tests for suite in self.test_suites)
        total_passed = sum(suite.passed_tests for suite in self.test_suites)
        total_failed = sum(suite.failed_tests for suite in self.test_suites)
        total_skipped = sum(suite.skipped_tests for suite in self.test_suites)
        total_errors = sum(suite.error_tests for suite in self.test_suites)
        
        # Calculate success rate
        success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        # Performance statistics
        all_tests = [test for suite in self.test_suites for test in suite.tests]
        test_durations = [test.duration for test in all_tests if test.duration > 0]
        
        performance_stats = {}
        if test_durations:
            performance_stats = {
                "average_test_duration": statistics.mean(test_durations),
                "median_test_duration": statistics.median(test_durations),
                "min_test_duration": min(test_durations),
                "max_test_duration": max(test_durations),
                "total_test_time": sum(test_durations)
            }
        
        # Suite statistics
        suite_stats = []
        for suite in self.test_suites:
            suite_duration = suite.end_time - suite.start_time
            suite_success_rate = (suite.passed_tests / suite.total_tests * 100) if suite.total_tests > 0 else 0
            
            suite_stats.append({
                "name": suite.suite_name,
                "total_tests": suite.total_tests,
                "passed": suite.passed_tests,
                "failed": suite.failed_tests,
                "skipped": suite.skipped_tests,
                "errors": suite.error_tests,
                "success_rate": suite_success_rate,
                "duration": suite_duration
            })
        
        return {
            "session": {
                "session_id": self.metadata["session_id"],
                "start_time": self.session_start_time,
                "end_time": session_end_time,
                "duration": session_duration,
                "framework_version": self.metadata["framework_version"]
            },
            "summary": {
                "total_suites": len(self.test_suites),
                "total_tests": total_tests,
                "passed_tests": total_passed,
                "failed_tests": total_failed,
                "skipped_tests": total_skipped,
                "error_tests": total_errors,
                "success_rate": success_rate
            },
            "performance": performance_stats,
            "suites": suite_stats,
            "platform": self.metadata["platform"]
        }
    
    def export_report(
        self,
        format_type: ReportFormat,
        filename: Optional[str] = None
    ) -> Path:
        """Export test report in specified format."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"mcp_test_report_{timestamp}.{format_type.value}"
        
        output_file = self.output_dir / filename
        
        if format_type == ReportFormat.JSON:
            self._export_json_report(output_file)
        elif format_type == ReportFormat.HTML:
            self._export_html_report(output_file)
        elif format_type == ReportFormat.CSV:
            self._export_csv_report(output_file)
        elif format_type == ReportFormat.MARKDOWN:
            self._export_markdown_report(output_file)
        else:
            raise ValueError(f"Unsupported report format: {format_type}")
        
        return output_file
    
    def _export_json_report(self, output_file: Path) -> None:
        """Export detailed JSON report."""
        report_data = {
            "metadata": self.metadata,
            "summary": self.generate_summary_report(),
            "test_suites": [asdict(suite) for suite in self.test_suites]
        }
        
        with open(output_file, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
    
    def _export_html_report(self, output_file: Path) -> None:
        """Export HTML report with interactive features."""
        summary = self.generate_summary_report()
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>MCP Test Report - {summary['session']['session_id']}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background: #f5f5f5; padding: 20px; border-radius: 5px; }}
        .summary {{ display: flex; gap: 20px; margin: 20px 0; }}
        .metric {{ background: white; padding: 15px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .passed {{ color: green; }}
        .failed {{ color: red; }}
        .skipped {{ color: orange; }}
        .error {{ color: darkred; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #f2f2f2; }}
        .suite-name {{ font-weight: bold; }}
        .success-rate {{ font-weight: bold; }}
        .progress-bar {{ width: 100%; height: 20px; background: #f0f0f0; border-radius: 10px; }}
        .progress-fill {{ height: 100%; background: linear-gradient(90deg, #4CAF50, #8BC34A); border-radius: 10px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>MCP Test Automation Report</h1>
        <p><strong>Session ID:</strong> {summary['session']['session_id']}</p>
        <p><strong>Duration:</strong> {summary['session']['duration']:.2f} seconds</p>
        <p><strong>Platform:</strong> {summary['platform']['system']} {summary['platform']['architecture']}</p>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="summary">
        <div class="metric">
            <h3>Total Tests</h3>
            <div style="font-size: 2em; font-weight: bold;">{summary['summary']['total_tests']}</div>
        </div>
        <div class="metric passed">
            <h3>Passed</h3>
            <div style="font-size: 2em; font-weight: bold;">{summary['summary']['passed_tests']}</div>
        </div>
        <div class="metric failed">
            <h3>Failed</h3>
            <div style="font-size: 2em; font-weight: bold;">{summary['summary']['failed_tests']}</div>
        </div>
        <div class="metric skipped">
            <h3>Skipped</h3>
            <div style="font-size: 2em; font-weight: bold;">{summary['summary']['skipped_tests']}</div>
        </div>
        <div class="metric">
            <h3>Success Rate</h3>
            <div style="font-size: 2em; font-weight: bold;">{summary['summary']['success_rate']:.1f}%</div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: {summary['summary']['success_rate']}%;"></div>
            </div>
        </div>
    </div>
    
    <h2>Test Suites</h2>
    <table>
        <thead>
            <tr>
                <th>Suite Name</th>
                <th>Total</th>
                <th>Passed</th>
                <th>Failed</th>
                <th>Skipped</th>
                <th>Errors</th>
                <th>Success Rate</th>
                <th>Duration (s)</th>
            </tr>
        </thead>
        <tbody>
"""
        
        for suite_stat in summary['suites']:
            html_content += f"""
            <tr>
                <td class="suite-name">{suite_stat['name']}</td>
                <td>{suite_stat['total_tests']}</td>
                <td class="passed">{suite_stat['passed']}</td>
                <td class="failed">{suite_stat['failed']}</td>
                <td class="skipped">{suite_stat['skipped']}</td>
                <td class="error">{suite_stat['errors']}</td>
                <td class="success-rate">{suite_stat['success_rate']:.1f}%</td>
                <td>{suite_stat['duration']:.2f}</td>
            </tr>
"""
        
        html_content += """
        </tbody>
    </table>
    
    <h2>Performance Metrics</h2>
    <div class="summary">
"""
        
        if summary['performance']:
            perf = summary['performance']
            html_content += f"""
        <div class="metric">
            <h3>Average Test Time</h3>
            <div style="font-size: 1.5em;">{perf['average_test_duration']:.2f}s</div>
        </div>
        <div class="metric">
            <h3>Fastest Test</h3>
            <div style="font-size: 1.5em;">{perf['min_test_duration']:.2f}s</div>
        </div>
        <div class="metric">
            <h3>Slowest Test</h3>
            <div style="font-size: 1.5em;">{perf['max_test_duration']:.2f}s</div>
        </div>
        <div class="metric">
            <h3>Total Test Time</h3>
            <div style="font-size: 1.5em;">{perf['total_test_time']:.2f}s</div>
        </div>
"""
        
        html_content += """
    </div>
    
    <footer style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; text-align: center; color: #666;">
        <p>Generated by MCP Test Automation Framework v1.0.0</p>
    </footer>
</body>
</html>
"""
        
        with open(output_file, 'w') as f:
            f.write(html_content)
    
    def _export_csv_report(self, output_file: Path) -> None:
        """Export CSV report for data analysis."""
        with open(output_file, 'w', newline='') as csvfile:
            fieldnames = [
                'suite_name', 'test_name', 'test_category', 'status',
                'duration', 'start_time', 'end_time', 'error_message'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for suite in self.test_suites:
                for test in suite.tests:
                    writer.writerow({
                        'suite_name': suite.suite_name,
                        'test_name': test.test_name,
                        'test_category': test.test_category,
                        'status': test.status.value,
                        'duration': test.duration,
                        'start_time': test.start_time,
                        'end_time': test.end_time,
                        'error_message': test.error_message or ''
                    })
    
    def _export_markdown_report(self, output_file: Path) -> None:
        """Export Markdown report for documentation."""
        summary = self.generate_summary_report()
        
        markdown_content = f"""# MCP Test Automation Report

## Session Information
- **Session ID:** {summary['session']['session_id']}
- **Duration:** {summary['session']['duration']:.2f} seconds
- **Platform:** {summary['platform']['system']} {summary['platform']['architecture']}
- **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

| Metric | Value |
|--------|--------|
| Total Tests | {summary['summary']['total_tests']} |
| Passed | {summary['summary']['passed_tests']} |
| Failed | {summary['summary']['failed_tests']} |
| Skipped | {summary['summary']['skipped_tests']} |
| Errors | {summary['summary']['error_tests']} |
| Success Rate | {summary['summary']['success_rate']:.1f}% |

## Test Suites

| Suite Name | Total | Passed | Failed | Skipped | Errors | Success Rate | Duration |
|------------|-------|--------|--------|---------|--------|--------------|----------|
"""
        
        for suite_stat in summary['suites']:
            markdown_content += f"| {suite_stat['name']} | {suite_stat['total_tests']} | {suite_stat['passed']} | {suite_stat['failed']} | {suite_stat['skipped']} | {suite_stat['errors']} | {suite_stat['success_rate']:.1f}% | {suite_stat['duration']:.2f}s |\n"
        
        if summary['performance']:
            perf = summary['performance']
            markdown_content += f"""
## Performance Metrics

| Metric | Value |
|--------|--------|
| Average Test Duration | {perf['average_test_duration']:.2f}s |
| Median Test Duration | {perf['median_test_duration']:.2f}s |
| Fastest Test | {perf['min_test_duration']:.2f}s |
| Slowest Test | {perf['max_test_duration']:.2f}s |
| Total Test Time | {perf['total_test_time']:.2f}s |
"""
        
        markdown_content += """
---
*Generated by MCP Test Automation Framework v1.0.0*
"""
        
        with open(output_file, 'w') as f:
            f.write(markdown_content)


class PerformanceReporter:
    """Specialized reporter for performance metrics and benchmarks."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics: List[PerformanceMetric] = []
        self.baselines: Dict[str, Dict[str, Any]] = {}
    
    def add_metric(
        self,
        metric_name: str,
        value: Union[int, float],
        unit: str,
        context: Optional[Dict[str, Any]] = None,
        threshold: Optional[Union[int, float]] = None
    ) -> None:
        """Add a performance metric."""
        metric = PerformanceMetric(
            metric_name=metric_name,
            value=value,
            unit=unit,
            timestamp=time.time(),
            context=context or {},
            threshold=threshold,
            passed_threshold=value <= threshold if threshold else None
        )
        self.metrics.append(metric)
    
    def set_baseline(self, baseline_name: str, baseline_data: Dict[str, Any]) -> None:
        """Set performance baseline for comparison."""
        self.baselines[baseline_name] = baseline_data
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        # Group metrics by name
        metric_groups = {}
        for metric in self.metrics:
            if metric.metric_name not in metric_groups:
                metric_groups[metric.metric_name] = []
            metric_groups[metric.metric_name].append(metric)
        
        # Calculate statistics for each metric group
        metric_stats = {}
        for metric_name, metric_list in metric_groups.items():
            values = [m.value for m in metric_list]
            
            if values:
                metric_stats[metric_name] = {
                    "count": len(values),
                    "average": statistics.mean(values),
                    "median": statistics.median(values),
                    "min": min(values),
                    "max": max(values),
                    "stddev": statistics.stdev(values) if len(values) > 1 else 0,
                    "unit": metric_list[0].unit,
                    "threshold": metric_list[0].threshold,
                    "threshold_violations": sum(1 for m in metric_list if m.passed_threshold is False)
                }
        
        # Performance summary
        total_metrics = len(self.metrics)
        threshold_violations = sum(1 for m in self.metrics if m.passed_threshold is False)
        performance_score = ((total_metrics - threshold_violations) / total_metrics * 100) if total_metrics > 0 else 100
        
        return {
            "summary": {
                "total_metrics": total_metrics,
                "unique_metric_types": len(metric_groups),
                "threshold_violations": threshold_violations,
                "performance_score": performance_score
            },
            "metrics": metric_stats,
            "baselines": self.baselines,
            "raw_data": [asdict(m) for m in self.metrics]
        }
    
    def export_performance_report(self, filename: Optional[str] = None) -> Path:
        """Export performance report to JSON."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"mcp_performance_report_{timestamp}.json"
        
        output_file = self.output_dir / filename
        report_data = self.generate_performance_report()
        
        with open(output_file, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        return output_file


class ComplianceReporter:
    """Specialized reporter for compliance validation and rule enforcement."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checks: List[ComplianceCheck] = []
    
    def add_compliance_check(
        self,
        check_name: str,
        rule_reference: str,
        status: bool,
        details: str,
        severity: str = "medium",
        remediation: Optional[str] = None
    ) -> None:
        """Add a compliance check result."""
        check = ComplianceCheck(
            check_name=check_name,
            rule_reference=rule_reference,
            status=status,
            details=details,
            severity=severity,
            remediation=remediation
        )
        self.checks.append(check)
    
    def generate_compliance_report(self) -> Dict[str, Any]:
        """Generate comprehensive compliance report."""
        total_checks = len(self.checks)
        passed_checks = sum(1 for check in self.checks if check.status)
        failed_checks = total_checks - passed_checks
        
        # Group by severity
        severity_stats = {}
        for check in self.checks:
            if check.severity not in severity_stats:
                severity_stats[check.severity] = {"total": 0, "passed": 0, "failed": 0}
            
            severity_stats[check.severity]["total"] += 1
            if check.status:
                severity_stats[check.severity]["passed"] += 1
            else:
                severity_stats[check.severity]["failed"] += 1
        
        # Compliance score
        compliance_score = (passed_checks / total_checks * 100) if total_checks > 0 else 100
        
        # Failed checks details
        failed_check_details = [
            asdict(check) for check in self.checks if not check.status
        ]
        
        return {
            "summary": {
                "total_checks": total_checks,
                "passed_checks": passed_checks,
                "failed_checks": failed_checks,
                "compliance_score": compliance_score
            },
            "severity_breakdown": severity_stats,
            "failed_checks": failed_check_details,
            "all_checks": [asdict(check) for check in self.checks]
        }
    
    def export_compliance_report(self, filename: Optional[str] = None) -> Path:
        """Export compliance report to JSON."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"mcp_compliance_report_{timestamp}.json"
        
        output_file = self.output_dir / filename
        report_data = self.generate_compliance_report()
        
        with open(output_file, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        return output_file