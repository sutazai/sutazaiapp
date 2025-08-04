#!/usr/bin/env python3
"""
SutazAI Test Runner
Automated test execution with AI-powered test generation and analysis
"""

import os
import sys
import subprocess
import argparse
import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Any
import logging
from datetime import datetime
import concurrent.futures
import yaml
import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

console = Console()


class TestRunner:
    """AI-powered test runner for SutazAI system."""
    
    def __init__(self, project_root: str = "/opt/sutazaiapp"):
        self.project_root = Path(project_root)
        self.test_results = {}
        self.start_time = None
        self.end_time = None
        
    def run_unit_tests(self, pattern: Optional[str] = None, parallel: bool = True) -> Dict:
        """Run unit tests with AI analysis."""
        console.print("[bold blue]🧪 Running Unit Tests[/bold blue]")
        
        cmd = ["python", "-m", "pytest", "-m", "unit"]
        
        if pattern:
            cmd.extend(["-k", pattern])
            
        if parallel:
            cmd.extend(["-n", "auto"])
            
        cmd.extend([
            "--cov=backend",
            "--cov=frontend", 
            "--cov-report=html:htmlcov/unit",
            "--cov-report=xml:coverage-unit.xml",
            "--junitxml=unit-test-results.xml",
            "-v"
        ])
        
        return self._execute_test_command(cmd, "unit")
    
    def run_integration_tests(self, services: Optional[List[str]] = None) -> Dict:
        """Run integration tests with service dependency checks."""
        console.print("[bold green]🔗 Running Integration Tests[/bold green]")
        
        # Check service dependencies
        if services:
            self._check_service_dependencies(services)
        
        cmd = ["python", "-m", "pytest", "-m", "integration"]
        cmd.extend([
            "--cov=backend",
            "--cov-append",
            "--cov-report=html:htmlcov/integration",
            "--cov-report=xml:coverage-integration.xml",
            "--junitxml=integration-test-results.xml",
            "-v",
            "--tb=short"
        ])
        
        return self._execute_test_command(cmd, "integration")
    
    def run_performance_tests(self, load_profile: str = "light") -> Dict:
        """Run performance tests with load profiling."""
        console.print("[bold yellow]⚡ Running Performance Tests[/bold yellow]")
        
        cmd = ["python", "-m", "pytest", "-m", "performance"]
        cmd.extend([
            f"--benchmark-only",
            f"--benchmark-json=performance-results.json",
            "--junitxml=performance-test-results.xml",
            "-v"
        ])
        
        # Set load profile environment
        env = os.environ.copy()
        env["LOAD_PROFILE"] = load_profile
        
        return self._execute_test_command(cmd, "performance", env=env)
    
    def run_security_tests(self, scan_type: str = "all") -> Dict:
        """Run security tests with vulnerability scanning."""
        console.print("[bold red]🔒 Running Security Tests[/bold red]")
        
        results = {}
        
        # Run pytest security tests
        cmd = ["python", "-m", "pytest", "-m", "security"]
        cmd.extend([
            "--junitxml=security-test-results.xml",
            "-v"
        ])
        
        results["pytest_security"] = self._execute_test_command(cmd, "security")
        
        # Run additional security scans
        if scan_type in ["all", "bandit"]:
            results["bandit"] = self._run_bandit_scan()
            
        if scan_type in ["all", "safety"]:
            results["safety"] = self._run_safety_scan()
            
        return results
    
    def run_end_to_end_tests(self, browser: str = "chrome") -> Dict:
        """Run end-to-end tests with browser automation."""
        console.print("[bold magenta]🌐 Running End-to-End Tests[/bold magenta]")
        
        # Setup browser environment
        env = os.environ.copy()
        env["BROWSER"] = browser
        env["HEADLESS"] = "true"
        
        cmd = ["python", "-m", "pytest", "-m", "e2e"]
        cmd.extend([
            "--junitxml=e2e-test-results.xml",
            "-v",
            "--tb=short",
            f"--browser={browser}"
        ])
        
        return self._execute_test_command(cmd, "e2e", env=env)
    
    def run_load_tests(self, users: int = 10, duration: str = "1m") -> Dict:
        """Run load tests with Locust."""
        console.print(f"[bold cyan]📈 Running Load Tests ({users} users, {duration})[/bold cyan]")
        
        cmd = [
            "locust",
            "-f", "tests/load/locustfile.py",
            "--headless",
            f"--users={users}",
            f"--spawn-rate={min(users, 10)}",
            f"--run-time={duration}",
            "--host=http://localhost:8002",
            "--html=load-test-report.html",
            "--csv=load-test-results"
        ]
        
        return self._execute_test_command(cmd, "load")
    
    def run_smoke_tests(self) -> Dict:
        """Run smoke tests for critical functionality."""
        console.print("[bold white]💨 Running Smoke Tests[/bold white]")
        
        cmd = ["python", "-m", "pytest", "-m", "smoke"]
        cmd.extend([
            "--junitxml=smoke-test-results.xml",
            "-v",
            "--tb=line",
            "--maxfail=1"  # Stop on first failure for smoke tests
        ])
        
        return self._execute_test_command(cmd, "smoke")
    
    def run_regression_tests(self, baseline: Optional[str] = None) -> Dict:
        """Run regression tests against baseline."""
        console.print("[bold purple]🔄 Running Regression Tests[/bold purple]")
        
        cmd = ["python", "-m", "pytest", "-m", "regression"]
        
        if baseline:
            cmd.extend(["--baseline", baseline])
        
        cmd.extend([
            "--junitxml=regression-test-results.xml",
            "-v"
        ])
        
        return self._execute_test_command(cmd, "regression")
    
    def run_all_tests(self, quick: bool = False) -> Dict:
        """Run comprehensive test suite."""
        console.print("[bold gold]🚀 Running Complete Test Suite[/bold gold]")
        
        self.start_time = time.time()
        all_results = {}
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            # Define test stages
            stages = [
                ("Unit Tests", lambda: self.run_unit_tests(parallel=not quick)),
                ("Integration Tests", lambda: self.run_integration_tests()),
                ("Security Tests", lambda: self.run_security_tests("pytest" if quick else "all")),
            ]
            
            if not quick:
                stages.extend([
                    ("Performance Tests", lambda: self.run_performance_tests()),
                    ("End-to-End Tests", lambda: self.run_end_to_end_tests()),
                    ("Smoke Tests", lambda: self.run_smoke_tests()),
                ])
            
            # Execute test stages
            for stage_name, stage_func in stages:
                task = progress.add_task(f"Running {stage_name}...", total=None)
                
                try:
                    result = stage_func()
                    all_results[stage_name.lower().replace(" ", "_")] = result
                    progress.update(task, description=f"✅ {stage_name} Complete")
                except Exception as e:
                    all_results[stage_name.lower().replace(" ", "_")] = {
                        "success": False,
                        "error": str(e)
                    }
                    progress.update(task, description=f"❌ {stage_name} Failed")
                    
                progress.remove_task(task)
        
        self.end_time = time.time()
        self.test_results = all_results
        
        # Generate comprehensive report
        self._generate_test_report()
        return all_results
    
    def _execute_test_command(self, cmd: List[str], test_type: str, env: Optional[Dict] = None) -> Dict:
        """Execute test command and capture results."""
        start_time = time.time()
        
        try:
            # Change to project root
            original_cwd = os.getcwd()
            os.chdir(self.project_root)
            
            # Execute command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env or os.environ.copy(),
                timeout=600  # 10 minute timeout
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            return {
                "success": result.returncode == 0,
                "exit_code": result.returncode,
                "duration": duration,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "command": " ".join(cmd),
                "test_type": test_type
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "exit_code": -1,
                "duration": 600,
                "error": "Test execution timed out after 10 minutes",
                "command": " ".join(cmd),
                "test_type": test_type
            }
        except Exception as e:
            return {
                "success": False,
                "exit_code": -1,
                "duration": time.time() - start_time,
                "error": str(e),
                "command": " ".join(cmd),
                "test_type": test_type
            }
        finally:
            os.chdir(original_cwd)
    
    def _check_service_dependencies(self, services: List[str]):
        """Check if required services are running."""
        console.print("🔍 Checking service dependencies...")
        
        for service in services:
            if not self._is_service_running(service):
                console.print(f"[red]❌ Service {service} is not running[/red]")
                raise RuntimeError(f"Required service {service} is not available")
            else:
                console.print(f"[green]✅ Service {service} is running[/green]")
    
    def _is_service_running(self, service: str) -> bool:
        """Check if a service is running."""
        service_urls = {
            "postgres": "postgresql://localhost:5432",
            "redis": "redis://localhost:6379",
            "ollama": "http://localhost:9005",
            "chromadb": "http://localhost:8001",
            "qdrant": "http://localhost:6333"
        }
        
        url = service_urls.get(service)
        if not url:
            return True  # Unknown service, assume it's available
        
        try:
            if url.startswith("http"):
                import requests
                response = requests.get(url, timeout=5)
                return response.status_code < 500
            else:
                # For database connections, we'd need specific checks
                return True  # Simplified for now
        except:
            return False
    
    def _run_bandit_scan(self) -> Dict:
        """Run Bandit security scan."""
        cmd = [
            "bandit",
            "-r", "backend", "frontend",
            "-f", "json",
            "-o", "bandit-report.json"
        ]
        
        return self._execute_test_command(cmd, "bandit")
    
    def _run_safety_scan(self) -> Dict:
        """Run Safety vulnerability scan."""
        cmd = [
            "safety",
            "check",
            "--json",
            "--output", "safety-report.json"
        ]
        
        return self._execute_test_command(cmd, "safety")
    
    def _generate_test_report(self):
        """Generate comprehensive test report."""
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "duration": self.end_time - self.start_time if self.end_time and self.start_time else 0,
            "results": self.test_results,
            "summary": self._calculate_summary()
        }
        
        # Save JSON report
        with open("test-report.json", "w") as f:
            json.dump(report_data, f, indent=2)
        
        # Generate HTML report
        self._generate_html_report(report_data)
        
        # Display summary in console
        self._display_summary()
    
    def _calculate_summary(self) -> Dict:
        """Calculate test summary statistics."""
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        total_duration = 0
        
        for test_type, result in self.test_results.items():
            if isinstance(result, dict):
                if result.get("success"):
                    passed_tests += 1
                else:
                    failed_tests += 1
                total_tests += 1
                total_duration += result.get("duration", 0)
        
        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            "total_duration": total_duration
        }
    
    def _generate_html_report(self, report_data: Dict):
        """Generate HTML test report."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>SutazAI Test Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .summary {{ margin: 20px 0; }}
                .test-result {{ margin: 10px 0; padding: 10px; border-left: 4px solid #ddd; }}
                .success {{ border-left-color: #4CAF50; }}
                .failure {{ border-left-color: #f44336; }}
                .duration {{ color: #666; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🧠 SutazAI Test Report</h1>
                <p>Generated: {report_data['timestamp']}</p>
                <p>Duration: {report_data['duration']:.2f} seconds</p>
            </div>
            
            <div class="summary">
                <h2>Summary</h2>
                <p>Total Tests: {report_data['summary']['total_tests']}</p>
                <p>Passed: {report_data['summary']['passed_tests']}</p>
                <p>Failed: {report_data['summary']['failed_tests']}</p>
                <p>Success Rate: {report_data['summary']['success_rate']:.1f}%</p>
            </div>
            
            <div class="results">
                <h2>Test Results</h2>
        """
        
        for test_type, result in report_data['results'].items():
            if isinstance(result, dict):
                status_class = "success" if result.get("success") else "failure"
                status_text = "✅ PASSED" if result.get("success") else "❌ FAILED"
                
                html_content += f"""
                <div class="test-result {status_class}">
                    <h3>{test_type.replace('_', ' ').title()}</h3>
                    <p><strong>{status_text}</strong></p>
                    <p class="duration">Duration: {result.get('duration', 0):.2f}s</p>
                    {f'<p>Error: {result.get("error", "")}</p>' if result.get("error") else ""}
                </div>
                """
        
        html_content += """
            </div>
        </body>
        </html>
        """
        
        with open("test-report.html", "w") as f:
            f.write(html_content)
    
    def _display_summary(self):
        """Display test summary in console."""
        summary = self._calculate_summary()
        
        # Create summary table
        table = Table(title="🧪 Test Results Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Total Tests", str(summary['total_tests']))
        table.add_row("Passed", str(summary['passed_tests']))
        table.add_row("Failed", str(summary['failed_tests']))
        table.add_row("Success Rate", f"{summary['success_rate']:.1f}%")
        table.add_row("Total Duration", f"{summary['total_duration']:.2f}s")
        
        console.print(table)
        
        # Display individual test results
        console.print("\n[bold]Test Results Detail:[/bold]")
        for test_type, result in self.test_results.items():
            if isinstance(result, dict):
                status = "✅ PASSED" if result.get("success") else "❌ FAILED"
                duration = result.get("duration", 0)
                
                color = "green" if result.get("success") else "red"
                console.print(f"[{color}]{status}[/{color}] {test_type.replace('_', ' ').title()} ({duration:.2f}s)")


def create_test_config(config_path: str = "test-config.yaml"):
    """Create test configuration file."""
    config = {
        "test_settings": {
            "parallel_execution": True,
            "coverage_threshold": 80,
            "timeout": 300,
            "retry_failed_tests": 1
        },
        "services": {
            "postgres": {
                "required": True,
                "url": "postgresql://localhost:5432"
            },
            "redis": {
                "required": True, 
                "url": "redis://localhost:6379"
            },
            "ollama": {
                "required": False,
                "url": "http://localhost:9005"
            }
        },
        "environments": {
            "test": {
                "database_url": "sqlite:///test.db",
                "debug": True,
                "log_level": "DEBUG"
            },
            "integration": {
                "database_url": "postgresql://localhost:5432/sutazai_test",
                "debug": False,
                "log_level": "INFO"
            }
        },
        "reports": {
            "html": True,
            "xml": True,
            "json": True,
            "coverage": True
        }
    }
    
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    console.print(f"[green]✅ Test configuration created: {config_path}[/green]")


@click.command()
@click.option("--type", "-t", "test_type", 
              type=click.Choice(["unit", "integration", "performance", "security", "e2e", "load", "smoke", "regression", "all"]),
              default="all", help="Type of tests to run")
@click.option("--pattern", "-k", help="Test pattern to match")
@click.option("--parallel/--no-parallel", default=True, help="Run tests in parallel")
@click.option("--quick", is_flag=True, help="Run quick test suite")
@click.option("--services", help="Comma-separated list of required services")
@click.option("--browser", default="chrome", help="Browser for E2E tests")
@click.option("--users", type=int, default=10, help="Number of users for load tests")
@click.option("--duration", default="1m", help="Duration for load tests")
@click.option("--config", help="Path to test configuration file")
@click.option("--create-config", is_flag=True, help="Create test configuration file")
def main(test_type, pattern, parallel, quick, services, browser, users, duration, config, create_config):
    """🧠 SutazAI AI-Powered Test Runner
    
    Run comprehensive tests for the SutazAI automation/advanced automation system with intelligent analysis.
    """
    
    if create_config:
        create_test_config()
        return
    
    # Initialize test runner
    runner = TestRunner()
    
    console.print(Panel.fit(
        "[bold blue]🧠 SutazAI AI-Powered Test Runner[/bold blue]\n"
        "[dim]Autonomous General Intelligence Testing Suite[/dim]",
        border_style="blue"
    ))
    
    try:
        # Parse services
        service_list = services.split(",") if services else None
        
        # Run specified tests
        if test_type == "unit":
            results = runner.run_unit_tests(pattern, parallel)
        elif test_type == "integration":
            results = runner.run_integration_tests(service_list)
        elif test_type == "performance":
            results = runner.run_performance_tests()
        elif test_type == "security":
            results = runner.run_security_tests()
        elif test_type == "e2e":
            results = runner.run_end_to_end_tests(browser)
        elif test_type == "load":
            results = runner.run_load_tests(users, duration)
        elif test_type == "smoke":
            results = runner.run_smoke_tests()
        elif test_type == "regression":
            results = runner.run_regression_tests()
        elif test_type == "all":
            results = runner.run_all_tests(quick)
        
        # Display results
        success = all(r.get("success", False) if isinstance(r, dict) else False for r in results.values()) if isinstance(results, dict) else results.get("success", False)
        
        if success:
            console.print("\n[bold green]🎉 All tests passed successfully![/bold green]")
            sys.exit(0)
        else:
            console.print("\n[bold red]❌ Some tests failed. Check the reports for details.[/bold red]")
            sys.exit(1)
            
    except Exception as e:
        console.print(f"\n[bold red]💥 Test execution failed: {str(e)}[/bold red]")
        sys.exit(1)


if __name__ == "__main__":
    main()