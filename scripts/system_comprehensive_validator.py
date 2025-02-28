#!/usr/bin/env python3
# cSpell:ignore semgrep Sutaz sutazai levelname Semgrep getloadavg

import hashlib
import json
import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict

import psutil  # type: ignore
import SpinnerColumn
import TextColumn
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress
from rich.table import Table

from misc.utils.subprocess_utils import run_command, run_python_module


class SystemComprehensiveValidator:
    def __init__(self, base_path: str = "/opt/sutazaiapp"):
        """
        Ultra-Comprehensive System Validation and Optimization Framework
        
        Args:
        base_path (str): Base path of the SutazAI project
        """
        self.base_path = base_path
        self.console = Console()
        self.log_dir = os.path.join(base_path, "logs", "system_validation")
        os.makedirs(self.log_dir, exist_ok=True)

        # Comprehensive logging setup
        self.validation_log = os.path.join(
            self.log_dir,
            f"system_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        )

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(self.validation_log),
                logging.StreamHandler(sys.stdout),
            ],
        )

        # Validation tracking
        self.validation_results = {
            "timestamp": datetime.now().isoformat(),
            "system_health": {},
            "code_quality": {},
            "performance_metrics": {},
            "optimization_recommendations": {},
        }

    def comprehensive_system_scan(self) -> Dict[str, Any]:
        """
        Perform an ultra-comprehensive system-wide scan
        
        Returns:
        Detailed system scan results
        """
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            progress.add_task(
                "[green]Performing Comprehensive System Scan...",
                total=None,
            )

            # Comprehensive system health check
            system_health = self._check_system_health()

            # Code quality analysis
            code_quality = self._analyze_code_quality()

            # Dependency analysis
            dependency_analysis = self._analyze_dependencies()

            # Performance metrics collection
            performance_metrics = self._collect_performance_metrics()

            self.validation_results["system_health"] = system_health
            self.validation_results["code_quality"] = code_quality
            self.validation_results["performance_metrics"] = performance_metrics
            self.validation_results["dependency_analysis"] = dependency_analysis

            return self.validation_results

    def _check_system_health(self) -> Dict[str, Any]:
        """
        Perform comprehensive system health check including resource
        monitoring.
        
        Returns:
        A dictionary containing system health metrics.
        """
        health = {}
        health["directory_structure"] = self._validate_directory_structure()
        health["file_integrity"] = self._check_file_integrity()
        health["environment_configuration"] = self._validate_environment_config()

        # Use psutil to check system memory (psutil import handled with type
        # ignore)
        health["memory"] = {
            "available": psutil.virtual_memory().available,
            "used": psutil.virtual_memory().used,
            "percent": psutil.virtual_memory().percent,
        }

        return health

    def _validate_directory_structure(self) -> Dict[str, Any]:
        """
        Validate project directory structure.
        
        Returns:
        Dictionary containing directory validation results.
        """
        required_dirs = [
            "ai_agents",
            "backend",
            "web_ui",
            "scripts",
            "model_management",
            "logs",
            "config",
            "docs",
        ]

        dir_validation = {
            "missing_directories": [],
            "unexpected_directories": [],
        }

        for req_dir in required_dirs:
            full_path = os.path.join(self.base_path, req_dir)
            if not os.path.exists(full_path):
                dir_validation["missing_directories"].append(req_dir)

        all_dirs = [
            d for d in os.listdir(self.base_path)
            if os.path.isdir(os.path.join(self.base_path, d))
        ]
        unexpected = [d for d in all_dirs if d not in required_dirs]
        dir_validation["unexpected_directories"] = unexpected

        return dir_validation

    def _check_file_integrity(self) -> Dict[str, Any]:
        """
        Check integrity of critical project files.
        
        Returns:
        Dictionary containing file integrity results.
        """
        critical_files = [
            "requirements.txt",
            "README.md",
            "LICENSE",
            "scripts/deploy.sh",
            "backend/main.py",
        ]

        integrity = {"missing_files": [], "file_checksums": {}}

        for file_path in critical_files:
            full_path = os.path.join(self.base_path, file_path)
            if not os.path.exists(full_path):
                integrity["missing_files"].append(file_path)
            else:
                with open(full_path, "rb") as f:
                    integrity["file_checksums"][file_path] = hashlib.sha256(
                        f.read(),
                    ).hexdigest()

        return integrity

    def _validate_environment_config(self) -> Dict[str, Any]:
        """
        Validate environment configuration.
        
        Returns:
        Dictionary containing environment configuration details.
        """
        import platform

        env_config = {
            "python_version": platform.python_version(),
            "required_version": "3.11.0",
            "is_compatible": platform.python_version().startswith("3.11"),
        }
        return env_config

    def _analyze_code_quality(self) -> Dict[str, Any]:
        """
        Analyze code quality using various formatters and linters.
        
        Returns:
        Dictionary containing code quality analysis results.
        """
        code_quality = {
            "black_formatting": self._run_black_formatting(),
            "isort_imports": self._run_isort_imports(),
            "mypy_type_checking": self._run_mypy_type_check(),
            "semgrep_analysis": self._run_semgrep_analysis(),
            "dependency_vulnerabilities": (self._check_dependency_vulnerabilities()),
        }
        return code_quality

    def _run_black_formatting(self) -> Dict[str, Any]:
        """
        Run Black code formatter.
        
        Returns:
        Dictionary with Black formatting results.
        """
        try:
            result = run_python_module(
                "black",
                ["--check", "--diff", self.base_path],
                check=False,
            )
            return {
                "passed": result.returncode == 0,
                "output": result.stdout,
            }
        except Exception as e:
            logging.warning(f"Black formatting check failed: {e}")
            return {"error": str(e)}

    def _run_isort_imports(self) -> Dict[str, Any]:
        """
        Run isort to check import order.
        
        Returns:
        Dictionary with isort results.
        """
        try:
            result = run_python_module(
                "isort",
                ["--check-only", "--diff", self.base_path],
                check=False,
            )
            return {
                "passed": result.returncode == 0,
                "output": result.stdout,
            }
        except Exception as e:
            logging.warning(f"Import order check failed: {e}")
            return {"error": str(e)}

    def _run_mypy_type_check(self) -> Dict[str, Any]:
        """
        Run mypy for type checking.
        
        Returns:
        Dictionary with mypy output.
        """
        try:
            result = run_python_module(
                "mypy",
                ["--strict", self.base_path],
                check=False,
            )
            return {
                "passed": result.returncode == 0,
                "output": result.stdout,
            }
        except Exception as e:
            logging.warning(f"Type checking failed: {e}")
            return {"error": str(e)}

    def _run_semgrep_analysis(self) -> Dict[str, Any]:
        """
        Run semgrep security analysis.
        
        Returns:
        Dictionary with semgrep analysis results.
        """
        try:
            result = run_command(
                ["semgrep", "scan", "--config=auto", self.base_path],
                check=False,
            )
            return {
                "passed": result.returncode == 0,
                "output": result.stdout,
            }
        except Exception as e:
            logging.warning(f"Security analysis failed: {e}")
            return {"error": str(e)}

    def _check_dependency_vulnerabilities(self) -> Dict[str, Any]:
        """
        Check for vulnerable dependencies using safety.
        
        Returns:
        Dictionary with vulnerability check results.
        """
        try:
            result = run_command(
                ["safety", "check", "--full-report"],
                check=False,
            )
            return {
                "passed": result.returncode == 0,
                "output": result.stdout,
            }
        except Exception as e:
            logging.warning(f"Vulnerability check failed: {e}")
            return {"error": str(e)}

    def _collect_performance_metrics(self) -> Dict[str, Any]:
        """
        Collect system performance metrics.
        
        Returns:
        Dictionary containing performance metrics.
        """
        metrics = {}
        # Example: CPU and load averages
        metrics["cpu"] = {"usage_percent": psutil.cpu_percent(interval=1)}
        metrics["load_average"] = os.getloadavg()
        return metrics

    def _analyze_dependencies(self) -> Dict[str, Any]:
        """
        Analyze project dependencies.
        
        Returns:
        Dictionary containing dependency analysis.
        """
        dependencies = {}
        # Implementation of dependency analysis and cross-referencing can be
        # added here.
        return dependencies

    def generate_optimization_recommendations(
        self,
        system_scan_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate comprehensive optimization recommendations
        
        Args:
        system_scan_results (Dict): Comprehensive system scan results
        
        Returns:
        Optimization recommendations
        """
        recommendations = {
            "system_structure": [],
            "code_quality": [],
            "performance": [],
        }

        # System structure recommendations
        if system_scan_results["system_health"]["directory_structure"][
            "missing_directories"
        ]:
            recommendations["system_structure"].append(
                "Create missing directories: "
                + ", ".join(
                    system_scan_results["system_health"]["directory_structure"][
                        "missing_directories"
                    ],
                ),
            )

        # Code quality recommendations
        if not system_scan_results["code_quality"]["black_formatting"]["passed"]:
            recommendations["code_quality"].append(
                "Run Black code formatter to ensure consistent code style",
            )

        if not system_scan_results["code_quality"]["isort_imports"]["passed"]:
            recommendations["code_quality"].append(
                "Use isort to organize and sort import statements",
            )

        # Dependency vulnerability check
        dependency_vuln_key = "dependency_vulnerabilities"
        if (
            dependency_vuln_key in system_scan_results["code_quality"]
            and not system_scan_results["code_quality"][dependency_vuln_key]["passed"]
        ):
            recommendations["code_quality"].append(
                "Update dependencies to resolve known vulnerabilities",
            )

        # Performance recommendations
        cpu_key = "cpu"
        if system_scan_results["performance_metrics"][cpu_key]["usage_percent"] > 70:
            metrics = system_scan_results["performance_metrics"]
            cpu_usage = metrics[cpu_key]["usage_percent"]
            recommendations["performance"].append(
                f"High CPU usage detected: {cpu_usage}%. "
                "Investigate and optimize resource-intensive processes.",
            )

        return recommendations

    def comprehensive_system_validation(self) -> Dict[str, Any]:
        """
        Perform ultra-comprehensive system validation
        
        Returns:
        Comprehensive validation results
        """
        logging.info("Starting Ultra-Comprehensive System Validation")

        # Perform comprehensive system scan
        system_scan_results = self.comprehensive_system_scan()

        # Generate optimization recommendations
        optimization_recommendations = self.generate_optimization_recommendations(
            system_scan_results,
        )

        # Combine all results
        comprehensive_results = {
            **system_scan_results,
            "optimization_recommendations": optimization_recommendations,
        }

        # Save validation results
        with open(self.validation_log, "w") as f:
            json.dump(comprehensive_results, f, indent=4)

        # Visualize results
        self._visualize_validation_results(comprehensive_results)

        return comprehensive_results

    def _visualize_validation_results(self, validation_results: Dict[str, Any]):
        """
        Create a rich, detailed visualization of validation results
        
        Args:
        validation_results (Dict): Comprehensive validation results
        """
        self.console.rule(
            "[bold blue]SutazAI Ultra-Comprehensive System Validation[/bold blue]",
        )

        # Get relevant system health components
        system_health = validation_results["system_health"]
        dir_struct = system_health["directory_structure"]
        file_integ = system_health["file_integrity"]
        env_conf = system_health["environment_configuration"]

        missing_dirs = dir_struct["missing_directories"]
        missing_files = file_integ["missing_files"]
        python_compatible = env_conf["python_version"]["is_compatible"]

        dir_status = "✅ Healthy" if not missing_dirs else "❌ Issues Detected"
        file_status = "✅ Intact" if not missing_files else "❌ Missing Files"
        env_status = "✅ Configured" if python_compatible else "❌ Incompatible"

        # System Health Panel
        health_panel = Panel(
            f"Directory Structure: {dir_status}\n"
            f"File Integrity: {file_status}\n"
            f"Environment Config: {env_status}",
            title="System Health Overview",
            border_style="green",
        )
        self.console.print(health_panel)

        # Code Quality Table
        code_quality_table = Table(title="Code Quality Analysis")
        code_quality_table.add_column("Check", style="cyan")
        code_quality_table.add_column("Status", style="magenta")

        code_quality_checks = [
            (
                "Black Formatting",
                validation_results["code_quality"]["black_formatting"]["passed"],
            ),
            (
                "Import Sorting",
                validation_results["code_quality"]["isort_imports"]["passed"],
            ),
            (
                "Type Checking",
                validation_results["code_quality"]["mypy_type_checking"]["passed"],
            ),
        ]

        for check, status in code_quality_checks:
            code_quality_table.add_row(check, "✅ Passed" if status else "❌ Failed")

        self.console.print(code_quality_table)

        # Optimization Recommendations
        if any(validation_results["optimization_recommendations"].values()):
            self.console.rule("[bold yellow]Optimization Recommendations[/bold yellow]")
            for category, recommendations in validation_results[
                "optimization_recommendations"
            ].items():
                if recommendations:
                    self.console.print(
                        f"[bold]{category.replace('_', ' ').title()}:[/bold]",
                    )
                    for rec in recommendations:
                        self.console.print(f"[red]➤[/red] {rec}")


def main():
    validator = SystemComprehensiveValidator()
    results = validator.comprehensive_system_scan()
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
