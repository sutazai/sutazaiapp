#!/usr/bin/env python3
# cSpell:ignore sutazai Sutaz levelname getloadavg semgrep Semgrep

import json
import logging
import os
import platform
import subprocess
import sys
from datetime import datetime
from typing import Any

import Dict
import networkx as nx
import psutil  # type: ignore
from rich.console import Console
from rich.panel import Panel


# Verify Python version
def verify_python_version():
    """
    Verify that Python 3.11 or higher is being used.
    """
    major, minor = sys.version_info.major, sys.version_info.minor
    if major < 3 or (major == 3 and minor < 11):
        print("❌ Error: Python 3.11 or higher is required.")
        print(f"Current Python version: {sys.version}")
        print("Please install Python 3.11 and try again.")
        sys.exit(1)
    print(f"✅ Python {major}.{minor} detected.")


class SystemDiagnosticOptimizer:
    def __init__(self, base_path: str = "/opt/sutazaiapp"):
        """
        Advanced System Diagnostic and Optimization Tool

        Args:
            base_path (str): Base path of the SutazAI project.
        """
        self.base_path = base_path
        self.console = Console()
        self.log_dir = os.path.join(base_path, "logs", "system_diagnostics")
        os.makedirs(self.log_dir, exist_ok=True)

        # Comprehensive logging setup with consistent formatting.
        self.diagnostic_log = os.path.join(
            self.log_dir,
            f"diagnostic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        )

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s: %(message)s",
            handlers=[
                logging.FileHandler(self.diagnostic_log),
                logging.StreamHandler(sys.stdout),
            ],
        )

    def system_health_check(self) -> Dict[str, Any]:
        """
        Perform a comprehensive system health check.

        Returns:
            Detailed system health metrics.
        """
        health_metrics = {
            "system_info": {
                "os": platform.platform(),
                "python_version": platform.python_version(),
                "architecture": platform.machine(),
                "processor": platform.processor(),
            },
            "resources": {
                "cpu": {
                    "cores": psutil.cpu_count(),
                    "usage_percent": psutil.cpu_percent(interval=1),
                    "load_average": os.getloadavg(),  # using getloadavg to capture system load
                },
                "memory": {
                    "total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
                    "available_gb": round(
                        psutil.virtual_memory().available / (1024**3), 2,
                    ),
                    "usage_percent": psutil.virtual_memory().percent,
                },
                "disk": {
                    "total_gb": round(psutil.disk_usage("/").total / (1024**3), 2),
                    "free_gb": round(psutil.disk_usage("/").free / (1024**3), 2),
                    "usage_percent": psutil.disk_usage("/").percent,
                },
            },
        }
        return health_metrics

    def project_structure_analysis(self) -> Dict[str, Any]:
        """
        Perform an in-depth project structure and dependency analysis.

        Returns:
            Comprehensive project structure insights.
        """
        structure_analysis = {
            "directories": {},
            "file_types": {},
            "total_files": 0,
            "total_directories": 0,
            "dependency_graph": {},
        }

        dependency_graph = nx.DiGraph()

        for root, dirs, files in os.walk(self.base_path):
            structure_analysis["total_directories"] += len(dirs)
            structure_analysis["total_files"] += len(files)

            relative_path = os.path.relpath(root, self.base_path)
            structure_analysis["directories"][relative_path] = {
                "subdirectories": dirs,
                "files": files,
            }

            # Track file extensions and Python import dependencies.
            for file in files:
                ext = os.path.splitext(file)[1]
                structure_analysis["file_types"][ext] = (
                    structure_analysis["file_types"].get(ext, 0) + 1
                )

                if file.endswith(".py"):
                    full_path = os.path.join(root, file)
                    try:
                        with open(full_path) as f:
                            content = f.read()
                            imports = [
                                line.split()[-1].strip()
                                for line in content.split("\n")
                                if line.startswith("import") or line.startswith("from")
                            ]

                            for imp in imports:
                                dependency_graph.add_edge(full_path, imp)
                    except Exception as e:
                        logging.warning(
                            f"Could not analyze imports in {full_path}: {e}",
                        )

        structure_analysis["dependency_graph"] = {
            "nodes": list(dependency_graph.nodes()),
            "edges": list(dependency_graph.edges()),
            "centrality": dict(nx.degree_centrality(dependency_graph)),
        }

        return structure_analysis

        """

        Returns:
        """
        try:
            # Run dependency safety check.
            safety_result = subprocess.run(
                [
                    "safety",
                    "check",
                    "-r",
                    os.path.join(self.base_path, "requirements.txt"),
                ],
                capture_output=True,
                text=True,
                check=False,
            )

            semgrep_result = subprocess.run(
                ["semgrep", "scan", "--config=auto", self.base_path],
                capture_output=True,
                text=True, check=False,
            )

            return {
                "safety_check": {
                    "passed": safety_result.returncode == 0,
                    "output": safety_result.stdout,
                    "errors": safety_result.stderr,
                },
                "semgrep_scan": {
                    "passed": semgrep_result.returncode == 0,
                    "output": semgrep_result.stdout,
                    "errors": semgrep_result.stderr,
                },
            }
        except Exception as e:
            return {"error": str(e)}

    def performance_optimization_recommendations(
        self,
        health_metrics: Dict[str, Any],
        structure_analysis: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate intelligent performance and optimization recommendations.

        Args:
            health_metrics (Dict): System health metrics.
            structure_analysis (Dict): Project structure analysis.

        Returns:
            Performance optimization recommendations.
        """
        recommendations = {
            "resource_optimization": [],
            "code_structure_improvements": [],
            "dependency_optimizations": [],
        }

        # Resource utilization recommendations.
        if health_metrics["resources"]["cpu"]["usage_percent"] > 80:
            recommendations["resource_optimization"].append(
                "High CPU usage detected. Consider optimizing CPU-intensive processes.",
            )

        if health_metrics["resources"]["memory"]["usage_percent"] > 85:
            recommendations["resource_optimization"].append(
                "High memory usage detected. Investigate memory leaks or optimize memory-intensive scripts.",
            )

        # Dependency optimization based on centrality in the dependency graph.
        for file_path, centrality in structure_analysis["dependency_graph"][
            "centrality"
        ].items():
            if centrality > 0.5:
                recommendations["dependency_optimizations"].append(
                    f"High dependency centrality for {file_path}. Consider modularizing the code.",
                )

        return recommendations

    def comprehensive_diagnostic(self) -> Dict[str, Any]:
        """
        Perform a comprehensive system diagnostic and optimization analysis

        Returns:
            Comprehensive diagnostic results
        """
        logging.info("Starting Comprehensive System Diagnostic")

        # Perform diagnostic steps
        diagnostic_results = {
            "timestamp": datetime.now().isoformat(),
            "system_health": self.system_health_check(),
            "project_structure": self.project_structure_analysis(),
        }

        # Generate optimization recommendations
        diagnostic_results["optimization_recommendations"] = (
            self.performance_optimization_recommendations(
                diagnostic_results["system_health"],
                diagnostic_results["project_structure"],
            )
        )

        # Save diagnostic results
        with open(self.diagnostic_log, "w") as f:
            json.dump(diagnostic_results, f, indent=4)

        # Visualize results
        self._visualize_diagnostic_results(diagnostic_results)

        return diagnostic_results

    def _visualize_diagnostic_results(self, diagnostic_results: Dict[str, Any]):
        """
        Visualize diagnostic results with a rich interface

        Args:
            diagnostic_results: Results from comprehensive diagnostic
        """
        self.console.rule("[bold blue]SutazAI System Diagnostic Results[/bold blue]")

        # System Health Panel
        cpu_usage = f"{diagnostic_results['system_health']['resources']['cpu']['usage_percent']}%"
        mem_usage = f"{diagnostic_results['system_health']['resources']['memory']['usage_percent']}%"
        disk_usage = f"{diagnostic_results['system_health']['resources']['disk']['usage_percent']}%"

        health_panel = Panel(
            f"CPU Usage: {cpu_usage}\n"
            f"Memory Usage: {mem_usage}\n"
            f"Disk Usage: {disk_usage}",
            title="System Resources",
            border_style="green",
        )
        self.console.print(health_panel)

        # Security Status
        security_status = (
            "✅ Passed"
            if diagnostic_results.get("security_check", {}).get("passed", False)
            else "❌ Vulnerabilities Detected"
        )
        self.console.print(f"Security Status: {security_status}")

        # Optimization Recommendations
        if any(diagnostic_results["optimization_recommendations"].values()):
            self.console.rule("[bold yellow]Optimization Recommendations[/bold yellow]")
            for category, recommendations in diagnostic_results[
                "optimization_recommendations"
            ].items():
                if recommendations:
                    self.console.print(
                        f"[bold]{category.replace('_', ' ').title()}:[/bold]",
                    )
                    for rec in recommendations:
                        self.console.print(f"[red]➤[/red] {rec}")


def main():
    # Verify Python version
    verify_python_version()

    optimizer = SystemDiagnosticOptimizer()
    optimizer.comprehensive_diagnostic()


if __name__ == "__main__":
    main()
