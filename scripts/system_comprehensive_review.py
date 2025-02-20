#!/usr/bin/env python3

import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from typing import Any, Dict, List

import networkx as nx
import yaml
from rich.console import Console
from rich.table import Table


class SystemComprehensiveReview:
    def __init__(self, base_path: str = "/opt/sutazai_project/SutazAI"):
        """
        Initialize Comprehensive System Review

        Args:
            base_path (str): Base path of the SutazAI project
        """
        self.base_path = base_path
        self.console = Console()
        self.review_log = os.path.join(
            base_path,
            "logs",
            "system_review",
            f"review_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        )

        # Ensure log directory exists
        os.makedirs(os.path.dirname(self.review_log), exist_ok=True)

        # Logging setup
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s: %(message)s",
            handlers=[
                logging.FileHandler(self.review_log),
                logging.StreamHandler(sys.stdout),
            ],
        )

    def analyze_project_structure(self) -> Dict[str, Any]:
        """
        Analyze the entire project structure

        Returns:
            Detailed project structure analysis
        """
        structure_analysis = {
            "directories": {},
            "file_types": {},
            "total_files": 0,
            "total_directories": 0,
        }

        for root, dirs, files in os.walk(self.base_path):
            structure_analysis["total_directories"] += len(dirs)
            structure_analysis["total_files"] += len(files)

            # Track directory structure
            relative_path = os.path.relpath(root, self.base_path)
            structure_analysis["directories"][relative_path] = {
                "subdirectories": dirs,
                "files": files,
            }

            # Track file types
            for file in files:
                ext = os.path.splitext(file)[1]
                structure_analysis["file_types"][ext] = (
                    structure_analysis["file_types"].get(ext, 0) + 1
                )

        return structure_analysis

    def dependency_graph_analysis(self) -> Dict[str, Any]:
        """
        Create and analyze project dependency graph

        Returns:
            Dependency graph analysis
        """
        dependency_graph = nx.DiGraph()

        # Scan Python files for imports
        for root, _, files in os.walk(self.base_path):
            for file in files:
                if file.endswith(".py"):
                    full_path = os.path.join(root, file)
                    try:
                        with open(full_path, "r") as f:
                            content = f.read()
                            imports = [
                                line.split()[-1].strip()
                                for line in content.split("\n")
                                if line.startswith("import") or line.startswith("from")
                            ]

                            for imp in imports:
                                dependency_graph.add_edge(full_path, imp)
                    except Exception as e:
                        logging.warning(f"Could not analyze {full_path}: {e}")

        return {
            "nodes": list(dependency_graph.nodes()),
            "edges": list(dependency_graph.edges()),
            "centrality": nx.degree_centrality(dependency_graph),
        }

    def security_vulnerability_scan(self) -> Dict[str, Any]:
        """
        Perform comprehensive security vulnerability scan

        Returns:
            Security scan results
        """
        try:
            # Run safety check on requirements
            safety_result = subprocess.run(
                [
                    "safety",
                    "check",
                    "-r",
                    os.path.join(self.base_path, "requirements.txt"),
                ],
                capture_output=True,
                text=True,
            )

            # Run Semgrep security scan
            semgrep_result = subprocess.run(
                ["semgrep", "scan", "--config=auto", self.base_path],
                capture_output=True,
                text=True,
            )

            return {
                "safety_check": {
                    "output": safety_result.stdout,
                    "errors": safety_result.stderr,
                    "passed": safety_result.returncode == 0,
                },
                "semgrep_scan": {
                    "output": semgrep_result.stdout,
                    "errors": semgrep_result.stderr,
                    "passed": semgrep_result.returncode == 0,
                },
            }
        except Exception as e:
            logging.error(f"Security scan failed: {e}")
            return {"error": str(e)}

    def performance_optimization_recommendations(
        self, structure_analysis: Dict[str, Any], dependency_graph: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate performance and optimization recommendations

        Args:
            structure_analysis (Dict): Project structure analysis
            dependency_graph (Dict): Dependency graph analysis

        Returns:
            Optimization recommendations
        """
        recommendations = {
            "code_structure": [],
            "dependency_optimization": [],
            "performance_hints": [],
        }

        # Large file detection
        for directory, content in structure_analysis["directories"].items():
            for file in content["files"]:
                full_path = os.path.join(self.base_path, directory, file)
                if os.path.getsize(full_path) > 10 * 1024 * 1024:  # 10MB
                    recommendations["code_structure"].append(
                        f"Large file detected: {full_path}. Consider refactoring."
                    )

        # Dependency complexity
        for node, centrality in dependency_graph.get("centrality", {}).items():
            if centrality > 0.5:
                recommendations["dependency_optimization"].append(
                    f"High dependency centrality for {node}. Review module design."
                )

        return recommendations

    def comprehensive_review(self) -> Dict[str, Any]:
        """
        Perform comprehensive system review

        Returns:
            Comprehensive review results
        """
        logging.info("Starting Comprehensive System Review")

        review_results = {
            "timestamp": datetime.now().isoformat(),
            "project_structure": self.analyze_project_structure(),
            "dependency_graph": self.dependency_graph_analysis(),
            "security_scan": self.security_vulnerability_scan(),
        }

        # Performance optimization recommendations
        review_results["optimization_recommendations"] = (
            self.performance_optimization_recommendations(
                review_results["project_structure"], review_results["dependency_graph"]
            )
        )

        # Save review results
        with open(self.review_log, "w") as f:
            json.dump(review_results, f, indent=4)

        # Rich console visualization
        self._visualize_review_results(review_results)

        return review_results

    def _visualize_review_results(self, review_results: Dict[str, Any]):
        """
        Visualize review results using Rich console

        Args:
            review_results (Dict): Comprehensive review results
        """
        self.console.rule("[bold blue]SutazAI Comprehensive System Review[/bold blue]")

        # Project Structure Overview
        structure_table = Table(title="Project Structure")
        structure_table.add_column("Metric", style="cyan")
        structure_table.add_column("Value", style="magenta")

        structure_table.add_row(
            "Total Directories",
            str(review_results["project_structure"]["total_directories"]),
        )
        structure_table.add_row(
            "Total Files", str(review_results["project_structure"]["total_files"])
        )

        self.console.print(structure_table)

        # Security Scan Results
        self.console.rule("[bold red]Security Scan Results[/bold red]")
        security_status = (
            "✅ Passed"
            if review_results["security_scan"]["safety_check"]["passed"]
            and review_results["security_scan"]["semgrep_scan"]["passed"]
            else "❌ Vulnerabilities Detected"
        )
        self.console.print(
            f"[yellow]Overall Security Status:[/yellow] {security_status}"
        )

        # Optimization Recommendations
        if review_results["optimization_recommendations"]["code_structure"]:
            self.console.rule("[bold yellow]Optimization Recommendations[/bold yellow]")
            for rec in review_results["optimization_recommendations"]["code_structure"]:
                self.console.print(f"[red]➤[/red] {rec}")


def main():
    review = SystemComprehensiveReview()
    review.comprehensive_review()


if __name__ == "__main__":
    main()
