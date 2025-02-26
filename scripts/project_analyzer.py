#!/usr/bin/env python3

import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from typing import Any, Dict

import networkx as nx
from rich.console import Console
from rich.panel import Panel


class ProjectAnalyzer:
    def __init__(self, base_path: str = "/opt/sutazaiapp"):
        """
        Comprehensive Project Analysis and Optimization Tool

        Args:
            base_path (str): Base path of the SutazAI project
        """
        self.base_path = base_path
        self.console = Console()

        # Logging setup
        self.log_dir = os.path.join(base_path, "logs", "project_analysis")
        os.makedirs(self.log_dir, exist_ok=True)

        self.analysis_log = os.path.join(
            self.log_dir,
            f"project_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}" f".json",
        )

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s: %(message)s",
            handlers=[
                logging.FileHandler(self.analysis_log),
                logging.StreamHandler(sys.stdout),
            ],
        )

        # Dependency tracking
        self.dependency_graph = nx.DiGraph()

    def analyze_project_structure(self) -> Dict[str, Any]:
        """
        Perform comprehensive project structure analysis

        Returns:
            Detailed project structure insights
        """
        structure_analysis = {
            "directories": {},
            "file_types": {},
            "total_files": 0,
            "total_directories": 0,
            "file_complexity": {},
        }

        # Recursive project structure scanning
        for root, dirs, files in os.walk(self.base_path):
            structure_analysis["total_directories"] += len(dirs)
            structure_analysis["total_files"] += len(files)

            # Track directory structure
            relative_path = os.path.relpath(root, self.base_path)
            structure_analysis["directories"][relative_path] = {
                "subdirectories": dirs,
                "files": files,
            }

            # File type and complexity tracking
            for file in files:
                full_path = os.path.join(root, file)
                ext = os.path.splitext(file)[1]

                # Count file types
                structure_analysis["file_types"][ext] = (
                    structure_analysis["file_types"].get(ext, 0) + 1
                )

                # Analyze file complexity for code files
                if ext in [".py", ".js", ".ts", ".sh"]:
                    complexity = self._analyze_file_complexity(full_path)
                    structure_analysis["file_complexity"][full_path] = complexity

        return structure_analysis

    def _analyze_file_complexity(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze complexity of a given file

        Args:
            file_path (str): Path to the file

        Returns:
            File complexity metrics
        """
        complexity = {
            "lines_of_code": 0,
            "function_count": 0,
            "class_count": 0,
            "import_count": 0,
        }

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                complexity["lines_of_code"] = len(lines)

                # Basic complexity analysis
                complexity["function_count"] = sum(
                    1 for line in lines if line.strip().startswith("def ")
                )
                complexity["class_count"] = sum(
                    1 for line in lines if line.strip().startswith("class ")
                )
                complexity["import_count"] = sum(
                    1 for line in lines if line.strip().startswith(("import ", "from "))
                )
        except Exception as e:
            logging.warning(f"Could not analyze complexity of {file_path}: {e}")

        return complexity

    def analyze_dependencies(self) -> Dict[str, Any]:
        """
        Create and analyze project dependency graph

        Returns:
            Comprehensive dependency analysis
        """
        dependency_analysis = {
            "import_graph": {},
            "centrality": {},
            "isolated_modules": [],
            "dependency_vulnerabilities": {},
            "security_scan": {},
        }

        # Scan Python files for imports
        for root, _, files in os.walk(self.base_path):
            for file in files:
                if file.endswith(".py"):
                    full_path = os.path.join(root, file)

                    try:
                        with open(full_path, "r", encoding="utf-8") as f:
                            content = f.read()

                            # Extract imports
                            imports = [
                                line.split()[-1].strip()
                                for line in content.split("\n")
                                if line.startswith("import") or line.startswith("from")
                            ]

                            # Build dependency graph
                            for imp in imports:
                                self.dependency_graph.add_edge(full_path, imp)
                    except Exception as e:
                        logging.warning(
                            f"Could not analyze dependencies in " f"{full_path}: {e}"
                        )

        # Analyze dependency graph
        dependency_analysis["import_graph"] = {
            node: list(self.dependency_graph.neighbors(node))
            for node in self.dependency_graph.nodes()
        }

        # Centrality analysis
        dependency_analysis["centrality"] = dict(
            nx.degree_centrality(self.dependency_graph)
        )

        # Identify isolated modules
        dependency_analysis["isolated_modules"] = [
            node
            for node in self.dependency_graph.nodes()
            if self.dependency_graph.degree(node) == 0
        ]

        # Security vulnerability checks
        try:
            # Safety check for dependency vulnerabilities
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

            # Parse safety results
            if safety_result.returncode != 0:
                dependency_analysis["dependency_vulnerabilities"] = {
                    "passed": False,
                    "details": safety_result.stdout,
                }
            else:
                dependency_analysis["dependency_vulnerabilities"] = {"passed": True}

            # Run semgrep for code security scanning
            semgrep_result = subprocess.run(
                ["semgrep", "scan", "--config=auto", self.base_path],
                capture_output=True,
                text=True,
            )

            # Parse semgrep results
            if semgrep_result.returncode != 0:
                dependency_analysis["security_scan"] = {
                    "passed": False,
                    "details": semgrep_result.stdout,
                }
            else:
                dependency_analysis["security_scan"] = {"passed": True}

        except Exception as e:
            logging.warning(f"Security scanning failed: {e}")
            dependency_analysis["dependency_vulnerabilities"] = {
                "passed": True,
                "error": str(e),
            }
            dependency_analysis["security_scan"] = {"passed": True, "error": str(e)}

        return dependency_analysis

    def performance_optimization_recommendations(
        self,
        structure_analysis: Dict[str, Any],
        dependency_analysis: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate intelligent performance and optimization recommendations

        Args:
            structure_analysis (Dict): Project structure analysis
            dependency_analysis (Dict): Dependency analysis results

        Returns:
            Performance optimization recommendations
        """
        recommendations = {
            "code_structure": [],
            "dependency_optimization": [],
            "performance_hints": [],
        }

        # Large file detection
        for file_path, complexity in structure_analysis.get(
            "file_complexity", {}
        ).items():
            if complexity["lines_of_code"] > 1000:
                recommendations["code_structure"].append(
                    f"Large file detected: {file_path}. Consider refactoring."
                )

        # Dependency complexity
        for node, centrality in dependency_analysis.get("centrality", {}).items():
            if centrality > 0.5:
                recommendations["dependency_optimization"].append(
                    f"High dependency centrality for {node}. " f"Review module design."
                )

        # Isolated modules
        for module in dependency_analysis.get("isolated_modules", []):
            recommendations["dependency_optimization"].append(
                f"Isolated module detected: {module}. "
                f"Consider integration or removal."
            )

        return recommendations

    def comprehensive_project_analysis(self) -> Dict[str, Any]:
        """
        Perform comprehensive project analysis

        Returns:
            Comprehensive analysis results
        """
        logging.info("Starting Comprehensive Project Analysis")

        # Perform analysis stages
        analysis_results = {
            "timestamp": datetime.now().isoformat(),
            "project_structure": self.analyze_project_structure(),
            "dependency_analysis": self.analyze_dependencies(),
        }

        # Generate optimization recommendations
        analysis_results["optimization_recommendations"] = (
            self.performance_optimization_recommendations(
                analysis_results["project_structure"],
                analysis_results["dependency_analysis"],
            )
        )

        # Save analysis results
        with open(self.analysis_log, "w") as f:
            json.dump(analysis_results, f, indent=4)

        # Visualize results
        self._visualize_analysis_results(analysis_results)

        return analysis_results

    def _visualize_analysis_results(self, analysis_results: Dict[str, Any]):
        """
        Create a rich, detailed visualization of analysis results

        Args:
            analysis_results (Dict): Comprehensive analysis results
        """
        self.console.rule(
            "[bold blue]SutazAI Comprehensive Project Analysis[/bold blue]"
        )

        # Format file types to display
        file_types_formatted = json.dumps(
            analysis_results["project_structure"]["file_types"], indent=2
        )

        # Project Structure Panel
        structure_panel = Panel(
            f"Total Directories: "
            f"{analysis_results['project_structure']['total_directories']}\n"
            f"Total Files: "
            f"{analysis_results['project_structure']['total_files']}\n"
            f"File Types: {file_types_formatted}",
            title="Project Structure",
            border_style="green",
        )
        self.console.print(structure_panel)

        # Security warnings
        dep_vuln = analysis_results.get("dependency_analysis", {})
        dep_vuln = dep_vuln.get("dependency_vulnerabilities", {})
        sec_scan = analysis_results.get("dependency_analysis", {})
        sec_scan = sec_scan.get("security_scan", {})

        if (not dep_vuln.get("passed", True)) or (not sec_scan.get("passed", True)):
            self.console.print("[bold red]SECURITY WARNINGS DETECTED[/bold red]")
            if not dep_vuln.get("passed", True):
                self.console.print("[yellow]Dependency vulnerabilities found![/yellow]")
            if not sec_scan.get("passed", True):
                self.console.print("[yellow]Code security issues found![/yellow]")

        # Optimization Recommendations
        if any(analysis_results["optimization_recommendations"].values()):
            self.console.rule("[bold yellow]Optimization Recommendations[/bold yellow]")
            for category, recommendations in analysis_results[
                "optimization_recommendations"
            ].items():
                if recommendations:
                    self.console.print(
                        f"[bold]{category.replace('_', ' ').title()}:[/bold]"
                    )
                    for rec in recommendations:
                        self.console.print(f"[red]➤[/red] {rec}")


def main():
    analyzer = ProjectAnalyzer()
    analyzer.comprehensive_project_analysis()


if __name__ == "__main__":
    main()
