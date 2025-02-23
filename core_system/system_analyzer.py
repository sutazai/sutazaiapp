#!/usr/bin/env python3
"""
SutazAI Advanced System Analysis Framework

Comprehensive system analysis tool providing:
- Deep architectural insights
- Dependency mapping
- Performance profiling
- Code quality assessment
- Security vulnerability scanning

Key Responsibilities:
- Holistic system architecture analysis
- Cross-component dependency tracking
- Code complexity evaluation
- Performance bottleneck identification
- Security risk assessment
"""

import ast
import importlib
import json
import os
import subprocess
import sys
import time
import typing
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from config.config_manager import ConfigurationManager
from core_system.monitoring.advanced_logger import AdvancedLogger

# Internal system imports
from system_integration.system_integrator import SystemIntegrator


@dataclass
class SystemAnalysisReport:
    """
    Comprehensive system analysis report tracking

    Captures detailed insights about system architecture,
    code quality, performance, and potential improvements
    """

    timestamp: str
    architectural_insights: Dict[str, Any]
    dependency_graph: Dict[str, List[str]]
    code_quality_metrics: Dict[str, Any]
    performance_analysis: Dict[str, Any]
    security_assessment: Dict[str, Any]
    optimization_recommendations: List[str]


class SystemAnalyzer:
    """
    Advanced system analysis framework

    Provides comprehensive system architecture and code quality assessment
    """

    def __init__(
        self,
        base_dir: str = "/opt/sutazai_project/SutazAI",
        logger: Optional[AdvancedLogger] = None,
    ):
        """
        Initialize system analysis framework

        Args:
            base_dir (str): Base directory of the project
            logger (AdvancedLogger): Advanced logging system
        """
        self.base_dir = base_dir
        self.logger = logger or AdvancedLogger()
        self.system_integrator = SystemIntegrator()

    def analyze_project_structure(self) -> Dict[str, Any]:
        """
        Analyze the overall project structure and architecture

        Returns:
            Detailed project structure insights
        """
        project_structure = {"directories": {}, "file_types": {}, "total_files": 0}

        for root, dirs, files in os.walk(self.base_dir):
            # Skip version control and hidden directories
            dirs[:] = [d for d in dirs if not d.startswith(".")]

            # Relative path from base directory
            rel_path = os.path.relpath(root, self.base_dir)

            # Track directories
            project_structure["directories"][rel_path] = {
                "subdirectories": dirs,
                "files": files,
            }

            # Count and categorize files
            for file in files:
                project_structure["total_files"] += 1
                file_ext = os.path.splitext(file)[1]
                project_structure["file_types"][file_ext] = (
                    project_structure["file_types"].get(file_ext, 0) + 1
                )

        return project_structure

    def generate_dependency_graph(self) -> Dict[str, List[str]]:
        """
        Generate a comprehensive dependency graph for the project

        Returns:
            Mapping of modules and their dependencies
        """
        dependency_graph = {}

        for root, _, files in os.walk(self.base_dir):
            for file in files:
                if file.endswith(".py") and not file.startswith("__"):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, "r") as f:
                            tree = ast.parse(f.read())

                        # Extract import statements
                        imports = [
                            node.names[0].name
                            for node in ast.walk(tree)
                            if isinstance(node, ast.Import)
                            or isinstance(node, ast.ImportFrom)
                        ]

                        # Relative path from base directory
                        rel_path = os.path.relpath(file_path, self.base_dir)
                        dependency_graph[rel_path] = imports

                    except Exception as e:
                        self.logger.log(
                            f"Could not analyze dependencies for {file_path}: {e}",
                            level="warning",
                        )

        return dependency_graph

    def assess_code_quality(self) -> Dict[str, Any]:
        """
        Perform comprehensive code quality assessment

        Returns:
            Code quality metrics and insights
        """
        try:
            # Use radon for code complexity analysis
            radon_cmd = ["radon", "cc", "-s", "-a", "-j", self.base_dir]

            radon_result = subprocess.run(radon_cmd, capture_output=True, text=True)

            # Use bandit for security vulnerability scanning
            bandit_cmd = ["bandit", "-r", "-f", "json", self.base_dir]

            bandit_result = subprocess.run(bandit_cmd, capture_output=True, text=True)

            return {
                "complexity_analysis": json.loads(radon_result.stdout),
                "security_vulnerabilities": json.loads(bandit_result.stdout),
            }

        except Exception as e:
            self.logger.log(f"Code quality assessment failed: {e}", level="error")
            return {"status": "error", "error_details": str(e)}

    def profile_system_performance(self) -> Dict[str, Any]:
        """
        Perform system-wide performance profiling

        Returns:
            Performance metrics and bottleneck identification
        """
        performance_metrics = {
            "cpu_usage": [],
            "memory_usage": [],
            "disk_io": [],
            "network_io": [],
        }

        try:
            # Use psutil for performance metrics
            import psutil

            # CPU Usage
            for _ in range(5):
                performance_metrics["cpu_usage"].append(psutil.cpu_percent(interval=1))

            # Memory Usage
            memory = psutil.virtual_memory()
            performance_metrics["memory_usage"] = {
                "total": memory.total,
                "available": memory.available,
                "percent": memory.percent,
            }

            # Disk I/O
            disk_io = psutil.disk_io_counters()
            performance_metrics["disk_io"] = {
                "read_bytes": disk_io.read_bytes,
                "write_bytes": disk_io.write_bytes,
            }

            # Network I/O
            net_io = psutil.net_io_counters()
            performance_metrics["network_io"] = {
                "bytes_sent": net_io.bytes_sent,
                "bytes_recv": net_io.bytes_recv,
            }

        except Exception as e:
            self.logger.log(f"Performance profiling failed: {e}", level="warning")

        return performance_metrics

    def generate_optimization_recommendations(
        self,
        project_structure: Dict[str, Any],
        dependency_graph: Dict[str, List[str]],
        code_quality: Dict[str, Any],
        performance_metrics: Dict[str, Any],
    ) -> List[str]:
        """
        Generate intelligent system optimization recommendations

        Args:
            project_structure (Dict): Project structure insights
            dependency_graph (Dict): Module dependency mapping
            code_quality (Dict): Code quality assessment
            performance_metrics (Dict): System performance metrics

        Returns:
            List of optimization recommendations
        """
        recommendations = []

        # Project structure recommendations
        if project_structure["total_files"] > 500:
            recommendations.append(
                "Consider modularizing the project to improve maintainability"
            )

        # Dependency graph recommendations
        for module, dependencies in dependency_graph.items():
            if len(dependencies) > 10:
                recommendations.append(
                    f"Reduce dependencies for module {module} to improve modularity"
                )

        # Code quality recommendations
        complexity_analysis = code_quality.get("complexity_analysis", {})
        for module, complexity in complexity_analysis.items():
            if complexity.get("rank", "A") not in ["A", "B"]:
                recommendations.append(
                    f"Refactor module {module} to reduce code complexity"
                )

        # Security vulnerability recommendations
        vulnerabilities = code_quality.get("security_vulnerabilities", {}).get(
            "results", []
        )
        for vuln in vulnerabilities:
            recommendations.append(
                f"Address security vulnerability: {vuln['issue_text']}"
            )

        # Performance recommendations
        if performance_metrics.get("memory_usage", {}).get("percent", 0) > 80:
            recommendations.append(
                "Optimize memory usage to improve system performance"
            )

        return recommendations

    def generate_comprehensive_analysis_report(self) -> SystemAnalysisReport:
        """
        Generate a comprehensive system analysis report

        Returns:
            Detailed system analysis report
        """
        with self.logger.trace("generate_comprehensive_analysis_report"):
            start_time = time.time()

            # Analyze project structure
            project_structure = self.analyze_project_structure()

            # Generate dependency graph
            dependency_graph = self.generate_dependency_graph()

            # Assess code quality
            code_quality = self.assess_code_quality()

            # Profile system performance
            performance_metrics = self.profile_system_performance()

            # Generate optimization recommendations
            optimization_recommendations = self.generate_optimization_recommendations(
                project_structure, dependency_graph, code_quality, performance_metrics
            )

            # Create comprehensive analysis report
            analysis_report = SystemAnalysisReport(
                timestamp=datetime.now().isoformat(),
                architectural_insights=project_structure,
                dependency_graph=dependency_graph,
                code_quality_metrics=code_quality,
                performance_analysis=performance_metrics,
                security_assessment=code_quality.get("security_vulnerabilities", {}),
                optimization_recommendations=optimization_recommendations,
            )

            # Persist analysis report
            report_path = f'/opt/sutazai_project/SutazAI/logs/system_analysis_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'

            with open(report_path, "w") as f:
                json.dump(asdict(analysis_report), f, indent=2)

            self.logger.log(
                f"System analysis report generated: {report_path}", level="info"
            )

            return analysis_report


def main():
    """
    Main execution point for system analysis
    """
    try:
        analyzer = SystemAnalyzer()

        # Generate comprehensive system analysis report
        report = analyzer.generate_comprehensive_analysis_report()

        # Print optimization recommendations
        print("System Optimization Recommendations:")
        for recommendation in report.optimization_recommendations:
            print(f"- {recommendation}")

    except Exception as e:
        print(f"System analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
