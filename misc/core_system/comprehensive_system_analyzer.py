#!/usr/bin/env python3
"""
SutazAI Advanced System Management Framework

Comprehensive Autonomous System Analysis, Optimization, and Self-Healing Platform

Key Capabilities:
- Ultra-intelligent system architecture analysis
- Proactive performance optimization
- Intelligent dependency management
- Deep code quality assessment
- Autonomous system remediation
- Self-documenting and self-healing mechanisms
"""

import ast
import json
import logging
import os
import re
import subprocess
import sys
from datetime import datetime
from typing import Any, Dict, List

from core_system.utils.safe_import import safe_import

# Safe import utility
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

# Safely import external libraries
networkx = safe_import("networkx")
yaml = safe_import("yaml")
pylint = safe_import("pylint.lint")
radon = safe_import("radon.metrics")
safety = safe_import("safety")
psutil = safe_import("psutil")

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    handlers=[
        logging.FileHandler("/opt/SutazAI/logs/system_management.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("SutazAI.SystemManagement")


class AdvancedSystemManager:
    """
    Ultra-Comprehensive Autonomous System Management Framework

    Provides intelligent, self-healing system management with:
    - Multi-dimensional semantic analysis
    - Predictive optimization
    - Intelligent dependency resolution
    """

    CRITICAL_SYSTEM_PATHS = [
        "/opt/SutazAI/core_system",
        "/opt/SutazAI/scripts",
        "/opt/SutazAI/ai_agents",
    ]

    SECURITY_PATTERNS = [
        (r"eval\(", "Potential code injection via eval()"),
        (r"os\.system\(", "Potential command injection"),
        (r"subprocess\..*shell=True", "Potential shell injection"),
        (r"pickle\.load\(", "Potential deserialization vulnerability"),
    ]

    def __init__(
        self,
        base_dir: str = "/opt/SutazAI",
        output_dir: str = "/opt/SutazAI/logs/system_analysis",
    ):
        """
        Initialize advanced system manager with comprehensive tracking

        Args:
            base_dir (str): Base project directory
            output_dir (str): Directory for storing analysis outputs
        """
        self.base_dir = base_dir
        self.output_dir = output_dir

        # Ensure critical directories exist
        for path in self.CRITICAL_SYSTEM_PATHS:
            os.makedirs(path, exist_ok=True)

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Initialize system state tracking
        self.system_state = {
            "initial_snapshot": self._capture_system_snapshot(),
            "analysis_history": [],
        }

    def _capture_system_snapshot(self) -> Dict[str, Any]:
        """
        Capture a comprehensive snapshot of the system state

        Returns:
            Dictionary containing system state details
        """
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "directories": {},
            "file_hashes": {},
            "system_resources": {
                "cpu_cores": psutil.cpu_count(),
                "total_memory": psutil.virtual_memory().total,
                "disk_partitions": [
                    p.mountpoint for p in psutil.disk_partitions()
                ],
            },
        }

        # Capture directory structures
        for path in self.CRITICAL_SYSTEM_PATHS:
            snapshot["directories"][path] = self._get_directory_structure(path)

        # Generate file hashes for critical files
        for root, _, files in os.walk(self.base_dir):
            for file in files:
                if file.endswith((".py", ".sh", ".yaml", ".json")):
                    full_path = os.path.join(root, file)
                    try:
                        with open(full_path, "rb") as f:
                            snapshot["file_hashes"][full_path] = hash(f.read())
                    except Exception as e:
                        logger.warning(f"Could not hash file {full_path}: {e}")

        return snapshot

    def _get_directory_structure(self, path: str) -> Dict[str, Any]:
        """
        Generate a detailed directory structure

        Args:
            path (str): Directory path to analyze

        Returns:
            Nested dictionary representing directory structure
        """
        structure = {}
        for root, dirs, files in os.walk(path):
            current = structure
            relative_path = os.path.relpath(root, path)

            # Navigate or create nested structure
            if relative_path != ".":
                path_parts = relative_path.split(os.sep)
                for part in path_parts:
                    current = current.setdefault(part, {})

            # Add files to current directory
            current["__files__"] = files
            current["__subdirs__"] = dirs

        return structure

    def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """
        Execute an ultra-comprehensive system-wide analysis

        Returns:
            Detailed system analysis report with advanced insights
        """
        comprehensive_report = {
            "timestamp": datetime.now().isoformat(),
            "system_components": self.analyze_system_components(),
            "code_quality": self.assess_code_quality(),
            "performance_metrics": self.collect_performance_metrics(),
            "dependency_analysis": self.analyze_dependencies(),
            "optimization_recommendations": [],
            "system_state_changes": self._detect_system_changes(),
        }

        # Generate optimization recommendations
        comprehensive_report["optimization_recommendations"] = (
            self.generate_optimization_recommendations(comprehensive_report)
        )

        # Log and persist report
        self._log_and_persist_report(comprehensive_report)

        return comprehensive_report

    def _detect_system_changes(self) -> Dict[str, Any]:
        """
        Detect and analyze changes in system state

        Returns:
            Dictionary of detected system changes
        """
        current_snapshot = self._capture_system_snapshot()
        changes = {
            "new_directories": [],
            "deleted_directories": [],
            "modified_files": [],
            "new_files": [],
            "deleted_files": [],
        }

        # Compare directory structures
        for path, old_structure in self.system_state["initial_snapshot"][
            "directories"
        ].items():
            current_structure = current_snapshot["directories"].get(path, {})

            # Detect directory changes
            self._compare_directory_structures(
                old_structure, current_structure, changes
            )

        # Compare file hashes
        old_hashes = self.system_state["initial_snapshot"]["file_hashes"]
        current_hashes = current_snapshot["file_hashes"]

        # Detect file changes
        for file_path, old_hash in old_hashes.items():
            current_hash = current_hashes.get(file_path)

            if current_hash is None:
                changes["deleted_files"].append(file_path)
            elif current_hash != old_hash:
                changes["modified_files"].append(file_path)

        # Detect new files
        for file_path in current_hashes:
            if file_path not in old_hashes:
                changes["new_files"].append(file_path)

        return changes

    def _compare_directory_structures(
        self,
        old_structure: Dict[str, Any],
        current_structure: Dict[str, Any],
        changes: Dict[str, List[str]],
        current_path: str = "",
    ):
        """
        Recursively compare directory structures

        Args:
            old_structure (Dict): Previous directory structure
            current_structure (Dict): Current directory structure
            changes (Dict): Tracking dictionary for changes
            current_path (str): Current path being compared
        """
        # Compare files
        old_files = set(old_structure.get("__files__", []))
        current_files = set(current_structure.get("__files__", []))

        # Detect deleted and new files
        deleted_files = old_files - current_files
        new_files = current_files - old_files

        if deleted_files:
            changes["deleted_files"].extend(
                [os.path.join(current_path, f) for f in deleted_files]
            )

        if new_files:
            changes["new_files"].extend(
                [os.path.join(current_path, f) for f in new_files]
            )

        # Compare subdirectories
        old_subdirs = {
            k: v
            for k, v in old_structure.items()
            if k not in ["__files__", "__subdirs__"]
        }
        current_subdirs = {
            k: v
            for k, v in current_structure.items()
            if k not in ["__files__", "__subdirs__"]
        }

        # Detect deleted and new directories
        deleted_dirs = set(old_subdirs.keys()) - set(current_subdirs.keys())
        new_dirs = set(current_subdirs.keys()) - set(old_subdirs.keys())

        if deleted_dirs:
            changes["deleted_directories"].extend(
                [os.path.join(current_path, d) for d in deleted_dirs]
            )

        if new_dirs:
            changes["new_directories"].extend(
                [os.path.join(current_path, d) for d in new_dirs]
            )

        # Recursively compare subdirectories
        for subdir, subdir_structure in old_subdirs.items():
            if subdir in current_subdirs:
                self._compare_directory_structures(
                    subdir_structure,
                    current_subdirs[subdir],
                    changes,
                    os.path.join(current_path, subdir),
                )

    def _log_and_persist_report(self, report: Dict[str, Any]):
        """
        Log and persist the comprehensive analysis report

        Args:
            report (Dict): Comprehensive analysis report
        """
        # Generate unique report filename
        report_filename = (
            f'system_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        )
        report_path = os.path.join(self.output_dir, report_filename)

        # Persist report
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        # Log key insights
        logger.info(
            f"Comprehensive System Analysis Report Generated: {report_path}"
        )
        logger.info(
        )
        logger.info(
            f"Optimization Recommendations: {len(report['optimization_recommendations'])}"
        )

    def analyze_system_components(self) -> Dict[str, Any]:
        """
        Analyze and catalog system components

        Returns:
            Dictionary of system component details
        """
        system_components = {"directories": {}, "files": {}, "file_types": {}}

        try:
            for root, dirs, files in os.walk(self.base_dir):
                # Skip version control and virtual environment directories
                if any(
                    skip in root for skip in [".git", "venv", "__pycache__"]
                ):
                    continue

                # Track directories
                relative_path = os.path.relpath(root, self.base_dir)
                system_components["directories"][relative_path] = {
                    "subdirectories": dirs,
                    "files": files,
                }

                # Analyze files
                for file in files:
                    full_path = os.path.join(root, file)
                    file_ext = os.path.splitext(file)[1]

                    # Track file types
                    system_components["file_types"][file_ext] = (
                        system_components["file_types"].get(file_ext, 0) + 1
                    )

                    # Collect file metadata
                    system_components["files"][full_path] = {
                        "size": os.path.getsize(full_path),
                        "modified": os.path.getmtime(full_path),
                        "extension": file_ext,
                    }

            return system_components

        except Exception as e:
            logger.warning(f"System component analysis failed: {e}")
            return system_components

    def assess_code_quality(self) -> Dict[str, Any]:
        """
        Perform comprehensive code quality assessment

        Returns:
            Dictionary of code quality metrics
        """
        code_quality = {
            "pylint_scores": {},
            "complexity_metrics": {},
            "code_smells": [],
        }

        try:
            # Find Python files
            python_files = [
                os.path.join(root, file)
                for root, _, files in os.walk(self.base_dir)
                for file in files
                if file.endswith(".py")
            ]

            # Pylint analysis
            for file_path in python_files:
                try:
                    # Run Pylint
                    pylint_output = pylint.Run([file_path], exit=False)

                    # Store Pylint score
                    code_quality["pylint_scores"][
                        file_path
                    ] = pylint_output.linter.stats.global_note
                except Exception as e:
                    logger.warning(
                        f"Pylint analysis failed for {file_path}: {e}"
                    )

            # Complexity analysis using Radon
            for file_path in python_files:
                try:
                    with open(file_path, "r") as f:
                        content = f.read()

                    # Analyze cyclomatic complexity
                    complexity_results = radon.cc_visit(content)
                    code_quality["complexity_metrics"][file_path] = [
                        {
                            "name": result.name,
                            "complexity": result.complexity,
                            "type": result.type,
                        }
                        for result in complexity_results
                    ]
                except Exception as e:
                    logger.warning(
                        f"Complexity analysis failed for {file_path}: {e}"
                    )

            return code_quality

        except Exception as e:
            logger.warning(f"Code quality assessment failed: {e}")
            return code_quality

        """

        Returns:
        """
            "dependency_vulnerabilities": [],
        }

        try:
            # Check dependencies for known vulnerabilities
            requirements_path = os.path.join(self.base_dir, "requirements.txt")
            if os.path.exists(requirements_path):
                vulnerabilities = safety.check(
                    files=[requirements_path], ignore_ids=[], cached=True
                )
                    {
                        "package": vuln[0],
                        "version": vuln[1],
                        "vulnerability_id": vuln[2],
                        "description": vuln[3],
                    }
                    for vuln in vulnerabilities
                ]

            python_files = [
                os.path.join(root, file)
                for root, _, files in os.walk(self.base_dir)
                for file in files
                if file.endswith(".py")
            ]

            for file_path in python_files:
                try:
                    with open(file_path, "r") as f:
                        content = f.read()

                    for pattern, description in self.SECURITY_PATTERNS:
                        if re.search(pattern, content):
                            ].append(
                                {
                                    "file": file_path,
                                    "issue": description,
                                    "pattern": pattern,
                                }
                            )
                except Exception as e:
                    logger.warning(
                    )


        except Exception as e:

    def collect_performance_metrics(self) -> Dict[str, Any]:
        """
        Collect system-wide performance metrics

        Returns:
            Dictionary of performance metrics
        """
        performance_metrics = {
            "cpu_usage": psutil.cpu_percent(interval=1),
            "memory_usage": {
                "total": psutil.virtual_memory().total,
                "available": psutil.virtual_memory().available,
                "percent": psutil.virtual_memory().percent,
            },
            "disk_io": dict(psutil.disk_io_counters().asdict()),
            "network_io": dict(psutil.net_io_counters().asdict()),
            "process_count": len(psutil.process_iter()),
        }

        return performance_metrics

    def analyze_dependencies(self) -> Dict[str, Any]:
        """
        Perform comprehensive dependency analysis

        Returns:
            Dictionary of dependency insights
        """
        dependency_analysis = {"python_dependencies": {}, "module_graph": {}}

        try:
            # Analyze Python dependencies
            requirements_path = os.path.join(self.base_dir, "requirements.txt")
            if os.path.exists(requirements_path):
                with open(requirements_path, "r") as f:
                    dependencies = f.readlines()

                for dep in dependencies:
                    dep = dep.strip()
                    if dep and not dep.startswith("#"):
                        try:
                            module_name = dep.split("==")[0]
                            module = importlib.import_module(module_name)
                            dependency_analysis["python_dependencies"][
                                module_name
                            ] = {
                                "version": (
                                    module.__version__
                                    if hasattr(module, "__version__")
                                    else "Unknown"
                                )
                            }
                        except Exception as e:
                            logger.warning(
                                f"Dependency analysis failed for {dep}: {e}"
                            )

            # Build module dependency graph
            python_files = [
                os.path.join(root, file)
                for root, _, files in os.walk(self.base_dir)
                for file in files
                if file.endswith(".py")
            ]

            module_graph = nx.DiGraph()

            for file_path in python_files:
                try:
                    with open(file_path, "r") as f:
                        tree = ast.parse(f.read())

                    # Track imports
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.Import, ast.ImportFrom)):
                            if isinstance(node, ast.Import):
                                for alias in node.names:
                                    module_graph.add_node(alias.name)
                            else:
                                module_name = node.module or ""
                                for alias in node.names:
                                    full_module = (
                                        f"{module_name}.{alias.name}"
                                        if module_name
                                        else alias.name
                                    )
                                    module_graph.add_node(full_module)
                except Exception as e:
                    logger.warning(
                        f"Module graph generation failed for {file_path}: {e}"
                    )

            dependency_analysis["module_graph"] = {
                "nodes": list(module_graph.nodes()),
                "edges": list(module_graph.edges()),
            }

            return dependency_analysis

        except Exception as e:
            logger.warning(f"Dependency analysis failed: {e}")
            return dependency_analysis

    def generate_optimization_recommendations(
        self, comprehensive_report: Dict[str, Any]
    ) -> List[str]:
        """
        Generate intelligent system optimization recommendations

        Args:
            comprehensive_report (Dict): Comprehensive system analysis report

        Returns:
            List of optimization recommendations
        """
        recommendations = []

        # Performance recommendations
        performance_metrics = comprehensive_report.get(
            "performance_metrics", {}
        )
        if performance_metrics.get("cpu_usage", 0) > 70:
            recommendations.append(
                f"High CPU usage detected: {performance_metrics['cpu_usage']}%. "
                "Consider optimizing resource-intensive processes."
            )

        if performance_metrics.get("memory_usage", {}).get("percent", 0) > 80:
            recommendations.append(
                f"High memory usage detected: {performance_metrics['memory_usage']['percent']}%. "
                "Implement memory optimization strategies."
            )

        # Code quality recommendations
        code_quality = comprehensive_report.get("code_quality", {})
        low_quality_files = [
            file
            for file, score in code_quality.get("pylint_scores", {}).items()
            if score < 7.0  # Adjust threshold as needed
        ]
        if low_quality_files:
            recommendations.append(
                f"{len(low_quality_files)} files have low code quality scores. "
                "Refactor and improve code quality."
            )

            recommendations.append(
            )

            recommendations.append(
            )

        # Dependency recommendations
        dependency_analysis = comprehensive_report.get(
            "dependency_analysis", {}
        )
        if dependency_analysis.get("python_dependencies"):
            outdated_deps = [
                dep
                for dep, info in dependency_analysis[
                    "python_dependencies"
                ].items()
                if info.get("version") == "Unknown"
            ]
            if outdated_deps:
                recommendations.append(
                    f"{len(outdated_deps)} dependencies have unknown versions. "
                    "Update and pin dependency versions."
                )

        return recommendations

    def apply_recommendations(self, recommendations: List[str]):
        """
        Attempt to automatically apply optimization recommendations

        Args:
            recommendations (List[str]): List of optimization recommendations
        """
        for recommendation in recommendations:
            try:
                # Performance optimization
                if "CPU usage" in recommendation:
                    self.optimize_cpu_usage()

                # Memory optimization
                if "memory usage" in recommendation:
                    self.optimize_memory_usage()

                # Code quality improvement
                if "code quality" in recommendation:
                    self.improve_code_quality()

                if "dependency vulnerabilities" in recommendation:
                    self.update_dependencies()


            except Exception as e:
                logger.warning(
                    f"Failed to apply recommendation: {recommendation}. Error: {e}"
                )

    def optimize_cpu_usage(self):
        """
        Attempt to optimize CPU usage
        """
        # Placeholder for CPU optimization strategies
        logger.info("Attempting CPU usage optimization")

    def optimize_memory_usage(self):
        """
        Attempt to optimize memory usage
        """
        # Placeholder for memory optimization strategies
        logger.info("Attempting memory usage optimization")

    def improve_code_quality(self):
        """
        Attempt to improve code quality
        """
        # Placeholder for code quality improvement strategies
        logger.info("Attempting code quality improvement")

    def update_dependencies(self):
        """
        Update project dependencies
        """
        try:
            subprocess.run(
                ["pip", "install", "--upgrade", "-r", "requirements.txt"],
                check=True,
                capture_output=True,
            )
            logger.info("Dependencies updated successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Dependency update failed: {e}")

        """
        """

    def generate_system_health_report(self) -> str:
        """
        Generate a human-readable system health report

        Returns:
            Markdown formatted system health report
        """
        report = self.run_comprehensive_analysis()

        markdown_report = f"""# SutazAI System Health Report
## Generated at {report['timestamp']}

### System Components
- Total Directories: {len(report['system_components'].get('directories', {}))}
- Total Files: {len(report['system_components'].get('files', {}))}

### Code Quality
- Files Analyzed: {len(report['code_quality'].get('pylint_scores', {}))}
- Average Pylint Score: {sum(report['code_quality'].get('pylint_scores', {}).values()) / max(len(report['code_quality'].get('pylint_scores', {})), 1):.2f}


### Performance Metrics
- CPU Usage: {report['performance_metrics'].get('cpu_usage', 0)}%
- Memory Usage: {report['performance_metrics'].get('memory_usage', {}).get('percent', 0)}%

### Optimization Recommendations
{chr(10).join(f"- {rec}" for rec in report['optimization_recommendations'])}
"""

        # Persist markdown report
        markdown_path = os.path.join(
            self.output_dir,
            f'system_health_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.md',
        )
        with open(markdown_path, "w") as f:
            f.write(markdown_report)

        return markdown_report


def main():
    """
    Main execution for advanced system management
    """
    try:
        system_manager = AdvancedSystemManager()

        # Run comprehensive analysis
        system_manager.run_comprehensive_analysis()

        # Generate system health report
        health_report = system_manager.generate_system_health_report()
        print(health_report)

        print("Advanced System Management Completed Successfully")

    except Exception as e:
        logger.error(f"Advanced System Management Failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
