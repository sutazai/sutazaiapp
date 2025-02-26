#!/usr/bin/env python3
"""
Autonomous Monitoring System for SutazAI
"""

import hashlib
import os
import subprocess
import sys
import time
from datetime import datetime
from typing import Any, Dict

import networkx as nx
import psutil
import yaml
from loguru import logger
from rich.console import Console
from rich.table import Table


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


class AutonomousMonitor:
    def __init__(
        self,
        config_path: str = "/opt/sutazaiapp/config/monitor_config.yaml",
    ):
        """
        Initialize the Autonomous Monitoring System

        Args:
            config_path (str): Path to the monitoring configuration file
        """
        self.config = self._load_configuration(config_path)
        self.base_path = "/opt/sutazaiapp"
        self.log_dir = os.path.join(
            self.base_path, "logs", "autonomous_monitor"
        )

        # Advanced logging setup
        os.makedirs(self.log_dir, exist_ok=True)
        log_file = os.path.join(
            self.log_dir,
            f"monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        )
        logger.add(log_file, rotation="10 MB", level="INFO")

        # Rich console for enhanced output
        self.console = Console()

        # Dependency and component tracking
        self.component_graph = nx.DiGraph()
        self.script_checksums = {}

        # Performance and health tracking
        self.performance_history = {
            "cpu_usage": [],
            "memory_usage": [],
            "script_execution_times": {},
        }

    def _load_configuration(self, config_path: str) -> Dict[str, Any]:
        """
        Load monitoring configuration from YAML file

        Args:
            config_path (str): Path to configuration file

        Returns:
            Dict containing monitoring configuration
        """
        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_path}")
            return {
                "monitoring_interval": 60,
                "critical_scripts": [],
                "performance_thresholds": {
                    "cpu_max": 80,
                    "memory_max": 85,
                    "script_timeout": 300,
                },
            }

    def _compute_file_checksum(self, filepath: str) -> str:
        """
        Compute SHA-256 checksum of a file

        Args:
            filepath (str): Path to the file

        Returns:
            Hexadecimal checksum of the file
        """
        hasher = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def analyze_script_dependencies(self, script_path: str) -> Dict[str, Any]:
        """
        Perform comprehensive script dependency and complexity analysis

        Args:
            script_path (str): Path to the script to analyze

        Returns:
            Dictionary with script analysis results
        """
        analysis = {
            "path": script_path,
            "imports": [],
            "function_count": 0,
            "complexity": 0,
            "dependencies": [],
        }

        try:
            with open(script_path, "r") as f:
                content = f.read()

            # Basic import detection
            import_lines = [
                line.strip()
                for line in content.split("\n")
                if line.startswith("import") or line.startswith("from")
            ]
            analysis["imports"] = import_lines

            # Function count and basic complexity estimation
            function_lines = [
                line
                for line in content.split("\n")
                if line.strip().startswith("def ")
            ]
            analysis["function_count"] = len(function_lines)
            analysis["complexity"] = len(content.split("\n"))

            # Dependency graph tracking
            self.component_graph.add_node(script_path)
            for imp in import_lines:
                self.component_graph.add_edge(script_path, imp)

        except Exception as e:
            logger.error(f"Script analysis failed for {script_path}: {e}")

        return analysis

    def monitor_script_execution(self, script_path: str) -> Dict[str, Any]:
        """
        Monitor script execution with advanced tracking

        Args:
            script_path (str): Path to the script to execute

        Returns:
            Execution monitoring results
        """
        start_time = time.time()
        result = {
            "script": script_path,
            "start_time": start_time,
            "status": "unknown",
            "execution_time": 0,
            "memory_usage": 0,
            "output": "",
            "error": "",
        }

        try:
            process = subprocess.Popen(
                [sys.executable, script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            # Timeout mechanism
            timeout = self.config["performance_thresholds"]["script_timeout"]
            try:
                stdout, stderr = process.communicate(timeout=timeout)
                result["output"] = stdout
                result["error"] = stderr
                result["status"] = (
                    "completed" if process.returncode == 0 else "failed"
                )
            except subprocess.TimeoutExpired:
                process.kill()
                result["status"] = "timeout"

            result["execution_time"] = time.time() - start_time

            # Memory tracking
            try:
                process_memory = psutil.Process(
                    process.pid
                ).memory_info().rss / (1024 * 1024)
                result["memory_usage"] = process_memory
            except Exception:
                pass

        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)

        # Performance history tracking
        self.performance_history["script_execution_times"][script_path] = (
            result["execution_time"]
        )

        return result

    def comprehensive_system_scan(self) -> Dict[str, Any]:
        """
        Perform a comprehensive system-wide scan

        Returns:
            Detailed system scan results
        """
        scan_results = {
            "timestamp": datetime.now().isoformat(),
            "scripts": {},
            "system_resources": {
                "cpu_usage": psutil.cpu_percent(interval=1),
                "memory_usage": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage("/").percent,
            },
            "dependency_graph": {},
        }

        # Scan all scripts in the project
        for root, _, files in os.walk(self.base_path):
            for file in files:
                if file.endswith(".py"):
                    script_path = os.path.join(root, file)

                    # Compute checksum and track changes
                    current_checksum = self._compute_file_checksum(script_path)
                    if (
                        script_path not in self.script_checksums
                        or self.script_checksums[script_path]
                        != current_checksum
                    ):
                        self.script_checksums[script_path] = current_checksum

                        # Analyze script
                        script_analysis = self.analyze_script_dependencies(
                            script_path
                        )
                        scan_results["scripts"][script_path] = script_analysis

        # Convert dependency graph to serializable format
        scan_results["dependency_graph"] = {
            node: list(self.component_graph.neighbors(node))
            for node in self.component_graph.nodes()
        }

        return scan_results

    def autonomous_optimization(
        self, scan_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Perform autonomous system optimization based on scan results

        Args:
            scan_results (Dict): Comprehensive system scan results

        Returns:
            Optimization recommendations
        """
        optimization_report = {"recommendations": [], "actions_taken": []}

        # CPU and Memory Optimization
        if (
            scan_results["system_resources"]["cpu_usage"]
            > self.config["performance_thresholds"]["cpu_max"]
            or scan_results["system_resources"]["memory_usage"]
            > self.config["performance_thresholds"]["memory_max"]
        ):

            optimization_report["recommendations"].append(
                "High resource utilization detected"
            )

            # Identify and terminate resource-intensive scripts
            for script_path, script_data in scan_results["scripts"].items():
                if (
                    script_data["complexity"] > 1000
                ):  # Arbitrary complexity threshold
                    optimization_report["recommendations"].append(
                        f"Optimize script: {script_path}"
                    )

        return optimization_report

    def run(self):
        """
        Main monitoring loop
        """
        logger.info("Autonomous Monitoring System Initialized")

        while True:
            try:
                # Comprehensive system scan
                scan_results = self.comprehensive_system_scan()

                # Autonomous optimization
                optimization_results = self.autonomous_optimization(
                    scan_results
                )

                # Visualization of results
                self._visualize_results(scan_results, optimization_results)

                # Sleep for configured interval
                time.sleep(self.config.get("monitoring_interval", 60))

            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(30)  # Backoff and retry

    def _visualize_results(
        self,
        scan_results: Dict[str, Any],
        optimization_results: Dict[str, Any],
    ):
        """
        Create a rich, detailed visualization of monitoring results

        Args:
            scan_results (Dict): Comprehensive system scan results
            optimization_results (Dict): Optimization recommendations
        """
        self.console.rule(
            "[bold blue]SutazAI Autonomous Monitoring System[/bold blue]"
        )

        # System Resources Table
        resources_table = Table(title="System Resources")
        resources_table.add_column("Resource", style="cyan")
        resources_table.add_column("Usage", style="magenta")

        for resource, usage in scan_results["system_resources"].items():
            resources_table.add_row(resource, f"{usage}%")

        self.console.print(resources_table)

        # Recommendations
        if optimization_results["recommendations"]:
            self.console.rule(
                "[bold red]Optimization Recommendations[/bold red]"
            )
            for recommendation in optimization_results["recommendations"]:
                self.console.print(f"[yellow]➤[/yellow] {recommendation}")


def main():
    """
    Main function for autonomous monitoring
    """
    # Verify Python version
    verify_python_version()
    
    monitor = AutonomousMonitor()
    monitor.run()


if __name__ == "__main__":
    main()
