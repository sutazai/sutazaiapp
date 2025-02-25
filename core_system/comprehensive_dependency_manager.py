#!/usr/bin/env python3
"""
SutazAI Comprehensive Dependency Manager
---------------------------------------
Manages all dependencies for the SutazAI system, providing unified
package management, version control, and dependency resolution.
"""

import importlib
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

try:
    import pkg_resources
except ImportError:
    pkg_resources = None
    logging.warning("pkg_resources not available, some features will be limited")

try:
    import networkx as nx
    import matplotlib.pyplot as plt
    GRAPH_AVAILABLE = True
except ImportError:
    GRAPH_AVAILABLE = False
    logging.warning("networkx and/or matplotlib not available, dependency graphs will be disabled")


class DependencyManager:
    """Comprehensive dependency management for SutazAI."""

    def __init__(self, project_root: Optional[Union[str, Path]] = None, log_level: str = "INFO"):
        """
        Initialize the dependency manager.
        
        Args:
            project_root: Path to the project root directory
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, log_level))
        
        self._setup_logging()
        
        if project_root:
            self.project_root = Path(project_root)
        else:
            self.project_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        self.requirements_files = self._find_requirements_files()
        self.all_requirements = self._load_all_requirements()
        self.installed_packages = self._get_installed_packages()
        self.dependency_graph = None
        
    def _setup_logging(self) -> None:
        """Set up logging for the dependency manager."""
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        formatter = logging.Formatter(log_format)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        try:
            log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
            os.makedirs(log_dir, exist_ok=True)
            
            log_file = os.path.join(log_dir, "dependency_manager.log")
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        except Exception as e:
            self.logger.warning(f"Failed to set up file logging: {str(e)}")
    
    def _find_requirements_files(self) -> List[Path]:
        """
        Find all requirements files in the project.
        
        Returns:
            List of paths to requirements files
        """
        req_files = []
        
        for path in self.project_root.glob("**/requirements*.txt"):
            if "venv" not in str(path) and ".git" not in str(path):
                req_files.append(path)
        
        self.logger.debug(f"Found {len(req_files)} requirements files")
        return req_files
    
    def _load_all_requirements(self) -> Dict[str, Dict[str, str]]:
        """
        Load all requirements from all requirements files.
        
        Returns:
            Dictionary mapping package names to versions and source files
        """
        all_reqs = {}
        
        for req_file in self.requirements_files:
            try:
                with open(req_file, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        
                        # Handle editable installs
                        if line.startswith("-e "):
                            line = line[3:]
                        
                        # Handle options like -r or --requirement
                        if line.startswith(("-r", "--requirement")):
                            continue
                        
                        # Parse package name and version
                        match = re.match(r"([A-Za-z0-9_.-]+)(?:[<>=!~]+([A-Za-z0-9_.-]+))?", line)
                        if match:
                            package = match.group(1).lower()
                            version = match.group(2) if match.group(2) else "latest"
                            
                            # Store package info
                            all_reqs[package] = {
                                "version": version,
                                "source": str(req_file.relative_to(self.project_root))
                            }
            except Exception as e:
                self.logger.error(f"Error parsing requirements file {req_file}: {str(e)}")
        
        self.logger.debug(f"Loaded {len(all_reqs)} requirements")
        return all_reqs
    
    def _get_installed_packages(self) -> Dict[str, str]:
        """
        Get all installed packages and their versions.
        
        Returns:
            Dictionary mapping package names to versions
        """
        installed = {}
        
        if pkg_resources:
            for dist in pkg_resources.working_set:
                installed[dist.key] = dist.version
        else:
            # Fallback to pip list
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "list", "--format=json"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                packages = json.loads(result.stdout)
                for pkg in packages:
                    installed[pkg["name"].lower()] = pkg["version"]
            except Exception as e:
                self.logger.error(f"Error getting installed packages: {str(e)}")
        
        self.logger.debug(f"Found {len(installed)} installed packages")
        return installed
    
    def check_missing_packages(self) -> Dict[str, Dict[str, str]]:
        """
        Check for missing packages that are required but not installed.
        
        Returns:
            Dictionary of missing packages with version and source
        """
        missing = {}
        
        for package, info in self.all_requirements.items():
            if package not in self.installed_packages:
                missing[package] = info
        
        if missing:
            self.logger.warning(f"Found {len(missing)} missing packages")
        else:
            self.logger.info("No missing packages found")
            
        return missing
    
    def check_version_mismatches(self) -> Dict[str, Dict[str, str]]:
        """
        Check for packages where the installed version differs from the required version.
        
        Returns:
            Dictionary of mismatched packages with required and installed versions
        """
        mismatched = {}
        
        for package, info in self.all_requirements.items():
            if package in self.installed_packages:
                required_version = info["version"]
                installed_version = self.installed_packages[package]
                
                if required_version != "latest" and required_version != installed_version:
                    mismatched[package] = {
                        "required": required_version,
                        "installed": installed_version,
                        "source": info["source"]
                    }
        
        if mismatched:
            self.logger.warning(f"Found {len(mismatched)} version mismatches")
        else:
            self.logger.info("No version mismatches found")
            
        return mismatched
    
    def check_unused_packages(self) -> List[str]:
        """
        Check for installed packages that are not listed in any requirements files.
        
        Returns:
            List of unused package names
        """
        unused = []
        
        for package in self.installed_packages:
            if package not in self.all_requirements:
                unused.append(package)
        
        if unused:
            self.logger.info(f"Found {len(unused)} potentially unused packages")
        
        return unused
    
    def find_module_dependencies(self, module_path: Union[str, Path]) -> Set[str]:
        """
        Find all Python package dependencies in a module.
        
        Args:
            module_path: Path to the module to analyze
            
        Returns:
            Set of package names that the module depends on
        """
        dependencies = set()
        module_path = Path(module_path)
        
        if not module_path.exists():
            self.logger.error(f"Module path {module_path} does not exist")
            return dependencies
        
        if module_path.is_file() and module_path.suffix == ".py":
            self._analyze_file_dependencies(module_path, dependencies)
        elif module_path.is_dir():
            for py_file in module_path.glob("**/*.py"):
                self._analyze_file_dependencies(py_file, dependencies)
        
        return dependencies
    
    def _analyze_file_dependencies(self, file_path: Path, dependencies: Set[str]) -> None:
        """
        Analyze a Python file for import statements and add dependencies to the set.
        
        Args:
            file_path: Path to the Python file
            dependencies: Set to add dependencies to
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Find all import statements
            import_pattern = r"(?:from|import)\s+([A-Za-z0-9_.]+)"
            for match in re.finditer(import_pattern, content):
                imported = match.group(1).split(".")[0]
                if imported and imported not in ["__future__", "typing"]:
                    # Skip standard library modules
                    if self._is_standard_library(imported):
                        continue
                    dependencies.add(imported)
        except Exception as e:
            self.logger.error(f"Error analyzing dependencies in {file_path}: {str(e)}")
    
    def _is_standard_library(self, module_name: str) -> bool:
        """
        Check if a module is part of the Python standard library.
        
        Args:
            module_name: Name of the module to check
            
        Returns:
            True if the module is part of the standard library, False otherwise
        """
        try:
            # Try to find the module's spec
            spec = importlib.util.find_spec(module_name)
            if spec is None:
                return False
            
            # If the spec has a path and it's in the standard library location, it's a stdlib module
            if spec.origin and ("lib/python" in spec.origin or "lib\\python" in spec.origin):
                if "site-packages" not in spec.origin and "dist-packages" not in spec.origin:
                    return True
            
            return False
        except (AttributeError, ImportError):
            return False
    
    def build_dependency_graph(self) -> None:
        """Build a graph of package dependencies."""
        if not GRAPH_AVAILABLE:
            self.logger.warning("Cannot build dependency graph: networkx or matplotlib not available")
            return
        
        try:
            # Create a directed graph
            G = nx.DiGraph()
            
            # Add nodes for all requirements
            for package in self.all_requirements:
                G.add_node(package)
            
            # Add edges for dependencies
            for package in self.all_requirements:
                try:
                    # Try to use pkg_resources to get dependencies
                    if pkg_resources:
                        dist = pkg_resources.get_distribution(package)
                        for req in dist.requires():
                            dep_name = req.project_name.lower()
                            if dep_name in self.all_requirements:
                                G.add_edge(package, dep_name)
                except Exception as e:
                    self.logger.debug(f"Error getting dependencies for {package}: {str(e)}")
            
            self.dependency_graph = G
            self.logger.info(f"Built dependency graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
            
        except Exception as e:
            self.logger.error(f"Error building dependency graph: {str(e)}")
    
    def save_dependency_graph(self, output_path: Union[str, Path] = None) -> None:
        """
        Save the dependency graph as an image.
        
        Args:
            output_path: Path to save the image to
        """
        if not GRAPH_AVAILABLE:
            self.logger.warning("Cannot save dependency graph: networkx or matplotlib not available")
            return
        
        if self.dependency_graph is None:
            self.build_dependency_graph()
        
        if self.dependency_graph is None:
            self.logger.error("Dependency graph could not be built")
            return
        
        try:
            if output_path is None:
                output_path = self.project_root / "module_dependency_graph.png"
            
            # Create a figure
            plt.figure(figsize=(12, 8))
            
            # Draw the graph
            pos = nx.spring_layout(self.dependency_graph, seed=42)
            nx.draw(
                self.dependency_graph,
                pos,
                with_labels=True,
                node_color="skyblue",
                node_size=1500,
                edge_color="gray",
                linewidths=1,
                font_size=8,
            )
            
            # Save the figure
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()
            
            self.logger.info(f"Dependency graph saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving dependency graph: {str(e)}")
    
    def generate_unified_requirements(self, output_path: Union[str, Path] = None) -> None:
        """
        Generate a unified requirements.txt file from all requirements files.
        
        Args:
            output_path: Path to save the unified requirements to
        """
        if output_path is None:
            output_path = self.project_root / "requirements_unified.txt"
        
        try:
            with open(output_path, "w") as f:
                f.write("# Unified requirements file generated by SutazAI Dependency Manager\n")
                f.write(f"# Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                for package, info in sorted(self.all_requirements.items()):
                    version = info["version"]
                    source = info["source"]
                    if version != "latest":
                        f.write(f"{package}=={version}  # from {source}\n")
                    else:
                        f.write(f"{package}  # from {source}\n")
            
            self.logger.info(f"Unified requirements saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error generating unified requirements: {str(e)}")
    
    def install_missing_packages(self) -> bool:
        """
        Install missing packages listed in requirements.
        
        Returns:
            True if all installations succeeded, False otherwise
        """
        missing = self.check_missing_packages()
        if not missing:
            self.logger.info("No missing packages to install")
            return True
        
        success = True
        for package, info in missing.items():
            version = info["version"]
            source = info["source"]
            
            try:
                if version != "latest":
                    package_spec = f"{package}=={version}"
                else:
                    package_spec = package
                
                self.logger.info(f"Installing {package_spec} from {source}")
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", package_spec],
                    check=True,
                    capture_output=True
                )
                self.logger.info(f"Successfully installed {package_spec}")
            except subprocess.CalledProcessError as e:
                self.logger.error(f"Failed to install {package}: {e.stderr.decode()}")
                success = False
        
        # Update installed packages
        self.installed_packages = self._get_installed_packages()
        return success
    
    def fix_version_mismatches(self) -> bool:
        """
        Fix packages where the installed version differs from the required version.
        
        Returns:
            True if all fixes succeeded, False otherwise
        """
        mismatched = self.check_version_mismatches()
        if not mismatched:
            self.logger.info("No version mismatches to fix")
            return True
        
        success = True
        for package, info in mismatched.items():
            required = info["required"]
            installed = info["installed"]
            source = info["source"]
            
            try:
                package_spec = f"{package}=={required}"
                self.logger.info(f"Updating {package} from {installed} to {required} (from {source})")
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "--upgrade", package_spec],
                    check=True,
                    capture_output=True
                )
                self.logger.info(f"Successfully updated {package} to {required}")
            except subprocess.CalledProcessError as e:
                self.logger.error(f"Failed to update {package}: {e.stderr.decode()}")
                success = False
        
        # Update installed packages
        self.installed_packages = self._get_installed_packages()
        return success
    
    def generate_dependency_report(self, output_path: Union[str, Path] = None) -> Dict:
        """
        Generate a comprehensive dependency report.
        
        Args:
            output_path: Path to save the report to (optional)
            
        Returns:
            Dictionary with dependency report data
        """
        missing = self.check_missing_packages()
        mismatched = self.check_version_mismatches()
        unused = self.check_unused_packages()
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "project_root": str(self.project_root),
            "requirements_files": [str(p.relative_to(self.project_root)) for p in self.requirements_files],
            "total_requirements": len(self.all_requirements),
            "total_installed": len(self.installed_packages),
            "missing_packages": missing,
            "version_mismatches": mismatched,
            "unused_packages": unused,
            "all_requirements": self.all_requirements,
        }
        
        if output_path:
            try:
                with open(output_path, "w") as f:
                    json.dump(report, f, indent=2)
                self.logger.info(f"Dependency report saved to {output_path}")
            except Exception as e:
                self.logger.error(f"Error saving dependency report: {str(e)}")
        
        return report


# For compatibility with Python < 3.8
if sys.version_info < (3, 8):
    from datetime import datetime
else:
    from datetime import datetime


def get_dependency_manager() -> DependencyManager:
    """Factory function to get a DependencyManager instance."""
    return DependencyManager()


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create the dependency manager
    manager = DependencyManager()
    
    # Check for issues
    missing = manager.check_missing_packages()
    mismatched = manager.check_version_mismatches()
    unused = manager.check_unused_packages()
    
    # Print a simple report
    print("\n=== SutazAI Dependency Report ===\n")
    print(f"Project root: {manager.project_root}")
    print(f"Requirements files: {len(manager.requirements_files)}")
    print(f"Total requirements: {len(manager.all_requirements)}")
    print(f"Total installed packages: {len(manager.installed_packages)}")
    print(f"Missing packages: {len(missing)}")
    print(f"Version mismatches: {len(mismatched)}")
    print(f"Unused packages: {len(unused)}")
    
    # Print details if issues were found
    if missing:
        print("\nMissing packages:")
        for package, info in missing.items():
            print(f"  - {package} {info['version']} (from {info['source']})")
    
    if mismatched:
        print("\nVersion mismatches:")
        for package, info in mismatched.items():
            print(f"  - {package}: required {info['required']}, installed {info['installed']} (from {info['source']})")
    
    # Ask to fix issues
    if missing or mismatched:
        print("\nIssues found. Would you like to fix them? [y/N]")
        choice = input().strip().lower()
        if choice in ["y", "yes"]:
            if missing:
                print("\nInstalling missing packages...")
                manager.install_missing_packages()
            
            if mismatched:
                print("\nFixing version mismatches...")
                manager.fix_version_mismatches()
                
            print("\nAll issues have been addressed.")
    else:
        print("\nNo dependency issues found!")
