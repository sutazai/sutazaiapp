#!/usr/bin/env python3
"""
SutazAI System Integration Manager

Provides a centralized coordination mechanism for system components,
ensuring seamless interaction, dependency management, and autonomous operation.
"""

import importlib
import inspect
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Type

# Import internal modules
from config.config_manager import ConfigurationManager

from core_system.monitoring.advanced_logger import AdvancedLogger
from scripts.dependency_manager import DependencyManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename="/opt/sutazai_project/SutazAI/logs/system_integration.log",
)
logger = logging.get_logger("SutazAI_IntegrationManager")


@dataclass
class ComponentMetadata:
    """Comprehensive metadata for system components"""

    name: str
    type: str
    module_path: str
    dependencies: List[str]
    last_loaded: str
    status: str
    version: Optional[str] = None
    configuration: Optional[Dict[str, Any]] = None


class SystemIntegrationManager:
    """
    Advanced system integration and component management framework

    Responsibilities:
    - Dynamic component discovery
    - Dependency resolution
    - Component lifecycle management
    - Autonomous system configuration
    """

    def __init__(
        self,
        base_dir: str = "/opt/sutazai_project/SutazAI",
        config_manager: Optional[ConfigurationManager] = None,
        dependency_manager: Optional[DependencyManager] = None,
        logger: Optional[AdvancedLogger] = None,
    ):
        """
        Initialize System Integration Manager

        Args:
            base_dir (str): Base directory of the project
            config_manager (ConfigurationManager, optional): Configuration management system
            dependency_manager (DependencyManager, optional): Dependency management system
            logger (AdvancedLogger, optional): Advanced logging system
        """
        self.base_dir = base_dir
        self.config_manager = config_manager or ConfigurationManager()
        self.dependency_manager = dependency_manager or DependencyManager()
        self.logger = logger or AdvancedLogger()

        # Component registries
        self._component_registry: Dict[str, ComponentMetadata] = {}
        self._loaded_components: Dict[str, Any] = {}

    def discover_components(
        self, base_path: Optional[str] = None
    ) -> List[ComponentMetadata]:
        """
        Dynamically discover system components

        Args:
            base_path (str, optional): Base path to search for components

        Returns:
            List of discovered component metadata
        """
        base_path = base_path or self.base_dir
        discovered_components = []

        # Recursive component discovery
        for root, dirs, files in os.walk(base_path):
            for file in files:
                if file.endswith(".py") and not file.startswith("__"):
                    module_path = os.path.join(root, file)
                    try:
                        module_name = os.path.relpath(
                            module_path, self.base_dir
                        ).replace("/", ".")[:-3]
                        module = importlib.import_module(module_name)

                        # Identify potential components
                        for name, obj in inspect.getmembers(module):
                            if (
                                inspect.isclass(obj)
                                and hasattr(obj, "__module__")
                                and obj.__module__ == module_name
                            ):
                                component_metadata = ComponentMetadata(
                                    name=name,
                                    type=obj.__class__.__name__,
                                    module_path=module_path,
                                    dependencies=self._extract_dependencies(obj),
                                    last_loaded=datetime.now().isoformat(),
                                    status="discovered",
                                    version=getattr(obj, "__version__", "unknown"),
                                )

                                discovered_components.append(component_metadata)
                                self._component_registry[name] = component_metadata

                    except Exception as e:
                        logger.warning(f"Could not process module {module_path}: {e}")

        return discovered_components

    def _extract_dependencies(self, component_class: Type) -> List[str]:
        """
        Extract dependencies for a given component

        Args:
            component_class (Type): Component class to analyze

        Returns:
            List of detected dependencies
        """
        dependencies = []

        # Analyze __init__ method signature
        try:
            signature = inspect.signature(component_class.__init__)
            for param_name, param in signature.parameters.items():
                if param.annotation and hasattr(param.annotation, "__name__"):
                    dependencies.append(param.annotation.__name__)
        except Exception:
            pass

        return dependencies

    def load_component(self, component_name: str) -> Any:
        """
        Dynamically load a system component

        Args:
            component_name (str): Name of the component to load

        Returns:
            Loaded component instance

        Raises:
            ValueError: If component not found or cannot be loaded
        """
        if component_name not in self._component_registry:
            raise ValueError(f"Component {component_name} not found")

        component_metadata = self._component_registry[component_name]

        try:
            module_name = os.path.relpath(
                component_metadata.module_path, self.base_dir
            ).replace("/", ".")[:-3]

            module = importlib.import_module(module_name)
            component_class = getattr(module, component_name)

            # Resolve dependencies
            dependencies = {}
            for dep_name in component_metadata.dependencies:
                if dep_name in self._loaded_components:
                    dependencies[dep_name.lower()] = self._loaded_components[dep_name]

            # Instantiate component
            component_instance = component_class(**dependencies)

            # Update metadata
            component_metadata.status = "loaded"
            component_metadata.last_loaded = datetime.now().isoformat()

            # Store loaded component
            self._loaded_components[component_name] = component_instance

            return component_instance

        except Exception as e:
            logger.error(f"Failed to load component {component_name}: {e}")
            component_metadata.status = "failed"
            raise

    def generate_system_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive system integration report

        Returns:
            Detailed system integration report
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_components": len(self._component_registry),
            "loaded_components": len(self._loaded_components),
            "component_details": [
                asdict(metadata) for metadata in self._component_registry.values()
            ],
            "dependency_report": self.dependency_manager.generate_dependency_report(),
            "configuration_report": self.config_manager.generate_configuration_report(),
        }

        # Persist system report
        report_path = os.path.join(
            self.base_dir,
            f'logs/system_integration_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',
        )

        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"System integration report generated: {report_path}")
        return report

    def autonomous_system_optimization(self):
        """
        Perform autonomous system optimization

        Coordinates:
        - Dependency updates
        - Configuration validation
        - Component health checks
        """
        with self.logger.trace("autonomous_system_optimization"):
            start_time = time.time()

            try:
                # Update dependencies
                self.dependency_manager.update_dependencies()

                # Validate configurations
                self.config_manager.load_configurations()

                # Rediscover and reload components
                self.discover_components()

                # Generate system report
                system_report = self.generate_system_report()

                self.logger.track_performance(
                    "autonomous_system_optimization", start_time, context=system_report
                )

            except Exception as e:
                self.logger.log(
                    "Autonomous system optimization failed", level="error", exception=e
                )


def main():
    """Main execution for system integration management"""
    integration_manager = SystemIntegrationManager()

    # Discover components
    discovered_components = integration_manager.discover_components()
    print("Discovered Components:", discovered_components)

    # Perform autonomous optimization
    integration_manager.autonomous_system_optimization()


if __name__ == "__main__":
    main()
