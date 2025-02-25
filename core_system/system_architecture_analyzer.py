import importlib
import inspect
import json
import logging
import os

import matplotlib.pyplot as plt
import networkx as nx

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s"
)


class ArchitectureAnalyzer:
    def __init__(self, root_path):
        self.root_path = root_path
        self.architecture_graph = nx.DiGraph()
        self.architecture_issues = []
        self.component_types = {}

    def identify_components(self):
        """Identify architectural components and their types."""
        for root, _, files in os.walk(self.root_path):
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    with open(file_path, "r") as f:
                        content = f.read()

                        # Identify component types
                        if "class" in content:
                            self._analyze_classes(file_path)
                        if "def" in content:
                            self._analyze_functions(file_path)
                        if "async def" in content:
                            self._analyze_async_functions(file_path)

    def _analyze_classes(self, file_path):
        """Analyze classes in a file."""
        try:
            module_name = (
                os.path.relpath(file_path, self.root_path)
                .replace("/", ".")
                .replace(".py", "")
            )
            module = importlib.import_module(module_name)

            for name, obj in inspect.getmembers(module, inspect.isclass):
                if obj.__module__ == module_name:
                    component_type = self._classify_component(obj)
                    self.component_types[name] = component_type
                    self.architecture_graph.add_node(name, type=component_type)
        except Exception as e:
            self.architecture_issues.append(
                f"Error analyzing classes in {file_path}: {str(e)}"
            )

    def _analyze_functions(self, file_path):
        """Analyze functions in a file."""
        try:
            module_name = (
                os.path.relpath(file_path, self.root_path)
                .replace("/", ".")
                .replace(".py", "")
            )
            module = importlib.import_module(module_name)

            for name, obj in inspect.getmembers(module, inspect.isfunction):
                if obj.__module__ == module_name:
                    component_type = self._classify_component(obj)
                    self.component_types[name] = component_type
                    self.architecture_graph.add_node(name, type=component_type)
        except Exception as e:
            self.architecture_issues.append(
                f"Error analyzing functions in {file_path}: {str(e)}"
            )

    def _analyze_async_functions(self, file_path):
        """Analyze async functions in a file."""
        try:
            module_name = (
                os.path.relpath(file_path, self.root_path)
                .replace("/", ".")
                .replace(".py", "")
            )
            module = importlib.import_module(module_name)

            for name, obj in inspect.getmembers(module):
                if (
                    inspect.iscoroutinefunction(obj)
                    and obj.__module__ == module_name
                ):
                    component_type = self._classify_component(obj)
                    self.component_types[name] = component_type
                    self.architecture_graph.add_node(name, type=component_type)
        except Exception as e:
            self.architecture_issues.append(
                f"Error analyzing async functions in {file_path}: {str(e)}"
            )

    def _classify_component(self, component):
        """Classify the type of architectural component."""
        if inspect.isclass(component):
            if "BaseModel" in [base.__name__ for base in component.__bases__]:
                return "Data Model"
            elif "APIRouter" in [
                base.__name__ for base in component.__bases__
            ]:
                return "API Router"
            else:
                return "Generic Class"
        elif inspect.isfunction(component) or inspect.iscoroutinefunction(
            component
        ):
            if "router" in component.__name__.lower():
                return "Route Handler"
            elif "service" in component.__name__.lower():
                return "Service Function"
            else:
                return "Generic Function"
        return "Unknown"

    def analyze_dependencies(self):
        """Analyze dependencies between components."""
        for name, component_type in self.component_types.items():
            try:
                module_name = name.split(".")[0]
                module = importlib.import_module(module_name)
                component = getattr(module, name)

                # Analyze source code for potential dependencies
                source_lines = inspect.getsource(component)
                for other_name in self.component_types.keys():
                    if other_name in source_lines:
                        self.architecture_graph.add_edge(name, other_name)
            except Exception as e:
                self.architecture_issues.append(
                    f"Error analyzing dependencies for {name}: {str(e)}"
                )

    def visualize_architecture(self):
        """Create a visualization of the system architecture."""
        plt.figure(figsize=(20, 15))
        pos = nx.spring_layout(self.architecture_graph, k=0.5)

        # Color mapping for component types
        color_map = {
            "Data Model": "lightblue",
            "API Router": "lightgreen",
            "Route Handler": "salmon",
            "Service Function": "lightpink",
            "Generic Class": "lightgray",
            "Generic Function": "lightyellow",
            "Unknown": "white",
        }

        node_colors = [
            color_map.get(self.architecture_graph.nodes[node]["type"], "white")
            for node in self.architecture_graph.nodes()
        ]

        nx.draw(
            self.architecture_graph,
            pos,
            node_color=node_colors,
            with_labels=True,
            node_size=500,
            font_size=8,
            font_weight="bold",
        )

        plt.title("System Architecture Dependency Graph")
        plt.tight_layout()
        plt.savefig("architecture_dependency_graph.png")

    def generate_architecture_report(self):
        """Generate a comprehensive architecture report."""
        report = {
            "total_components": len(self.architecture_graph.nodes),
            "total_dependencies": len(self.architecture_graph.edges),
            "component_types": dict(self.component_types),
            "architecture_issues": self.architecture_issues,
        }

        with open("architecture_analysis_report.json", "w") as f:
            json.dump(report, f, indent=2)

        logging.info(f"Total Components: {report['total_components']}")
        logging.info(f"Total Dependencies: {report['total_dependencies']}")

        for issue in self.architecture_issues:
            logging.warning(issue)


def main():
    root_path = os.getcwd()
    analyzer = ArchitectureAnalyzer(root_path)

    try:
        analyzer.identify_components()
        analyzer.analyze_dependencies()
        analyzer.visualize_architecture()
        analyzer.generate_architecture_report()
    except Exception as e:
        logging.error(f"Architecture analysis failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
