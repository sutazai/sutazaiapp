#!/usr/bin/env python3
import importlib
import json
import platform
import subprocess
import sys
from typing import Any, Dict, List, Tuple

import defusedxml.xmlrpc

# Whitelist of allowed modules for dynamic import
ALLOWED_MODULES = {
    "numpy",
    "pandas",
    "tensorflow",
    "torch",
    "sklearn",
    "scipy",
    "matplotlib",
    "seaborn",
    "requests",
    "fastapi",
    "sqlalchemy",
    "psycopg2",
    "redis",
    "celery",
    "pydantic",
}

defusedxml.xmlrpc.monkey_patch()


def validate_module_import(module_name: str) -> bool:
    """Validate if a module is in the whitelist before importing."""
    return module_name in ALLOWED_MODULES


def safe_import_module(module_name: str):
    """Safely import a module after validation."""
    if not validate_module_import(module_name):
            f"Module {module_name} is not in the allowed modules list"
        )
    return importlib.import_module(module_name)




class SystemValidator:
    def __init__(self):
        self.validation_results: Dict[str, Any] = {
            "python_version": None,
            "critical_dependencies": {},
            "gpu_availability": None,
            "system_tools": {},
            "compatibility_warnings": [],
        }

    def check_python_version(self) -> bool:
        """Validate Python version compatibility."""
        version = platform.python_version()
        major, minor = map(int, version.split(".")[:2])

        is_compatible = (3, 9) <= (major, minor) <= (3, 11)
        self.validation_results["python_version"] = {
            "current": version,
            "is_compatible": is_compatible,
            "recommended_range": "3.9 - 3.11",
        }

        if not is_compatible:
            self.validation_results["compatibility_warnings"].append(
                f"Python version {version} is outside recommended range"
            )

        return is_compatible

    def validate_dependencies(self) -> bool:
        """Validate critical dependencies and their versions."""
        critical_deps = [
            "torch",
            "numpy",
            "pandas",
            "fastapi",
            "sqlalchemy",
            "transformers",
            "langchain",
            "tensorflow",
            "chromadb",
        ]

        all_passed = True

        for dep in critical_deps:
            try:
                module = safe_import_module(dep)
                version = getattr(module, "__version__", "Unknown")

                self.validation_results["critical_dependencies"][dep] = {
                    "version": version,
                    "status": "✅ Installed",
                }
            except ImportError:
                self.validation_results["critical_dependencies"][dep] = {
                    "version": None,
                    "status": "❌ Not Installed",
                }
                all_passed = False
                self.validation_results["compatibility_warnings"].append(
                    f"{dep} is not installed"
                )

        return all_passed

    def check_gpu_availability(self) -> bool:
        """Check GPU availability for deep learning frameworks."""
        try:
            import torch

            cuda_available = torch.cuda.is_available()
            device_name = (
                torch.cuda.get_device_name(0) if cuda_available else None
            )

            self.validation_results["gpu_availability"] = {
                "cuda_available": cuda_available,
                "device_name": device_name,
            }

            return cuda_available
        except ImportError:
            self.validation_results["gpu_availability"] = {
                "cuda_available": False,
                "device_name": None,
            }
            self.validation_results["compatibility_warnings"].append(
                "PyTorch GPU support not available"
            )
            return False

    def _run_command(self, command: List[str]) -> Tuple[str, int]:
        """Run command safely without shell=True."""
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,  # Don't raise exception on non-zero exit
            )
            return result.stdout.strip(), result.returncode
        except Exception as e:
            return str(e), 1

    def run_system_tool_checks(self) -> None:
        """Check availability and versions of system tools."""
        tools = [
            ("pytest", ["pytest", "--version"]),
            ("black", ["black", "--version"]),
            ("ruff", ["ruff", "--version"]),
            ("git", ["git", "--version"]),
        ]

        for name, cmd in tools:
            try:
                result = self._run_command(cmd)
                self.validation_results["system_tools"][name] = {
                    "version": result[0],
                    "status": (
                        "✅ Installed"
                        if result[1] == 0
                        else "❌ Not Installed"
                    ),
                }
            except Exception as e:
                self.validation_results["system_tools"][name] = {
                    "version": str(e),
                    "status": "❌ Not Installed",
                }
                self.validation_results["compatibility_warnings"].append(
                    f"{name} tool error: {e}"
                )

    def generate_report(self) -> None:
        """Generate a comprehensive validation report."""
        report_path = "system_validation_report.json"
        with open(report_path, "w") as f:
            json.dump(self.validation_results, f, indent=2)
        print(f"Validation report saved to {report_path}")

    def validate_system(self) -> bool:
        """Perform comprehensive system validation."""
        checks = [
            self.check_python_version(),
            self.validate_dependencies(),
            self.check_gpu_availability(),
        ]

        self.run_system_tool_checks()
        self.generate_report()

        return all(checks)


def main():
    validator = SystemValidator()
    is_system_valid = validator.validate_system()

    print("\n🔍 SutazAI System Validation Report")
    print("=" * 40)

    if is_system_valid:
        print("\n✨ System Ready for Deployment ✨")
        sys.exit(0)
    else:
        print("\n⚠️ System Validation Failed")
        print("Warnings and Issues:")
        for warning in validator.validation_results.get(
            "compatibility_warnings", []
        ):
            print(f" - {warning}")
        sys.exit(1)


if __name__ == "__main__":
    main()
