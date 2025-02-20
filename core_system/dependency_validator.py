#!/usr/bin/env python3
import sys
import importlib
import platform
import subprocess
import pkg_resources
import json
from typing import Dict, Any

class SystemValidator:
    def __init__(self):
        self.validation_results: Dict[str, Any] = {
            "python_version": None,
            "critical_dependencies": {},
            "gpu_availability": None,
            "system_tools": {},
            "compatibility_warnings": []
        }

    def check_python_version(self) -> bool:
        """Validate Python version compatibility."""
        version = platform.python_version()
        major, minor = map(int, version.split('.')[:2])
        
        is_compatible = (3, 9) <= (major, minor) <= (3, 11)
        self.validation_results["python_version"] = {
            "current": version,
            "is_compatible": is_compatible,
            "recommended_range": "3.9 - 3.11"
        }
        
        if not is_compatible:
            self.validation_results["compatibility_warnings"].append(
                f"Python version {version} is outside recommended range"
            )
        
        return is_compatible

    def validate_dependencies(self) -> bool:
        """Validate critical dependencies and their versions."""
        critical_deps = [
            'torch', 'numpy', 'pandas', 'fastapi', 
            'sqlalchemy', 'transformers', 'langchain',
            'tensorflow', 'chromadb'
        ]
        
        all_passed = True
        
        for dep in critical_deps:
            try:
                module = importlib.import_module(dep)
                version = getattr(module, '__version__', 'Unknown')
                
                self.validation_results["critical_dependencies"][dep] = {
                    "version": version,
                    "status": "✅ Installed"
                }
            except ImportError:
                self.validation_results["critical_dependencies"][dep] = {
                    "version": None,
                    "status": "❌ Not Installed"
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
            device_name = torch.cuda.get_device_name(0) if cuda_available else None
            
            self.validation_results["gpu_availability"] = {
                "cuda_available": cuda_available,
                "device_name": device_name
            }
            
            return cuda_available
        except ImportError:
            self.validation_results["gpu_availability"] = {
                "cuda_available": False,
                "device_name": None
            }
            self.validation_results["compatibility_warnings"].append(
                "PyTorch GPU support not available"
            )
            return False

    def run_system_tool_checks(self) -> None:
        """Check availability and versions of system tools."""
        tools = [
            ('pytest', ['pytest', '--version']),
            ('black', ['black', '--version']),
            ('ruff', ['ruff', '--version']),
            ('git', ['git', '--version'])
        ]
        
        for name, cmd in tools:
            try:
                result = subprocess.run(cmd, capture_output=True, text=True)
                self.validation_results["system_tools"][name] = {
                    "version": result.stdout.strip(),
                    "status": "✅ Installed"
                }
            except FileNotFoundError:
                self.validation_results["system_tools"][name] = {
                    "version": None,
                    "status": "❌ Not Installed"
                }
                self.validation_results["compatibility_warnings"].append(
                    f"{name} tool not found"
                )

    def generate_report(self) -> None:
        """Generate a comprehensive validation report."""
        report_path = "system_validation_report.json"
        with open(report_path, 'w') as f:
            json.dump(self.validation_results, f, indent=2)
        print(f"Validation report saved to {report_path}")

    def validate_system(self) -> bool:
        """Perform comprehensive system validation."""
        checks = [
            self.check_python_version(),
            self.validate_dependencies(),
            self.check_gpu_availability()
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
        for warning in validator.validation_results.get("compatibility_warnings", []):
            print(f" - {warning}")
        sys.exit(1)

if __name__ == '__main__':
    main()