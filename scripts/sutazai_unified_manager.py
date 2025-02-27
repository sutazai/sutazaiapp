#!/usr/bin/env python3.11
"""
SutazAI Unified System Manager

This script consolidates functionality from multiple system management scripts
into a single, comprehensive solution.
"""

import ast
import importlib
import json
import logging
import os
import re
import subprocess
import sys
from datetime import datetime
from typing import Any, Dict, List

# Configure logging
logging.basicConfig(
level=logging.INFO,
format="%(asctime)s - %(levelname)s: %(message)s",
handlers=[
logging.FileHandler("/opt/sutazaiapp/logs/unified_manager.log"),
logging.StreamHandler(sys.stdout),
],
)
logger = logging.getLogger("SutazAI.UnifiedManager")


    class UnifiedSystemManager:
    """Unified system management framework"""
    
        def __init__(self, base_path: str = "/opt/sutazaiapp"):
        self.base_path = base_path
        self.log_dir = os.path.join(base_path, "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        
            def verify_python_version(self) -> bool:
            """Verify Python 3.11 compatibility"""
            major, minor = sys.version_info.major, sys.version_info.minor
                if major != 3 or minor != 11:
                logger.error(
                "Unsupported Python version. "
                f"Required: 3.11, Current: {major}.{minor}",
            )
            return False
            return True
            
                def find_python_files(self) -> List[str]:
                """Find all Python files in project"""
                python_files = []
                    for root, _, files in os.walk(self.base_path):
                        for file in files:
                            if file.endswith(".py"):
                            python_files.append(os.path.join(root, file))
                            return python_files
                            
                                def analyze_code_structure(self, file_path: str) -> Dict[str, Any]:
                                """Analyze code structure and complexity"""
                                    try:
                                    with open(file_path) as f:
                                    source = f.read()
                                    
                                    tree = ast.parse(source)
                                    structure = {
                                    "filename": os.path.basename(file_path),
                                    "classes": [],
                                    "functions": [],
                                    "imports": [],
                                    "complexity": {
                                    "total_lines": len(source.splitlines()),
                                    "class_count": 0,
                                    "function_count": 0,
                                    "import_count": 0,
                                    },
                                }
                                
                                    for node in ast.walk(tree):
                                        if isinstance(node, ast.ClassDef):
                                        class_info = {
                                        "name": node.name,
                                        "methods": [
                                        method.name
                                            for method in node.body
                                                if isinstance(method, ast.FunctionDef)
                                                ],
                                            }
                                            structure["classes"].append(class_info)
                                            structure["complexity"]["class_count"] += 1
                                                elif isinstance(node, ast.FunctionDef):
                                                structure["functions"].append(
                                                {
                                                "name": node.name,
                                                "args": [arg.arg for arg in node.args.args],
                                                "line_number": node.lineno,
                                            }
                                        )
                                        structure["complexity"]["function_count"] += 1
                                            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                                                if isinstance(node, ast.Import):
                                                imports = [alias.name for alias in node.names]
                                                    else:
                                                    imports = [
                                                    f"{node.module}.{alias.name}" for alias in node.names
                                                ]
                                                structure["imports"].extend(imports)
                                                structure["complexity"]["import_count"] += len(imports)
                                                
                                                return structure
                                                except Exception as e:
                                                logger.error(f"Error analyzing {file_path}: {e}")
                                                return {}
                                                
                                                    def fix_code_issues(self, file_path: str) -> bool:
                                                    """Fix common code issues"""
                                                        try:
                                                        with open(file_path) as f:
                                                        content = f.read()
                                                        
                                                        # Fix imports
                                                        content = re.sub(
                                                        r"import\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*,\s*" r"([a-zA-Z_][a-zA-Z0-9_]*)",
                                                        r"import \1\nimport \2",
                                                        content,
                                                    )
                                                    
                                                    # Fix logging
                                                    content = re.sub(
                                                    r"logger\.(info|warning|error|debug)\((.*?)\)",
                                                    lambda m: f"logger.{m.group(1)}(f{m.group(2)})",
                                                    content,
                                                )
                                                
                                                # Fix exception handling
                                                content = re.sub(
                                                r"except Exception:\n\s*logger\.error\((.*?)\)",
                                                r"except Exception as e:\n    logger.exception(\1)",
                                                content,
                                            )
                                            
                                                if content != open(file_path).read():
                                                with open(file_path, "w") as f:
                                                f.write(content)
                                                return True
                                                return False
                                                
                                                except Exception as e:
                                                logger.error(f"Error fixing {file_path}: {e}")
                                                return False
                                                
                                                    def run_security_checks(self, file_path: str) -> List[Dict[str, Any]]:
                                                    """Run security checks using bandit"""
                                                        try:
                                                        result = subprocess.run(
                                                        ["bandit", "-r", file_path],
                                                        capture_output=True,
                                                        text=True,
                                                        check=False,
                                                    )
                                                    return self._parse_bandit_output(result.stdout)
                                                    except Exception as e:
                                                    logger.error(f"Security check failed for {file_path}: {e}")
                                                    return []
                                                    
                                                        def _parse_bandit_output(self, output: str) -> List[Dict[str, Any]]:
                                                        """Parse bandit output into structured format"""
                                                        issues = []
                                                            for line in output.split("\n"):
                                                                if ">> Issue: [" in line:
                                                                parts = line.split("]")
                                                                    if len(parts) >= 2:
                                                                    issues.append(
                                                                    {
                                                                    "type": parts[0].split("[")[1].strip(),
                                                                    "description": parts[1].strip(),
                                                                }
                                                            )
                                                            return issues
                                                            
                                                                def optimize_performance(self) -> Dict[str, Any]:
                                                                """Optimize system performance"""
                                                                    try:
                                                                    import psutil
                                                                    
                                                                    # Collect system metrics
                                                                    cpu_percent = psutil.cpu_percent(interval=1)
                                                                    memory = psutil.virtual_memory()
                                                                    disk = psutil.disk_usage("/")
                                                                    
                                                                    return {
                                                                    "cpu_usage": cpu_percent,
                                                                    "memory_usage": {
                                                                    "total": memory.total,
                                                                    "available": memory.available,
                                                                    "percent": memory.percent,
                                                                    },
                                                                    "disk_usage": {
                                                                    "total": disk.total,
                                                                    "used": disk.used,
                                                                    "free": disk.free,
                                                                    "percent": disk.percent,
                                                                    },
                                                                }
                                                                except Exception as e:
                                                                logger.error(f"Performance optimization failed: {e}")
                                                                return {}
                                                                
                                                                    def comprehensive_system_check(self) -> Dict[str, Any]:
                                                                    """Run comprehensive system check"""
                                                                        if not self.verify_python_version():
                                                                        logger.warning("Python version check failed")
                                                                        
                                                                        python_files = self.find_python_files()
                                                                        report = {
                                                                        "timestamp": datetime.now().isoformat(),
                                                                        "total_files": len(python_files),
                                                                        "code_structure": [],
                                                                        "security_issues": [],
                                                                        "performance": self.optimize_performance(),
                                                                    }
                                                                    
                                                                        for file_path in python_files:
                                                                        # Analyze code structure
                                                                        structure = self.analyze_code_structure(file_path)
                                                                        report["code_structure"].append(structure)
                                                                        
                                                                        # Fix code issues
                                                                            if self.fix_code_issues(file_path):
                                                                            logger.info(f"Fixed issues in {file_path}")
                                                                            
                                                                            # Security checks
                                                                            security_issues = self.run_security_checks(file_path)
                                                                                if security_issues:
                                                                                report["security_issues"].extend(security_issues)
                                                                                
                                                                                return report
                                                                                
                                                                                    def generate_report(self, report: Dict[str, Any]) -> None:
                                                                                    """Generate comprehensive report"""
                                                                                    report_path = os.path.join(
                                                                                    self.log_dir,
                                                                                    f"unified_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                                                                )
                                                                                
                                                                                with open(report_path, "w") as f:
                                                                                json.dump(report, f, indent=2)
                                                                                
                                                                                logger.info(f"Report generated: {report_path}")
                                                                                
                                                                                    def main(self) -> None:
                                                                                    """Main execution function"""
                                                                                    logger.info("Starting unified system management")
                                                                                    report = self.comprehensive_system_check()
                                                                                    self.generate_report(report)
                                                                                    logger.info("Unified system management completed")
                                                                                    
                                                                                    
                                                                                        def main():
                                                                                        """Main entry point"""
                                                                                        manager = UnifiedSystemManager()
                                                                                        manager.main()
                                                                                        
                                                                                        
                                                                                            if __name__ == "__main__":
                                                                                            main()
                                                                                            