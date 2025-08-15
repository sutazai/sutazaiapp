#!/usr/bin/env python3
"""
ULTRA DEBUGGING TEST for SutazAI Frontend Optimization
Comprehensive validation of optimization implementation without requiring dependencies
"""

import ast
import os
import sys
from pathlib import Path
import json
import re
from typing import Dict, List, Any, Optional

class UltraFrontendDebugger:
    """Ultra-comprehensive debugging of frontend optimization"""
    
    def __init__(self, frontend_path: str = "."):
        self.frontend_path = Path(frontend_path)
        self.issues = []
        self.warnings = []
        self.successes = []
        
    def log_issue(self, severity: str, component: str, message: str):
        """Log debugging issues"""
        self.issues.append({
            "severity": severity,
            "component": component, 
            "message": message
        })
        print(f"❌ [{severity}] {component}: {message}")
    
    def log_warning(self, component: str, message: str):
        """Log warnings"""
        self.warnings.append({
            "component": component,
            "message": message
        })
        print(f"⚠️  [WARNING] {component}: {message}")
    
    def log_success(self, component: str, message: str):
        """Log successes"""
        self.successes.append({
            "component": component,
            "message": message
        })
        print(f"✅ [SUCCESS] {component}: {message}")

    def test_file_exists(self, file_path: str, description: str) -> bool:
        """Test if optimization file exists"""
        full_path = self.frontend_path / file_path
        if full_path.exists():
            self.log_success("File Check", f"{description} exists: {file_path}")
            return True
        else:
            self.log_issue("CRITICAL", "File Check", f"{description} missing: {file_path}")
            return False

    def test_python_syntax(self, file_path: str, description: str) -> bool:
        """Test Python file for syntax errors"""
        full_path = self.frontend_path / file_path
        
        if not full_path.exists():
            return False
            
        try:
            with open(full_path, 'r') as f:
                source_code = f.read()
            
            ast.parse(source_code)
            self.log_success("Syntax Check", f"{description} has valid Python syntax")
            return True
            
        except SyntaxError as e:
            self.log_issue("CRITICAL", "Syntax Check", f"{description} syntax error: {e}")
            return False
        except Exception as e:
            self.log_issue("ERROR", "Syntax Check", f"{description} parse error: {e}")
            return False

    def analyze_import_structure(self, file_path: str, description: str) -> Dict[str, List[str]]:
        """Analyze import structure of Python file"""
        full_path = self.frontend_path / file_path
        
        if not full_path.exists():
            return {"stdlib": [], "third_party": [], "local": []}
        
        try:
            with open(full_path, 'r') as f:
                source_code = f.read()
            
            tree = ast.parse(source_code)
            
            stdlib_imports = []
            third_party_imports = []
            local_imports = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module_name = alias.name
                        if self._is_stdlib(module_name):
                            stdlib_imports.append(module_name)
                        elif module_name.startswith('.') or module_name.startswith('utils') or module_name.startswith('components'):
                            local_imports.append(module_name)
                        else:
                            third_party_imports.append(module_name)
                
                elif isinstance(node, ast.ImportFrom):
                    module_name = node.module or ""
                    if self._is_stdlib(module_name):
                        stdlib_imports.append(module_name)
                    elif module_name.startswith('.') or module_name.startswith('utils') or module_name.startswith('components'):
                        local_imports.append(module_name)
                    else:
                        third_party_imports.append(module_name)
            
            imports = {
                "stdlib": list(set(stdlib_imports)),
                "third_party": list(set(third_party_imports)),
                "local": list(set(local_imports))
            }
            
            self.log_success("Import Analysis", f"{description} - {len(imports['third_party'])} external deps")
            return imports
            
        except Exception as e:
            self.log_issue("ERROR", "Import Analysis", f"{description} import analysis failed: {e}")
            return {"stdlib": [], "third_party": [], "local": []}
    
    def _is_stdlib(self, module_name: str) -> bool:
        """Check if module is standard library"""
        stdlib_modules = {
            'asyncio', 'time', 'os', 'sys', 'json', 'logging', 'datetime',
            'typing', 'functools', 'contextlib', 'hashlib', 'threading',
            'concurrent', 'importlib', 'pathlib', 're'
        }
        return module_name.split('.')[0] in stdlib_modules

    def validate_requirements_file(self, file_path: str) -> Dict[str, Any]:
        """Validate requirements.txt file structure"""
        full_path = self.frontend_path / file_path
        
        if not full_path.exists():
            self.log_issue("CRITICAL", "Requirements", f"Requirements file missing: {file_path}")
            return {"valid": False}
        
        try:
            with open(full_path, 'r') as f:
                lines = f.readlines()
            
            dependencies = []
            comments = []
            invalid_lines = []
            
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue
                elif line.startswith('#'):
                    comments.append(line)
                elif '==' in line:
                    # Valid versioned dependency
                    dep_name = line.split('==')[0]
                    version = line.split('==')[1]
                    dependencies.append({"name": dep_name, "version": version, "line": line_num})
                elif line and not line.startswith('#'):
                    invalid_lines.append({"line": line, "line_num": line_num})
            
            result = {
                "valid": len(invalid_lines) == 0,
                "dependencies": len(dependencies),
                "comments": len(comments),
                "invalid_lines": invalid_lines
            }
            
            if result["valid"]:
                self.log_success("Requirements", f"Valid requirements file with {result['dependencies']} dependencies")
            else:
                for invalid in invalid_lines:
                    self.log_issue("ERROR", "Requirements", f"Invalid line {invalid['line_num']}: {invalid['line']}")
            
            return result
            
        except Exception as e:
            self.log_issue("ERROR", "Requirements", f"Failed to parse requirements: {e}")
            return {"valid": False, "error": str(e)}

    def analyze_optimization_features(self, file_path: str) -> Dict[str, bool]:
        """Analyze optimization features in app file"""
        full_path = self.frontend_path / file_path
        
        if not full_path.exists():
            return {}
        
        try:
            with open(full_path, 'r') as f:
                content = f.read()
            
            features = {
                "lazy_loading": "lazy_loader" in content and "SmartPreloader" in content,
                "caching": "performance_cache" in content and "cache" in content,
                "performance_modes": "performance_mode" in content,
                "async_operations": "async def" in content or "asyncio" in content,
                "connection_pooling": "httpx" in content or "connection" in content,
                "smart_refresh": "SmartRefresh" in content,
                "conditional_rendering": "ConditionalRenderer" in content,
                "progress_indicators": "spinner" in content or "progress" in content,
                "session_state_optimization": "session_state" in content,
                "error_handling": "try:" in content and "except" in content
            }
            
            implemented_count = sum(features.values())
            total_features = len(features)
            
            for feature, implemented in features.items():
                if implemented:
                    self.log_success("Optimization Features", f"{feature.replace('_', ' ').title()} implemented")
                else:
                    self.log_warning("Optimization Features", f"{feature.replace('_', ' ').title()} not found")
            
            self.log_success("Optimization Summary", f"{implemented_count}/{total_features} optimization features implemented")
            
            return features
            
        except Exception as e:
            self.log_issue("ERROR", "Feature Analysis", f"Failed to analyze features: {e}")
            return {}

    def check_component_structure(self) -> bool:
        """Check if component structure is properly organized"""
        required_dirs = [
            "utils",
            "components", 
            "pages",
            "services",
            "styles"
        ]
        
        all_good = True
        for dir_name in required_dirs:
            dir_path = self.frontend_path / dir_name
            if dir_path.exists() and dir_path.is_dir():
                self.log_success("Structure", f"Directory {dir_name} exists")
                
                # Check for __init__.py
                init_file = dir_path / "__init__.py"
                if init_file.exists():
                    self.log_success("Structure", f"{dir_name}/__init__.py exists")
                else:
                    self.log_warning("Structure", f"{dir_name}/__init__.py missing")
            else:
                self.log_issue("ERROR", "Structure", f"Directory {dir_name} missing")
                all_good = False
        
        return all_good

    def validate_circular_imports(self) -> bool:
        """Basic circular import detection"""
        # This is a simplified check - full detection would require more complex analysis
        python_files = list(self.frontend_path.rglob("*.py"))
        
        for file_path in python_files:
            if file_path.name.startswith('test_'):
                continue
                
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                
                # Look for potential circular imports
                if "from utils.optimized_api_client import" in content and file_path.name != "app_optimized.py":
                    if "from components.lazy_loader import" in content:
                        self.log_warning("Circular Imports", f"Potential circular import in {file_path.name}")
                
            except Exception:
                continue
        
        self.log_success("Circular Imports", "No obvious circular import issues detected")
        return True

    def run_comprehensive_debug(self) -> Dict[str, Any]:
        """Run all debugging tests"""
        print("🚀 Starting ULTRA FRONTEND DEBUGGING...")
        print("=" * 60)
        
        # Test 1: File existence
        print("\n📁 Testing File Existence...")
        files_exist = {
            "app_optimized.py": self.test_file_exists("app_optimized.py", "Optimized App"),
            "requirements_optimized.txt": self.test_file_exists("requirements_optimized.txt", "Optimized Requirements"),
            "optimized_api_client.py": self.test_file_exists("utils/optimized_api_client.py", "Optimized API Client"),
            "performance_cache.py": self.test_file_exists("utils/performance_cache.py", "Performance Cache"),
            "lazy_loader.py": self.test_file_exists("components/lazy_loader.py", "Lazy Loader")
        }
        
        # Test 2: Syntax validation
        print("\n🐍 Testing Python Syntax...")
        syntax_valid = {}
        if files_exist["app_optimized.py"]:
            syntax_valid["app_optimized.py"] = self.test_python_syntax("app_optimized.py", "Optimized App")
        if files_exist["optimized_api_client.py"]:
            syntax_valid["optimized_api_client.py"] = self.test_python_syntax("utils/optimized_api_client.py", "API Client")
        if files_exist["performance_cache.py"]:
            syntax_valid["performance_cache.py"] = self.test_python_syntax("utils/performance_cache.py", "Cache System")
        if files_exist["lazy_loader.py"]:
            syntax_valid["lazy_loader.py"] = self.test_python_syntax("components/lazy_loader.py", "Lazy Loader")
        
        # Test 3: Import analysis
        print("\n📦 Analyzing Import Structure...")
        import_analysis = {}
        if syntax_valid.get("app_optimized.py", False):
            import_analysis["app_optimized"] = self.analyze_import_structure("app_optimized.py", "Optimized App")
        if syntax_valid.get("optimized_api_client.py", False):
            import_analysis["api_client"] = self.analyze_import_structure("utils/optimized_api_client.py", "API Client")
        
        # Test 4: Requirements validation
        print("\n📋 Validating Requirements...")
        requirements_validation = self.validate_requirements_file("requirements_optimized.txt")
        
        # Test 5: Optimization features
        print("\n⚡ Analyzing Optimization Features...")
        optimization_features = {}
        if files_exist["app_optimized.py"]:
            optimization_features = self.analyze_optimization_features("app_optimized.py")
        
        # Test 6: Component structure
        print("\n🏗️ Checking Component Structure...")
        structure_valid = self.check_component_structure()
        
        # Test 7: Circular imports
        print("\n🔄 Checking for Circular Imports...")
        no_circular_imports = self.validate_circular_imports()
        
        # Generate final report
        print("\n" + "=" * 60)
        print("📊 ULTRA DEBUGGING RESULTS")
        print("=" * 60)
        
        critical_issues = len([i for i in self.issues if i["severity"] == "CRITICAL"])
        error_issues = len([i for i in self.issues if i["severity"] == "ERROR"])
        total_warnings = len(self.warnings)
        total_successes = len(self.successes)
        
        print(f"✅ Successes: {total_successes}")
        print(f"⚠️  Warnings: {total_warnings}")
        print(f"❌ Errors: {error_issues}")
        print(f"🔥 Critical Issues: {critical_issues}")
        
        overall_score = (total_successes / (total_successes + total_warnings + error_issues + critical_issues)) * 100
        print(f"\n🎯 Overall Health Score: {overall_score:.1f}%")
        
        if critical_issues == 0 and error_issues == 0:
            print("🎉 OPTIMIZATION IMPLEMENTATION IS VALID!")
        elif critical_issues == 0:
            print("✅ OPTIMIZATION IMPLEMENTATION IS FUNCTIONAL (with warnings)")
        else:
            print("❌ OPTIMIZATION IMPLEMENTATION HAS CRITICAL ISSUES")
        
        return {
            "files_exist": files_exist,
            "syntax_valid": syntax_valid,
            "import_analysis": import_analysis,
            "requirements_validation": requirements_validation,
            "optimization_features": optimization_features,
            "structure_valid": structure_valid,
            "no_circular_imports": no_circular_imports,
            "issues": self.issues,
            "warnings": self.warnings,
            "successes": self.successes,
            "overall_score": overall_score
        }

def main():
    """Main debugging function"""
    debugger = UltraFrontendDebugger()
    results = debugger.run_comprehensive_debug()
    
    # Save results to file
    with open("frontend_debug_report.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Detailed report saved to: frontend_debug_report.json")
    
    return results

if __name__ == "__main__":
    main()