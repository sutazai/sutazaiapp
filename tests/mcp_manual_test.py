#!/usr/bin/env python3
"""
Manual MCP Testing Script - Direct testing without backend dependency
Tests wrapper scripts directly and documents real functionality
"""

import subprocess
import json
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MCPManualTester:
    """Direct MCP wrapper script testing"""
    
    def __init__(self):
        self.wrapper_dir = Path("/opt/sutazaiapp/scripts/mcp/wrappers")
        self.results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "wrapper_tests": {},
            "summary": {},
            "detailed_analysis": {}
        }
    
    def test_wrapper_script_detailed(self, script_path: Path) -> Dict[str, Any]:
        """Test wrapper script with multiple methods"""
        script_name = script_path.stem
        logger.info(f"Testing wrapper script: {script_name}")
        
        test_result = {
            "script": script_name,
            "path": str(script_path),
            "exists": script_path.exists(),
            "executable": script_path.exists() and script_path.stat().st_mode & 0o111,
            "tests": {}
        }
        
        if not test_result["exists"]:
            test_result["error"] = "Script file not found"
            return test_result
        
        if not test_result["executable"]:
            test_result["error"] = "Script not executable"
            return test_result
        
        # Test multiple methods
        test_methods = [
            {"name": "list", "payload": '{"method":"list"}'},
            {"name": "ping", "payload": '{"method":"ping"}'},
            {"name": "capabilities", "payload": '{"method":"capabilities"}'},
            {"name": "selfcheck", "command": ["--selfcheck"]},
            {"name": "help", "command": ["--help"]},
        ]
        
        for test_method in test_methods:
            method_result = self._execute_test_method(script_path, test_method)
            test_result["tests"][test_method["name"]] = method_result
        
        # Determine overall status
        successful_tests = sum(1 for test in test_result["tests"].values() 
                             if test.get("status") == "success")
        test_result["overall_status"] = "working" if successful_tests > 0 else "broken"
        test_result["success_count"] = successful_tests
        test_result["total_tests"] = len(test_methods)
        
        return test_result
    
    def _execute_test_method(self, script_path: Path, test_method: Dict) -> Dict[str, Any]:
        """Execute a single test method"""
        method_result = {
            "method": test_method["name"],
            "status": "unknown",
            "execution_time": None,
            "return_code": None,
            "stdout": None,
            "stderr": None,
            "error": None
        }
        
        try:
            start_time = time.time()
            
            if "payload" in test_method:
                # Test with JSON input
                process = subprocess.run(
                    [str(script_path)],
                    input=test_method["payload"],
                    text=True,
                    capture_output=True,
                    timeout=15
                )
            elif "command" in test_method:
                # Test with command line arguments
                process = subprocess.run(
                    [str(script_path)] + test_method["command"],
                    text=True,
                    capture_output=True,
                    timeout=15
                )
            else:
                # Basic execution
                process = subprocess.run(
                    [str(script_path)],
                    text=True,
                    capture_output=True,
                    timeout=15
                )
            
            execution_time = time.time() - start_time
            
            method_result.update({
                "status": "success" if process.returncode == 0 else "failed",
                "execution_time": execution_time,
                "return_code": process.returncode,
                "stdout": process.stdout[:1000] if process.stdout else None,
                "stderr": process.stderr[:500] if process.stderr else None
            })
            
            # Try to parse JSON output
            if process.stdout and process.stdout.strip():
                try:
                    json_output = json.loads(process.stdout)
                    method_result["json_response"] = json_output
                    method_result["response_type"] = "json"
                except json.JSONDecodeError:
                    method_result["response_type"] = "text"
            
        except subprocess.TimeoutExpired:
            method_result.update({
                "status": "timeout",
                "error": "Execution timeout after 15 seconds"
            })
        except FileNotFoundError:
            method_result.update({
                "status": "not_found",
                "error": "Script file not found"
            })
        except PermissionError:
            method_result.update({
                "status": "permission_denied",
                "error": "Permission denied executing script"
            })
        except Exception as e:
            method_result.update({
                "status": "error",
                "error": str(e)
            })
        
        return method_result
    
    def analyze_script_dependencies(self, script_path: Path) -> Dict[str, Any]:
        """Analyze script dependencies and requirements"""
        analysis = {
            "script": script_path.stem,
            "dependencies": [],
            "requirements": [],
            "shell_type": "unknown",
            "npm_commands": [],
            "python_commands": [],
            "system_commands": []
        }
        
        if not script_path.exists():
            return analysis
        
        try:
            with open(script_path, 'r') as f:
                content = f.read()
            
            # Analyze shebang
            lines = content.split('\n')
            if lines and lines[0].startswith('#!'):
                analysis["shell_type"] = lines[0]
            
            # Look for common patterns
            if 'npx' in content:
                analysis["npm_commands"] = [line.strip() for line in lines 
                                          if 'npx' in line and not line.strip().startswith('#')]
            
            if 'python' in content:
                analysis["python_commands"] = [line.strip() for line in lines 
                                             if 'python' in line and not line.strip().startswith('#')]
            
            # Look for external commands
            common_commands = ['curl', 'wget', 'git', 'docker', 'ssh']
            for cmd in common_commands:
                if cmd in content:
                    analysis["system_commands"].append(cmd)
            
            # Look for package requirements
            if '@' in content:
                analysis["npm_packages"] = [word for word in content.split() 
                                          if word.startswith('@') or '/' in word]
            
        except Exception as e:
            analysis["analysis_error"] = str(e)
        
        return analysis
    
    def run_comprehensive_manual_test(self) -> Dict[str, Any]:
        """Run comprehensive manual testing of all wrapper scripts"""
        logger.info("Starting comprehensive manual MCP wrapper testing...")
        
        if not self.wrapper_dir.exists():
            self.results["error"] = f"Wrapper directory not found: {self.wrapper_dir}"
            return self.results
        
        # Find all wrapper scripts
        wrapper_scripts = list(self.wrapper_dir.glob("*.sh"))
        logger.info(f"Found {len(wrapper_scripts)} wrapper scripts")
        
        # Test each script
        for script_path in wrapper_scripts:
            script_result = self.test_wrapper_script_detailed(script_path)
            script_analysis = self.analyze_script_dependencies(script_path)
            
            script_result["dependency_analysis"] = script_analysis
            self.results["wrapper_tests"][script_path.stem] = script_result
        
        # Generate detailed analysis
        self._generate_detailed_analysis()
        
        # Generate summary
        self._generate_summary()
        
        return self.results
    
    def _generate_detailed_analysis(self):
        """Generate detailed analysis of test results"""
        wrapper_tests = self.results["wrapper_tests"]
        
        # Categorize scripts by status
        working_scripts = []
        broken_scripts = []
        timeout_scripts = []
        permission_scripts = []
        
        for script_name, result in wrapper_tests.items():
            if result.get("overall_status") == "working":
                working_scripts.append(script_name)
            elif any(test.get("status") == "timeout" for test in result.get("tests", {}).values()):
                timeout_scripts.append(script_name)
            elif any(test.get("status") == "permission_denied" for test in result.get("tests", {}).values()):
                permission_scripts.append(script_name)
            else:
                broken_scripts.append(script_name)
        
        # Analyze common patterns
        npm_scripts = []
        python_scripts = []
        shell_scripts = []
        
        for script_name, result in wrapper_tests.items():
            analysis = result.get("dependency_analysis", {})
            if analysis.get("npm_commands"):
                npm_scripts.append(script_name)
            if analysis.get("python_commands"):
                python_scripts.append(script_name)
            if "bash" in analysis.get("shell_type", "").lower():
                shell_scripts.append(script_name)
        
        self.results["detailed_analysis"] = {
            "categorization": {
                "working": working_scripts,
                "broken": broken_scripts,
                "timeout": timeout_scripts,
                "permission_issues": permission_scripts
            },
            "technology_breakdown": {
                "npm_based": npm_scripts,
                "python_based": python_scripts,
                "shell_based": shell_scripts
            },
            "statistics": {
                "total_scripts": len(wrapper_tests),
                "working_count": len(working_scripts),
                "broken_count": len(broken_scripts),
                "timeout_count": len(timeout_scripts),
                "permission_count": len(permission_scripts)
            }
        }
    
    def _generate_summary(self):
        """Generate test summary statistics"""
        wrapper_tests = self.results["wrapper_tests"]
        analysis = self.results["detailed_analysis"]
        
        total_scripts = len(wrapper_tests)
        working_scripts = len(analysis["categorization"]["working"])
        
        self.results["summary"] = {
            "total_wrapper_scripts": total_scripts,
            "working_scripts": working_scripts,
            "broken_scripts": total_scripts - working_scripts,
            "success_rate": (working_scripts / total_scripts * 100) if total_scripts > 0 else 0,
            "test_timestamp": datetime.now(timezone.utc).isoformat(),
            "key_findings": self._generate_key_findings()
        }
    
    def _generate_key_findings(self) -> List[str]:
        """Generate key findings from test results"""
        findings = []
        analysis = self.results["detailed_analysis"]
        stats = analysis["statistics"]
        
        if stats["working_count"] > 0:
            findings.append(f"{stats['working_count']} wrapper scripts are functional")
        
        if stats["broken_count"] > 0:
            findings.append(f"{stats['broken_count']} wrapper scripts have issues")
        
        if stats["timeout_count"] > 0:
            findings.append(f"{stats['timeout_count']} scripts experience timeouts")
        
        if stats["permission_count"] > 0:
            findings.append(f"{stats['permission_count']} scripts have permission issues")
        
        # Technology findings
        tech = analysis["technology_breakdown"]
        if tech["npm_based"]:
            findings.append(f"{len(tech['npm_based'])} scripts use NPM/Node.js")
        
        if tech["python_based"]:
            findings.append(f"{len(tech['python_based'])} scripts use Python")
        
        return findings
    
    def save_results(self, output_path: str = "/opt/sutazaiapp/tests/results/mcp_manual_test_results.json"):
        """Save test results to JSON file"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Manual test results saved to: {output_file}")
        return str(output_file)
    
    def print_summary_report(self):
        """Print a human-readable summary report"""
        summary = self.results["summary"]
        analysis = self.results["detailed_analysis"]
        
        print("\n" + "="*80)
        print("MCP WRAPPER SCRIPTS - MANUAL TEST RESULTS")
        print("="*80)
        print(f"Test Date: {summary['test_timestamp']}")
        print(f"Total Scripts: {summary['total_wrapper_scripts']}")
        print(f"Working Scripts: {summary['working_scripts']}")
        print(f"Broken Scripts: {summary['broken_scripts']}")
        print(f"Success Rate: {summary['success_rate']:.1f}%")
        print()
        
        print("WORKING SCRIPTS:")
        for script in analysis["categorization"]["working"]:
            print(f"  ✅ {script}")
        print()
        
        print("BROKEN SCRIPTS:")
        for script in analysis["categorization"]["broken"]:
            print(f"  ❌ {script}")
        print()
        
        if analysis["categorization"]["timeout"]:
            print("TIMEOUT ISSUES:")
            for script in analysis["categorization"]["timeout"]:
                print(f"  ⏱️  {script}")
            print()
        
        if analysis["categorization"]["permission_issues"]:
            print("PERMISSION ISSUES:")
            for script in analysis["categorization"]["permission_issues"]:
                print(f"  🚫 {script}")
            print()
        
        print("KEY FINDINGS:")
        for finding in summary["key_findings"]:
            print(f"  • {finding}")
        
        print("="*80)

def main():
    """Main execution function"""
    tester = MCPManualTester()
    results = tester.run_comprehensive_manual_test()
    output_file = tester.save_results()
    tester.print_summary_report()
    
    return results

if __name__ == "__main__":
    main()