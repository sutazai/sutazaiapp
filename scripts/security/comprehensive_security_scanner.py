#!/usr/bin/env python3
"""
COMPREHENSIVE SECURITY SCANNER
Multi-tool security analysis for SutazAI codebase
Version: 1.0.0 - Enterprise Security Validation
"""

import os
import sys
import json
import subprocess
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SecurityTool:
    """Base class for security tools"""
    
    def __init__(self, name: str, command: str, required: bool = True):
        self.name = name
        self.command = command
        self.required = required
        self.available = self._check_availability()
    
    def _check_availability(self) -> bool:
        """Check if tool is available in the system"""
        try:
            subprocess.run([self.command, '--version'], 
                         capture_output=True, timeout=10)
            return True
        except (subprocess.SubprocessError, FileNotFoundError):
            return False
    
    def run(self, target_path: str, **kwargs) -> Dict[str, Any]:
        """Run the security tool - to be implemented by subclasses"""
        raise NotImplementedError

class BanditScanner(SecurityTool):
    """Python security scanner using Bandit"""
    
    def __init__(self):
        super().__init__("Bandit", "bandit")
    
    def run(self, target_path: str, **kwargs) -> Dict[str, Any]:
        """Run Bandit security scan"""
        if not self.available:
            return {"status": "unavailable", "tool": self.name}
        
        output_file = kwargs.get('output_dir', '/tmp') + f'/bandit_report_{os.getpid()}.json'
        
        try:
            cmd = [
                'bandit',
                '-r', target_path,
                '-f', 'json',
                '-o', output_file,
                '--severity-level', 'medium',
                '--confidence-level', 'medium'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            # Read and parse results
            if os.path.exists(output_file):
                with open(output_file, 'r') as f:
                    data = json.load(f)
                os.unlink(output_file)  # Clean up
                
                return {
                    "status": "completed",
                    "tool": self.name,
                    "issues_count": len(data.get('results', [])),
                    "high_severity": len([r for r in data.get('results', []) 
                                       if r.get('issue_severity') == 'HIGH']),
                    "medium_severity": len([r for r in data.get('results', []) 
                                         if r.get('issue_severity') == 'MEDIUM']),
                    "results": data
                }
        except Exception as e:
            logger.error(f"Bandit scan failed: {e}")
            return {"status": "error", "tool": self.name, "error": str(e)}

class SafetyScanner(SecurityTool):
    """Dependency vulnerability scanner using Safety"""
    
    def __init__(self):
        super().__init__("Safety", "safety")
    
    def run(self, target_path: str, **kwargs) -> Dict[str, Any]:
        """Run Safety dependency scan"""
        if not self.available:
            return {"status": "unavailable", "tool": self.name}
        
        try:
            # Find requirements files
            requirements_files = []
            for root, dirs, files in os.walk(target_path):
                for file in files:
                    if file.startswith('requirements') and file.endswith('.txt'):
                        requirements_files.append(os.path.join(root, file))
            
            if not requirements_files:
                return {"status": "no_requirements", "tool": self.name}
            
            all_vulnerabilities = []
            
            for req_file in requirements_files:
                cmd = ['safety', 'check', '-r', req_file, '--json']
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                
                if result.stdout:
                    try:
                        vulnerabilities = json.loads(result.stdout)
                        all_vulnerabilities.extend(vulnerabilities)
                    except json.JSONDecodeError:
                        continue
            
            return {
                "status": "completed",
                "tool": self.name,
                "vulnerabilities_count": len(all_vulnerabilities),
                "requirements_files": requirements_files,
                "vulnerabilities": all_vulnerabilities
            }
            
        except Exception as e:
            logger.error(f"Safety scan failed: {e}")
            return {"status": "error", "tool": self.name, "error": str(e)}

class SemgrepScanner(SecurityTool):
    """Advanced static analysis using Semgrep"""
    
    def __init__(self):
        super().__init__("Semgrep", "semgrep", required=False)
    
    def run(self, target_path: str, **kwargs) -> Dict[str, Any]:
        """Run Semgrep security analysis"""
        if not self.available:
            return {"status": "unavailable", "tool": self.name}
        
        try:
            cmd = [
                'semgrep',
                '--config=auto',
                '--json',
                '--severity=ERROR',
                '--severity=WARNING',
                target_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.stdout:
                try:
                    data = json.loads(result.stdout)
                    results = data.get('results', [])
                    
                    return {
                        "status": "completed",
                        "tool": self.name,
                        "findings_count": len(results),
                        "error_count": len([r for r in results if r.get('extra', {}).get('severity') == 'ERROR']),
                        "warning_count": len([r for r in results if r.get('extra', {}).get('severity') == 'WARNING']),
                        "results": data
                    }
                except json.JSONDecodeError:
                    pass
            
            return {"status": "no_findings", "tool": self.name}
            
        except Exception as e:
            logger.error(f"Semgrep scan failed: {e}")
            return {"status": "error", "tool": self.name, "error": str(e)}

class SecretsScanner(SecurityTool):
    """Secrets detection using detect-secrets"""
    
    def __init__(self):
        super().__init__("detect-secrets", "detect-secrets", required=False)
    
    def run(self, target_path: str, **kwargs) -> Dict[str, Any]:
        """Run secrets detection scan"""
        if not self.available:
            return {"status": "unavailable", "tool": self.name}
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
                baseline_file = tmp.name
            
            # Create baseline
            cmd_baseline = [
                'detect-secrets', 'scan',
                '--all-files',
                target_path
            ]
            
            result = subprocess.run(cmd_baseline, capture_output=True, text=True, timeout=180)
            
            if result.stdout:
                with open(baseline_file, 'w') as f:
                    f.write(result.stdout)
                
                try:
                    data = json.loads(result.stdout)
                    secrets_count = sum(len(secrets) for secrets in data.get('results', {}).values())
                    
                    return {
                        "status": "completed",
                        "tool": self.name,
                        "secrets_count": secrets_count,
                        "files_scanned": len(data.get('results', {})),
                        "results": data
                    }
                except json.JSONDecodeError:
                    pass
            
            return {"status": "no_secrets", "tool": self.name}
            
        except Exception as e:
            logger.error(f"Secrets scan failed: {e}")
            return {"status": "error", "tool": self.name, "error": str(e)}
        finally:
            if os.path.exists(baseline_file):
                os.unlink(baseline_file)

class DockerScanner:
    """Docker security scanner for Dockerfiles and images"""
    
    def __init__(self):
        self.name = "Docker Security Scanner"
    
    def scan_dockerfiles(self, target_path: str) -> Dict[str, Any]:
        """Scan Dockerfiles for security issues"""
        dockerfiles = []
        for root, dirs, files in os.walk(target_path):
            for file in files:
                if file.lower().startswith('dockerfile'):
                    dockerfiles.append(os.path.join(root, file))
        
        if not dockerfiles:
            return {"status": "no_dockerfiles", "tool": self.name}
        
        security_issues = []
        
        for dockerfile in dockerfiles:
            try:
                with open(dockerfile, 'r') as f:
                    content = f.read()
                
                file_issues = []
                
                # Check for security anti-patterns
                if 'USER root' in content:
                    file_issues.append({
                        "severity": "HIGH",
                        "issue": "Running as root user",
                        "description": "Container runs with root privileges"
                    })
                
                if '--privileged' in content:
                    file_issues.append({
                        "severity": "CRITICAL",
                        "issue": "Privileged container",
                        "description": "Container has elevated privileges"
                    })
                
                if 'ADD http' in content:
                    file_issues.append({
                        "severity": "MEDIUM",
                        "issue": "Using ADD with HTTP",
                        "description": "Use COPY instead of ADD for better security"
                    })
                
                if 'COPY . /' in content:
                    file_issues.append({
                        "severity": "MEDIUM",
                        "issue": "Copying entire context to root",
                        "description": "Avoid copying entire build context to container root"
                    })
                
                if file_issues:
                    security_issues.append({
                        "file": dockerfile,
                        "issues": file_issues
                    })
                    
            except Exception as e:
                logger.warning(f"Could not scan {dockerfile}: {e}")
        
        return {
            "status": "completed",
            "tool": self.name,
            "dockerfiles_scanned": len(dockerfiles),
            "issues_count": sum(len(item['issues']) for item in security_issues),
            "critical_issues": sum(
                len([issue for issue in item['issues'] if issue['severity'] == 'CRITICAL']) 
                for item in security_issues
            ),
            "high_issues": sum(
                len([issue for issue in item['issues'] if issue['severity'] == 'HIGH']) 
                for item in security_issues
            ),
            "results": security_issues
        }

class ComprehensiveSecurityScanner:
    """Main security scanner orchestrator"""
    
    def __init__(self, target_path: str, output_dir: str = None):
        self.target_path = Path(target_path).resolve()
        self.output_dir = output_dir or str(self.target_path / 'security_reports')
        self.scanners = [
            BanditScanner(),
            SafetyScanner(),
            SemgrepScanner(),
            SecretsScanner()
        ]
        self.docker_scanner = DockerScanner()
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
    
    def run_all_scans(self, parallel: bool = True) -> Dict[str, Any]:
        """Run all security scans"""
        logger.info(f"Starting comprehensive security scan of {self.target_path}")
        
        start_time = datetime.now()
        results = {
            "scan_info": {
                "target": str(self.target_path),
                "start_time": start_time.isoformat(),
                "scanner_version": "1.0.0"
            },
            "tool_results": {},
            "summary": {}
        }
        
        # Run security tool scans
        if parallel:
            results["tool_results"] = self._run_parallel_scans()
        else:
            results["tool_results"] = self._run_sequential_scans()
        
        # Run Docker security scan
        logger.info("Running Docker security scan...")
        results["tool_results"]["docker_security"] = self.docker_scanner.scan_dockerfiles(
            str(self.target_path)
        )
        
        # Generate summary
        results["summary"] = self._generate_summary(results["tool_results"])
        results["scan_info"]["end_time"] = datetime.now().isoformat()
        results["scan_info"]["duration_seconds"] = (
            datetime.now() - start_time
        ).total_seconds()
        
        # Save results
        self._save_results(results)
        
        logger.info("Comprehensive security scan completed")
        return results
    
    def _run_parallel_scans(self) -> Dict[str, Any]:
        """Run security scans in parallel"""
        results = {}
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_scanner = {
                executor.submit(scanner.run, str(self.target_path), output_dir=self.output_dir): scanner
                for scanner in self.scanners
            }
            
            for future in as_completed(future_to_scanner):
                scanner = future_to_scanner[future]
                try:
                    result = future.result()
                    results[scanner.name.lower().replace(' ', '_')] = result
                    logger.info(f"{scanner.name} scan completed: {result['status']}")
                except Exception as e:
                    logger.error(f"{scanner.name} scan failed: {e}")
                    results[scanner.name.lower().replace(' ', '_')] = {
                        "status": "error",
                        "tool": scanner.name,
                        "error": str(e)
                    }
        
        return results
    
    def _run_sequential_scans(self) -> Dict[str, Any]:
        """Run security scans sequentially"""
        results = {}
        
        for scanner in self.scanners:
            logger.info(f"Running {scanner.name} scan...")
            try:
                result = scanner.run(str(self.target_path), output_dir=self.output_dir)
                results[scanner.name.lower().replace(' ', '_')] = result
                logger.info(f"{scanner.name} scan completed: {result['status']}")
            except Exception as e:
                logger.error(f"{scanner.name} scan failed: {e}")
                results[scanner.name.lower().replace(' ', '_')] = {
                    "status": "error",
                    "tool": scanner.name,
                    "error": str(e)
                }
        
        return results
    
    def _generate_summary(self, tool_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive security summary"""
        summary = {
            "total_tools_run": len(tool_results),
            "successful_scans": 0,
            "failed_scans": 0,
            "critical_issues": 0,
            "high_issues": 0,
            "medium_issues": 0,
            "vulnerabilities": 0,
            "secrets_detected": 0,
            "recommendations": []
        }
        
        for tool_name, result in tool_results.items():
            if result.get("status") == "completed":
                summary["successful_scans"] += 1
            else:
                summary["failed_scans"] += 1
            
            # Aggregate security metrics
            if tool_name == "bandit":
                summary["high_issues"] += result.get("high_severity", 0)
                summary["medium_issues"] += result.get("medium_severity", 0)
            elif tool_name == "safety":
                summary["vulnerabilities"] += result.get("vulnerabilities_count", 0)
            elif tool_name == "detect-secrets":
                summary["secrets_detected"] += result.get("secrets_count", 0)
            elif tool_name == "semgrep":
                summary["critical_issues"] += result.get("error_count", 0)
                summary["medium_issues"] += result.get("warning_count", 0)
            elif tool_name == "docker_security":
                summary["critical_issues"] += result.get("critical_issues", 0)
                summary["high_issues"] += result.get("high_issues", 0)
        
        # Generate recommendations
        if summary["critical_issues"] > 0:
            summary["recommendations"].append(
                f"URGENT: {summary['critical_issues']} critical security issues detected"
            )
        
        if summary["vulnerabilities"] > 0:
            summary["recommendations"].append(
                f"Update dependencies: {summary['vulnerabilities']} vulnerable packages found"
            )
        
        if summary["secrets_detected"] > 0:
            summary["recommendations"].append(
                f"Remove secrets: {summary['secrets_detected']} potential secrets detected"
            )
        
        if summary["high_issues"] > 10:
            summary["recommendations"].append(
                "Consider security code review: High number of security issues detected"
            )
        
        # Calculate overall security score (0-100)
        total_issues = (summary["critical_issues"] * 10 + 
                       summary["high_issues"] * 5 + 
                       summary["medium_issues"] * 2 + 
                       summary["vulnerabilities"] * 3 + 
                       summary["secrets_detected"] * 8)
        
        summary["security_score"] = max(0, 100 - total_issues)
        
        return summary
    
    def _save_results(self, results: Dict[str, Any]):
        """Save scan results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save comprehensive JSON report
        json_file = os.path.join(self.output_dir, f"security_scan_{timestamp}.json")
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Security scan report saved to: {json_file}")
        
        # Save summary report
        summary_file = os.path.join(self.output_dir, f"security_summary_{timestamp}.md")
        self._save_markdown_summary(results, summary_file)
        
        logger.info(f"Security summary saved to: {summary_file}")
    
    def _save_markdown_summary(self, results: Dict[str, Any], output_file: str):
        """Save human-readable summary in Markdown format"""
        summary = results["summary"]
        
        with open(output_file, 'w') as f:
            f.write("# Comprehensive Security Scan Report\n\n")
            f.write(f"**Target:** {results['scan_info']['target']}\n")
            f.write(f"**Date:** {results['scan_info']['start_time']}\n")
            f.write(f"**Duration:** {results['scan_info']['duration_seconds']:.1f} seconds\n\n")
            
            f.write("## Security Score\n\n")
            f.write(f"**Overall Security Score: {summary['security_score']}/100**\n\n")
            
            if summary['security_score'] >= 90:
                f.write("✅ **EXCELLENT** - Your codebase has strong security posture\n\n")
            elif summary['security_score'] >= 75:
                f.write("⚠️ **GOOD** - Minor security improvements recommended\n\n")
            elif summary['security_score'] >= 50:
                f.write("🔶 **NEEDS ATTENTION** - Several security issues detected\n\n")
            else:
                f.write("🚨 **CRITICAL** - Immediate security attention required\n\n")
            
            f.write("## Summary\n\n")
            f.write(f"- **Critical Issues:** {summary['critical_issues']}\n")
            f.write(f"- **High Severity Issues:** {summary['high_issues']}\n")
            f.write(f"- **Medium Severity Issues:** {summary['medium_issues']}\n")
            f.write(f"- **Vulnerable Dependencies:** {summary['vulnerabilities']}\n")
            f.write(f"- **Potential Secrets:** {summary['secrets_detected']}\n\n")
            
            f.write("## Tool Results\n\n")
            for tool_name, result in results["tool_results"].items():
                status = result.get("status", "unknown")
                f.write(f"### {tool_name.replace('_', ' ').title()}\n")
                f.write(f"**Status:** {status}\n")
                
                if status == "completed":
                    if "issues_count" in result:
                        f.write(f"**Issues Found:** {result['issues_count']}\n")
                    if "vulnerabilities_count" in result:
                        f.write(f"**Vulnerabilities:** {result['vulnerabilities_count']}\n")
                    if "secrets_count" in result:
                        f.write(f"**Secrets Detected:** {result['secrets_count']}\n")
                
                f.write("\n")
            
            if summary["recommendations"]:
                f.write("## Recommendations\n\n")
                for i, rec in enumerate(summary["recommendations"], 1):
                    f.write(f"{i}. {rec}\n")
                f.write("\n")
            
            f.write("---\n")
            f.write("*Generated by SutazAI Comprehensive Security Scanner v1.0.0*\n")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Comprehensive Security Scanner")
    parser.add_argument("target", help="Target directory to scan")
    parser.add_argument("--output-dir", help="Output directory for reports")
    parser.add_argument("--sequential", action="store_true", 
                       help="Run scans sequentially instead of parallel")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if not os.path.exists(args.target):
        logger.error(f"Target path does not exist: {args.target}")
        sys.exit(1)
    
    scanner = ComprehensiveSecurityScanner(args.target, args.output_dir)
    results = scanner.run_all_scans(parallel=not args.sequential)
    
    # Print summary
    summary = results["summary"]
    print(f"\n🔐 SECURITY SCAN SUMMARY")
    print(f"{'=' * 50}")
    print(f"Security Score: {summary['security_score']}/100")
    print(f"Critical Issues: {summary['critical_issues']}")
    print(f"High Issues: {summary['high_issues']}")
    print(f"Medium Issues: {summary['medium_issues']}")
    print(f"Vulnerabilities: {summary['vulnerabilities']}")
    print(f"Secrets Detected: {summary['secrets_detected']}")
    
    if summary["recommendations"]:
        print(f"\n📋 RECOMMENDATIONS:")
        for rec in summary["recommendations"]:
            print(f"  • {rec}")
    
    # Exit with appropriate code
    if summary["critical_issues"] > 0 or summary["vulnerabilities"] > 0:
        print(f"\n❌ Security scan detected critical issues!")
        sys.exit(1)
    elif summary["high_issues"] > 5:
        print(f"\n⚠️ Security scan detected multiple high-severity issues")
        sys.exit(1)
    else:
        print(f"\n✅ Security scan completed successfully")
        sys.exit(0)

if __name__ == "__main__":
    main()