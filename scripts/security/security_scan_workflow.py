#!/usr/bin/env python3
"""
Security Scanning Workflow
Automated security analysis using local AI agents
"""

import asyncio
import httpx
import json
import os
from datetime import datetime
from pathlib import Path

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


class SecurityScanWorkflow:
    """Comprehensive security scanning workflow"""
    
    def __init__(self):
        self.api_url = f"{API_BASE_URL}/api/v1"
        self.client = httpx.AsyncClient(timeout=60.0)
    
    async def scan_codebase(self, target_dir: str = ".") -> Dict[str, Any]:
        """Perform comprehensive security scan of codebase"""
        print(f"🔍 Starting security scan of: {target_dir}")
        
        scan_results = {
            "scan_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "target": target_dir,
            "started_at": datetime.now().isoformat(),
            "scans": {}
        }
        
        # Run different types of security scans
        
        # 1. Static code analysis
        print("  📝 Running static code analysis...")
        static_results = await self._static_code_analysis(target_dir)
        scan_results["scans"]["static_analysis"] = static_results
        
        # 2. Dependency vulnerability scan
        print("  📦 Scanning dependencies...")
        dep_results = await self._dependency_scan(target_dir)
        scan_results["scans"]["dependencies"] = dep_results
        
        # 3. Secrets detection
        print("  🔑 Scanning for secrets...")
        secrets_results = await self._scan_for_secrets(target_dir)
        scan_results["scans"]["secrets"] = secrets_results
        
        # 4. Container security (if Docker files exist)
        print("  🐳 Scanning Docker configurations...")
        docker_results = await self._scan_docker_files(target_dir)
        scan_results["scans"]["docker"] = docker_results
        
        # 5. Configuration security
        print("  ⚙️  Scanning configurations...")
        config_results = await self._scan_configurations(target_dir)
        scan_results["scans"]["configurations"] = config_results
        
        scan_results["completed_at"] = datetime.now().isoformat()
        
        # Generate summary
        scan_results["summary"] = self._generate_summary(scan_results["scans"])
        
        return scan_results
    
    async def _static_code_analysis(self, target_dir: str) -> Dict[str, Any]:
        """Run static code analysis using semgrep agent"""
        try:
            response = await self.client.post(
                f"{self.api_url}/agents/execute",
                json={
                    "agent": "semgrep-security-analyzer",
                    "task": "analyze",
                    "data": {
                        "target": target_dir,
                        "rules": [
                            "security",
                            "owasp-top-10",
                            "python-security"
                        ]
                    }
                }
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"Static analysis failed: {response.status_code}"}
        except Exception as e:
            return {"error": f"Static analysis error: {e}"}
    
    async def _dependency_scan(self, target_dir: str) -> Dict[str, Any]:
        """Scan for vulnerable dependencies"""
        results = {
            "vulnerabilities": [],
            "total_dependencies": 0
        }
        
        # Check Python dependencies
        requirements_files = [
            "requirements.txt",
            "pyproject.toml",
            "Pipfile",
            "poetry.lock"
        ]
        
        for req_file in requirements_files:
            file_path = Path(target_dir) / req_file
            if file_path.exists():
                print(f"    Found {req_file}")
                try:
                    response = await self.client.post(
                        f"{self.api_url}/agents/execute",
                        json={
                            "agent": "security-pentesting-specialist",
                            "task": "dependency_scan",
                            "data": {
                                "file": str(file_path),
                                "type": "python"
                            }
                        }
                    )
                    
                    if response.status_code == 200:
                        scan_result = response.json()
                        results["vulnerabilities"].extend(
                            scan_result.get("vulnerabilities", [])
                        )
                        results["total_dependencies"] += scan_result.get("total_dependencies", 0)
                except Exception as e:
                    results["errors"] = results.get("errors", [])
                    results["errors"].append(f"Failed to scan {req_file}: {e}")
        
        return results
    
    async def _scan_for_secrets(self, target_dir: str) -> Dict[str, Any]:
        """Scan for hardcoded secrets and credentials"""
        secrets_patterns = [
            {"name": "AWS Keys", "pattern": r"AKIA[0-9A-Z]{16}"},
            {"name": "API Keys", "pattern": r"api[_-]?key[_-]?[=:]\s*['\"][0-9a-zA-Z]{32,}['\"]"},
            {"name": "Passwords", "pattern": r"password[_-]?[=:]\s*['\"][^'\"]{8,}['\"]"},
            {"name": "Private Keys", "pattern": r"-----BEGIN (RSA|DSA|EC|OPENSSH) PRIVATE KEY-----"},
            {"name": "JWT Tokens", "pattern": r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"},
        ]
        
        findings = []
        
        # Scan common file types
        extensions = [".py", ".js", ".env", ".yml", ".yaml", ".json", ".config"]
        
        for ext in extensions:
            files = Path(target_dir).rglob(f"*{ext}")
            for file_path in files:
                # Skip test files and node_modules
                if "test" in str(file_path) or "node_modules" in str(file_path):
                    continue
                
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        
                    # Simple pattern matching (in production, use proper secret scanning)
                    for pattern_info in secrets_patterns:
                        # This is a simplified check - real implementation would use regex
                        if pattern_info["name"] == "Passwords" and "password" in content.lower():
                            # Check if it looks like a hardcoded password
                            lines = content.split('\n')
                            for i, line in enumerate(lines):
                                if 'password' in line.lower() and '=' in line and ('"' in line or "'" in line):
                                    findings.append({
                                        "type": pattern_info["name"],
                                        "file": str(file_path),
                                        "line": i + 1,
                                        "severity": "HIGH",
                                        "message": "Possible hardcoded password"
                                    })
                except Exception:
                    pass
        
        return {
            "findings": findings,
            "total_findings": len(findings)
        }
    
    async def _scan_docker_files(self, target_dir: str) -> Dict[str, Any]:
        """Scan Docker files for security issues"""
        docker_files = list(Path(target_dir).rglob("Dockerfile*"))
        docker_compose_files = list(Path(target_dir).rglob("docker-compose*.yml"))
        
        issues = []
        
        for dockerfile in docker_files:
            try:
                with open(dockerfile, 'r') as f:
                    content = f.read()
                
                # Check for common Docker security issues
                if "USER root" in content or not "USER " in content:
                    issues.append({
                        "file": str(dockerfile),
                        "issue": "Container runs as root",
                        "severity": "HIGH",
                        "recommendation": "Add a non-root USER directive"
                    })
                
                if "latest" in content:
                    issues.append({
                        "file": str(dockerfile),
                        "issue": "Using 'latest' tag",
                        "severity": "MEDIUM",
                        "recommendation": "Pin specific versions for reproducibility"
                    })
                
                if "--no-cache-dir" not in content and "pip install" in content:
                    issues.append({
                        "file": str(dockerfile),
                        "issue": "Pip cache not disabled",
                        "severity": "LOW",
                        "recommendation": "Use --no-cache-dir with pip install"
                    })
            except Exception:
                pass
        
        for compose_file in docker_compose_files:
            try:
                with open(compose_file, 'r') as f:
                    content = f.read()
                
                # Check for privileged containers
                if "privileged: true" in content:
                    issues.append({
                        "file": str(compose_file),
                        "issue": "Privileged container",
                        "severity": "CRITICAL",
                        "recommendation": "Avoid privileged containers"
                    })
            except Exception:
                pass
        
        return {
            "docker_files_scanned": len(docker_files) + len(docker_compose_files),
            "issues": issues
        }
    
    async def _scan_configurations(self, target_dir: str) -> Dict[str, Any]:
        """Scan configuration files for security issues"""
        config_issues = []
        
        # Check for common configuration issues
        config_patterns = {
            "debug_enabled": {
                "files": ["*.py", "*.js", "*.yml"],
                "patterns": ["DEBUG = True", "debug: true", "debug=true"],
                "severity": "MEDIUM",
                "message": "Debug mode enabled in production"
            },
            "cors_wildcard": {
                "files": ["*.py", "*.js"],
                "patterns": ["CORS(.*)\\*", "Access-Control-Allow-Origin.*\\*"],
                "severity": "HIGH",
                "message": "CORS wildcard origin allowed"
            }
        }
        
        for issue_type, config in config_patterns.items():
            for file_pattern in config["files"]:
                files = Path(target_dir).rglob(file_pattern)
                for file_path in files:
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        
                        for pattern in config["patterns"]:
                            if pattern in content:
                                config_issues.append({
                                    "file": str(file_path),
                                    "issue": config["message"],
                                    "severity": config["severity"],
                                    "type": issue_type
                                })
                    except Exception:
                        pass
        
        return {
            "configuration_issues": config_issues,
            "total_issues": len(config_issues)
        }
    
    def _generate_summary(self, scans: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of all scan results"""
        summary = {
            "total_issues": 0,
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0,
            "categories": {}
        }
        
        # Aggregate results from all scans
        for scan_type, results in scans.items():
            if "error" in results:
                continue
            
            category_count = 0
            
            # Count issues by severity
            for key in ["vulnerabilities", "findings", "issues", "configuration_issues"]:
                if key in results:
                    for issue in results[key]:
                        severity = issue.get("severity", "LOW").upper()
                        summary[severity.lower()] = summary.get(severity.lower(), 0) + 1
                        summary["total_issues"] += 1
                        category_count += 1
            
            summary["categories"][scan_type] = category_count
        
        # Risk level assessment
        if summary["critical"] > 0:
            summary["risk_level"] = "CRITICAL"
        elif summary["high"] > 5:
            summary["risk_level"] = "HIGH"
        elif summary["medium"] > 10:
            summary["risk_level"] = "MEDIUM"
        else:
            summary["risk_level"] = "LOW"
        
        return summary
    
    async def generate_report(self, scan_results: Dict[str, Any]) -> str:
        """Generate a comprehensive security report"""
        report = [
            "# Security Scan Report",
            f"\n**Scan ID**: {scan_results['scan_id']}",
            f"**Target**: `{scan_results['target']}`",
            f"**Date**: {scan_results['started_at']}",
            ""
        ]
        
        # Executive Summary
        summary = scan_results["summary"]
        report.extend([
            "## Executive Summary",
            f"**Risk Level**: {summary['risk_level']}",
            f"**Total Issues**: {summary['total_issues']}",
            "",
            "### Issues by Severity",
            f"- 🔴 Critical: {summary['critical']}",
            f"- 🟠 High: {summary['high']}",
            f"- 🟡 Medium: {summary['medium']}",
            f"- 🟢 Low: {summary['low']}",
            ""
        ])
        
        # Detailed Findings
        report.append("## Detailed Findings")
        
        for scan_type, results in scan_results["scans"].items():
            report.append(f"\n### {scan_type.replace('_', ' ').title()}")
            
            if "error" in results:
                report.append(f"❌ Error: {results['error']}")
                continue
            
            # Format results based on scan type
            if scan_type == "secrets" and "findings" in results:
                if results["findings"]:
                    for finding in results["findings"]:
                        report.append(f"\n- **{finding['type']}** in `{finding['file']}`")
                        report.append(f"  - Line: {finding.get('line', 'N/A')}")
                        report.append(f"  - Severity: {finding['severity']}")
                else:
                    report.append("✅ No secrets found")
            
            elif scan_type == "docker" and "issues" in results:
                if results["issues"]:
                    for issue in results["issues"]:
                        report.append(f"\n- **{issue['issue']}** in `{issue['file']}`")
                        report.append(f"  - Severity: {issue['severity']}")
                        report.append(f"  - Recommendation: {issue['recommendation']}")
                else:
                    report.append("✅ No Docker security issues found")
            
            elif "vulnerabilities" in results:
                if results["vulnerabilities"]:
                    for vuln in results["vulnerabilities"]:
                        report.append(f"\n- **{vuln.get('package', 'Unknown')}**")
                        report.append(f"  - Severity: {vuln.get('severity', 'Unknown')}")
                        report.append(f"  - Description: {vuln.get('description', 'N/A')}")
                else:
                    report.append("✅ No vulnerabilities found")
        
        # Recommendations
        report.extend([
            "\n## Recommendations",
            ""
        ])
        
        if summary["critical"] > 0:
            report.append("1. **Immediate Action Required**: Address all critical issues before deployment")
        if summary["high"] > 0:
            report.append("2. **High Priority**: Fix high severity issues within 24-48 hours")
        if summary["total_issues"] > 0:
            report.append("3. **Security Review**: Conduct thorough review of all findings")
        
        report.append("\n---\n*Generated by SutazAI Security Scanner*")
        
        return "\n".join(report)
    
    async def close(self):
        """Clean up resources"""
        await self.client.aclose()


async def main():
    """Run security scanning workflow"""
    scanner = SecurityScanWorkflow()
    
    try:
        # Scan the current directory (or specify a different target)
        target_directory = "."
        
        print("🛡️  Starting comprehensive security scan...")
        print(f"📁 Target: {os.path.abspath(target_directory)}")
        print()
        
        # Run the scan
        scan_results = await scanner.scan_codebase(target_directory)
        
        # Generate report
        report = await scanner.generate_report(scan_results)
        
        # Save report
        report_filename = f"security_report_{scan_results['scan_id']}.md"
        with open(report_filename, 'w') as f:
            f.write(report)
        
        print(f"\n✅ Security scan complete!")
        print(f"📄 Report saved to: {report_filename}")
        
        # Print summary
        summary = scan_results["summary"]
        print(f"\n🎯 Risk Level: {summary['risk_level']}")
        print(f"📊 Total Issues: {summary['total_issues']}")
        
        if summary['critical'] > 0:
            print(f"\n⚠️  CRITICAL ISSUES FOUND! Immediate action required.")
        
        # Save raw results
        json_filename = f"security_scan_{scan_results['scan_id']}.json"
        with open(json_filename, 'w') as f:
            json.dump(scan_results, f, indent=2)
        
        print(f"📄 Raw results saved to: {json_filename}")
        
    except Exception as e:
        print(f"❌ Security scan failed: {e}")
    
    finally:
        await scanner.close()


if __name__ == "__main__":
    asyncio.run(main())