#!/usr/bin/env python3
"""
Comprehensive Agent Compliance Checker for CLAUDE.md Rules

This script checks all agent .md files in the /opt/sutazaiapp/.claude/agents directory
for compliance with CLAUDE.md rules and standards.
"""

import os
import re
import json
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import yaml
from datetime import datetime

class AgentComplianceChecker:
    """Comprehensive checker for agent compliance with CLAUDE.md rules"""
    
    def __init__(self, agents_dir: str = "/opt/sutazaiapp/.claude/agents"):
        self.agents_dir = Path(agents_dir)
        self.claude_md_path = Path("/opt/sutazaiapp/CLAUDE.md")
        self.compliance_report = {
            "timestamp": datetime.now().isoformat(),
            "total_agents": 0,
            "compliant_agents": 0,
            "non_compliant_agents": 0,
            "agents": {},
            "summary": {}
        }
        
    def find_all_agent_files(self) -> List[Path]:
        """Find all agent .md files (excluding reports and documentation)"""
        agent_files = []
        
        # Get all .md files but exclude reports and documentation
        exclude_patterns = [
            'AGENT_CLEANUP_SUMMARY.md',
            'AGENT_COMPLIANCE_REPORT.md',
            'COMPREHENSIVE_INVESTIGATION_PROTOCOL.md',
            'COMPLETE_CLEANUP_STATUS.md',
            'DUPLICATE_AGENTS_REPORT.md',
            'FINAL_COMPLIANCE_SUMMARY.md',
            'team_collaboration_standards.md'
        ]
        
        for md_file in self.agents_dir.glob("*.md"):
            if md_file.name not in exclude_patterns:
                agent_files.append(md_file)
                
        return sorted(agent_files)
    
    def check_claude_md_header(self, file_path: Path) -> Tuple[bool, str]:
        """Check if agent file has proper CLAUDE.md compliance header"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Check for the mandatory compliance header
            expected_header_patterns = [
                r"## Important: Codebase Standards",
                r"\*\*MANDATORY\*\*: Before performing any task",
                r"`/opt/sutazaiapp/CLAUDE\.md`",
                r"Codebase standards and conventions",
                r"Rules for avoiding fantasy elements",
                r"System stability and performance guidelines"
            ]
            
            missing_patterns = []
            for pattern in expected_header_patterns:
                if not re.search(pattern, content, re.IGNORECASE):
                    missing_patterns.append(pattern)
            
            if missing_patterns:
                return False, f"Missing compliance header elements: {missing_patterns}"
            else:
                return True, "Compliance header present and complete"
                
        except Exception as e:
            return False, f"Error reading file: {str(e)}"
    
    def check_agent_structure(self, file_path: Path) -> Tuple[bool, List[str]]:
        """Check if agent file has proper YAML structure"""
        issues = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for required fields
            required_fields = ['name:', 'description:', 'model:', 'version:']
            
            for field in required_fields:
                if field not in content:
                    issues.append(f"Missing required field: {field}")
            
            # Check for proper YAML structure (starts with ---)
            if not content.strip().startswith('---'):
                issues.append("File should start with YAML front matter (---)")
            
            return len(issues) == 0, issues
            
        except Exception as e:
            return False, [f"Error checking structure: {str(e)}"]
    
    def check_rules_integration(self, file_path: Path) -> Tuple[bool, List[str]]:
        """Check if agent has proper integration with claude_rules_checker.py"""
        issues = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for rules checker integration indicators
            rules_indicators = [
                'claude_rules_checker',
                'CLAUDE.md',
                'rules enforcement',
                'compliance'
            ]
            
            found_indicators = sum(1 for indicator in rules_indicators 
                                 if indicator.lower() in content.lower())
            
            if found_indicators < 2:
                issues.append("Insufficient rules enforcement integration")
            
            return len(issues) == 0, issues
            
        except Exception as e:
            return False, [f"Error checking rules integration: {str(e)}"]
    
    def check_environment_variables(self, file_path: Path) -> Tuple[bool, List[str]]:
        """Check if agent has proper environment variable handling"""
        issues = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for environment variable patterns
            env_patterns = [
                r'environment:',
                r'env_file:',
                r'\$\{[^}]+\}',  # ${VAR} pattern
                r'CLAUDE_RULES_ENABLED'
            ]
            
            found_env_patterns = sum(1 for pattern in env_patterns 
                                   if re.search(pattern, content))
            
            if found_env_patterns == 0:
                issues.append("No environment variable configuration found")
            
            return len(issues) == 0, issues
            
        except Exception as e:
            return False, [f"Error checking environment variables: {str(e)}"]
    
    def analyze_agent_file(self, file_path: Path) -> Dict:
        """Comprehensive analysis of a single agent file"""
        analysis = {
            "file_path": str(file_path),
            "file_name": file_path.name,
            "compliant": True,
            "issues": [],
            "checks": {}
        }
        
        # Check CLAUDE.md header compliance
        header_ok, header_msg = self.check_claude_md_header(file_path)
        analysis["checks"]["claude_md_header"] = {
            "passed": header_ok,
            "message": header_msg
        }
        if not header_ok:
            analysis["compliant"] = False
            analysis["issues"].append(f"Header: {header_msg}")
        
        # Check agent structure
        structure_ok, structure_issues = self.check_agent_structure(file_path)
        analysis["checks"]["agent_structure"] = {
            "passed": structure_ok,
            "issues": structure_issues
        }
        if not structure_ok:
            analysis["compliant"] = False
            analysis["issues"].extend(structure_issues)
        
        # Check rules integration
        rules_ok, rules_issues = self.check_rules_integration(file_path)
        analysis["checks"]["rules_integration"] = {
            "passed": rules_ok,
            "issues": rules_issues
        }
        if not rules_ok:
            analysis["compliant"] = False
            analysis["issues"].extend(rules_issues)
        
        # Check environment variables
        env_ok, env_issues = self.check_environment_variables(file_path)
        analysis["checks"]["environment_variables"] = {
            "passed": env_ok,
            "issues": env_issues
        }
        if not env_ok:
            analysis["compliant"] = False
            analysis["issues"].extend(env_issues)
        
        return analysis
    
    def run_full_compliance_check(self) -> Dict:
        """Run comprehensive compliance check on all agents"""
        print("🔍 Starting comprehensive agent compliance check...")
        
        agent_files = self.find_all_agent_files()
        self.compliance_report["total_agents"] = len(agent_files)
        
        print(f"Found {len(agent_files)} agent files to check")
        
        compliant_count = 0
        non_compliant_count = 0
        
        for agent_file in agent_files:
            print(f"Checking: {agent_file.name}")
            analysis = self.analyze_agent_file(agent_file)
            
            self.compliance_report["agents"][agent_file.name] = analysis
            
            if analysis["compliant"]:
                compliant_count += 1
                print(f"  ✅ Compliant")
            else:
                non_compliant_count += 1
                print(f"  ❌ Non-compliant: {len(analysis['issues'])} issues")
                for issue in analysis["issues"]:
                    print(f"    • {issue}")
        
        self.compliance_report["compliant_agents"] = compliant_count
        self.compliance_report["non_compliant_agents"] = non_compliant_count
        
        # Generate summary
        self.compliance_report["summary"] = {
            "compliance_rate": f"{(compliant_count / len(agent_files)) * 100:.1f}%",
            "critical_issues": self._identify_critical_issues(),
            "recommendations": self._generate_recommendations()
        }
        
        return self.compliance_report
    
    def _identify_critical_issues(self) -> List[str]:
        """Identify the most critical compliance issues"""
        critical_issues = []
        
        header_issues = 0
        structure_issues = 0
        rules_issues = 0
        
        for agent_name, analysis in self.compliance_report["agents"].items():
            if not analysis["checks"]["claude_md_header"]["passed"]:
                header_issues += 1
            if not analysis["checks"]["agent_structure"]["passed"]:
                structure_issues += 1
            if not analysis["checks"]["rules_integration"]["passed"]:
                rules_issues += 1
        
        if header_issues > 0:
            critical_issues.append(f"{header_issues} agents missing CLAUDE.md compliance headers")
        if structure_issues > 0:
            critical_issues.append(f"{structure_issues} agents have structural issues")
        if rules_issues > 0:
            critical_issues.append(f"{rules_issues} agents lack proper rules integration")
        
        return critical_issues
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations for fixing compliance issues"""
        recommendations = [
            "1. Run the agent update scripts to add missing compliance headers",
            "2. Ensure all agents reference claude_rules_checker.py",
            "3. Add CLAUDE_RULES_ENABLED environment variable to all agents",
            "4. Update agent startup scripts to check CLAUDE.md before execution",
            "5. Implement automated compliance checking in CI/CD pipeline"
        ]
        return recommendations
    
    def save_report(self, output_file: str = None) -> str:
        """Save compliance report to JSON file"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"agent_compliance_report_{timestamp}.json"
        
        output_path = self.agents_dir / output_file
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.compliance_report, f, indent=2, ensure_ascii=False)
        
        print(f"📊 Compliance report saved to: {output_path}")
        return str(output_path)
    
    def print_summary(self):
        """Print a summary of the compliance check"""
        report = self.compliance_report
        
        print("\n" + "="*60)
        print("🔍 AGENT COMPLIANCE SUMMARY")
        print("="*60)
        print(f"Total Agents: {report['total_agents']}")
        print(f"Compliant: {report['compliant_agents']} ({report['summary']['compliance_rate']})")
        print(f"Non-compliant: {report['non_compliant_agents']}")
        print()
        
        if report['summary']['critical_issues']:
            print("🚨 CRITICAL ISSUES:")
            for issue in report['summary']['critical_issues']:
                print(f"  • {issue}")
            print()
        
        print("💡 RECOMMENDATIONS:")
        for rec in report['summary']['recommendations']:
            print(f"  {rec}")
        print()

def main():
    """Main function to run compliance check"""
    checker = AgentComplianceChecker()
    
    # Run the compliance check
    report = checker.run_full_compliance_check()
    
    # Print summary
    checker.print_summary()
    
    # Save detailed report
    report_file = checker.save_report()
    
    return report_file

if __name__ == "__main__":
    main()