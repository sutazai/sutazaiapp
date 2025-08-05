#!/usr/bin/env python3
"""
Comprehensive Agent Analysis Script
Analyzes all agents in /opt/sutazaiapp/agents/ directory for:
1. Real implementation vs stubs
2. Requirements conflicts
3. Security vulnerabilities
4. Value assessment
"""

import os
import json
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Set, Any
import ast

class AgentAnalyzer:
    def __init__(self, agents_dir: str = "/opt/sutazaiapp/agents"):
        self.agents_dir = Path(agents_dir)
        self.agent_dirs = [d for d in self.agents_dir.iterdir() if d.is_dir() and '-' in d.name]
        self.results = {
            'total_agents': len(self.agent_dirs),
            'stub_agents': [],
            'working_agents': [],
            'requirements_conflicts': {},
            'security_issues': [],
            'value_assessment': {},
            'recommendations': {
                'keep': [],
                'remove': [],
                'consolidate': []
            }
        }
        
    def analyze_app_py(self, agent_path: Path) -> Dict[str, Any]:
        """Analyze app.py for implementation vs stub detection"""
        app_py = agent_path / "app.py"
        if not app_py.exists():
            return {"type": "missing", "has_logic": False, "endpoints": []}
        
        try:
            with open(app_py, 'r') as f:
                content = f.read()
            
            # Check for stub indicators
            stub_indicators = [
                'return jsonify({"status": "stub"',
                'return {"status": "stub"',
                'placeholder implementation',
                'TODO: implement',
                'stub response',
                'not implemented',
                'return "stub"',
                'mock response'
            ]
            
            is_stub = any(indicator.lower() in content.lower() for indicator in stub_indicators)
            
            # Check for real logic indicators
            logic_indicators = [
                'import requests',
                'import subprocess',
                'import docker',
                'import kubernetes',
                'import ollama',
                'def process(',
                'def analyze(',
                'def optimize(',
                'class Agent',
                'SQLAlchemy',
                'psycopg2',
                'redis',
                'celery'
            ]
            
            has_logic = any(indicator in content for indicator in logic_indicators)
            
            # Extract endpoints
            endpoints = []
            flask_routes = re.findall(r"@app\.route\(['\"]([^'\"]+)['\"]", content)
            fastapi_routes = re.findall(r"@router\.[get|post|put|delete]+\(['\"]([^'\"]+)['\"]", content)
            endpoints.extend(flask_routes)
            endpoints.extend(fastapi_routes)
            
            # Count lines of actual logic (excluding imports, comments, empty lines)
            lines = content.split('\n')
            logic_lines = 0
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#') and not line.startswith('import') and not line.startswith('from'):
                    logic_lines += 1
            
            return {
                "type": "stub" if is_stub else ("working" if has_logic else "minimal"),
                "has_logic": has_logic,
                "endpoints": endpoints,
                "logic_lines": logic_lines,
                "file_size": app_py.stat().st_size,
                "is_stub": is_stub
            }
            
        except Exception as e:
            return {"type": "error", "error": str(e), "has_logic": False, "endpoints": []}
    
    def analyze_requirements(self, agent_path: Path) -> Dict[str, Any]:
        """Analyze requirements.txt for conflicts and security issues"""
        req_file = agent_path / "requirements.txt"
        if not req_file.exists():
            return {"exists": False, "packages": []}
        
        try:
            with open(req_file, 'r') as f:
                content = f.read()
            
            packages = []
            for line in content.split('\n'):
                line = line.strip()
                if line and not line.startswith('#'):
                    packages.append(line)
            
            # Check for known vulnerable packages
            vulnerable_packages = [
                'flask==0.12',  # Old vulnerable versions
                'django<3.0',
                'requests<2.20',
                'pyyaml<5.1',
                'jinja2<2.10.1'
            ]
            
            security_issues = []
            for pkg in packages:
                for vuln in vulnerable_packages:
                    if pkg.lower().startswith(vuln.split('=')[0].split('<')[0]):
                        security_issues.append(f"Potentially vulnerable: {pkg}")
            
            return {
                "exists": True,
                "packages": packages,
                "package_count": len(packages),
                "security_issues": security_issues
            }
            
        except Exception as e:
            return {"exists": True, "error": str(e), "packages": []}
    
    def check_security_issues(self, agent_path: Path) -> List[str]:
        """Check for security issues in agent code"""
        issues = []
        
        # Check app.py for security issues
        app_py = agent_path / "app.py"
        if app_py.exists():
            try:
                with open(app_py, 'r') as f:
                    content = f.read()
                
                # Security red flags
                security_patterns = [
                    (r'eval\(', 'Uses eval() - code injection risk'),
                    (r'exec\(', 'Uses exec() - code injection risk'),
                    (r'subprocess\.call\([^)]*shell=True', 'Shell injection risk'),
                    (r'os\.system\(', 'Command injection risk'),
                    (r'pickle\.loads?\(', 'Pickle deserialization risk'),
                    (r'__import__\(', 'Dynamic import risk'),
                    (r'open\([^)]*[\'\"]/etc/', 'Reading system files'),
                    (r'app\.run\(.*debug=True', 'Debug mode enabled'),
                    (r'CORS\(.*origins=\*', 'Overly permissive CORS'),
                ]
                
                for pattern, description in security_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        issues.append(f"{description}: {agent_path.name}")
                        
            except Exception as e:
                issues.append(f"Error reading {agent_path.name}/app.py: {str(e)}")
        
        return issues
    
    def assess_agent_value(self, agent_name: str, analysis: Dict[str, Any]) -> str:
        """Assess the value/utility of an agent"""
        app_analysis = analysis.get('app_analysis', {})
        req_analysis = analysis.get('requirements_analysis', {})
        
        # High value agents (core functionality)
        high_value_keywords = [
            'orchestrator', 'health-monitor', 'backend-developer', 
            'frontend-developer', 'system-architect', 'qa', 'testing',
            'security', 'deployment', 'infrastructure', 'monitoring'
        ]
        
        # Low value agents (theoretical/experimental)
        low_value_keywords = [
            'quantum', 'neuromorphic', 'agi', 'teleport', 'magic',
            'experimental', 'theoretical', 'research', 'brain-architect'
        ]
        
        score = 0
        
        # Implementation quality
        if app_analysis.get('has_logic', False):
            score += 3
        if app_analysis.get('logic_lines', 0) > 50:
            score += 2
        if len(app_analysis.get('endpoints', [])) > 2:
            score += 1
        
        # Requirements quality
        if req_analysis.get('exists', False):
            score += 1
        if req_analysis.get('package_count', 0) > 0 and req_analysis.get('package_count', 0) < 20:
            score += 1
        
        # Keyword analysis
        for keyword in high_value_keywords:
            if keyword in agent_name:
                score += 2
                break
        
        for keyword in low_value_keywords:
            if keyword in agent_name:
                score -= 2
                break
        
        # Security issues penalty
        if analysis.get('security_issues'):
            score -= 3
        
        if score >= 5:
            return "HIGH"
        elif score >= 2:
            return "MEDIUM"
        else:
            return "LOW"
    
    def find_requirements_conflicts(self) -> Dict[str, List[str]]:
        """Find conflicting package versions across agents"""
        package_versions = {}
        
        for agent_dir in self.agent_dirs:
            req_file = agent_dir / "requirements.txt"
            if req_file.exists():
                try:
                    with open(req_file, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if '==' in line:
                                pkg, version = line.split('==', 1)
                                pkg = pkg.strip()
                                version = version.strip()
                                
                                if pkg not in package_versions:
                                    package_versions[pkg] = {}
                                if version not in package_versions[pkg]:
                                    package_versions[pkg][version] = []
                                package_versions[pkg][version].append(agent_dir.name)
                except:
                    continue
        
        conflicts = {}
        for pkg, versions in package_versions.items():
            if len(versions) > 1:
                conflicts[pkg] = versions
        
        return conflicts
    
    def analyze_all_agents(self):
        """Main analysis function"""
        print(f"Analyzing {len(self.agent_dirs)} agents...")
        
        for i, agent_dir in enumerate(self.agent_dirs):
            print(f"Analyzing {i+1}/{len(self.agent_dirs)}: {agent_dir.name}")
            
            # Analyze app.py
            app_analysis = self.analyze_app_py(agent_dir)
            
            # Analyze requirements.txt
            req_analysis = self.analyze_requirements(agent_dir)
            
            # Check security issues
            security_issues = self.check_security_issues(agent_dir)
            
            # Store analysis
            agent_analysis = {
                'name': agent_dir.name,
                'path': str(agent_dir),
                'app_analysis': app_analysis,
                'requirements_analysis': req_analysis,
                'security_issues': security_issues
            }
            
            # Categorize agents
            if app_analysis.get('is_stub', False) or app_analysis.get('type') == 'stub':
                self.results['stub_agents'].append(agent_analysis)
            elif app_analysis.get('has_logic', False):
                self.results['working_agents'].append(agent_analysis)
            else:
                self.results['stub_agents'].append(agent_analysis)  # Treat minimal as stub
            
            # Assess value
            value = self.assess_agent_value(agent_dir.name, agent_analysis)
            self.results['value_assessment'][agent_dir.name] = {
                'value': value,
                'analysis': agent_analysis
            }
            
            # Security issues
            if security_issues:
                self.results['security_issues'].extend(security_issues)
        
        # Find requirements conflicts
        self.results['requirements_conflicts'] = self.find_requirements_conflicts()
        
        # Generate recommendations
        self.generate_recommendations()
    
    def generate_recommendations(self):
        """Generate keep/remove/consolidate recommendations"""
        
        # Keep: High value with working implementation
        for agent_name, assessment in self.results['value_assessment'].items():
            if assessment['value'] == 'HIGH' and assessment['analysis']['app_analysis'].get('has_logic', False):
                self.results['recommendations']['keep'].append(agent_name)
            elif assessment['value'] == 'LOW' or assessment['analysis']['app_analysis'].get('is_stub', False):
                self.results['recommendations']['remove'].append(agent_name)
            else:
                # Medium value - needs review
                pass
        
        # Find similar agents for consolidation
        agent_groups = {}
        for agent_name in self.results['value_assessment'].keys():
            # Group by prefix (e.g., 'ai-senior-', 'system-', etc.)
            prefix = '-'.join(agent_name.split('-')[:2])
            if prefix not in agent_groups:
                agent_groups[prefix] = []
            agent_groups[prefix].append(agent_name)
        
        for group, agents in agent_groups.items():
            if len(agents) > 2:  # Multiple similar agents
                self.results['recommendations']['consolidate'].append({
                    'group': group,
                    'agents': agents,
                    'suggestion': f"Consider consolidating {len(agents)} {group}* agents"
                })
    
    def generate_report(self) -> str:
        """Generate comprehensive analysis report"""
        report = []
        report.append("# COMPREHENSIVE AGENT ANALYSIS REPORT")
        report.append("=" * 50)
        report.append("")
        
        # Summary
        report.append("## EXECUTIVE SUMMARY")
        report.append(f"- Total Agents Analyzed: {self.results['total_agents']}")
        report.append(f"- Working Agents: {len(self.results['working_agents'])}")
        report.append(f"- Stub Agents: {len(self.results['stub_agents'])}")
        report.append(f"- Security Issues Found: {len(self.results['security_issues'])}")
        report.append(f"- Requirements Conflicts: {len(self.results['requirements_conflicts'])}")
        report.append("")
        
        # Working Agents
        report.append("## WORKING AGENTS (Real Implementation)")
        report.append("-" * 40)
        for agent in self.results['working_agents']:
            name = agent['name']
            logic_lines = agent['app_analysis'].get('logic_lines', 0)
            endpoints = len(agent['app_analysis'].get('endpoints', []))
            value = self.results['value_assessment'][name]['value']
            report.append(f"✅ {name}")
            report.append(f"   - Logic Lines: {logic_lines}")
            report.append(f"   - Endpoints: {endpoints}")
            report.append(f"   - Value: {value}")
            report.append("")
        
        # Stub Agents
        report.append("## STUB AGENTS (Placeholder Implementation)")
        report.append("-" * 40)
        for agent in self.results['stub_agents']:
            name = agent['name']
            value = self.results['value_assessment'][name]['value']
            report.append(f"❌ {name} (Value: {value})")
        report.append("")
        
        # Requirements Conflicts
        report.append("## REQUIREMENTS CONFLICTS")
        report.append("-" * 40)
        for package, versions in self.results['requirements_conflicts'].items():
            report.append(f"⚠️  {package}:")
            for version, agents in versions.items():
                report.append(f"   {version}: {', '.join(agents[:3])}{'...' if len(agents) > 3 else ''}")
            report.append("")
        
        # Security Issues
        report.append("## SECURITY ISSUES")
        report.append("-" * 40)
        for issue in self.results['security_issues']:
            report.append(f"🔒 {issue}")
        report.append("")
        
        # Recommendations
        report.append("## RECOMMENDATIONS")
        report.append("-" * 40)
        
        report.append("### KEEP (High Value + Working):")
        for agent in self.results['recommendations']['keep']:
            report.append(f"✅ {agent}")
        report.append("")
        
        report.append("### REMOVE (Low Value / Stubs):")
        for agent in self.results['recommendations']['remove']:
            report.append(f"❌ {agent}")
        report.append("")
        
        report.append("### CONSOLIDATE:")
        for consolidation in self.results['recommendations']['consolidate']:
            report.append(f"🔄 {consolidation['suggestion']}")
            for agent in consolidation['agents']:
                report.append(f"   - {agent}")
            report.append("")
        
        # Detailed Analysis
        report.append("## DETAILED AGENT BREAKDOWN")
        report.append("-" * 40)
        
        for agent_name, assessment in self.results['value_assessment'].items():
            analysis = assessment['analysis']
            report.append(f"### {agent_name}")
            report.append(f"**Value:** {assessment['value']}")
            report.append(f"**Type:** {analysis['app_analysis'].get('type', 'unknown')}")
            report.append(f"**Logic Lines:** {analysis['app_analysis'].get('logic_lines', 0)}")
            report.append(f"**Endpoints:** {len(analysis['app_analysis'].get('endpoints', []))}")
            report.append(f"**Requirements:** {'✅' if analysis['requirements_analysis'].get('exists') else '❌'}")
            if analysis['security_issues']:
                report.append(f"**Security Issues:** {len(analysis['security_issues'])}")
            report.append("")
        
        return "\n".join(report)

if __name__ == "__main__":
    analyzer = AgentAnalyzer()
    analyzer.analyze_all_agents()
    
    # Save detailed results
    with open('/opt/sutazaiapp/agent_analysis_results.json', 'w') as f:
        json.dump(analyzer.results, f, indent=2, default=str)
    
    # Generate and save report
    report = analyzer.generate_report()
    with open('/opt/sutazaiapp/AGENT_ANALYSIS_REPORT.md', 'w') as f:
        f.write(report)
    
    print("\n" + "="*50)
    print("ANALYSIS COMPLETE")
    print("="*50)
    print(f"Results saved to: agent_analysis_results.json")
    print(f"Report saved to: AGENT_ANALYSIS_REPORT.md")
    print("\nQuick Summary:")
    print(f"- Total: {analyzer.results['total_agents']} agents")
    print(f"- Working: {len(analyzer.results['working_agents'])} agents")
    print(f"- Stubs: {len(analyzer.results['stub_agents'])} agents")
    print(f"- Security Issues: {len(analyzer.results['security_issues'])}")
    print(f"- Requirements Conflicts: {len(analyzer.results['requirements_conflicts'])}")