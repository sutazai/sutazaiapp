#!/usr/bin/env python3
"""
Comprehensive AI Agent QA Validation Script
Testing QA Validator Agent for SutazAI platform

This script performs exhaustive validation of all AI agents to ensure
codebase hygiene compliance and system integrity.
"""

import os
import json
import yaml
import re
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime
import hashlib

class AgentQAValidator:
    """Comprehensive AI Agent Quality Assurance Validator"""
    
    def __init__(self):
        self.agents_dir = Path("/opt/sutazaiapp/.claude/agents")
        self.test_results = {
            'total_agents': 0,
            'passed_agents': 0,
            'failed_agents': 0,
            'compliance_percentage': 0.0,
            'validation_timestamp': datetime.now().isoformat(),
            'detailed_results': {},
            'critical_issues': [],
            'recommendations': [],
            'agent_inventory': {},
            'test_categories': {
                'yaml_frontmatter': {'passed': 0, 'failed': 0, 'issues': []},
                'naming_conventions': {'passed': 0, 'failed': 0, 'issues': []},
                'hygiene_enforcement': {'passed': 0, 'failed': 0, 'issues': []},
                'duplicate_detection': {'passed': 0, 'failed': 0, 'issues': []},
                'agent_behavior': {'passed': 0, 'failed': 0, 'issues': []},
                'integration_compliance': {'passed': 0, 'failed': 0, 'issues': []},
                'performance_standards': {'passed': 0, 'failed': 0, 'issues': []}
            }
        }
        
        # Essential agent list from COMPREHENSIVE_INVESTIGATION_PROTOCOL.md
        self.essential_agents = {
            'senior-ai-engineer',
            'senior-backend-developer', 
            'senior-frontend-developer',
            'infrastructure-devops-manager',
            'autonomous-system-controller',
            'ai-agent-orchestrator',
            'self-healing-orchestrator'
        }
        
        # Required hygiene sections
        self.required_hygiene_sections = [
            "Clean Code Principles",
            "Zero Duplication Policy", 
            "File Organization Standards",
            "Professional Standards"
        ]
        
    def validate_all_agents(self) -> Dict[str, Any]:
        """Run comprehensive validation on all AI agents"""
        
        print("🔍 Starting Comprehensive AI Agent QA Validation...")
        print("=" * 60)
        
        # Get all agent files
        agent_files = self.get_agent_files()
        self.test_results['total_agents'] = len(agent_files)
        
        print(f"📊 Found {len(agent_files)} agents to validate")
        
        # Run all validation tests
        for agent_file in agent_files:
            agent_name = agent_file.stem
            print(f"\n🧪 Testing agent: {agent_name}")
            
            agent_results = self.validate_single_agent(agent_file)
            self.test_results['detailed_results'][agent_name] = agent_results
            
            if agent_results['overall_pass']:
                self.test_results['passed_agents'] += 1
                print(f"✅ {agent_name}: PASSED")
            else:
                self.test_results['failed_agents'] += 1
                print(f"❌ {agent_name}: FAILED")
                
        # Calculate compliance percentage
        if self.test_results['total_agents'] > 0:
            self.test_results['compliance_percentage'] = (
                self.test_results['passed_agents'] / self.test_results['total_agents']
            ) * 100
            
        # Run system-wide tests
        self.validate_system_integration()
        self.validate_performance_standards()
        self.generate_recommendations()
        
        return self.test_results
        
    def get_agent_files(self) -> List[Path]:
        """Get list of all agent definition files"""
        
        agent_files = []
        for file_path in self.agents_dir.glob("*.md"):
            # Skip non-agent files
            if any(skip in file_path.name for skip in [
                'detailed', 'COMPREHENSIVE_INVESTIGATION', 'AGENT_', 
                'COMPLETE_', 'DUPLICATE_', 'FINAL_', 'team_collaboration',
                'README', 'IMPLEMENTATION_SUMMARY'
            ]):
                continue
                
            # Skip backup directories
            if any(backup in str(file_path) for backup in [
                'backups', 'compliance_fixes_backup', 'fixes'
            ]):
                continue
                
            agent_files.append(file_path)
            
        return sorted(agent_files)
        
    def validate_single_agent(self, agent_file: Path) -> Dict[str, Any]:
        """Validate a single agent file comprehensively"""
        
        agent_name = agent_file.stem
        results = {
            'agent_name': agent_name,
            'file_path': str(agent_file),
            'tests': {},
            'overall_pass': True,
            'issues': [],
            'score': 0
        }
        
        try:
            with open(agent_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Test 1: YAML Frontmatter Validation
            results['tests']['yaml_frontmatter'] = self.test_yaml_frontmatter(content, agent_name)
            
            # Test 2: Naming Convention Validation
            results['tests']['naming_conventions'] = self.test_naming_conventions(agent_file, content)
            
            # Test 3: Hygiene Enforcement Section Validation
            results['tests']['hygiene_enforcement'] = self.test_hygiene_enforcement(content, agent_name)
            
            # Test 4: Content Quality Analysis
            results['tests']['content_quality'] = self.test_content_quality(content, agent_name)
            
            # Test 5: Agent Behavior Compliance
            results['tests']['behavior_compliance'] = self.test_behavior_compliance(content, agent_name)
            
            # Calculate overall score and pass/fail
            total_score = sum(test['score'] for test in results['tests'].values())
            max_score = len(results['tests']) * 100
            results['score'] = (total_score / max_score) * 100
            
            # Agent passes if score >= 85% and no critical issues
            results['overall_pass'] = (
                results['score'] >= 85.0 and 
                not any(test['critical'] for test in results['tests'].values() if test.get('critical', False))
            )
            
            # Update category results
            for test_name, test_result in results['tests'].items():
                if test_name in self.test_results['test_categories']:
                    if test_result['passed']:
                        self.test_results['test_categories'][test_name]['passed'] += 1
                    else:
                        self.test_results['test_categories'][test_name]['failed'] += 1
                        self.test_results['test_categories'][test_name]['issues'].extend(test_result['issues'])
                        
        except Exception as e:
            results['overall_pass'] = False
            results['issues'].append(f"Failed to process agent file: {str(e)}")
            
        return results
        
    def test_yaml_frontmatter(self, content: str, agent_name: str) -> Dict[str, Any]:
        """Test YAML frontmatter structure and required fields"""
        
        result = {
            'passed': False,
            'score': 0,
            'issues': [],
            'critical': False
        }
        
        # Extract frontmatter
        frontmatter_match = re.match(r'^---\n(.*?)\n---', content, re.DOTALL)
        
        if not frontmatter_match:
            result['issues'].append("Missing YAML frontmatter")
            result['critical'] = True
            return result
            
        try:
            frontmatter = yaml.safe_load(frontmatter_match.group(1))
            
            # Required fields
            required_fields = ['name', 'version', 'description', 'category', 'tags', 'model']
            missing_fields = []
            
            for field in required_fields:
                if field not in frontmatter:
                    missing_fields.append(field)
                    
            if missing_fields:
                result['issues'].append(f"Missing required fields: {missing_fields}")
            else:
                result['score'] += 50
                
            # Validate name matches filename
            if frontmatter.get('name') != agent_name:
                result['issues'].append(f"Name mismatch: {frontmatter.get('name')} != {agent_name}")
            else:
                result['score'] += 50
                
            result['passed'] = len(result['issues']) == 0
            
        except yaml.YAMLError as e:
            result['issues'].append(f"Invalid YAML syntax: {str(e)}")
            result['critical'] = True
            
        return result
        
    def test_naming_conventions(self, agent_file: Path, content: str) -> Dict[str, Any]:
        """Test naming conventions and filename compliance"""
        
        result = {
            'passed': False,
            'score': 0,
            'issues': []
        }
        
        agent_name = agent_file.stem
        
        # Test naming pattern (kebab-case)
        if not re.match(r'^[a-z0-9]+(-[a-z0-9]+)*$', agent_name):
            result['issues'].append(f"Invalid naming pattern: {agent_name} (should be kebab-case)")
        else:
            result['score'] += 50
            
        # Test for meaningful name
        if len(agent_name) < 5:
            result['issues'].append(f"Agent name too short: {agent_name}")
        else:
            result['score'] += 25
            
        # Check for reserved words or conflicts
        reserved_words = ['agent', 'system', 'main', 'core', 'base']
        if agent_name in reserved_words:
            result['issues'].append(f"Reserved word used in name: {agent_name}")
        else:
            result['score'] += 25
            
        result['passed'] = len(result['issues']) == 0
        return result
        
    def test_hygiene_enforcement(self, content: str, agent_name: str) -> Dict[str, Any]:
        """Test presence and quality of codebase hygiene enforcement section"""
        
        result = {
            'passed': False,
            'score': 0,
            'issues': [],
            'critical': False
        }
        
        # Check for main hygiene section
        hygiene_pattern = r'## 🧼 MANDATORY: Codebase Hygiene Enforcement'
        if not re.search(hygiene_pattern, content):
            result['issues'].append("Missing mandatory '🧼 MANDATORY: Codebase Hygiene Enforcement' section")
            result['critical'] = True
            return result
        else:
            result['score'] += 25
            
        # Check for required subsections
        missing_sections = []
        for section in self.required_hygiene_sections:
            section_pattern = f'### {section}'
            if not re.search(section_pattern, content):
                missing_sections.append(section)
                
        if missing_sections:
            result['issues'].append(f"Missing hygiene subsections: {missing_sections}")
            result['score'] -= 10 * len(missing_sections)
        else:
            result['score'] += 25
            
        # Check for specific hygiene principles
        hygiene_keywords = [
            'self-documenting code', 'DRY principle', 'single responsibility',
            'error handling', 'consistent formatting', 'backwards compatibility'
        ]
        
        found_keywords = 0
        for keyword in hygiene_keywords:
            if keyword.lower() in content.lower():
                found_keywords += 1
                
        result['score'] += (found_keywords / len(hygiene_keywords)) * 50
        
        result['passed'] = result['score'] >= 85 and not result['critical']
        return result
        
    def test_content_quality(self, content: str, agent_name: str) -> Dict[str, Any]:
        """Test overall content quality and completeness"""
        
        result = {
            'passed': False,
            'score': 0,
            'issues': []
        }
        
        # Test content length (should be substantial)
        if len(content) < 1000:
            result['issues'].append("Agent definition too short (< 1000 characters)")
        else:
            result['score'] += 20
            
        # Test for core sections
        required_sections = [
            'Core Responsibilities',
            'Technical Implementation', 
            'Best Practices'
        ]
        
        found_sections = 0
        for section in required_sections:
            if section in content:
                found_sections += 1
                
        result['score'] += (found_sections / len(required_sections)) * 40
        
        # Test for code examples or implementation details
        if '```python' in content or '```yaml' in content or '```javascript' in content:
            result['score'] += 20
        else:
            result['issues'].append("Missing code examples or implementation details")
            
        # Test for Docker/deployment configuration
        if 'docker' in content.lower() or 'container' in content.lower():
            result['score'] += 20
        else:
            result['issues'].append("Missing Docker/deployment configuration")
            
        result['passed'] = result['score'] >= 60
        return result
        
    def test_behavior_compliance(self, content: str, agent_name: str) -> Dict[str, Any]:
        """Test agent behavior compliance with hygiene standards"""
        
        result = {
            'passed': False,
            'score': 0,
            'issues': []
        }
        
        # Test for investigation protocol compliance
        investigation_keywords = [
            'comprehensive investigation', 'system analysis', 'investigation protocol'
        ]
        
        found_investigation = any(keyword in content.lower() for keyword in investigation_keywords)
        if found_investigation:
            result['score'] += 30
        else:
            result['issues'].append("Missing investigation protocol references")
            
        # Test for agent coordination mentions
        coordination_keywords = [
            'agent coordination', 'senior agent', 'infrastructure-devops-manager'
        ]
        
        found_coordination = any(keyword in content.lower() for keyword in coordination_keywords)
        if found_coordination:
            result['score'] += 30
        else:
            result['issues'].append("Missing agent coordination guidelines")
            
        # Test for clean code principles implementation
        clean_code_keywords = [
            'clean code', 'code quality', 'refactor', 'optimize'
        ]
        
        found_clean_code = any(keyword in content.lower() for keyword in clean_code_keywords)
        if found_clean_code:
            result['score'] += 40
        else:
            result['issues'].append("Missing clean code implementation details")
            
        result['passed'] = result['score'] >= 70
        return result
        
    def validate_system_integration(self):
        """Validate system-wide integration and coordination"""
        
        # Test essential agents presence
        missing_essential = []
        for essential in self.essential_agents:
            essential_file = self.agents_dir / f"{essential}.md"
            if not essential_file.exists():
                missing_essential.append(essential)
                
        if missing_essential:
            self.test_results['critical_issues'].append({
                'type': 'missing_essential_agents',
                'agents': missing_essential,
                'severity': 'critical',
                'impact': 'System stability compromised'
            })
            
        # Test for duplicate agent definitions
        agent_names = set()
        duplicates = []
        
        for agent_name in self.test_results['detailed_results'].keys():
            if agent_name in agent_names:
                duplicates.append(agent_name)
            agent_names.add(agent_name)
            
        if duplicates:
            self.test_results['critical_issues'].append({
                'type': 'duplicate_agents',
                'agents': duplicates,
                'severity': 'high',
                'impact': 'Resource conflicts and confusion'
            })
            
    def validate_performance_standards(self):
        """Validate performance and efficiency standards"""
        
        performance_issues = []
        
        # Check for agents with missing performance configurations
        for agent_name, results in self.test_results['detailed_results'].items():
            content_file = self.agents_dir / f"{agent_name}.md"
            
            try:
                with open(content_file, 'r') as f:
                    content = f.read()
                    
                # Check for performance configuration
                if 'response_time' not in content and 'performance' not in content:
                    performance_issues.append(agent_name)
                    
            except Exception:
                continue
                
        if performance_issues:
            self.test_results['critical_issues'].append({
                'type': 'missing_performance_config',
                'agents': performance_issues,
                'severity': 'medium',
                'impact': 'Unoptimized resource usage'
            })
            
    def generate_recommendations(self):
        """Generate recommendations for improvement"""
        
        recommendations = []
        
        # Based on compliance percentage
        if self.test_results['compliance_percentage'] < 90:
            recommendations.append(
                "Overall compliance is below 90%. Focus on improving agent hygiene enforcement sections."
            )
            
        # Based on critical issues
        if len(self.test_results['critical_issues']) > 0:
            recommendations.append(
                "Critical issues detected. Address essential agent availability and duplicate definitions immediately."
            )
            
        # Based on test category failures
        for category, results in self.test_results['test_categories'].items():
            if results['failed'] > results['passed']:
                recommendations.append(
                    f"Category '{category}' has high failure rate. Review and improve {category} implementation."
                )
                
        # General recommendations
        recommendations.extend([
            "Implement automated compliance checking in CI/CD pipeline",
            "Create agent template to ensure consistency across new agents",
            "Regular hygiene audits should be scheduled monthly",
            "Consider agent performance optimization for resource efficiency"
        ])
        
        self.test_results['recommendations'] = recommendations
        
    def generate_report(self) -> str:
        """Generate comprehensive QA validation report"""
        
        report = f"""
# 🧪 COMPREHENSIVE AI AGENT QA VALIDATION REPORT

**Generated:** {self.test_results['validation_timestamp']}
**Total Agents Tested:** {self.test_results['total_agents']}
**Compliance Score:** {self.test_results['compliance_percentage']:.1f}%

## 📊 EXECUTIVE SUMMARY

- ✅ **Passed:** {self.test_results['passed_agents']} agents
- ❌ **Failed:** {self.test_results['failed_agents']} agents  
- 🎯 **Success Rate:** {self.test_results['compliance_percentage']:.1f}%
- 🚨 **Critical Issues:** {len(self.test_results['critical_issues'])}

## 🏷️ TEST CATEGORY RESULTS

"""
        
        for category, results in self.test_results['test_categories'].items():
            total = results['passed'] + results['failed']
            if total > 0:
                success_rate = (results['passed'] / total) * 100
                status = "✅ PASS" if success_rate >= 85 else "⚠️ WARN" if success_rate >= 70 else "❌ FAIL"
                
                report += f"""
### {category.replace('_', ' ').title()}
- **Status:** {status}
- **Success Rate:** {success_rate:.1f}%
- **Passed:** {results['passed']} | **Failed:** {results['failed']}
"""
                
                if results['issues']:
                    report += f"- **Top Issues:** {', '.join(results['issues'][:3])}\n"
                    
        # Critical Issues Section
        if self.test_results['critical_issues']:
            report += "\n## 🚨 CRITICAL ISSUES\n\n"
            for issue in self.test_results['critical_issues']:
                report += f"- **{issue['type']}:** {issue.get('impact', 'Unknown impact')}\n"
                if 'agents' in issue:
                    report += f"  - Affected agents: {', '.join(issue['agents'])}\n"
                    
        # Top Failed Agents
        failed_agents = [
            (name, results) for name, results in self.test_results['detailed_results'].items()
            if not results['overall_pass']
        ]
        
        if failed_agents:
            report += "\n## ❌ FAILED AGENTS\n\n"
            for name, results in sorted(failed_agents, key=lambda x: x[1]['score'])[:10]:
                report += f"- **{name}:** Score {results['score']:.1f}% - {', '.join(results['issues'][:2])}\n"
                
        # Recommendations
        report += "\n## 💡 RECOMMENDATIONS\n\n"
        for i, rec in enumerate(self.test_results['recommendations'], 1):
            report += f"{i}. {rec}\n"
            
        # Detailed Test Results
        report += "\n## 📋 DETAILED TEST RESULTS\n\n"
        for agent_name, results in sorted(self.test_results['detailed_results'].items()):
            status = "✅ PASS" if results['overall_pass'] else "❌ FAIL"
            report += f"### {agent_name} - {status} ({results['score']:.1f}%)\n\n"
            
            for test_name, test_result in results['tests'].items():
                test_status = "✅" if test_result['passed'] else "❌"
                report += f"- {test_status} **{test_name.replace('_', ' ').title()}:** {test_result['score']}%\n"
                
                if test_result['issues']:
                    for issue in test_result['issues']:
                        report += f"  - ⚠️ {issue}\n"
            report += "\n"
            
        report += f"""
## 🎯 FINAL VERDICT

**Overall System Status:** {"✅ COMPLIANT" if self.test_results['compliance_percentage'] >= 85 else "❌ NON-COMPLIANT"}

**Next Steps:**
1. {"✅ System ready for production" if self.test_results['compliance_percentage'] >= 90 and len(self.test_results['critical_issues']) == 0 else "⚠️ Address critical issues before production deployment"}
2. {"✅ Agent hygiene standards met" if self.test_results['test_categories']['hygiene_enforcement']['passed'] > self.test_results['test_categories']['hygiene_enforcement']['failed'] else "❌ Improve agent hygiene enforcement"}
3. {"✅ Performance standards acceptable" if len([i for i in self.test_results['critical_issues'] if i['type'] == 'missing_performance_config']) == 0 else "⚠️ Optimize performance configurations"}

---
Generated by Testing QA Validator Agent | SutazAI Platform
"""
        
        return report

def main():
    """Run comprehensive AI agent QA validation"""
    
    validator = AgentQAValidator()
    
    print("🚀 Starting Comprehensive AI Agent QA Validation...")
    print("Following COMPREHENSIVE_INVESTIGATION_PROTOCOL.md requirements")
    print("=" * 70)
    
    # Run validation
    results = validator.validate_all_agents()
    
    # Generate and save report
    report = validator.generate_report()
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save JSON results
    json_file = f"/opt/sutazaiapp/comprehensive_agent_qa_report_{timestamp}.json"
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
        
    # Save markdown report
    md_file = f"/opt/sutazaiapp/COMPREHENSIVE_AGENT_QA_VALIDATION_REPORT_{timestamp}.md"
    with open(md_file, 'w') as f:
        f.write(report)
        
    print("\n" + "=" * 70)
    print("🎉 QA VALIDATION COMPLETE!")
    print(f"📊 Compliance Score: {results['compliance_percentage']:.1f}%")
    print(f"✅ Passed: {results['passed_agents']}/{results['total_agents']} agents")
    print(f"📁 Detailed report: {md_file}")
    print(f"📁 JSON results: {json_file}")
    
    # Print summary
    print("\n📋 QUICK SUMMARY:")
    if results['compliance_percentage'] >= 90:
        print("🟢 EXCELLENT: System highly compliant with hygiene standards")
    elif results['compliance_percentage'] >= 75:
        print("🟡 GOOD: System mostly compliant, minor improvements needed")
    elif results['compliance_percentage'] >= 60:
        print("🟠 FAIR: System needs significant hygiene improvements")
    else:
        print("🔴 POOR: System requires major hygiene enforcement overhaul")
        
    return results

if __name__ == "__main__":
    main()