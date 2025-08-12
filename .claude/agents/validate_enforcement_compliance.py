#!/usr/bin/env python3
"""
ENFORCEMENT COMPLIANCE VALIDATOR
Validates that all agents have proper rule enforcement and identifies violations.
Part of the Zero-Tolerance Rule Enforcement System.
"""

import os
import re
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

class EnforcementValidator:
    """Validates agent compliance with zero-tolerance enforcement rules."""
    
    # Critical enforcement markers that MUST be present
    REQUIRED_MARKERS = [
        "MANDATORY RULE ENFORCEMENT SYSTEM",
        "PRE-EXECUTION VALIDATION",
        "CLAUDE.md",
        "IMPORTANT/*",
        "CHANGELOG",
        "ZERO TOLERANCE"
    ]
    
    # Specific rules that must be referenced
    REQUIRED_RULES = [
        "Rule 1.*NO FANTASY.*CONCEPTUAL",
        "Rule 2.*NEVER BREAK",
        "Rule 3.*ANALYZE EVERYTHING",
        "Rule 4.*REUSE BEFORE CREATING",
        "Rule 19.*MANDATORY CHANGELOG"
    ]
    
    # Proactive trigger indicators
    TRIGGER_INDICATORS = [
        "PROACTIVE TRIGGER",
        "Automatically activate",
        "trigger",
        "Monitor",
        "on: file_modification",
        "on: deployment"
    ]
    
    # Cross-validation requirements
    CROSS_VALIDATION = [
        "code-reviewer",
        "testing-qa-validator",
        "rules-enforcer",
        "Cross.*validation",
        "trigger validation"
    ]
    
    def __init__(self):
        self.agents_dir = Path('/opt/sutazaiapp/.claude/agents')
        self.results = {
            'compliant': [],
            'non_compliant': [],
            'partial_compliant': [],
            'missing_triggers': [],
            'missing_cross_validation': []
        }
        self.detailed_report = []
    
    def validate_agent(self, filepath: Path) -> Dict:
        """Validate a single agent file for enforcement compliance."""
        agent_name = filepath.stem
        
        try:
            with open(filepath, 'r') as f:
                content = f.read()
            
            # Check for required markers
            markers_found = sum(1 for marker in self.REQUIRED_MARKERS 
                              if marker.lower() in content.lower())
            marker_compliance = markers_found / len(self.REQUIRED_MARKERS)
            
            # Check for specific rules
            rules_found = sum(1 for rule in self.REQUIRED_RULES 
                            if re.search(rule, content, re.IGNORECASE))
            rule_compliance = rules_found / len(self.REQUIRED_RULES)
            
            # Check for proactive triggers
            has_triggers = any(indicator.lower() in content.lower() 
                             for indicator in self.TRIGGER_INDICATORS)
            
            # Check for cross-validation
            cross_val_found = sum(1 for val in self.CROSS_VALIDATION 
                                if re.search(val, content, re.IGNORECASE))
            has_cross_validation = cross_val_found >= 3  # At least 3 validators mentioned
            
            # Determine overall compliance
            is_compliant = (
                marker_compliance == 1.0 and 
                rule_compliance >= 0.8 and 
                has_triggers and 
                has_cross_validation
            )
            
            # Check if it's a supreme validator
            is_supreme = agent_name in ['code-reviewer', 'testing-qa-validator', 
                                       'rules-enforcer', 'mega-code-auditor']
            
            return {
                'agent': agent_name,
                'filepath': str(filepath),
                'marker_compliance': marker_compliance,
                'rule_compliance': rule_compliance,
                'has_triggers': has_triggers,
                'has_cross_validation': has_cross_validation,
                'is_compliant': is_compliant,
                'is_supreme': is_supreme,
                'missing_markers': [m for m in self.REQUIRED_MARKERS 
                                  if m.lower() not in content.lower()],
                'missing_rules': [r.split('.*')[0] for r in self.REQUIRED_RULES 
                                if not re.search(r, content, re.IGNORECASE)]
            }
            
        except Exception as e:
            return {
                'agent': agent_name,
                'filepath': str(filepath),
                'error': str(e),
                'is_compliant': False
            }
    
    def categorize_results(self, validation: Dict) -> None:
        """Categorize validation results for reporting."""
        agent = validation['agent']
        
        if validation.get('error'):
            self.results['non_compliant'].append(agent)
        elif validation['is_compliant']:
            self.results['compliant'].append(agent)
        elif validation['marker_compliance'] >= 0.5 and validation['rule_compliance'] >= 0.5:
            self.results['partial_compliant'].append(agent)
        else:
            self.results['non_compliant'].append(agent)
        
        if not validation.get('has_triggers', False):
            self.results['missing_triggers'].append(agent)
        
        if not validation.get('has_cross_validation', False):
            self.results['missing_cross_validation'].append(agent)
        
        self.detailed_report.append(validation)
    
    def run_validation(self) -> None:
        """Run validation on all agent files."""
        if not self.agents_dir.exists():
            print(f"❌ Agents directory not found: {self.agents_dir}")
            return
        
        # Get all agent files
        agent_files = sorted(self.agents_dir.glob('*.md'))
        
        # Skip meta files
        skip_files = {'ZERO_TOLERANCE_ENFORCEMENT_STRATEGY.md', 'agent-overview.md'}
        agent_files = [f for f in agent_files if f.name not in skip_files]
        
        print(f"🔍 Validating {len(agent_files)} agents for enforcement compliance...\n")
        
        # Validate each agent
        for filepath in agent_files:
            validation = self.validate_agent(filepath)
            self.categorize_results(validation)
    
    def generate_report(self) -> str:
        """Generate comprehensive compliance report."""
        total = len(self.detailed_report)
        compliant_count = len(self.results['compliant'])
        partial_count = len(self.results['partial_compliant'])
        non_compliant_count = len(self.results['non_compliant'])
        
        report = []
        report.append("=" * 80)
        report.append("🚨 ZERO-TOLERANCE ENFORCEMENT COMPLIANCE REPORT")
        report.append("=" * 80)
        report.append(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Agents Validated: {total}")
        report.append("")
        
        # Summary Statistics
        report.append("📊 COMPLIANCE SUMMARY")
        report.append("-" * 40)
        compliance_rate = (compliant_count / total * 100) if total > 0 else 0
        report.append(f"✅ Fully Compliant: {compliant_count} ({compliance_rate:.1f}%)")
        report.append(f"🟡 Partially Compliant: {partial_count} ({partial_count/total*100:.1f}%)")
        report.append(f"❌ Non-Compliant: {non_compliant_count} ({non_compliant_count/total*100:.1f}%)")
        report.append("")
        
        # Trigger and Cross-Validation Status
        report.append("🔄 PROACTIVE ENFORCEMENT STATUS")
        report.append("-" * 40)
        triggers_missing = len(self.results['missing_triggers'])
        cross_val_missing = len(self.results['missing_cross_validation'])
        report.append(f"Agents with Proactive Triggers: {total - triggers_missing}/{total}")
        report.append(f"Agents with Cross-Validation: {total - cross_val_missing}/{total}")
        report.append("")
        
        # Supreme Validators Status
        report.append("👑 SUPREME VALIDATORS STATUS")
        report.append("-" * 40)
        supreme_validators = [r for r in self.detailed_report if r.get('is_supreme')]
        for validator in supreme_validators:
            status = "✅ COMPLIANT" if validator['is_compliant'] else "❌ NON-COMPLIANT"
            report.append(f"{validator['agent']}: {status}")
        report.append("")
        
        # Non-Compliant Agents (Critical)
        if self.results['non_compliant']:
            report.append("❌ NON-COMPLIANT AGENTS (REQUIRES IMMEDIATE ACTION)")
            report.append("-" * 40)
            for agent in self.results['non_compliant'][:10]:  # Show first 10
                agent_data = next((r for r in self.detailed_report if r['agent'] == agent), {})
                if agent_data.get('missing_markers'):
                    report.append(f"• {agent}")
                    report.append(f"  Missing: {', '.join(agent_data['missing_markers'][:3])}")
            if len(self.results['non_compliant']) > 10:
                report.append(f"  ... and {len(self.results['non_compliant']) - 10} more")
            report.append("")
        
        # Partially Compliant (Need Enhancement)
        if self.results['partial_compliant']:
            report.append("🟡 PARTIALLY COMPLIANT AGENTS (ENHANCEMENT NEEDED)")
            report.append("-" * 40)
            for agent in self.results['partial_compliant'][:5]:
                agent_data = next((r for r in self.detailed_report if r['agent'] == agent), {})
                report.append(f"• {agent}")
                if agent_data.get('missing_rules'):
                    report.append(f"  Missing Rules: {', '.join(agent_data['missing_rules'])}")
            if len(self.results['partial_compliant']) > 5:
                report.append(f"  ... and {len(self.results['partial_compliant']) - 5} more")
            report.append("")
        
        # Enforcement Gaps
        report.append("⚠️ ENFORCEMENT GAPS")
        report.append("-" * 40)
        if triggers_missing > 0:
            report.append(f"Missing Proactive Triggers: {triggers_missing} agents")
            report.append(f"  Examples: {', '.join(self.results['missing_triggers'][:5])}")
        if cross_val_missing > 0:
            report.append(f"Missing Cross-Validation: {cross_val_missing} agents")
            report.append(f"  Examples: {', '.join(self.results['missing_cross_validation'][:5])}")
        report.append("")
        
        # Recommendations
        report.append("💡 RECOMMENDATIONS")
        report.append("-" * 40)
        if compliance_rate < 100:
            report.append(f"1. Run enforce_rules_in_all_agents.py to inject enforcement")
            report.append(f"2. Priority: Fix {non_compliant_count} non-compliant agents first")
            report.append(f"3. Enhance {partial_count} partially compliant agents")
            report.append(f"4. Add proactive triggers to {triggers_missing} agents")
            report.append(f"5. Implement cross-validation in {cross_val_missing} agents")
        else:
            report.append("✅ All agents are fully compliant!")
            report.append("🎉 Zero-Tolerance Enforcement System is fully operational!")
        report.append("")
        
        # Final Status
        report.append("=" * 80)
        if compliance_rate == 100:
            report.append("🟢 SYSTEM STATUS: FULLY PROTECTED - ZERO TOLERANCE ACTIVE")
        elif compliance_rate >= 80:
            report.append("🟡 SYSTEM STATUS: PARTIALLY PROTECTED - ENHANCEMENT NEEDED")
        else:
            report.append("🔴 SYSTEM STATUS: VULNERABLE - IMMEDIATE ACTION REQUIRED")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def save_detailed_report(self) -> None:
        """Save detailed JSON report for analysis."""
        report_path = Path('/opt/sutazaiapp/.claude/agents/enforcement_compliance_report.json')
        
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_agents': len(self.detailed_report),
                'compliant': len(self.results['compliant']),
                'partial_compliant': len(self.results['partial_compliant']),
                'non_compliant': len(self.results['non_compliant']),
                'missing_triggers': len(self.results['missing_triggers']),
                'missing_cross_validation': len(self.results['missing_cross_validation'])
            },
            'results': self.results,
            'detailed_validations': self.detailed_report
        }
        
        try:
            with open(report_path, 'w') as f:
                json.dump(report_data, f, indent=2)
            print(f"📄 Detailed report saved to: {report_path}")
        except Exception as e:
            print(f"⚠️ Could not save detailed report: {e}")

def main():
    """Main execution."""
    validator = EnforcementValidator()
    
    # Run validation
    validator.run_validation()
    
    # Generate and print report
    report = validator.generate_report()
    print(report)
    
    # Save detailed report
    validator.save_detailed_report()
    
    # Return exit code based on compliance
    compliant_count = len(validator.results['compliant'])
    total = len(validator.detailed_report)
    if total > 0:
        compliance_rate = compliant_count / total
        if compliance_rate == 1.0:
            exit(0)  # Full compliance
        elif compliance_rate >= 0.8:
            exit(1)  # Partial compliance
        else:
            exit(2)  # Non-compliance

if __name__ == "__main__":
    main()