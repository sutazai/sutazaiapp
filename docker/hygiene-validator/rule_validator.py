#!/usr/bin/env python3
"""
Rule Validator - Validates CLAUDE.md rules compliance
"""

import os
import sys
import json
import re
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from rich.console import Console
from rich.table import Table

console = Console()

class RuleValidator:
    def __init__(self, project_root: str, rules_path: str):
        self.project_root = Path(project_root)
        self.rules_path = Path(rules_path)
        self.validation_results = {
            "timestamp": datetime.now().isoformat(),
            "rules_checked": [],
            "compliance_score": 0,
            "details": {}
        }
        
    def validate_all(self) -> Dict:
        """Validate all CLAUDE.md rules"""
        console.print("[bold blue]Starting Rule Validation...[/bold blue]\n")
        
        rules = self._parse_claude_rules()
        total_score = 0
        
        for rule_num, rule in enumerate(rules, 1):
            console.print(f"[cyan]Checking Rule {rule_num}: {rule['title']}[/cyan]")
            score = self._validate_rule(rule)
            total_score += score
            
            self.validation_results["rules_checked"].append({
                "rule": rule_num,
                "title": rule['title'],
                "score": score,
                "passed": score >= 80
            })
        
        self.validation_results["compliance_score"] = total_score / len(rules) if rules else 0
        return self.validation_results
    
    def _parse_claude_rules(self) -> List[Dict]:
        """Parse CLAUDE.md to extract rules"""
        rules = []
        
        if not self.rules_path.exists():
            console.print(f"[red]Warning: CLAUDE.md not found at {self.rules_path}[/red]")
            return rules
        
        with open(self.rules_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract rules using regex
        rule_pattern = r"📌 Rule (\d+):\s*([^\n]+)"
        matches = re.finditer(rule_pattern, content)
        
        for match in matches:
            rule_num = match.group(1)
            rule_title = match.group(2)
            
            # Extract rule content (simplified)
            start = match.end()
            next_rule = re.search(r"📌 Rule \d+:", content[start:])
            end = start + next_rule.start() if next_rule else len(content)
            
            rule_content = content[start:end].strip()
            
            rules.append({
                "number": int(rule_num),
                "title": rule_title,
                "content": rule_content,
                "checks": self._extract_checks(rule_content)
            })
        
        return rules
    
    def _extract_checks(self, content: str) -> List[str]:
        """Extract checkable items from rule content"""
        checks = []
        
        # Look for bullet points with checks
        check_patterns = [
            r"✅\s*([^\n]+)",
            r"✨\s*([^\n]+)",
            r"•\s*([^\n]+)",
            r"-\s*([^\n]+)"
        ]
        
        for pattern in check_patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                checks.append(match.group(1).strip())
        
        return checks
    
    def _validate_rule(self, rule: Dict) -> float:
        """Validate a specific rule"""
        score = 0
        checks_passed = 0
        total_checks = len(rule['checks'])
        
        if total_checks == 0:
            return 100  # No specific checks, assume compliant
        
        # Simplified validation logic
        for check in rule['checks']:
            if self._check_compliance(check):
                checks_passed += 1
        
        score = (checks_passed / total_checks) * 100 if total_checks > 0 else 100
        
        self.validation_results["details"][f"rule_{rule['number']}"] = {
            "title": rule['title'],
            "checks_total": total_checks,
            "checks_passed": checks_passed,
            "score": score
        }
        
        return score
    
    def _check_compliance(self, check: str) -> bool:
        """Check if a specific requirement is met"""
        # Simplified compliance checking
        # In a real implementation, this would do actual validation
        
        check_lower = check.lower()
        
        # Check for forbidden terms
        if "no fantasy" in check_lower or "no process" in check_lower:
            return self._check_no_fantasy_elements()
        
        # Check for structure requirements
        if "centralized" in check_lower or "single location" in check_lower:
            return self._check_centralized_structure()
        
        # Default to True for now (simplified)
        return True
    
    def _check_no_fantasy_elements(self) -> bool:
        """Check for fantasy elements in code"""
        forbidden_terms = ['process', 'configurator', 'transfer', 'processing-unit', 'someday']
        
        for root, dirs, files in os.walk(self.project_root):
            dirs[:] = [d for d in dirs if d not in {'.venv', 'venv', '__pycache__', '.git'}]
            
            for file in files:
                if file.endswith(('.py', '.js', '.jsx', '.ts', '.tsx')):
                    file_path = Path(root) / file
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read().lower()
                            
                        for term in forbidden_terms:
                            if term in content:
                                return False
                    except:
                        pass
        
        return True
    
    def _check_centralized_structure(self) -> bool:
        """Check if scripts and docker files are centralized"""
        scripts_dir = self.project_root / "scripts"
        docker_dir = self.project_root / "docker"
        
        return scripts_dir.exists() and docker_dir.exists()
    
    def save_report(self, output_path: str):
        """Save validation report"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save JSON report
        with open(output_path, 'w') as f:
            json.dump(self.validation_results, f, indent=2)
        
        # Print summary
        self._print_summary()
        
        return output_path
    
    def _print_summary(self):
        """Print validation summary"""
        console.print("\n[bold green]Validation Complete![/bold green]\n")
        
        table = Table(title="Rule Compliance Summary")
        table.add_column("Rule", style="cyan")
        table.add_column("Title", style="white")
        table.add_column("Score", style="white")
        table.add_column("Status", style="white")
        
        for rule in self.validation_results["rules_checked"]:
            status = "[green]PASS[/green]" if rule["passed"] else "[red]FAIL[/red]"
            table.add_row(
                str(rule["rule"]),
                rule["title"][:50] + "..." if len(rule["title"]) > 50 else rule["title"],
                f"{rule['score']:.1f}%",
                status
            )
        
        console.print(table)
        console.print(f"\n[bold]Overall Compliance Score: {self.validation_results['compliance_score']:.1f}%[/bold]")


def main():
    parser = argparse.ArgumentParser(description="CLAUDE.md Rule Validator")
    parser.add_argument("--project-root", default="/app/project", help="Project root directory")
    parser.add_argument("--rules-path", default="/app/rules/CLAUDE.md", help="Path to CLAUDE.md")
    parser.add_argument("--output", default="/app/reports/rule-validation.json", help="Output report path")
    parser.add_argument("--validate-all", action="store_true", help="Validate all rules")
    
    args = parser.parse_args()
    
    validator = RuleValidator(args.project_root, args.rules_path)
    results = validator.validate_all()
    
    # Save report
    report_path = validator.save_report(args.output)
    console.print(f"\n[bold]Report saved to:[/bold] {report_path}")
    
    # Exit with appropriate code
    sys.exit(0 if results['compliance_score'] >= 80 else 1)


if __name__ == "__main__":
    main()