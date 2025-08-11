#!/usr/bin/env python3
"""
Standalone Hygiene Scanner
A lightweight, on-demand scanner for codebase hygiene validation
"""

import os
import sys
import json
import yaml
import argparse
import logging
from datetime import datetime
from pathlib import Path
import re
from typing import Dict, List, Set, Tuple
import git
from rich.console import Console
from rich.table import Table
from rich.progress import Progress
from tabulate import tabulate

console = Console()

class HygieneScanner:
    def __init__(self, project_root: str, rules_path: str = None):
        self.project_root = Path(project_root)
        self.rules_path = Path(rules_path) if rules_path else self.project_root / "CLAUDE.md"
        self.report_data = {
            "scan_time": datetime.now().isoformat(),
            "project_root": str(self.project_root),
            "violations": [],
            "statistics": {},
            "summary": {}
        }
        
        # Load rule definitions
        self.rules = self._load_rules()
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def _load_rules(self) -> Dict:
        """Load rule definitions from YAML config"""
        rules_file = Path(__file__).parent / "rule_definitions.yaml"
        if rules_file.exists():
            with open(rules_file, 'r') as f:
                return yaml.safe_load(f)
        return self._get_default_rules()
    
    def _get_default_rules(self) -> Dict:
        """Default rule definitions based on CLAUDE.md"""
        return {
            "forbidden_patterns": {
                "fantasy_elements": [
                    r"process\w*", r"configurator\w*", r"transfer\w*", 
                    r"black[_-]?box", r"someday", r"TODO.*later"
                ],
                "garbage_patterns": [
                    r"test_?final.*", r".*\.bak$", r".*_?copy\d*\.", 
                    r"temp_?.*", r"old_?.*", r".*_?v\d+\."
                ],
                "duplicate_patterns": [
                    r"utils?\d+\.", r"helper\d+\.", r"service\d+\."
                ]
            },
            "structure_rules": {
                "required_dirs": ["scripts", "docker", "backend", "frontend"],
                "centralized_dirs": {
                    "scripts": ["*.py", "*.sh"],
                    "docker": ["Dockerfile*", "*.yml"],
                    "docs": ["*.md"]
                }
            },
            "naming_conventions": {
                "python_files": r"^[a-z_]+\.py$",
                "scripts": r"^[a-z-]+\.(py|sh)$",
                "dockerfiles": r"^Dockerfile(\.[a-z-]+)?$"
            }
        }
    
    def scan(self) -> Dict:
        """Run comprehensive hygiene scan"""
        console.print("[bold blue]Starting Hygiene Scan...[/bold blue]")
        
        with Progress() as progress:
            task = progress.add_task("[cyan]Scanning codebase...", total=5)
            
            # 1. Check for forbidden patterns
            progress.update(task, advance=1, description="[cyan]Checking forbidden patterns...")
            self._check_forbidden_patterns()
            
            # 2. Check structure compliance
            progress.update(task, advance=1, description="[cyan]Validating structure...")
            self._check_structure_compliance()
            
            # 3. Check for duplicates
            progress.update(task, advance=1, description="[cyan]Finding duplicates...")
            self._check_duplicates()
            
            # 4. Check naming conventions
            progress.update(task, advance=1, description="[cyan]Validating naming conventions...")
            self._check_naming_conventions()
            
            # 5. Generate summary
            progress.update(task, advance=1, description="[cyan]Generating report...")
            self._generate_summary()
        
        return self.report_data
    
    def _check_forbidden_patterns(self):
        """Check for forbidden patterns in code"""
        violations = []
        
        for category, patterns in self.rules["forbidden_patterns"].items():
            for root, dirs, files in os.walk(self.project_root):
                # Skip virtual environments and build directories
                dirs[:] = [d for d in dirs if d not in {'.venv', 'venv', '__pycache__', 'node_modules', '.git'}]
                
                for file in files:
                    if file.endswith(('.py', '.js', '.jsx', '.ts', '.tsx', '.sh')):
                        file_path = Path(root) / file
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                                
                            for pattern in patterns:
                                matches = re.finditer(pattern, content, re.IGNORECASE)
                                for match in matches:
                                    line_no = content[:match.start()].count('\n') + 1
                                    violations.append({
                                        "type": f"forbidden_{category}",
                                        "file": str(file_path.relative_to(self.project_root)),
                                        "line": line_no,
                                        "pattern": pattern,
                                        "match": match.group()
                                    })
                        except Exception as e:
                            self.logger.warning(f"Could not read {file_path}: {e}")
        
        self.report_data["violations"].extend(violations)
        self.report_data["statistics"]["forbidden_patterns"] = len(violations)
    
    def _check_structure_compliance(self):
        """Check directory structure compliance"""
        violations = []
        
        # Check required directories
        for required_dir in self.rules["structure_rules"]["required_dirs"]:
            dir_path = self.project_root / required_dir
            if not dir_path.exists():
                violations.append({
                    "type": "missing_required_dir",
                    "path": required_dir,
                    "message": f"Required directory '{required_dir}' is missing"
                })
        
        # Check for misplaced files
        for dir_name, patterns in self.rules["structure_rules"]["centralized_dirs"].items():
            expected_dir = self.project_root / dir_name
            
            for root, dirs, files in os.walk(self.project_root):
                current_path = Path(root)
                
                # Skip the expected directory itself
                if current_path == expected_dir:
                    continue
                
                # Skip virtual environments
                if any(part in current_path.parts for part in ['.venv', 'venv', 'node_modules']):
                    continue
                
                for pattern in patterns:
                    for file in files:
                        if Path(file).match(pattern):
                            if current_path != expected_dir:
                                violations.append({
                                    "type": "misplaced_file",
                                    "file": str((current_path / file).relative_to(self.project_root)),
                                    "expected_location": dir_name,
                                    "pattern": pattern
                                })
        
        self.report_data["violations"].extend(violations)
        self.report_data["statistics"]["structure_violations"] = len(violations)
    
    def _check_duplicates(self):
        """Check for duplicate files and code"""
        violations = []
        file_hashes = {}
        
        for root, dirs, files in os.walk(self.project_root):
            dirs[:] = [d for d in dirs if d not in {'.venv', 'venv', '__pycache__', 'node_modules', '.git'}]
            
            for file in files:
                if file.endswith(('.py', '.js', '.jsx', '.ts', '.tsx')):
                    file_path = Path(root) / file
                    
                    # Check for duplicate naming patterns
                    for pattern in self.rules["forbidden_patterns"]["duplicate_patterns"]:
                        if re.match(pattern, file, re.IGNORECASE):
                            violations.append({
                                "type": "duplicate_pattern",
                                "file": str(file_path.relative_to(self.project_root)),
                                "pattern": pattern
                            })
        
        self.report_data["violations"].extend(violations)
        self.report_data["statistics"]["duplicate_violations"] = len(violations)
    
    def _check_naming_conventions(self):
        """Check file naming conventions"""
        violations = []
        
        for root, dirs, files in os.walk(self.project_root):
            dirs[:] = [d for d in dirs if d not in {'.venv', 'venv', '__pycache__', 'node_modules', '.git'}]
            
            for file in files:
                file_path = Path(root) / file
                relative_path = file_path.relative_to(self.project_root)
                
                # Check Python files
                if file.endswith('.py') and not re.match(self.rules["naming_conventions"]["python_files"], file):
                    violations.append({
                        "type": "naming_convention",
                        "file": str(relative_path),
                        "expected_pattern": self.rules["naming_conventions"]["python_files"]
                    })
                
                # Check scripts
                if relative_path.parts[0] == "scripts" and not re.match(self.rules["naming_conventions"]["scripts"], file):
                    violations.append({
                        "type": "naming_convention",
                        "file": str(relative_path),
                        "expected_pattern": self.rules["naming_conventions"]["scripts"]
                    })
        
        self.report_data["violations"].extend(violations)
        self.report_data["statistics"]["naming_violations"] = len(violations)
    
    def _generate_summary(self):
        """Generate scan summary"""
        total_violations = len(self.report_data["violations"])
        
        self.report_data["summary"] = {
            "total_violations": total_violations,
            "violation_types": {},
            "severity": "PASS" if total_violations == 0 else ("WARNING" if total_violations < 50 else "FAIL"),
            "scan_duration": (datetime.now() - datetime.fromisoformat(self.report_data["scan_time"])).total_seconds()
        }
        
        # Count violations by type
        for violation in self.report_data["violations"]:
            vtype = violation["type"]
            self.report_data["summary"]["violation_types"][vtype] = \
                self.report_data["summary"]["violation_types"].get(vtype, 0) + 1
    
    def save_report(self, output_path: str):
        """Save report to file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save JSON report
        json_path = output_path.with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump(self.report_data, f, indent=2)
        
        # Save HTML report
        html_path = output_path.with_suffix('.html')
        self._generate_html_report(html_path)
        
        # Print summary to console
        self._print_summary()
        
        return json_path, html_path
    
    def _generate_html_report(self, output_path: Path):
        """Generate HTML report"""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Hygiene Scan Report - {datetime.now().strftime('%Y-%m-%d %H:%M')}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1, h2, h3 {{ color: #333; }}
        .summary {{ background-color: #e8f4f8; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .pass {{ color: #28a745; font-weight: bold; }}
        .warning {{ color: #ffc107; font-weight: bold; }}
        .fail {{ color: #dc3545; font-weight: bold; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ text-align: left; padding: 12px; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f8f9fa; font-weight: bold; }}
        tr:hover {{ background-color: #f5f5f5; }}
        .violation-type {{ padding: 2px 8px; border-radius: 3px; font-size: 12px; }}
        .forbidden {{ background-color: #ffebee; color: #c62828; }}
        .structure {{ background-color: #e3f2fd; color: #1565c0; }}
        .duplicate {{ background-color: #fff3e0; color: #e65100; }}
        .naming {{ background-color: #f3e5f5; color: #6a1b9a; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Hygiene Scan Report</h1>
        <div class="summary">
            <p><strong>Scan Time:</strong> {self.report_data['scan_time']}</p>
            <p><strong>Project Root:</strong> {self.report_data['project_root']}</p>
            <p><strong>Total Violations:</strong> {self.report_data['summary']['total_violations']}</p>
            <p><strong>Status:</strong> <span class="{self.report_data['summary']['severity'].lower()}">{self.report_data['summary']['severity']}</span></p>
        </div>
        
        <h2>Violation Summary</h2>
        <table>
            <tr>
                <th>Violation Type</th>
                <th>Count</th>
            </tr>
"""
        
        for vtype, count in self.report_data['summary']['violation_types'].items():
            category = vtype.split('_')[0]
            html_content += f"""
            <tr>
                <td><span class="violation-type {category}">{vtype}</span></td>
                <td>{count}</td>
            </tr>
"""
        
        html_content += """
        </table>
        
        <h2>Detailed Violations</h2>
        <table>
            <tr>
                <th>Type</th>
                <th>File</th>
                <th>Details</th>
            </tr>
"""
        
        for violation in self.report_data['violations'][:100]:  # Show first 100
            details = violation.get('message', '')
            if 'line' in violation:
                details = f"Line {violation['line']}: {violation.get('match', '')}"
            elif 'pattern' in violation:
                details = f"Pattern: {violation['pattern']}"
            
            category = violation['type'].split('_')[0]
            html_content += f"""
            <tr>
                <td><span class="violation-type {category}">{violation['type']}</span></td>
                <td>{violation.get('file', violation.get('path', 'N/A'))}</td>
                <td>{details}</td>
            </tr>
"""
        
        if len(self.report_data['violations']) > 100:
            html_content += f"""
            <tr>
                <td colspan="3"><em>... and {len(self.report_data['violations']) - 100} more violations</em></td>
            </tr>
"""
        
        html_content += """
        </table>
    </div>
</body>
</html>
"""
        
        with open(output_path, 'w') as f:
            f.write(html_content)
    
    def _print_summary(self):
        """Print summary to console"""
        console.print("\n[bold green]Scan Complete![/bold green]\n")
        
        # Create summary table
        table = Table(title="Hygiene Scan Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        
        table.add_row("Total Violations", str(self.report_data['summary']['total_violations']))
        table.add_row("Scan Duration", f"{self.report_data['summary']['scan_duration']:.2f}s")
        table.add_row("Status", self.report_data['summary']['severity'])
        
        console.print(table)
        
        # Show violation breakdown
        if self.report_data['summary']['violation_types']:
            console.print("\n[bold]Violations by Type:[/bold]")
            for vtype, count in self.report_data['summary']['violation_types'].items():
                console.print(f"  • {vtype}: {count}")


def main():
    parser = argparse.ArgumentParser(description="Standalone Hygiene Scanner")
    parser.add_argument("--project-root", default="/app/project", help="Project root directory")
    parser.add_argument("--rules-path", help="Path to CLAUDE.md rules file")
    parser.add_argument("--output", default="/app/reports/hygiene-report", help="Output report path")
    parser.add_argument("--scan-once", action="store_true", help="Run once and exit")
    
    args = parser.parse_args()
    
    scanner = HygieneScanner(args.project_root, args.rules_path)
    report = scanner.scan()
    
    # Save report
    json_path, html_path = scanner.save_report(args.output)
    console.print(f"\n[bold]Reports saved:[/bold]")
    console.print(f"  • JSON: {json_path}")
    console.print(f"  • HTML: {html_path}")
    
    # Exit with appropriate code
    sys.exit(0 if report['summary']['severity'] == 'PASS' else 1)


if __name__ == "__main__":
    main()