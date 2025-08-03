#!/usr/bin/env python3
"""
Purpose: Consolidates multiple hygiene reports into a single comprehensive report
Usage: python consolidate-reports.py --input-dir ./reports --output report.md --format markdown
Requirements: Python 3.8+, pyyaml, jinja2
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import yaml

try:
    from jinja2 import Template
except ImportError:
    print("Warning: jinja2 not installed. HTML output will be limited.")
    Template = None


class ReportConsolidator:
    """Consolidates multiple hygiene analysis and enforcement reports"""
    
    def __init__(self, input_dir: str = None, input_pattern: str = None):
        self.input_dir = Path(input_dir) if input_dir else Path.cwd()
        self.input_pattern = input_pattern or "*.json"
        self.consolidated_data = {
            "timestamp": datetime.now().isoformat(),
            "analysis_reports": [],
            "enforcement_reports": [],
            "summary": {},
            "violations": {},
            "recommendations": []
        }
    
    def find_reports(self) -> List[Path]:
        """Find all report files matching the pattern"""
        if self.input_dir.is_dir():
            return list(self.input_dir.glob(self.input_pattern))
        else:
            # Handle comma-separated list of files
            return [Path(f) for f in self.input_pattern.split(",") if Path(f).exists()]
    
    def load_report(self, file_path: Path) -> Dict[str, Any]:
        """Load a JSON report file"""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return {}
    
    def consolidate(self):
        """Consolidate all reports into a single structure"""
        report_files = self.find_reports()
        
        for report_file in report_files:
            report_data = self.load_report(report_file)
            
            if "enforcement" in report_file.name:
                self.consolidated_data["enforcement_reports"].append({
                    "file": str(report_file),
                    "data": report_data
                })
            else:
                self.consolidated_data["analysis_reports"].append({
                    "file": str(report_file),
                    "data": report_data
                })
        
        # Aggregate violations
        self._aggregate_violations()
        
        # Calculate summary statistics
        self._calculate_summary()
        
        # Generate recommendations
        self._generate_recommendations()
    
    def _aggregate_violations(self):
        """Aggregate violations from all reports"""
        violations_by_rule = {}
        
        for report in self.consolidated_data["analysis_reports"]:
            report_violations = report["data"].get("violations", {})
            
            for rule, rule_violations in report_violations.items():
                if rule not in violations_by_rule:
                    violations_by_rule[rule] = []
                
                if isinstance(rule_violations, list):
                    violations_by_rule[rule].extend(rule_violations)
                elif isinstance(rule_violations, dict):
                    violations_by_rule[rule].append(rule_violations)
        
        # Organize by priority
        self.consolidated_data["violations"] = {
            "critical": [],
            "high": [],
            "medium": [],
            "low": []
        }
        
        # Map rules to priorities
        rule_priorities = {
            "1": "critical", "2": "critical", "3": "critical",
            "4": "high", "5": "high", "8": "high", "11": "high", "12": "high",
            "6": "medium", "7": "medium", "9": "medium", "10": "medium",
            "13": "critical",  # Special case - garbage is critical
            "14": "low", "15": "low", "16": "low"
        }
        
        for rule, violations in violations_by_rule.items():
            priority = rule_priorities.get(str(rule), "medium")
            self.consolidated_data["violations"][priority].extend([
                {**v, "rule": rule} for v in violations if isinstance(v, dict)
            ])
    
    def _calculate_summary(self):
        """Calculate summary statistics"""
        violations = self.consolidated_data["violations"]
        
        total_violations = sum(len(v) for v in violations.values())
        
        # Calculate hygiene score (100 - penalties)
        penalties = {
            "critical": 10,  # Each critical violation = -10 points
            "high": 5,       # Each high violation = -5 points
            "medium": 2,     # Each medium violation = -2 points
            "low": 1         # Each low violation = -1 point
        }
        
        score = 100
        for priority, violation_list in violations.items():
            score -= len(violation_list) * penalties.get(priority, 0)
        
        score = max(0, score)  # Don't go below 0
        
        self.consolidated_data["summary"] = {
            "total_violations": total_violations,
            "critical_violations": len(violations["critical"]),
            "high_violations": len(violations["high"]),
            "medium_violations": len(violations["medium"]),
            "low_violations": len(violations["low"]),
            "hygiene_score": score,
            "reports_analyzed": len(self.consolidated_data["analysis_reports"]),
            "enforcement_actions": len(self.consolidated_data["enforcement_reports"])
        }
    
    def _generate_recommendations(self):
        """Generate actionable recommendations based on violations"""
        recommendations = []
        violations = self.consolidated_data["violations"]
        
        if violations["critical"]:
            recommendations.append({
                "priority": "URGENT",
                "title": "Address Critical Violations Immediately",
                "description": f"Found {len(violations['critical'])} critical violations that must be resolved before merge.",
                "actions": [
                    "Review each critical violation in the detailed report",
                    "Fix violations or provide justification for exceptions",
                    "Re-run hygiene checks after fixes"
                ]
            })
        
        if len(violations["high"]) > 5:
            recommendations.append({
                "priority": "HIGH",
                "title": "Reduce High Priority Technical Debt",
                "description": f"{len(violations['high'])} high priority violations indicate significant technical debt.",
                "actions": [
                    "Schedule a technical debt sprint",
                    "Prioritize fixing structural issues (Rules 4-8, 11-12)",
                    "Consider running comprehensive enforcement weekly"
                ]
            })
        
        if self.consolidated_data["summary"]["hygiene_score"] < 70:
            recommendations.append({
                "priority": "HIGH",
                "title": "Improve Overall Hygiene Score",
                "description": f"Current score of {self.consolidated_data['summary']['hygiene_score']}% is below acceptable threshold.",
                "actions": [
                    "Enable pre-commit hooks for local validation",
                    "Conduct team training on CLAUDE.md rules",
                    "Implement automated fixes for safe rules (6, 8, 13, 15)"
                ]
            })
        
        self.consolidated_data["recommendations"] = recommendations
    
    def format_markdown(self, include_recommendations: bool = True) -> str:
        """Format report as Markdown"""
        summary = self.consolidated_data["summary"]
        violations = self.consolidated_data["violations"]
        
        md = f"""# 🧹 Codebase Hygiene Report

Generated: {self.consolidated_data['timestamp']}

## Summary

| Metric | Value | Status |
|--------|-------|--------|
| Hygiene Score | {summary['hygiene_score']:.1f}% | {'✅' if summary['hygiene_score'] >= 70 else '❌'} |
| Critical Violations | {summary['critical_violations']} | {'✅' if summary['critical_violations'] == 0 else '❌'} |
| High Violations | {summary['high_violations']} | {'⚠️' if summary['high_violations'] > 5 else '✅'} |
| Medium Violations | {summary['medium_violations']} | - |
| Low Violations | {summary['low_violations']} | - |

## Violations by Priority

"""
        
        # Add violation details
        for priority in ["critical", "high", "medium", "low"]:
            if violations[priority]:
                md += f"### {priority.title()} Priority Violations ({len(violations[priority])})\n\n"
                
                # Group by rule
                by_rule = {}
                for v in violations[priority]:
                    rule = v.get("rule", "unknown")
                    if rule not in by_rule:
                        by_rule[rule] = []
                    by_rule[rule].append(v)
                
                for rule, rule_violations in sorted(by_rule.items()):
                    md += f"#### Rule {rule} ({len(rule_violations)} violations)\n\n"
                    
                    for i, v in enumerate(rule_violations[:5]):  # Show first 5
                        md += f"- {v.get('file', 'Unknown file')}: {v.get('message', 'No message')}\n"
                    
                    if len(rule_violations) > 5:
                        md += f"- ... and {len(rule_violations) - 5} more\n"
                    
                    md += "\n"
        
        # Add recommendations
        if include_recommendations and self.consolidated_data["recommendations"]:
            md += "## Recommendations\n\n"
            
            for rec in self.consolidated_data["recommendations"]:
                md += f"### {rec['priority']}: {rec['title']}\n\n"
                md += f"{rec['description']}\n\n"
                md += "**Actions:**\n"
                for action in rec['actions']:
                    md += f"- {action}\n"
                md += "\n"
        
        # Add enforcement results if any
        if self.consolidated_data["enforcement_reports"]:
            md += "## Enforcement Actions\n\n"
            
            for report in self.consolidated_data["enforcement_reports"]:
                agent = report["data"].get("agent", "Unknown")
                status = report["data"].get("status", "Unknown")
                md += f"- **{agent}**: {status}\n"
        
        return md
    
    def format_json(self) -> str:
        """Format report as JSON"""
        return json.dumps(self.consolidated_data, indent=2)
    
    def format_html(self, template: str = "default") -> str:
        """Format report as HTML"""
        if not Template:
            return "<html><body><pre>" + self.format_markdown() + "</pre></body></html>"
        
        # Default HTML template
        html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Hygiene Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .summary { background: #f0f0f0; padding: 15px; border-radius: 5px; }
        .score { font-size: 48px; font-weight: bold; }
        .good { color: #28a745; }
        .warning { color: #ffc107; }
        .bad { color: #dc3545; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .violation { margin: 10px 0; padding: 10px; border-left: 4px solid; }
        .critical { border-color: #dc3545; background: #f8d7da; }
        .high { border-color: #ffc107; background: #fff3cd; }
        .medium { border-color: #17a2b8; background: #d1ecf1; }
        .low { border-color: #6c757d; background: #e2e3e5; }
    </style>
</head>
<body>
    <h1>🧹 Codebase Hygiene Report</h1>
    
    <div class="summary">
        <div class="score {{ 'good' if summary.hygiene_score >= 80 else 'warning' if summary.hygiene_score >= 70 else 'bad' }}">
            {{ summary.hygiene_score }}%
        </div>
        <p>Generated: {{ timestamp }}</p>
    </div>
    
    <table>
        <tr>
            <th>Metric</th>
            <th>Value</th>
            <th>Status</th>
        </tr>
        <tr>
            <td>Critical Violations</td>
            <td>{{ summary.critical_violations }}</td>
            <td>{{ '✅' if summary.critical_violations == 0 else '❌' }}</td>
        </tr>
        <tr>
            <td>High Violations</td>
            <td>{{ summary.high_violations }}</td>
            <td>{{ '⚠️' if summary.high_violations > 5 else '✅' }}</td>
        </tr>
        <tr>
            <td>Medium Violations</td>
            <td>{{ summary.medium_violations }}</td>
            <td>-</td>
        </tr>
        <tr>
            <td>Low Violations</td>
            <td>{{ summary.low_violations }}</td>
            <td>-</td>
        </tr>
    </table>
    
    <!-- Add more content here -->
</body>
</html>
"""
        
        template_obj = Template(html_template)
        return template_obj.render(**self.consolidated_data)
    
    def format_gitlab(self) -> str:
        """Format for GitLab-specific features"""
        # Generate both Markdown and JUnit XML
        return self.format_markdown()
    
    def format_jenkins(self) -> str:
        """Format for Jenkins-specific features"""
        # Similar to default but with Jenkins-specific formatting
        return self.format_html(template="jenkins")


def main():
    parser = argparse.ArgumentParser(description="Consolidate hygiene reports")
    parser.add_argument("--input-dir", help="Directory containing reports")
    parser.add_argument("--input-pattern", default="*.json", help="File pattern to match")
    parser.add_argument("--enforcement-pattern", help="Pattern for enforcement reports")
    parser.add_argument("--output", required=True, help="Output file path")
    parser.add_argument("--format", choices=["markdown", "json", "html"], default="markdown")
    parser.add_argument("--include-recommendations", action="store_true", default=True)
    parser.add_argument("--gitlab-format", action="store_true", help="GitLab-specific formatting")
    parser.add_argument("--jenkins-format", action="store_true", help="Jenkins-specific formatting")
    parser.add_argument("--template", help="HTML template to use")
    
    args = parser.parse_args()
    
    consolidator = ReportConsolidator(args.input_dir, args.input_pattern)
    
    # Load enforcement reports separately if specified
    if args.enforcement_pattern:
        enforcement_files = Path(args.input_dir).glob(args.enforcement_pattern)
        for ef in enforcement_files:
            report_data = consolidator.load_report(ef)
            consolidator.consolidated_data["enforcement_reports"].append({
                "file": str(ef),
                "data": report_data
            })
    
    consolidator.consolidate()
    
    # Format output
    if args.format == "json":
        output = consolidator.format_json()
    elif args.format == "html":
        output = consolidator.format_html(args.template or "default")
    else:  # markdown
        if args.gitlab_format:
            output = consolidator.format_gitlab()
        elif args.jenkins_format:
            output = consolidator.format_jenkins()
        else:
            output = consolidator.format_markdown(args.include_recommendations)
    
    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(output)
    
    print(f"Consolidated report written to: {output_path}")
    
    # Exit with appropriate code based on violations
    if consolidator.consolidated_data["summary"]["critical_violations"] > 0:
        sys.exit(1)
    
    sys.exit(0)


if __name__ == "__main__":
    main()