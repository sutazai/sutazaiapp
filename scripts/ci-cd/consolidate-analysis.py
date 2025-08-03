#!/usr/bin/env python3
"""
Purpose: Consolidates individual rule analysis results into a comprehensive report
Usage: python consolidate-analysis.py --input-dir ./temp --output report.json --priority high
Requirements: Python 3.8+, json
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any


def consolidate_analysis(input_dir: str, priority: str, ci_platform: str) -> Dict[str, Any]:
    """Consolidate individual rule analysis results"""
    
    input_path = Path(input_dir)
    all_violations = {}
    total_by_severity = {
        "critical": 0,
        "high": 0,
        "medium": 0,
        "low": 0,
        "info": 0
    }
    
    # Load all rule analysis files
    for rule_file in input_path.glob("rule-*.json"):
        try:
            with open(rule_file, 'r') as f:
                data = json.load(f)
                
                rule_num = data.get("rule")
                violations = data.get("violations", [])
                
                if violations:
                    all_violations[str(rule_num)] = violations
                
                # Update severity counts
                for severity, count in data.get("summary", {}).get("by_severity", {}).items():
                    total_by_severity[severity] = total_by_severity.get(severity, 0) + count
                    
        except Exception as e:
            print(f"Error loading {rule_file}: {e}")
    
    # Calculate hygiene score
    penalties = {
        "critical": 10,
        "high": 5,
        "medium": 2,
        "low": 1,
        "info": 0
    }
    
    score = 100
    for severity, count in total_by_severity.items():
        score -= count * penalties.get(severity, 0)
    score = max(0, score)
    
    # Create consolidated report
    report = {
        "timestamp": datetime.now().isoformat(),
        "priority": priority,
        "ci_platform": ci_platform,
        "violations": all_violations,
        "summary": {
            "hygiene_score": score,
            "total_violations": sum(total_by_severity.values()),
            "critical_violations": total_by_severity.get("critical", 0),
            "high_violations": total_by_severity.get("high", 0),
            "medium_violations": total_by_severity.get("medium", 0),
            "low_violations": total_by_severity.get("low", 0),
            "by_severity": total_by_severity,
            "rules_analyzed": len(all_violations)
        }
    }
    
    # Add CI platform-specific data
    if ci_platform == "github":
        report["github_annotations"] = []
        for rule, violations in all_violations.items():
            for v in violations[:10]:  # Limit annotations
                if v.get("severity") in ["critical", "high"]:
                    report["github_annotations"].append({
                        "path": v.get("file", ""),
                        "start_line": v.get("line", 1),
                        "end_line": v.get("line", 1),
                        "annotation_level": "error" if v["severity"] == "critical" else "warning",
                        "message": f"Rule {rule}: {v.get('message', '')}",
                        "title": f"Hygiene Rule {rule} Violation"
                    })
    
    return report


def main():
    parser = argparse.ArgumentParser(description="Consolidate rule analysis results")
    parser.add_argument("--input-dir", required=True, help="Directory with rule analysis files")
    parser.add_argument("--output", required=True, help="Output JSON file")
    parser.add_argument("--priority", default="medium", help="Analysis priority level")
    parser.add_argument("--ci-platform", default="unknown", help="CI platform")
    
    args = parser.parse_args()
    
    report = consolidate_analysis(args.input_dir, args.priority, args.ci_platform)
    
    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Consolidated report written to: {output_path}")
    print(f"Hygiene Score: {report['summary']['hygiene_score']}%")
    print(f"Total Violations: {report['summary']['total_violations']}")
    
    # Exit with error if critical violations
    if report['summary']['critical_violations'] > 0:
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())