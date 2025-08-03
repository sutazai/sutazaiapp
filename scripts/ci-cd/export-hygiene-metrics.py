#!/usr/bin/env python3
"""
Purpose: Export hygiene metrics in various formats for monitoring systems
Usage: python export-hygiene-metrics.py --input report.json --output metrics.txt --format prometheus
Requirements: Python 3.8+, json
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any


class MetricsExporter:
    """Export hygiene metrics in various monitoring formats"""
    
    def __init__(self, report_data: Dict[str, Any]):
        self.report_data = report_data
        self.timestamp = int(datetime.now().timestamp())
    
    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format"""
        summary = self.report_data.get("summary", {})
        
        metrics = []
        
        # Add HELP and TYPE annotations
        metrics.extend([
            "# HELP hygiene_score Overall codebase hygiene score percentage",
            "# TYPE hygiene_score gauge",
            f"hygiene_score {summary.get('hygiene_score', 0)}"
        ])
        
        metrics.extend([
            "",
            "# HELP hygiene_violations_total Total number of hygiene violations by priority",
            "# TYPE hygiene_violations_total gauge"
        ])
        
        for priority in ["critical", "high", "medium", "low"]:
            count = summary.get(f"{priority}_violations", 0)
            metrics.append(f'hygiene_violations_total{{priority="{priority}"}} {count}')
        
        metrics.extend([
            "",
            "# HELP hygiene_total_violations Total number of all violations",
            "# TYPE hygiene_total_violations gauge",
            f"hygiene_total_violations {summary.get('total_violations', 0)}"
        ])
        
        # Rule-specific metrics
        violations = self.report_data.get("violations", {})
        rule_counts = {}
        
        for priority, priority_violations in violations.items():
            for violation in priority_violations:
                rule = violation.get("rule", "unknown")
                rule_counts[rule] = rule_counts.get(rule, 0) + 1
        
        if rule_counts:
            metrics.extend([
                "",
                "# HELP hygiene_violations_by_rule Number of violations per rule",
                "# TYPE hygiene_violations_by_rule gauge"
            ])
            
            for rule, count in sorted(rule_counts.items()):
                metrics.append(f'hygiene_violations_by_rule{{rule="{rule}"}} {count}')
        
        # Enforcement metrics
        enforcement_reports = self.report_data.get("enforcement_reports", [])
        if enforcement_reports:
            metrics.extend([
                "",
                "# HELP hygiene_enforcement_total Number of enforcement actions by agent",
                "# TYPE hygiene_enforcement_total counter"
            ])
            
            agent_counts = {}
            for report in enforcement_reports:
                agent = report.get("data", {}).get("agent", "unknown")
                status = report.get("data", {}).get("status", "unknown")
                key = f"{agent}_{status}"
                agent_counts[key] = agent_counts.get(key, 0) + 1
            
            for key, count in agent_counts.items():
                agent, status = key.rsplit("_", 1)
                metrics.append(f'hygiene_enforcement_total{{agent="{agent}",status="{status}"}} {count}')
        
        # Add timestamp
        metrics.extend([
            "",
            "# HELP hygiene_last_run_timestamp Timestamp of last hygiene check",
            "# TYPE hygiene_last_run_timestamp gauge",
            f"hygiene_last_run_timestamp {self.timestamp}"
        ])
        
        return "\n".join(metrics)
    
    def export_influxdb(self) -> str:
        """Export metrics in InfluxDB line protocol format"""
        summary = self.report_data.get("summary", {})
        
        lines = []
        
        # Main metrics
        lines.append(
            f"hygiene_score value={summary.get('hygiene_score', 0)} {self.timestamp}"
        )
        
        # Violations by priority
        for priority in ["critical", "high", "medium", "low"]:
            count = summary.get(f"{priority}_violations", 0)
            lines.append(
                f"hygiene_violations,priority={priority} count={count} {self.timestamp}"
            )
        
        # Total violations
        lines.append(
            f"hygiene_total_violations value={summary.get('total_violations', 0)} {self.timestamp}"
        )
        
        return "\n".join(lines)
    
    def export_json(self) -> str:
        """Export metrics as JSON"""
        summary = self.report_data.get("summary", {})
        
        metrics = {
            "timestamp": self.timestamp,
            "hygiene_score": summary.get("hygiene_score", 0),
            "violations": {
                "total": summary.get("total_violations", 0),
                "critical": summary.get("critical_violations", 0),
                "high": summary.get("high_violations", 0),
                "medium": summary.get("medium_violations", 0),
                "low": summary.get("low_violations", 0)
            },
            "reports_analyzed": summary.get("reports_analyzed", 0),
            "enforcement_actions": summary.get("enforcement_actions", 0)
        }
        
        # Add rule breakdown
        violations = self.report_data.get("violations", {})
        rule_counts = {}
        
        for priority, priority_violations in violations.items():
            for violation in priority_violations:
                rule = violation.get("rule", "unknown")
                rule_counts[rule] = rule_counts.get(rule, 0) + 1
        
        metrics["violations_by_rule"] = rule_counts
        
        return json.dumps(metrics, indent=2)
    
    def export_jenkins_properties(self) -> str:
        """Export metrics as Jenkins properties file"""
        summary = self.report_data.get("summary", {})
        
        properties = []
        
        properties.append(f"hygiene_score={summary.get('hygiene_score', 0)}")
        properties.append(f"critical_violations={summary.get('critical_violations', 0)}")
        properties.append(f"high_violations={summary.get('high_violations', 0)}")
        properties.append(f"medium_violations={summary.get('medium_violations', 0)}")
        properties.append(f"low_violations={summary.get('low_violations', 0)}")
        properties.append(f"total_violations={summary.get('total_violations', 0)}")
        properties.append(f"timestamp={self.timestamp}")
        
        # Add pass/fail status
        passed = summary.get("critical_violations", 0) == 0
        properties.append(f"quality_gates_passed={str(passed).lower()}")
        
        return "\n".join(properties)
    
    def export_graphite(self) -> str:
        """Export metrics in Graphite format"""
        summary = self.report_data.get("summary", {})
        
        metrics = []
        prefix = "hygiene"
        
        metrics.append(f"{prefix}.score {summary.get('hygiene_score', 0)} {self.timestamp}")
        metrics.append(f"{prefix}.violations.total {summary.get('total_violations', 0)} {self.timestamp}")
        
        for priority in ["critical", "high", "medium", "low"]:
            count = summary.get(f"{priority}_violations", 0)
            metrics.append(f"{prefix}.violations.{priority} {count} {self.timestamp}")
        
        return "\n".join(metrics)
    
    def export_cloudwatch(self) -> str:
        """Export metrics in AWS CloudWatch JSON format"""
        summary = self.report_data.get("summary", {})
        
        metrics = []
        
        # Standard CloudWatch metric format
        base_metric = {
            "Namespace": "CodebaseHygiene",
            "Timestamp": self.timestamp * 1000,  # CloudWatch uses milliseconds
            "Unit": "None"
        }
        
        # Hygiene score
        metrics.append({
            **base_metric,
            "MetricName": "HygieneScore",
            "Value": summary.get("hygiene_score", 0),
            "Unit": "Percent"
        })
        
        # Violations by priority
        for priority in ["critical", "high", "medium", "low"]:
            metrics.append({
                **base_metric,
                "MetricName": "Violations",
                "Value": summary.get(f"{priority}_violations", 0),
                "Dimensions": [{"Name": "Priority", "Value": priority.title()}]
            })
        
        return json.dumps(metrics, indent=2)
    
    def export_statsd(self) -> str:
        """Export metrics in StatsD format"""
        summary = self.report_data.get("summary", {})
        
        metrics = []
        
        # Gauges
        metrics.append(f"hygiene.score:{summary.get('hygiene_score', 0)}|g")
        metrics.append(f"hygiene.violations.total:{summary.get('total_violations', 0)}|g")
        
        for priority in ["critical", "high", "medium", "low"]:
            count = summary.get(f"{priority}_violations", 0)
            metrics.append(f"hygiene.violations.{priority}:{count}|g")
        
        # Counters for enforcement
        enforcement_count = len(self.report_data.get("enforcement_reports", []))
        if enforcement_count > 0:
            metrics.append(f"hygiene.enforcement.executed:{enforcement_count}|c")
        
        return "\n".join(metrics)


def main():
    parser = argparse.ArgumentParser(description="Export hygiene metrics for monitoring")
    parser.add_argument("--input", required=True, help="Input JSON report file")
    parser.add_argument("--output", required=True, help="Output metrics file")
    parser.add_argument(
        "--format", 
        choices=["prometheus", "influxdb", "json", "jenkins-properties", 
                 "graphite", "cloudwatch", "statsd"],
        default="prometheus",
        help="Output format for metrics"
    )
    
    args = parser.parse_args()
    
    # Load report data
    with open(args.input, 'r') as f:
        report_data = json.load(f)
    
    # Create exporter
    exporter = MetricsExporter(report_data)
    
    # Export metrics
    format_methods = {
        "prometheus": exporter.export_prometheus,
        "influxdb": exporter.export_influxdb,
        "json": exporter.export_json,
        "jenkins-properties": exporter.export_jenkins_properties,
        "graphite": exporter.export_graphite,
        "cloudwatch": exporter.export_cloudwatch,
        "statsd": exporter.export_statsd
    }
    
    export_method = format_methods.get(args.format)
    if export_method:
        metrics = export_method()
    else:
        print(f"Unsupported format: {args.format}")
        return 1
    
    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(metrics)
    
    print(f"Metrics exported to: {output_path}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())