#!/usr/bin/env python3
"""
Test coverage reporter for SutazAI system
Generates comprehensive coverage reports and analysis
"""

import os
import sys
import subprocess
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
import coverage


console = Console()


class CoverageReporter:
    """Comprehensive coverage reporter with analysis."""

    def __init__(self, project_root: str = "/opt/sutazaiapp"):
        self.project_root = Path(project_root)
        self.coverage_data_file = self.project_root / ".coverage"
        self.reports_dir = self.project_root / "coverage_reports"
        self.reports_dir.mkdir(exist_ok=True)
        
        # Initialize coverage
        self.cov = coverage.Coverage(data_file=str(self.coverage_data_file))

    def run_tests_with_coverage(self, test_type: str = "all", parallel: bool = True) -> Dict[str, Any]:
        """Run tests with coverage collection."""
        console.print(f"[bold blue]Running {test_type} tests with coverage collection[/bold blue]")
        
        # Build pytest command
        cmd = ["python", "-m", "pytest"]
        
        # Add test type filters
        if test_type == "unit":
            cmd.extend(["-m", "unit"])
        elif test_type == "integration":
            cmd.extend(["-m", "integration"])
        elif test_type == "backend":
            cmd.extend(["backend/tests/"])
        elif test_type == "frontend":
            cmd.extend(["frontend/tests/"])
        elif test_type != "all":
            cmd.extend(["-k", test_type])
        
        # Add coverage options
        cmd.extend([
            "--cov=backend",
            "--cov=frontend",
            "--cov-config=.coveragerc",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov",
            "--cov-report=xml:coverage.xml",
            "--cov-report=json:coverage.json"
        ])
        
        if parallel:
            cmd.extend(["-n", "auto"])
        
        cmd.extend(["-v", "--tb=short"])
        
        # Change to project root
        original_cwd = os.getcwd()
        os.chdir(self.project_root)
        
        try:
            console.print(f"[dim]Command: {' '.join(cmd)}[/dim]")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minutes timeout
            )
            
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.returncode
            }
            
        except subprocess.TimeoutExpired:
            console.print("[red]Test execution timed out after 30 minutes[/red]")
            return {
                "success": False,
                "error": "Timeout",
                "exit_code": -1
            }
        finally:
            os.chdir(original_cwd)

    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive coverage report."""
        console.print("[bold green]Generating comprehensive coverage report[/bold green]")
        
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "summary": self._get_coverage_summary(),
            "detailed_analysis": self._analyze_coverage_details(),
            "uncovered_lines": self._get_uncovered_lines(),
            "coverage_trends": self._analyze_coverage_trends(),
            "recommendations": self._generate_coverage_recommendations()
        }
        
        # Save comprehensive report
        report_file = self.reports_dir / f"coverage_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        console.print(f"[green]Comprehensive report saved: {report_file}[/green]")
        return report_data

    def _get_coverage_summary(self) -> Dict[str, Any]:
        """Get coverage summary statistics."""
        try:
            # Try to load coverage data
            self.cov.load()
            
            # Get overall statistics
            total = self.cov.report(show_missing=False, file=None)
            
            # Get detailed file statistics
            file_stats = {}
            for filename in self.cov.get_data().measured_files():
                try:
                    analysis = self.cov._analyze(filename)
                    statements = len(analysis.statements)
                    missing = len(analysis.missing)
                    covered = statements - missing
                    percent = (covered / statements * 100) if statements > 0 else 0
                    
                    # Normalize filename
                    rel_filename = os.path.relpath(filename, self.project_root)
                    
                    file_stats[rel_filename] = {
                        "statements": statements,
                        "covered": covered,
                        "missing": missing,
                        "percent": round(percent, 2)
                    }
                except Exception as e:
                    console.print(f"[yellow]Warning: Could not analyze {filename}: {e}[/yellow]")
            
            return {
                "overall_percent": round(total, 2),
                "total_files": len(file_stats),
                "files": file_stats
            }
            
        except Exception as e:
            console.print(f"[red]Error getting coverage summary: {e}[/red]")
            return self._fallback_coverage_summary()

    def _fallback_coverage_summary(self) -> Dict[str, Any]:
        """Fallback coverage summary from XML/JSON files."""
        summary = {"overall_percent": 0, "total_files": 0, "files": {}}
        
        # Try XML file
        xml_file = self.project_root / "coverage.xml"
        if xml_file.exists():
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                
                # Get overall coverage
                overall = root.attrib.get("line-rate", "0")
                summary["overall_percent"] = round(float(overall) * 100, 2)
                
                # Get file details
                for package in root.findall(".//package"):
                    for class_elem in package.findall(".//class"):
                        filename = class_elem.attrib.get("filename", "")
                        line_rate = float(class_elem.attrib.get("line-rate", "0"))
                        
                        # Count lines (approximation)
                        lines = len(class_elem.findall(".//line"))
                        covered = int(lines * line_rate)
                        
                        summary["files"][filename] = {
                            "statements": lines,
                            "covered": covered,
                            "missing": lines - covered,
                            "percent": round(line_rate * 100, 2)
                        }
                
                summary["total_files"] = len(summary["files"])
                return summary
                
            except Exception as e:
                console.print(f"[yellow]Could not parse XML coverage: {e}[/yellow]")
        
        # Try JSON file
        json_file = self.project_root / "coverage.json"
        if json_file.exists():
            try:
                with open(json_file) as f:
                    data = json.load(f)
                
                if "totals" in data:
                    totals = data["totals"]
                    summary["overall_percent"] = totals.get("percent_covered", 0)
                
                if "files" in data:
                    for filename, file_data in data["files"].items():
                        summary_data = file_data.get("summary", {})
                        summary["files"][filename] = {
                            "statements": summary_data.get("num_statements", 0),
                            "covered": summary_data.get("covered_lines", 0),
                            "missing": summary_data.get("missing_lines", 0),
                            "percent": summary_data.get("percent_covered", 0)
                        }
                
                summary["total_files"] = len(summary["files"])
                return summary
                
            except Exception as e:
                console.print(f"[yellow]Could not parse JSON coverage: {e}[/yellow]")
        
        return summary

    def _analyze_coverage_details(self) -> Dict[str, Any]:
        """Analyze detailed coverage information."""
        summary = self._get_coverage_summary()
        files = summary.get("files", {})
        
        if not files:
            return {"analysis": "No coverage data available"}
        
        # Categorize files by coverage level
        excellent = []  # 90%+
        good = []       # 80-89%
        fair = []       # 60-79%
        poor = []       # <60%
        
        for filename, stats in files.items():
            percent = stats["percent"]
            if percent >= 90:
                excellent.append((filename, percent))
            elif percent >= 80:
                good.append((filename, percent))
            elif percent >= 60:
                fair.append((filename, percent))
            else:
                poor.append((filename, percent))
        
        # Sort by coverage percentage
        excellent.sort(key=lambda x: x[1], reverse=True)
        good.sort(key=lambda x: x[1], reverse=True)
        fair.sort(key=lambda x: x[1], reverse=True)
        poor.sort(key=lambda x: x[1], reverse=True)
        
        return {
            "coverage_distribution": {
                "excellent_90_plus": {"count": len(excellent), "files": excellent[:10]},
                "good_80_89": {"count": len(good), "files": good[:10]},
                "fair_60_79": {"count": len(fair), "files": fair[:10]},
                "poor_below_60": {"count": len(poor), "files": poor[:10]}
            },
            "statistics": {
                "avg_coverage": sum(f["percent"] for f in files.values()) / len(files),
                "median_coverage": sorted([f["percent"] for f in files.values()])[len(files)//2],
                "total_statements": sum(f["statements"] for f in files.values()),
                "total_covered": sum(f["covered"] for f in files.values()),
                "total_missing": sum(f["missing"] for f in files.values())
            }
        }

    def _get_uncovered_lines(self) -> Dict[str, List[int]]:
        """Get uncovered lines for each file."""
        uncovered = {}
        
        try:
            self.cov.load()
            
            for filename in self.cov.get_data().measured_files():
                try:
                    analysis = self.cov._analyze(filename)
                    if analysis.missing:
                        rel_filename = os.path.relpath(filename, self.project_root)
                        uncovered[rel_filename] = sorted(list(analysis.missing))
                except Exception:
                    continue
                    
        except Exception as e:
            console.print(f"[yellow]Could not get uncovered lines: {e}[/yellow]")
        
        return uncovered

    def _analyze_coverage_trends(self) -> Dict[str, Any]:
        """Analyze coverage trends over time."""
        # Look for historical coverage data
        historical_reports = list(self.reports_dir.glob("coverage_report_*.json"))
        
        if len(historical_reports) < 2:
            return {"trend": "Insufficient historical data"}
        
        # Sort by date
        historical_reports.sort()
        
        trends = []
        for report_file in historical_reports[-5:]:  # Last 5 reports
            try:
                with open(report_file) as f:
                    data = json.load(f)
                
                trends.append({
                    "timestamp": data.get("timestamp"),
                    "overall_percent": data.get("summary", {}).get("overall_percent", 0),
                    "total_files": data.get("summary", {}).get("total_files", 0)
                })
            except Exception:
                continue
        
        if len(trends) >= 2:
            recent = trends[-1]["overall_percent"]
            previous = trends[-2]["overall_percent"]
            change = recent - previous
            
            return {
                "trend_direction": "increasing" if change > 0 else "decreasing" if change < 0 else "stable",
                "change_percent": round(change, 2),
                "historical_data": trends
            }
        
        return {"trend": "Insufficient trend data"}

    def _generate_coverage_recommendations(self) -> List[str]:
        """Generate coverage improvement recommendations."""
        recommendations = []
        summary = self._get_coverage_summary()
        overall_percent = summary.get("overall_percent", 0)
        files = summary.get("files", {})
        
        # Overall coverage recommendations
        if overall_percent < 60:
            recommendations.append("CRITICAL: Overall coverage is below 60%. Immediate action required to improve test coverage.")
        elif overall_percent < 80:
            recommendations.append("WARNING: Overall coverage is below 80%. Consider adding more comprehensive tests.")
        elif overall_percent < 90:
            recommendations.append("GOOD: Coverage is above 80%. Focus on testing critical paths and edge cases.")
        else:
            recommendations.append("EXCELLENT: Coverage is above 90%. Maintain current testing practices.")
        
        # File-specific recommendations
        if files:
            poor_files = [(f, s["percent"]) for f, s in files.items() if s["percent"] < 60]
            if poor_files:
                poor_files.sort(key=lambda x: x[1])
                recommendations.append(f"Priority files needing tests: {', '.join([f[0] for f in poor_files[:5]])}")
            
            untested_files = [f for f, s in files.items() if s["percent"] == 0]
            if untested_files:
                recommendations.append(f"Completely untested files: {len(untested_files)} files need initial test coverage.")
            
            large_untested = [(f, s["missing"]) for f, s in files.items() if s["missing"] > 50 and s["percent"] < 50]
            if large_untested:
                large_untested.sort(key=lambda x: x[1], reverse=True)
                recommendations.append(f"Large files with low coverage: {large_untested[0][0]} has {large_untested[0][1]} untested lines.")
        
        # Test type recommendations
        detailed = self._analyze_coverage_details()
        if "statistics" in detailed:
            stats = detailed["statistics"]
            if stats["total_statements"] > 10000:
                recommendations.append("Large codebase detected. Consider implementing incremental coverage targets.")
            
            if stats["avg_coverage"] < stats.get("median_coverage", 0) - 10:
                recommendations.append("Coverage distribution is uneven. Some files have much lower coverage than others.")
        
        # Testing strategy recommendations
        recommendations.extend([
            "Focus on testing critical business logic and error handling paths.",
            "Add integration tests for complex workflows and API endpoints.",
            "Consider property-based testing for complex algorithms.",
            "Implement mutation testing to verify test quality.",
            "Use coverage-guided fuzzing for security-critical components."
        ])
        
        return recommendations

    def display_coverage_summary(self, data: Dict[str, Any]):
        """Display coverage summary in console."""
        console.print("\n" + "="*60)
        console.print(Panel.fit(
            "[bold blue]SutazAI Test Coverage Summary[/bold blue]",
            border_style="blue"
        ))
        
        summary = data.get("summary", {})
        overall_percent = summary.get("overall_percent", 0)
        
        # Overall coverage
        color = "green" if overall_percent >= 80 else "yellow" if overall_percent >= 60 else "red"
        console.print(f"\n[bold]Overall Coverage: [{color}]{overall_percent:.2f}%[/{color}][/bold]")
        console.print(f"Total Files: {summary.get('total_files', 0)}")
        
        # Coverage distribution
        detailed = data.get("detailed_analysis", {})
        if "coverage_distribution" in detailed:
            dist = detailed["coverage_distribution"]
            
            table = Table(title="Coverage Distribution")
            table.add_column("Coverage Range", style="cyan")
            table.add_column("File Count", style="magenta")
            table.add_column("Percentage", style="green")
            
            total_files = summary.get("total_files", 1)
            
            table.add_row(
                "Excellent (90%+)",
                str(dist["excellent_90_plus"]["count"]),
                f"{dist['excellent_90_plus']['count']/total_files*100:.1f}%"
            )
            table.add_row(
                "Good (80-89%)",
                str(dist["good_80_89"]["count"]),
                f"{dist['good_80_89']['count']/total_files*100:.1f}%"
            )
            table.add_row(
                "Fair (60-79%)",
                str(dist["fair_60_79"]["count"]),
                f"{dist['fair_60_79']['count']/total_files*100:.1f}%"
            )
            table.add_row(
                "Poor (<60%)",
                str(dist["poor_below_60"]["count"]),
                f"{dist['poor_below_60']['count']/total_files*100:.1f}%"
            )
            
            console.print(table)
        
        # Top recommendations
        recommendations = data.get("recommendations", [])
        if recommendations:
            console.print(f"\n[bold yellow]Top Recommendations:[/bold yellow]")
            for i, rec in enumerate(recommendations[:5], 1):
                console.print(f"  {i}. {rec}")
        
        # Coverage trend
        trends = data.get("coverage_trends", {})
        if "change_percent" in trends:
            change = trends["change_percent"]
            trend_color = "green" if change > 0 else "red" if change < 0 else "yellow"
            trend_symbol = "↑" if change > 0 else "↓" if change < 0 else "→"
            console.print(f"\n[bold]Coverage Trend: [{trend_color}]{trend_symbol} {change:+.2f}%[/{trend_color}][/bold]")

    def generate_html_dashboard(self, data: Dict[str, Any]) -> str:
        """Generate HTML coverage dashboard."""
        dashboard_file = self.reports_dir / "coverage_dashboard.html"
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SutazAI Coverage Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ background: #2c3e50; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
        .metric-card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }}
        .metric-value {{ font-size: 2em; font-weight: bold; }}
        .metric-label {{ color: #666; font-size: 0.9em; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; }}
        .chart-container {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .file-list {{ max-height: 300px; overflow-y: auto; }}
        .file-item {{ padding: 8px; border-bottom: 1px solid #eee; display: flex; justify-content: space-between; }}
        .coverage-bar {{ width: 100px; height: 20px; background: #e0e0e0; border-radius: 10px; overflow: hidden; }}
        .coverage-fill {{ height: 100%; background: linear-gradient(90deg, #ff4444 0%, #ffaa00 50%, #44ff44 100%); }}
        .recommendations {{ background: #fff3cd; border: 1px solid #ffeaa7; padding: 20px; border-radius: 8px; }}
        .trend-up {{ color: #27ae60; }}
        .trend-down {{ color: #e74c3c; }}
        .trend-stable {{ color: #f39c12; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 SutazAI Test Coverage Dashboard</h1>
            <p>Generated: {data.get('timestamp', '')}</p>
        </div>
        
        <div class="grid">
            <div class="metric-card">
                <div class="metric-value" style="color: {'#27ae60' if data.get('summary', {}).get('overall_percent', 0) >= 80 else '#e74c3c'}">
                    {data.get('summary', {}).get('overall_percent', 0):.1f}%
                </div>
                <div class="metric-label">Overall Coverage</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-value">{data.get('summary', {}).get('total_files', 0)}</div>
                <div class="metric-label">Total Files</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-value">{data.get('detailed_analysis', {}).get('statistics', {}).get('total_statements', 0)}</div>
                <div class="metric-label">Total Statements</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-value">{data.get('detailed_analysis', {}).get('statistics', {}).get('total_covered', 0)}</div>
                <div class="metric-label">Covered Lines</div>
            </div>
        </div>
        
        <div class="grid">
            <div class="chart-container">
                <h3>Coverage Distribution</h3>
                <canvas id="distributionChart" width="400" height="200"></canvas>
            </div>
            
            <div class="chart-container">
                <h3>Coverage Trend</h3>
                <canvas id="trendChart" width="400" height="200"></canvas>
            </div>
        </div>
        
        <div class="chart-container">
            <h3>Files with Low Coverage</h3>
            <div class="file-list">
"""
        
        # Add file list
        files = data.get("summary", {}).get("files", {})
        low_coverage_files = [(f, s) for f, s in files.items() if s["percent"] < 80]
        low_coverage_files.sort(key=lambda x: x[1]["percent"])
        
        for filename, stats in low_coverage_files[:20]:  # Top 20 files
            html_content += f"""
                <div class="file-item">
                    <span>{filename}</span>
                    <div>
                        <div class="coverage-bar">
                            <div class="coverage-fill" style="width: {stats['percent']}%"></div>
                        </div>
                        <span>{stats['percent']:.1f}%</span>
                    </div>
                </div>
"""
        
        html_content += f"""
            </div>
        </div>
        
        <div class="recommendations">
            <h3>📋 Recommendations</h3>
            <ul>
"""
        
        for rec in data.get("recommendations", [])[:10]:
            html_content += f"<li>{rec}</li>"
        
        html_content += """
            </ul>
        </div>
    </div>
    
    <script>
        // Coverage Distribution Chart
        const distCtx = document.getElementById('distributionChart').getContext('2d');
        new Chart(distCtx, {
            type: 'doughnut',
            data: {
                labels: ['Excellent (90%+)', 'Good (80-89%)', 'Fair (60-79%)', 'Poor (<60%)'],
                datasets: [{
                    data: [""" + f"""
                        {data.get('detailed_analysis', {}).get('coverage_distribution', {}).get('excellent_90_plus', {}).get('count', 0)},
                        {data.get('detailed_analysis', {}).get('coverage_distribution', {}).get('good_80_89', {}).get('count', 0)},
                        {data.get('detailed_analysis', {}).get('coverage_distribution', {}).get('fair_60_79', {}).get('count', 0)},
                        {data.get('detailed_analysis', {}).get('coverage_distribution', {}).get('poor_below_60', {}).get('count', 0)}
                    """ + """],
                    backgroundColor: ['#27ae60', '#f39c12', '#e67e22', '#e74c3c']
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        position: 'bottom'
                    }
                }
            }
        });
        
        // Coverage Trend Chart (placeholder)
        const trendCtx = document.getElementById('trendChart').getContext('2d');
        new Chart(trendCtx, {
            type: 'line',
            data: {
                labels: ['Previous', 'Current'],
                datasets: [{
                    label: 'Coverage %',
                    data: [75, """ + f"{data.get('summary', {}).get('overall_percent', 0)}" + """],
                    borderColor: '#3498db',
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: false,
                        min: 0,
                        max: 100
                    }
                }
            }
        });
    </script>
</body>
</html>
"""
        
        with open(dashboard_file, 'w') as f:
            f.write(html_content)
        
        console.print(f"[green]HTML dashboard generated: {dashboard_file}[/green]")
        return str(dashboard_file)

    def run_coverage_analysis(self, test_type: str = "all", generate_reports: bool = True) -> Dict[str, Any]:
        """Run complete coverage analysis."""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            # Step 1: Run tests with coverage
            task1 = progress.add_task("Running tests with coverage...", total=None)
            test_result = self.run_tests_with_coverage(test_type)
            progress.update(task1, description="✅ Tests completed")
            
            if not test_result["success"]:
                console.print(f"[red]Tests failed with exit code {test_result['exit_code']}[/red]")
                console.print("[yellow]Continuing with coverage analysis of existing data...[/yellow]")
            
            # Step 2: Generate comprehensive report
            task2 = progress.add_task("Generating coverage report...", total=None)
            report_data = self.generate_comprehensive_report()
            progress.update(task2, description="✅ Report generated")
            
            # Step 3: Generate additional reports
            if generate_reports:
                task3 = progress.add_task("Generating HTML dashboard...", total=None)
                dashboard_file = self.generate_html_dashboard(report_data)
                progress.update(task3, description="✅ Dashboard generated")
        
        # Display summary
        self.display_coverage_summary(report_data)
        
        # Add test result info
        report_data["test_execution"] = test_result
        
        return report_data


@click.command()
@click.option("--test-type", default="all", help="Type of tests to run (all, unit, integration, backend, frontend)")
@click.option("--no-reports", is_flag=True, help="Skip generating HTML reports")
@click.option("--project-root", default="/opt/sutazaiapp", help="Project root directory")
@click.option("--threshold", type=float, default=80.0, help="Coverage threshold percentage")
def main(test_type, no_reports, project_root, threshold):
    """SutazAI Coverage Reporter - Comprehensive test coverage analysis."""
    
    console.print(Panel.fit(
        "[bold blue]SutazAI Coverage Reporter[/bold blue]\n[dim]Comprehensive Test Coverage Analysis[/dim]",
        border_style="blue"
    ))
    
    reporter = CoverageReporter(project_root)
    
    try:
        # Run coverage analysis
        results = reporter.run_coverage_analysis(
            test_type=test_type,
            generate_reports=not no_reports
        )
        
        # Check threshold
        overall_coverage = results.get("summary", {}).get("overall_percent", 0)
        
        if overall_coverage >= threshold:
            console.print(f"\n[bold green]✅ Coverage threshold met: {overall_coverage:.2f}% >= {threshold}%[/bold green]")
            exit_code = 0
        else:
            console.print(f"\n[bold red]❌ Coverage below threshold: {overall_coverage:.2f}% < {threshold}%[/bold red]")
            exit_code = 1
        
        # Show report locations
        console.print(f"\n[bold]Report Files:[/bold]")
        console.print(f"  📊 HTML Coverage: {project_root}/htmlcov/index.html")
        console.print(f"  📈 Dashboard: {project_root}/coverage_reports/coverage_dashboard.html")
        console.print(f"  📋 JSON Report: Latest in {project_root}/coverage_reports/")
        
        return exit_code
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Coverage analysis interrupted by user[/yellow]")
        return 130
    except Exception as e:
        console.print(f"\n[red]Coverage analysis failed: {str(e)}[/red]")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)