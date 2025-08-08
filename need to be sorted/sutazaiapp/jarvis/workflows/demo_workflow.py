#!/usr/bin/env python3
"""
Simple demonstration of the code improvement workflow
Shows how to analyze code and get actionable improvements
"""

import asyncio
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from workflows.code_improvement_workflow import CodeImprovementWorkflow


async def demonstrate_workflow():
    """Demonstrate the code improvement workflow"""
    
    print("SutazAI Code Improvement Workflow Demo")
    print("=" * 50)
    
    # Create workflow instance
    workflow = CodeImprovementWorkflow()
    await workflow.initialize()
    
    # Analyze the backend/app directory
    target_dir = "/opt/sutazaiapp/backend/app"
    print(f"\nAnalyzing directory: {target_dir}")
    print("This will use multiple AI agents to analyze code quality...\n")
    
    # Run analysis
    report = await workflow.analyze_directory(target_dir)
    
    # Display results
    print("\n" + "=" * 50)
    print("ANALYSIS COMPLETE")
    print("=" * 50)
    
    print(f"\nCode Metrics:")
    print(f"  - Lines of Code: {report.metrics.lines_of_code:,}")
    print(f"  - Complexity Score: {report.metrics.complexity_score:.2f}/100")
    print(f"  - Security Issues: {report.metrics.security_issues}")
    print(f"  - Performance Issues: {report.metrics.performance_issues}")
    print(f"  - Style Violations: {report.metrics.style_violations}")
    
    # Count issues by severity
    severity_counts = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
    for issue in report.issues:
        severity_counts[issue.severity] += 1
    
    print(f"\nIssues Found ({len(report.issues)} total):")
    for severity, count in severity_counts.items():
        if count > 0:
            print(f"  - {severity.capitalize()}: {count}")
    
    # Show top critical/high issues
    print("\n" + "-" * 50)
    print("TOP PRIORITY ISSUES:")
    print("-" * 50)
    
    critical_issues = [i for i in report.issues if i.severity in ['critical', 'high']]
    
    if critical_issues:
        for i, issue in enumerate(critical_issues[:5], 1):
            print(f"\n{i}. [{issue.severity.upper()}] {issue.description}")
            print(f"   File: {os.path.basename(issue.file_path)}:{issue.line_number}")
            print(f"   Type: {issue.issue_type}")
            print(f"   Found by: {issue.agent}")
            if issue.suggested_fix:
                print(f"   Fix: {issue.suggested_fix}")
    else:
        print("\nNo critical or high severity issues found!")
    
    # Show agent recommendations
    print("\n" + "-" * 50)
    print("AGENT RECOMMENDATIONS:")
    print("-" * 50)
    
    for agent, analysis in report.agent_analyses.items():
        if 'recommendations' in analysis and analysis['recommendations']:
            print(f"\n{agent}:")
            for rec in analysis['recommendations'][:2]:
                print(f"  • {rec}")
    
    # Show actionable improvements
    print("\n" + "-" * 50)
    print("ACTIONABLE IMPROVEMENTS:")
    print("-" * 50)
    
    high_priority_improvements = [
        imp for imp in report.improvements 
        if imp.get('priority') in ['critical', 'high']
    ]
    
    if high_priority_improvements:
        for i, improvement in enumerate(high_priority_improvements[:5], 1):
            print(f"\n{i}. {improvement['description']}")
            if 'file' in improvement:
                print(f"   File: {os.path.basename(improvement['file'])}")
            if 'suggested_fix' in improvement:
                print(f"   Action: {improvement['suggested_fix']}")
    
    # Save report
    output_dir = Path("/opt/sutazaiapp/data/workflow_reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "demo_code_improvement_report.md"
    workflow.save_report(report, str(output_file))
    
    print(f"\n" + "=" * 50)
    print(f"Full report saved to:")
    print(f"  Markdown: {output_file}")
    print(f"  JSON: {str(output_file).replace('.md', '.json')}")
    print("=" * 50)
    
    # Show example of actual code change
    if critical_issues:
        print("\n" + "=" * 50)
        print("EXAMPLE CODE CHANGE:")
        print("=" * 50)
        
        example_issue = critical_issues[0]
        print(f"\nFile: {example_issue.file_path}")
        print(f"Line {example_issue.line_number}: {example_issue.description}")
        
        # Read the problematic line
        try:
            with open(example_issue.file_path, 'r') as f:
                lines = f.readlines()
                if example_issue.line_number <= len(lines):
                    print(f"\nCurrent code:")
                    start = max(0, example_issue.line_number - 2)
                    end = min(len(lines), example_issue.line_number + 1)
                    for i in range(start, end):
                        prefix = ">>> " if i == example_issue.line_number - 1 else "    "
                        print(f"{prefix}{i+1}: {lines[i].rstrip()}")
                    
                    if example_issue.suggested_fix:
                        print(f"\nSuggested fix:")
                        print(f"    {example_issue.suggested_fix}")
        except:
            pass


async def main():
    """Main function"""
    try:
        await demonstrate_workflow()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print(f"\n\nError during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())