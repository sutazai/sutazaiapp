#!/usr/bin/env python3
"""
SutazAI Codebase Coordination Tracker
====================================

Tracks progress on the codebase cleanup and standardization efforts.
Used by specialized agents to report status and coordinate work.

Author: Codebase Team Lead Agent
Date: 2025-08-02
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Compliance with CLAUDE.md requirements
def validate_file_structure() -> Tuple[bool, List[str]]:
    """
    Validate the current file structure against CLAUDE.md standards.
    
    Returns:
        Tuple of (is_compliant, list_of_issues)
    """
    issues = []
    base_path = Path("/opt/sutazaiapp")
    
    # Check for required directories
    required_dirs = [
        "agents", "backend", "frontend", "config", 
        "scripts", "tests", "docs", "deployment"
    ]
    
    for required_dir in required_dirs:
        if not (base_path / required_dir).exists():
            issues.append(f"Missing required directory: {required_dir}")
    
    # Check for forbidden files/directories
    forbidden_patterns = [
        "*.backup", "*.agi_backup", "*.fantasy_backup",
        "*_backup_*", "archive/", "temp/", "tmp/"
    ]
    
    try:
        result = subprocess.run(
            ["find", str(base_path), "-name", "*.backup", "-o", "-name", "*.fantasy_backup"],
            capture_output=True, text=True, timeout=30
        )
        if result.stdout.strip():
            backup_files = result.stdout.strip().split('\n')
            issues.extend([f"Found backup file: {f}" for f in backup_files])
    except subprocess.TimeoutExpired:
        issues.append("Timeout during backup file search")
    except Exception as e:
        issues.append(f"Error checking backup files: {e}")
    
    return len(issues) == 0, issues

def check_dependency_consolidation() -> Dict[str, any]:
    """
    Check the status of dependency consolidation across the project.
    
    Returns:
        Dictionary with consolidation status and metrics
    """
    base_path = Path("/opt/sutazaiapp")
    requirements_files = list(base_path.rglob("requirements*.txt"))
    
    return {
        "total_files": len(requirements_files),
        "files": [str(f) for f in requirements_files],
        "needs_consolidation": len(requirements_files) > 5,
        "critical_files": [
            str(base_path / "requirements.txt"),
            str(base_path / "backend" / "requirements.txt"),
            str(base_path / "frontend" / "requirements.txt")
        ]
    }

def check_agent_compliance() -> Dict[str, any]:
    """
    Check compliance status of all agent configurations.
    
    Returns:
        Dictionary with agent compliance metrics
    """
    agents_path = Path("/opt/sutazaiapp/.claude/agents")
    
    if not agents_path.exists():
        return {"error": "Agents directory not found"}
    
    agent_files = list(agents_path.glob("*.md"))
    compliant_agents = []
    non_compliant_agents = []
    
    for agent_file in agent_files:
        try:
            content = agent_file.read_text()
            # Check for YAML frontmatter
            if content.startswith("---") and "\n---\n" in content:
                compliant_agents.append(agent_file.name)
            else:
                non_compliant_agents.append(agent_file.name)
        except Exception as e:
            non_compliant_agents.append(f"{agent_file.name} (error: {e})")
    
    return {
        "total_agents": len(agent_files),
        "compliant": len(compliant_agents),
        "non_compliant": len(non_compliant_agents),
        "compliance_rate": len(compliant_agents) / len(agent_files) * 100 if agent_files else 0,
        "compliant_agents": compliant_agents,
        "non_compliant_agents": non_compliant_agents
    }

def check_code_quality() -> Dict[str, any]:
    """
    Check code quality metrics across the codebase.
    
    Returns:
        Dictionary with code quality metrics
    """
    base_path = Path("/opt/sutazaiapp")
    python_files = []
    
    # Find Python files in key directories
    for pattern in ["backend/**/*.py", "agents/**/*.py", "scripts/**/*.py", "frontend/**/*.py"]:
        python_files.extend(base_path.glob(pattern))
    
    missing_docstrings = []
    total_files = len(python_files)
    
    for py_file in python_files[:50]:  # Limit to first 50 for performance
        try:
            content = py_file.read_text()
            # Simple check for module docstring
            if not ('"""' in content or "'''" in content):
                missing_docstrings.append(str(py_file))
        except Exception:
            continue
    
    return {
        "total_python_files": total_files,
        "checked_files": min(50, total_files),
        "missing_docstrings": len(missing_docstrings),
        "docstring_compliance": (50 - len(missing_docstrings)) / 50 * 100 if total_files > 0 else 100,
        "sample_missing": missing_docstrings[:10]  # First 10 examples
    }

def check_script_organization() -> Dict[str, any]:
    """
    Check the organization status of scripts directory.
    
    Returns:
        Dictionary with script organization metrics
    """
    scripts_path = Path("/opt/sutazaiapp/scripts")
    
    if not scripts_path.exists():
        return {"error": "Scripts directory not found"}
    
    all_scripts = list(scripts_path.rglob("*.py")) + list(scripts_path.rglob("*.sh"))
    organized_scripts = []
    unorganized_scripts = []
    
    # Check for proper headers and documentation
    for script in all_scripts:
        try:
            content = script.read_text()
            lines = content.split('\n')
            
            # Check for shebang and description
            has_shebang = any(line.startswith('#!') for line in lines[:3])
            has_description = any('"""' in line or 'description' in line.lower() for line in lines[:20])
            
            if has_shebang and has_description:
                organized_scripts.append(str(script))
            else:
                unorganized_scripts.append(str(script))
        except Exception:
            unorganized_scripts.append(str(script))
    
    return {
        "total_scripts": len(all_scripts),
        "organized": len(organized_scripts),
        "unorganized": len(unorganized_scripts),
        "organization_rate": len(organized_scripts) / len(all_scripts) * 100 if all_scripts else 100,
        "sample_unorganized": unorganized_scripts[:10]
    }

def generate_progress_report() -> Dict[str, any]:
    """
    Generate a comprehensive progress report for all coordination tasks.
    
    Returns:
        Dictionary with complete progress status
    """
    report = {
        "timestamp": datetime.now().isoformat(),
        "coordinator": "Codebase Team Lead Agent",
        "tasks": {}
    }
    
    # Task 1: File Structure Validation
    structure_compliant, structure_issues = validate_file_structure()
    report["tasks"]["file_structure"] = {
        "status": "COMPLETED" if structure_compliant else "IN_PROGRESS",
        "compliant": structure_compliant,
        "issues": structure_issues,
        "priority": "HIGH"
    }
    
    # Task 2: Dependency Management
    dependency_status = check_dependency_consolidation()
    report["tasks"]["dependency_management"] = {
        "status": "PENDING" if dependency_status["needs_consolidation"] else "COMPLETED",
        "metrics": dependency_status,
        "priority": "HIGH"
    }
    
    # Task 3: Agent Compliance
    agent_status = check_agent_compliance()
    report["tasks"]["agent_compliance"] = {
        "status": "COMPLETED" if agent_status.get("compliance_rate", 0) > 95 else "IN_PROGRESS",
        "metrics": agent_status,
        "priority": "HIGH"
    }
    
    # Task 4: Code Quality
    quality_status = check_code_quality()
    report["tasks"]["code_quality"] = {
        "status": "IN_PROGRESS" if quality_status.get("docstring_compliance", 0) < 90 else "COMPLETED",
        "metrics": quality_status,
        "priority": "HIGH"
    }
    
    # Task 5: Script Organization
    script_status = check_script_organization()
    report["tasks"]["script_organization"] = {
        "status": "IN_PROGRESS" if script_status.get("organization_rate", 0) < 80 else "COMPLETED",
        "metrics": script_status,
        "priority": "MEDIUM"
    }
    
    # Overall Progress
    completed_tasks = sum(1 for task in report["tasks"].values() if task["status"] == "COMPLETED")
    total_tasks = len(report["tasks"])
    report["overall_progress"] = {
        "completed": completed_tasks,
        "total": total_tasks,
        "percentage": completed_tasks / total_tasks * 100,
        "status": "ON_TRACK" if completed_tasks >= total_tasks * 0.6 else "NEEDS_ATTENTION"
    }
    
    return report

def print_summary(report: Dict[str, any]) -> None:
    """
    Print a human-readable summary of the progress report.
    
    Args:
        report: Progress report dictionary
    """
    print("=" * 60)
    print("SutazAI Codebase Coordination Progress Report")
    print("=" * 60)
    print(f"Generated: {report['timestamp']}")
    print(f"Coordinator: {report['coordinator']}")
    print()
    
    print(f"Overall Progress: {report['overall_progress']['percentage']:.1f}% "
          f"({report['overall_progress']['completed']}/{report['overall_progress']['total']} tasks)")
    print(f"Status: {report['overall_progress']['status']}")
    print()
    
    print("Task Breakdown:")
    print("-" * 40)
    
    for task_name, task_info in report["tasks"].items():
        status_emoji = "✅" if task_info["status"] == "COMPLETED" else "🔄" if task_info["status"] == "IN_PROGRESS" else "⏳"
        print(f"{status_emoji} {task_name.replace('_', ' ').title()}: {task_info['status']} ({task_info['priority']} priority)")
        
        if "metrics" in task_info:
            metrics = task_info["metrics"]
            if "compliance_rate" in metrics:
                print(f"   Compliance: {metrics['compliance_rate']:.1f}%")
            if "total_files" in metrics:
                print(f"   Files: {metrics['total_files']}")
            if "organization_rate" in metrics:
                print(f"   Organization: {metrics['organization_rate']:.1f}%")
        print()

def save_report(report: Dict[str, any], filepath: Optional[str] = None) -> None:
    """
    Save the progress report to a JSON file.
    
    Args:
        report: Progress report dictionary
        filepath: Optional custom filepath for the report
    """
    if filepath is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f"/opt/sutazaiapp/reports/coordination_progress_{timestamp}.json"
    
    # Ensure reports directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Report saved to: {filepath}")

def main():
    """Main function to run the coordination tracker."""
    print("Running SutazAI Codebase Coordination Tracker...")
    print()
    
    try:
        # Generate progress report
        report = generate_progress_report()
        
        # Print summary
        print_summary(report)
        
        # Save detailed report
        save_report(report)
        
        # Exit with appropriate code
        if report["overall_progress"]["status"] == "NEEDS_ATTENTION":
            print("⚠️  Some tasks need immediate attention!")
            sys.exit(1)
        else:
            print("✅ Coordination efforts are on track!")
            sys.exit(0)
            
    except Exception as e:
        print(f"❌ Error running coordination tracker: {e}")
        sys.exit(2)

if __name__ == "__main__":
    main()