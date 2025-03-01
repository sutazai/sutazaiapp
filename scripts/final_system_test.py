#!/usr/bin/env python3.11
"""
SutazAI Final System Test Script
This script performs a comprehensive system test and audit of the SutazAI application.
"""
import json
from datetime import datetime
from typing import Dict, List, Tuple
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from misc.utils.subprocess_utils import run_command
def run_python_script(script_path: str) -> Tuple[bool, str]:    """    Run a Python script and return whether it succeeded and its output.    Args:    script_path: Path to the Python script to run    Returns:    Tuple of (success, output)    """    try:    result = run_command(        ["python3", script_path],        check=True,
    )
    return True, result.stdout
except Exception as e:    return False, str(e)
    def main() -> None:    """Main execution function for final system test"""        console = Console()        console.rule(        "[bold blue]Final System Test: Comprehensive Audit[/bold blue]")        summary: Dict[str, Dict[str, str]] = {}        tasks: List[Tuple[str, str]] = [        (            "System Validation",
        "/opt/sutazaiapp/scripts/system_comprehensive_validator.py",
        ),
        (
        "Project Analysis",
        "/opt/sutazaiapp/scripts/project_analyzer.py",
        ),
        (
        "Documentation Generation",
        "/opt/sutazaiapp/scripts/documentation_generator.py",
        ),
        (
        "Spell Checking",
        "/opt/sutazaiapp/scripts/spell_checker.py",
        ),
        ]
        console.print("[bold]Detailed Task Execution:[/bold]")
        for task_name, script_path in tasks:    console.print(
        f"[bold]{task_name}:[/bold] Running script: {script_path}")
        success, output = run_python_script(script_path)
        summary[task_name] = {"success": str(success), "output": output}
        status = "[green]Success[/green]" if success else "[red]Failed[/red]"
        console.print(f"{task_name}: {status}")
        # Print detailed output inside a collapsible panel
        console.print(
        Panel(
        output,
        title=f"{task_name} Output",
        expand=False))
        # Execute unit tests via pytest
        console.print(
        "[bold]Unit Tests:[/bold] Running pytest with detailed logs...")
        try:    result = run_command(
            ["pytest", "--maxfail=1", "--disable-warnings", "-q"],
            check=True,
            )
            success = True
            output = result.stdout
            except Exception as e:    success = False
                output = str(e)
                summary["Unit Tests"] = {
                "success": str(success), "output": output}
                status = "[green]Success[/green]" if success else "[red]Failed[/red]"
                console.print(f"Unit Tests: {status}")
                console.print(
                Panel(
                output,
                title="Unit Tests Output",
                expand=False))
                # Prepare a more detailed summary table
                table = Table(title="Final System Test Summary")
                table.add_column("Task", style="cyan")
                table.add_column("Status", style="magenta")
                table.add_column("Details", style="green")
                for task, result in summary.items():    detail = result["output"].splitlines(
                )[0] if result["output"] else "No Output"
                status_text = "Success" if result["success"] == "True" else "Failed"
                table.add_row(task, status_text, detail)
                console.print(table)
                # Save JSON summary with full details
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                summary_file = f"/opt/sutazaiapp/logs/final_system_test_summary_{timestamp}.json"
                with open(summary_file, "w") as f:    json.dump(summary, f, indent=4)
                console.print(
                f"[bold green]Final system test summary saved to {summary_file}[/bold green]")
                if __name__ == "__main__":    main()

