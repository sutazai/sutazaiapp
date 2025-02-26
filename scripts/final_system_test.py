#!/usr/bin/env python3
# cSpell:ignore sutazai Sutaz maxfail

import json
import subprocess
from datetime import datetime
from typing import Tuple

from rich.console import Console
from rich.panel import Panel
from rich.table import Table


def run_command(command: str) -> Tuple[bool, str]:
    """
    Run a shell command and return whether it succeeded and its output.
    """
    try:
        output = subprocess.check_output(
            command, stderr=subprocess.STDOUT, shell=True, text=True
        )
        return True, output
    except subprocess.CalledProcessError as e:
        return False, e.output


def main():
    console = Console()
    console.rule("[bold blue]Final System Test: Comprehensive Audit[/bold blue]")

    summary = {}
    tasks = [
        (
            "System Validation",
            "python3 /opt/sutazaiapp/scripts/system_comprehensive_validator.py",
        ),
        ("Project Analysis", "python3 /opt/sutazaiapp/scripts/project_analyzer.py"),
        (
            "Documentation Generation",
            "python3 /opt/sutazaiapp/scripts/documentation_generator.py",
        ),
        ("Spell Checking", "python3 /opt/sutazaiapp/scripts/spell_checker.py"),
    ]

    console.print("[bold]Detailed Task Execution:[/bold]")
    for task_name, command in tasks:
        console.print(f"[bold]{task_name}:[/bold] Running command: {command}")
        success, output = run_command(command)
        summary[task_name] = {"success": success, "output": output}
        status = "[green]Success[/green]" if success else "[red]Failed[/red]"
        console.print(f"{task_name}: {status}")
        # Print detailed output inside a collapsible panel
        console.print(Panel(output, title=f"{task_name} Output", expand=False))

    # Execute unit tests via pytest
    console.print("[bold]Unit Tests:[/bold] Running pytest with detailed logs...")
    success, output = run_command("pytest --maxfail=1 --disable-warnings -q")
    summary["Unit Tests"] = {"success": success, "output": output}
    status = "[green]Success[/green]" if success else "[red]Failed[/red]"
    console.print(f"Unit Tests: {status}")
    console.print(Panel(output, title="Unit Tests Output", expand=False))

    # Prepare a more detailed summary table
    table = Table(title="Final System Test Summary")
    table.add_column("Task", style="cyan")
    table.add_column("Status", style="magenta")
    table.add_column("Details", style="green")

    for task, result in summary.items():
        detail = result["output"].splitlines()[0] if result["output"] else "No Output"
        status_text = "Success" if result["success"] else "Failed"
        table.add_row(task, status_text, detail)

    console.print(table)

    # Save JSON summary with full details
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = f"/opt/sutazaiapp/logs/final_system_test_summary_{timestamp}.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=4)
    console.print(
        f"[bold green]Final system test summary saved to {summary_file}[/bold green]"
    )


if __name__ == "__main__":
    main()
