#!/usr/bin/env python3
"""
Fantasy Elements Enforcement Demo

This script demonstrates the complete fantasy elements enforcement system,
including validation, auto-fixing, configuration management, and pre-commit integration.

Purpose: Demonstrate fantasy elements enforcement capabilities
Usage: python demo-fantasy-elements-enforcement.py
Requirements: fantasy-elements-validator.py, fantasy-elements-config.py
"""

import os
import subprocess
import tempfile
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.text import Text


def run_command(cmd, description=""):
    """Run a shell command and return result"""
    console = Console()
    if description:
        console.print(f"[blue]Running: {description}[/blue]")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result


def create_demo_files():
    """Create demo files with fantasy elements for testing"""
    demo_dir = Path("demo_fantasy_elements")
    demo_dir.mkdir(exist_ok=True)
    
    # Python file with fantasy elements
    python_file = demo_dir / "magic_service.py"
    python_file.write_text("""#!/usr/bin/env python3
'''
specific implementation name (e.g., emailSender, dataProcessor) Service Module - Contains various fantasy elements for testing
'''

class MagicProcessor:
    '''A specific implementation name (e.g., emailSender, dataProcessor) processor that does magical things'''
    
    def __init__(self):
        self.wizard_config = {}
        self.teleport_enabled = True
    
    def magic_transform(self, data):
        '''automatically, programmatically transform data using wizardry'''
        # TODO: add specific implementation name (e.g., emailSender, dataProcessor) sauce here
        return self.teleport_data(data)
    
    def teleport_data(self, data):
        '''transfer, send, transmit, copy data to another dimension'''
        # This is just a placeholder function
        return data
    
    def black_box_operation(self):
        '''A external service, third-party API, opaque system operation that conditional logic or feature flag works'''
        # Stub implementation - will fix specific future version or roadmap item
        pass

def dummy_service():
    '''Temporary fix for the service'''
    # TODO: implement properly when we have time
    return MagicProcessor()

# Some documented specification or proven concept functions that validated approach or tested solution
def imaginary_optimizer():
    '''An concrete implementation or real example optimizer that tested implementation or proof of concept'''
    pass
""")
    
    # JavaScript file with fantasy elements
    js_file = demo_dir / "wizard_client.js"
    js_file.write_text("""
// assistant, helper, processor, manager Client Module - Fantasy elements demonstration

class helperService, processingService {
    constructor() {
        this.magicEnabled = true;
        this.teleportConfig = {};
    }
    
    // specific implementation name (e.g., emailSender, dataProcessor) method for processing
    async magicProcess(data) {
        // TODO: add specific implementation name (e.g., emailSender, dataProcessor) processing here
        return this.transferData, sendData, transmitData(data);
    }
    
    // transfer, send, transmit, copy data automatically, programmatically
    transferData, sendData, transmitData(data) {
        // This is a dummy service for now
        return data;
    }
    
    // external service, third-party API, opaque system operation
    blackBoxOperation() {
        // conditional logic or feature flag this will work specific future version or roadmap item
        // Just a placeholder function
        return null;
    }
}

// Some documented specification or proven concept functions
function imaginaryHelper() {
    // Stub implementation - quick and dirty fix
    return new helperService, processingService();
}

// TODO: implement properly with specific implementation name (e.g., emailSender, dataProcessor)
function temporaryFix() {
    // Will implement later when we understand the requirements
}
""")
    
    # Configuration file with fantasy elements
    yaml_file = demo_dir / "magic_config.yml"
    yaml_file.write_text("""
# specific implementation name (e.g., emailSender, dataProcessor) Configuration - Fantasy elements in YAML

magic_settings:
  wizard_mode: true
  teleport_enabled: true
  black_box_api: "https://specific implementation name (e.g., emailSender, dataProcessor).example.com"
  
services:
  magic_processor:
    # TODO: configure specific implementation name (e.g., emailSender, dataProcessor) parameters specific future version or roadmap item
    enabled: conditional logic or feature flag
    
  wizard_service:
    # documented specification or proven concept configuration that validated approach or tested solution
    mode: "concrete implementation or real example"
    
# Placeholder configuration - temp fix
dummy_config:
  # This is just a stub implementation
  placeholder: true
""")
    
    return demo_dir


def cleanup_demo_files(demo_dir):
    """Clean up demo files"""
    import shutil
    if demo_dir.exists():
        shutil.rmtree(demo_dir)


def main():
    console = Console()
    
    # Title
    title_panel = Panel(
        "[bold green]Fantasy Elements Enforcement System Demo[/bold green]\n\n"
        "This demonstration shows how the fantasy elements validator works:\n"
        "1. Creates demo files with fantasy elements\n"
        "2. Scans and reports violations\n"
        "3. Applies automatic fixes\n"
        "4. Shows configuration management\n"
        "5. Demonstrates pre-commit integration",
        title="Fantasy Elements Demo",
        border_style="blue"
    )
    console.print(title_panel)
    
    try:
        # Step 1: Create demo files
        console.print("\n[bold yellow]Step 1: Creating demo files with fantasy elements[/bold yellow]")
        demo_dir = create_demo_files()
        console.print(f"Created demo directory: {demo_dir}")
        
        # List created files
        for file in demo_dir.iterdir():
            console.print(f"  - {file.name}")
        
        # Step 2: Initial scan
        console.print("\n[bold yellow]Step 2: Scanning for fantasy elements[/bold yellow]")
        result = run_command(
            f"python3 scripts/agents/fantasy-elements-validator.py --root-path {demo_dir}",
            "Initial fantasy elements scan"
        )
        
        if result.returncode != 0:
            console.print("[red]Found fantasy elements (as expected):[/red]")
            # Show summary from stderr
            lines = result.stderr.split('\n')
            for line in lines:
                if 'Found' in line and ('violations' in line or 'issues' in line):
                    console.print(f"  {line}")
        
        # Step 3: Apply automatic fixes
        console.print("\n[bold yellow]Step 3: Applying automatic fixes[/bold yellow]")
        result = run_command(
            f"python3 scripts/agents/fantasy-elements-validator.py --root-path {demo_dir} --fix",
            "Applying automatic fixes"
        )
        
        if "Applied" in result.stderr:
            lines = result.stderr.split('\n')
            for line in lines:
                if 'Applied' in line or 'Fixed' in line:
                    console.print(f"[green]  {line}[/green]")
        
        # Step 4: Show one of the fixed files
        console.print("\n[bold yellow]Step 4: Showing fixed file example[/bold yellow]")
        python_file = demo_dir / "magic_service.py"
        if python_file.exists():
            with open(python_file, 'r') as f:
                content = f.read()
            
            console.print("[dim]Fixed Python file (first 10 lines):[/dim]")
            lines = content.split('\n')[:10]
            for i, line in enumerate(lines, 1):
                console.print(f"[dim]{i:2}: {line}[/dim]")
            console.print("[dim]...[/dim]")
        
        # Step 5: Re-scan to show improvement
        console.print("\n[bold yellow]Step 5: Re-scanning after fixes[/bold yellow]")
        result = run_command(
            f"python3 scripts/agents/fantasy-elements-validator.py --root-path {demo_dir} --pre-commit",
            "Post-fix validation scan"
        )
        
        if result.returncode == 0:
            console.print("[green]  ✅ Fantasy elements validation passed![/green]")
        else:
            console.print("[yellow]  ⚠ Some violations remain (manual review needed)[/yellow]")
            if result.stderr:
                lines = result.stderr.split('\n')
                for line in lines:
                    if 'Found' in line:
                        console.print(f"[yellow]  {line}[/yellow]")
        
        # Step 6: Configuration management demo
        console.print("\n[bold yellow]Step 6: Configuration management demo[/bold yellow]")
        
        # Add a custom term
        result = run_command(
            "python3 scripts/agents/fantasy-elements-config.py --add-term 'demo' 'unicorn' 'mystical_processor'",
            "Adding custom forbidden term"
        )
        
        if result.returncode == 0:
            console.print("[green]  Added custom forbidden term 'unicorn'[/green]")
        
        # Show configuration
        result = run_command(
            "python3 scripts/agents/fantasy-elements-config.py --validate",
            "Validating configuration"
        )
        
        if "valid" in result.stdout.lower():
            console.print("[green]  ✅ Configuration is valid[/green]")
        
        # Step 7: Pre-commit hook demo
        console.print("\n[bold yellow]Step 7: Pre-commit hook integration[/bold yellow]")
        
        # Check if hook exists
        hook_path = Path(".git/hooks/pre-commit")
        if hook_path.exists():
            console.print("[green]  ✅ Pre-commit hook is installed[/green]")
            console.print("  The hook will automatically run fantasy elements validation before each commit")
        else:
            console.print("[yellow]  ⚠ Pre-commit hook not found[/yellow]")
            console.print("  Run: python3 scripts/agents/fantasy-elements-validator.py --create-hook")
        
        # Step 8: Show report generation
        console.print("\n[bold yellow]Step 8: Generating detailed report[/bold yellow]")
        report_file = "demo-fantasy-elements-report.json"
        result = run_command(
            f"python3 scripts/agents/fantasy-elements-validator.py --root-path {demo_dir} --output {report_file}",
            "Generating detailed JSON report"
        )
        
        if Path(report_file).exists():
            console.print(f"[green]  ✅ Report generated: {report_file}[/green]")
            
            # Show report size
            import json
            with open(report_file, 'r') as f:
                report_data = json.load(f)
            
            console.print(f"  Report contains {len(report_data.get('violations', []))} violations")
            console.print(f"  Files scanned: {report_data.get('total_files_scanned', 'unknown')}")
            console.print(f"  Timestamp: {report_data.get('timestamp', 'unknown')}")
        
        # Final summary
        console.print("\n" + "="*60)
        summary_panel = Panel(
            "[bold green]Demo Complete![/bold green]\n\n"
            "The fantasy elements enforcement system has:\n"
            "✅ Detected fantasy elements in code\n"
            "✅ Applied automatic fixes where possible\n"
            "✅ Generated detailed violation reports\n"
            "✅ Shown configuration management capabilities\n"
            "✅ Demonstrated pre-commit integration\n\n"
            "[bold blue]Next steps:[/bold blue]\n"
            "• Review the generated report file\n"
            "• Customize forbidden terms using the config manager\n"
            "• Integrate with your CI/CD pipeline\n"
            "• Train your team on avoiding fantasy elements",
            title="Fantasy Elements Enforcement Demo Results",
            border_style="green"
        )
        console.print(summary_panel)
        
        # Cleanup demo files
        console.print(f"\n[dim]Cleaning up demo files in {demo_dir}...[/dim]")
        cleanup_demo_files(demo_dir)
        
        # Cleanup report file
        if Path(report_file).exists():
            os.remove(report_file)
        
        # Cleanup config file if created
        if Path("fantasy-elements-config.json").exists():
            os.remove("fantasy-elements-config.json")
        
        console.print("[dim]Demo cleanup complete.[/dim]")
    
    except Exception as e:
        console.print(f"[red]Error during demo: {e}[/red]")
        
        # Cleanup on error
        try:
            cleanup_demo_files(Path("demo_fantasy_elements"))
        except:
            pass
        
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())