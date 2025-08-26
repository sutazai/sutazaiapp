"""
Command Line Interface for MCP Manager

Provides comprehensive CLI commands for managing MCP servers,
monitoring health, and inspecting system state.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.text import Text
from rich.tree import Tree

from .manager import MCPManager
from .models import MCPManagerConfig, ServerConfig, ServerStatus, HealthStatus

# Initialize Rich console and Typer app
console = Console()
app = typer.Typer(help="Dynamic MCP Management System")

# Global manager instance (initialized per command)
manager: Optional[MCPManager] = None


async def get_manager(config_file: Optional[Path] = None) -> MCPManager:
    """Get or create manager instance"""
    global manager
    
    if manager is None:
        # Load configuration
        if config_file and config_file.exists():
            # TODO: Load config from file
            config = MCPManagerConfig()
        else:
            config = MCPManagerConfig()
        
        manager = MCPManager(config)
    
    return manager


@app.command()
def start(
    config_file: Optional[Path] = typer.Option(None, "--config", "-c", help="Configuration file path"),
    daemon: bool = typer.Option(False, "--daemon", "-d", help="Run as daemon"),
    no_monitoring: bool = typer.Option(False, "--no-monitoring", help="Disable health monitoring"),
) -> None:
    """Start the MCP Manager and all configured servers"""
    
    async def start_manager():
        mgr = await get_manager(config_file)
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Starting MCP Manager...", total=None)
                
                await mgr.start()
                
                progress.update(task, description="MCP Manager started successfully")
                
                if daemon:
                    console.print("[green]✓[/green] MCP Manager started in daemon mode")
                    # Keep running until interrupted
                    try:
                        while mgr.is_running():
                            await asyncio.sleep(1)
                    except KeyboardInterrupt:
                        console.print("\n[yellow]Shutting down...[/yellow]")
                        await mgr.stop()
                else:
                    console.print("[green]✓[/green] MCP Manager started successfully")
                    
        except Exception as e:
            console.print(f"[red]✗[/red] Failed to start MCP Manager: {e}")
            sys.exit(1)
    
    asyncio.run(start_manager())


@app.command()
def stop() -> None:
    """Stop the MCP Manager and all servers"""
    
    async def stop_manager():
        mgr = await get_manager()
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Stopping MCP Manager...", total=None)
                
                await mgr.stop()
                
                progress.update(task, description="MCP Manager stopped successfully")
            
            console.print("[green]✓[/green] MCP Manager stopped successfully")
            
        except Exception as e:
            console.print(f"[red]✗[/red] Failed to stop MCP Manager: {e}")
            sys.exit(1)
    
    asyncio.run(stop_manager())


@app.command()
def status(
    detailed: bool = typer.Option(False, "--detailed", "-d", help="Show detailed status"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON")
) -> None:
    """Show MCP Manager and server status"""
    
    async def show_status():
        mgr = await get_manager()
        
        try:
            system_status = mgr.get_system_status()
            server_states = mgr.get_server_states()
            
            if json_output:
                output = {
                    "system": system_status,
                    "servers": {
                        name: {
                            "status": state.status.value,
                            "health": state.health.status.value,
                            "uptime": state.uptime.total_seconds(),
                            "capabilities": len(state.capabilities)
                        }
                        for name, state in server_states.items()
                    }
                }
                console.print_json(json.dumps(output, indent=2))
                return
            
            # Rich output
            console.print(Panel.fit(
                f"[bold]MCP Manager Status[/bold]\n"
                f"Running: {'[green]Yes[/green]' if system_status['manager_running'] else '[red]No[/red]'}\n"
                f"Total Servers: {system_status['total_servers']}\n"
                f"Running Servers: {system_status['running_servers']}\n"
                f"Healthy Servers: {system_status['healthy_servers']}\n"
                f"Active Connections: {system_status['active_connections']}",
                title="System Status",
                border_style="blue"
            ))
            
            if server_states:
                table = Table(title="Server Status")
                table.add_column("Server", style="cyan")
                table.add_column("Status", style="magenta")
                table.add_column("Health", style="green")
                table.add_column("Uptime", style="blue")
                
                if detailed:
                    table.add_column("Capabilities", style="yellow")
                    table.add_column("Requests", style="red")
                
                for name, state in server_states.items():
                    # Format status with colors
                    status_color = {
                        ServerStatus.RUNNING: "green",
                        ServerStatus.STOPPED: "red",
                        ServerStatus.ERROR: "red",
                        ServerStatus.STARTING: "yellow"
                    }.get(state.status, "white")
                    
                    health_color = {
                        HealthStatus.HEALTHY: "green",
                        HealthStatus.DEGRADED: "yellow",
                        HealthStatus.UNHEALTHY: "red",
                        HealthStatus.CRITICAL: "red"
                    }.get(state.health.status, "white")
                    
                    # Format uptime
                    uptime = state.uptime
                    if uptime.days > 0:
                        uptime_str = f"{uptime.days}d {uptime.seconds // 3600}h"
                    elif uptime.seconds > 3600:
                        uptime_str = f"{uptime.seconds // 3600}h {(uptime.seconds % 3600) // 60}m"
                    else:
                        uptime_str = f"{uptime.seconds // 60}m {uptime.seconds % 60}s"
                    
                    row = [
                        name,
                        f"[{status_color}]{state.status.value}[/{status_color}]",
                        f"[{health_color}]{state.health.status.value}[/{health_color}]",
                        uptime_str
                    ]
                    
                    if detailed:
                        row.extend([
                            str(len(state.capabilities)),
                            str(state.metrics.total_requests)
                        ])
                    
                    table.add_row(*row)
                
                console.print(table)
            else:
                console.print("[yellow]No servers configured[/yellow]")
                
        except Exception as e:
            console.print(f"[red]✗[/red] Error getting status: {e}")
            sys.exit(1)
    
    asyncio.run(show_status())


@app.command()
def list_servers(
    filter_status: Optional[str] = typer.Option(None, "--status", help="Filter by status"),
    show_config: bool = typer.Option(False, "--config", help="Show configuration details")
) -> None:
    """List all MCP servers"""
    
    async def list_all_servers():
        mgr = await get_manager()
        
        try:
            server_states = mgr.get_server_states()
            
            if not server_states:
                console.print("[yellow]No servers found[/yellow]")
                return
            
            # Filter by status if requested
            if filter_status:
                try:
                    status_filter = ServerStatus(filter_status.lower())
                    server_states = {
                        name: state for name, state in server_states.items()
                        if state.status == status_filter
                    }
                except ValueError:
                    console.print(f"[red]Invalid status: {filter_status}[/red]")
                    return
            
            for name, state in server_states.items():
                # Create server panel
                config = state.config
                
                info_lines = [
                    f"[bold]Type:[/bold] {config.server_type.value}",
                    f"[bold]Connection:[/bold] {config.connection_type.value}",
                    f"[bold]Status:[/bold] {state.status.value}",
                    f"[bold]Health:[/bold] {state.health.status.value}",
                    f"[bold]Enabled:[/bold] {'Yes' if config.enabled else 'No'}",
                ]
                
                if show_config:
                    info_lines.extend([
                        f"[bold]Command:[/bold] {config.command}",
                        f"[bold]Args:[/bold] {' '.join(config.args)}",
                        f"[bold]Working Dir:[/bold] {config.working_directory or 'N/A'}",
                    ])
                
                console.print(Panel(
                    "\n".join(info_lines),
                    title=f"[cyan]{name}[/cyan]",
                    subtitle=config.description or "No description",
                    border_style="blue"
                ))
                
        except Exception as e:
            console.print(f"[red]✗[/red] Error listing servers: {e}")
            sys.exit(1)
    
    asyncio.run(list_all_servers())


@app.command()
def start_server(server_name: str) -> None:
    """Start a specific MCP server"""
    
    async def start_single_server():
        mgr = await get_manager()
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task(f"Starting {server_name}...", total=None)
                
                success, error = await mgr.start_server(server_name)
                
                if success:
                    progress.update(task, description=f"✓ {server_name} started successfully")
                    console.print(f"[green]✓[/green] Server '{server_name}' started successfully")
                else:
                    console.print(f"[red]✗[/red] Failed to start '{server_name}': {error}")
                    sys.exit(1)
                    
        except Exception as e:
            console.print(f"[red]✗[/red] Error starting server: {e}")
            sys.exit(1)
    
    asyncio.run(start_single_server())


@app.command()
def stop_server(server_name: str) -> None:
    """Stop a specific MCP server"""
    
    async def stop_single_server():
        mgr = await get_manager()
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task(f"Stopping {server_name}...", total=None)
                
                success, error = await mgr.stop_server(server_name)
                
                if success:
                    progress.update(task, description=f"✓ {server_name} stopped successfully")
                    console.print(f"[green]✓[/green] Server '{server_name}' stopped successfully")
                else:
                    console.print(f"[red]✗[/red] Failed to stop '{server_name}': {error}")
                    sys.exit(1)
                    
        except Exception as e:
            console.print(f"[red]✗[/red] Error stopping server: {e}")
            sys.exit(1)
    
    asyncio.run(stop_single_server())


@app.command()
def restart_server(server_name: str) -> None:
    """Restart a specific MCP server"""
    
    async def restart_single_server():
        mgr = await get_manager()
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task(f"Restarting {server_name}...", total=None)
                
                success, error = await mgr.restart_server(server_name)
                
                if success:
                    progress.update(task, description=f"✓ {server_name} restarted successfully")
                    console.print(f"[green]✓[/green] Server '{server_name}' restarted successfully")
                else:
                    console.print(f"[red]✗[/red] Failed to restart '{server_name}': {error}")
                    sys.exit(1)
                    
        except Exception as e:
            console.print(f"[red]✗[/red] Error restarting server: {e}")
            sys.exit(1)
    
    asyncio.run(restart_single_server())


@app.command()
def discover(
    rescan: bool = typer.Option(False, "--rescan", help="Force rescan of all directories")
) -> None:
    """Discover MCP servers from configuration files"""
    
    async def discover_servers():
        mgr = await get_manager()
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Discovering servers...", total=None)
                
                discovered = await mgr.discovery_engine.discover_servers(force_refresh=rescan)
                
                progress.update(task, description=f"✓ Discovered {len(discovered)} servers")
            
            if discovered:
                table = Table(title="Discovered Servers")
                table.add_column("Name", style="cyan")
                table.add_column("Type", style="green")
                table.add_column("Command", style="yellow")
                table.add_column("Enabled", style="blue")
                
                for name, config in discovered.items():
                    table.add_row(
                        name,
                        config.server_type.value,
                        config.command,
                        "Yes" if config.enabled else "No"
                    )
                
                console.print(table)
            else:
                console.print("[yellow]No servers discovered[/yellow]")
                
        except Exception as e:
            console.print(f"[red]✗[/red] Error discovering servers: {e}")
            sys.exit(1)
    
    asyncio.run(discover_servers())


@app.command()
def capabilities(
    server_name: Optional[str] = typer.Option(None, "--server", help="Show capabilities for specific server"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON")
) -> None:
    """List available MCP capabilities"""
    
    async def show_capabilities():
        mgr = await get_manager()
        
        try:
            if server_name:
                # Show capabilities for specific server
                server_state = mgr.get_server_state(server_name)
                if not server_state:
                    console.print(f"[red]✗[/red] Server '{server_name}' not found")
                    return
                
                capabilities_list = server_state.capabilities
                
                if json_output:
                    output = [
                        {
                            "name": cap.name,
                            "description": cap.description,
                            "parameters": cap.parameters
                        }
                        for cap in capabilities_list
                    ]
                    console.print_json(json.dumps(output, indent=2))
                else:
                    if capabilities_list:
                        for cap in capabilities_list:
                            console.print(Panel(
                                f"[bold]Description:[/bold] {cap.description or 'No description'}\n"
                                f"[bold]Parameters:[/bold] {len(cap.parameters)} parameters",
                                title=f"[cyan]{cap.name}[/cyan]",
                                border_style="blue"
                            ))
                    else:
                        console.print(f"[yellow]No capabilities found for {server_name}[/yellow]")
            else:
                # Show all capabilities
                summary = await mgr.list_capabilities()
                
                if json_output:
                    console.print_json(json.dumps(summary, indent=2))
                else:
                    total_caps = summary.get("total_capabilities", 0)
                    total_servers = summary.get("total_servers", 0)
                    
                    console.print(Panel.fit(
                        f"[bold]Capability Summary[/bold]\n"
                        f"Total Capabilities: {total_caps}\n"
                        f"Total Servers: {total_servers}",
                        title="System Capabilities",
                        border_style="green"
                    ))
                    
                    capabilities = summary.get("capabilities", {})
                    
                    if capabilities:
                        table = Table(title="Available Capabilities")
                        table.add_column("Capability", style="cyan")
                        table.add_column("Servers", style="green")
                        table.add_column("Healthy", style="yellow")
                        table.add_column("Available", style="blue")
                        
                        for cap_name, cap_info in capabilities.items():
                            available = "Yes" if cap_info.get("available", False) else "No"
                            table.add_row(
                                cap_name,
                                str(cap_info.get("total_servers", 0)),
                                str(cap_info.get("healthy_servers", 0)),
                                available
                            )
                        
                        console.print(table)
                    else:
                        console.print("[yellow]No capabilities available[/yellow]")
                        
        except Exception as e:
            console.print(f"[red]✗[/red] Error listing capabilities: {e}")
            sys.exit(1)
    
    asyncio.run(show_capabilities())


@app.command()
def health(
    server_name: Optional[str] = typer.Option(None, "--server", help="Check health of specific server"),
    watch: bool = typer.Option(False, "--watch", "-w", help="Watch health status continuously")
) -> None:
    """Check health status of servers"""
    
    async def check_health():
        mgr = await get_manager()
        
        try:
            if server_name:
                # Check specific server
                health_result = mgr.health_monitor.get_server_health(server_name)
                
                if health_result:
                    status_color = {
                        HealthStatus.HEALTHY: "green",
                        HealthStatus.DEGRADED: "yellow", 
                        HealthStatus.UNHEALTHY: "red",
                        HealthStatus.CRITICAL: "red"
                    }.get(health_result.status, "white")
                    
                    console.print(Panel(
                        f"[bold]Status:[/bold] [{status_color}]{health_result.status.value}[/{status_color}]\n"
                        f"[bold]Response Time:[/bold] {health_result.response_time:.3f}s\n"
                        f"[bold]Running:[/bold] {'Yes' if health_result.is_running else 'No'}\n"
                        f"[bold]Responsive:[/bold] {'Yes' if health_result.is_responsive else 'No'}\n"
                        f"[bold]Capabilities:[/bold] {health_result.capabilities_count}\n"
                        f"[bold]Last Check:[/bold] {health_result.timestamp}\n"
                        f"[bold]Error:[/bold] {health_result.error_message or 'None'}",
                        title=f"[cyan]{server_name}[/cyan] Health",
                        border_style=status_color
                    ))
                else:
                    console.print(f"[yellow]No health data available for {server_name}[/yellow]")
            else:
                # Check all servers
                health_summary = mgr.health_monitor.get_overall_health_summary()
                all_health = mgr.health_monitor.get_all_server_health()
                
                # Overall summary
                overall_color = {
                    "healthy": "green",
                    "degraded": "yellow",
                    "unhealthy": "red",
                    "critical": "red"
                }.get(health_summary.get("status", "unknown"), "white")
                
                console.print(Panel.fit(
                    f"[bold]Overall Health:[/bold] [{overall_color}]{health_summary.get('status', 'Unknown').upper()}[/{overall_color}]\n"
                    f"[bold]Total Servers:[/bold] {health_summary.get('total_servers', 0)}\n"
                    f"[bold]Healthy:[/bold] {health_summary.get('healthy_servers', 0)}\n"
                    f"[bold]Degraded:[/bold] {health_summary.get('degraded_servers', 0)}\n"
                    f"[bold]Unhealthy:[/bold] {health_summary.get('unhealthy_servers', 0)}\n"
                    f"[bold]Critical:[/bold] {health_summary.get('critical_servers', 0)}",
                    title="System Health",
                    border_style=overall_color
                ))
                
                # Individual server health
                if all_health:
                    table = Table(title="Server Health Details")
                    table.add_column("Server", style="cyan")
                    table.add_column("Status", style="magenta")
                    table.add_column("Response Time", style="blue")
                    table.add_column("Error", style="red")
                    
                    for name, health in all_health.items():
                        status_color = {
                            HealthStatus.HEALTHY: "green",
                            HealthStatus.DEGRADED: "yellow",
                            HealthStatus.UNHEALTHY: "red",
                            HealthStatus.CRITICAL: "red"
                        }.get(health.status, "white")
                        
                        table.add_row(
                            name,
                            f"[{status_color}]{health.status.value}[/{status_color}]",
                            f"{health.response_time:.3f}s",
                            health.error_message or "-"
                        )
                    
                    console.print(table)
                        
        except Exception as e:
            console.print(f"[red]✗[/red] Error checking health: {e}")
            sys.exit(1)
    
    if watch:
        # TODO: Implement watch mode
        console.print("[yellow]Watch mode not yet implemented[/yellow]")
    else:
        asyncio.run(check_health())


def main() -> None:
    """Main entry point"""
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()