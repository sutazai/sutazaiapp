#!/usr/bin/env python3
"""
Energy Management CLI - Command-line interface for energy optimization
"""

import asyncio
import click
import json
import sys
import os
from datetime import datetime, timedelta
from typing import Optional

# Add the parent directory to the path to import energy modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from energy.energy_profiler import get_global_profiler, start_global_monitoring, stop_global_monitoring
from energy.power_optimizer import get_global_optimizer, OptimizationStrategy
from energy.agent_hibernation import get_hibernation_manager
from energy.workload_scheduler import get_global_scheduler, SchedulingPolicy
from energy.resource_allocator import get_global_allocator, AllocationStrategy
from energy.sustainability_manager import get_global_sustainability_manager
from energy.monitoring_dashboard import run_dashboard

@click.group()
@click.version_option(version="1.0.0")
def cli():
    """SutazAI Energy Management CLI
    
    Comprehensive energy optimization and monitoring tools for the SutazAI system.
    """
    pass

# Energy Profiling Commands
@cli.group()
def profile():
    """Energy profiling and monitoring commands"""
    pass

@profile.command()
@click.option('--duration', '-d', default=60, help='Monitoring duration in seconds')
@click.option('--interval', '-i', default=1.0, help='Measurement interval in seconds')
@click.option('--output', '-o', help='Output file for results')
def start(duration: int, interval: float, output: Optional[str]):
    """Start energy profiling"""
    click.echo(f"Starting energy profiling for {duration} seconds...")
    
    profiler = get_global_profiler()
    profiler.measurement_interval = interval
    profiler.start_monitoring()
    
    try:
        import time
        time.sleep(duration)
    except KeyboardInterrupt:
        click.echo("\nProfiling interrupted by user")
    finally:
        profiler.stop_monitoring()
    
    # Get results
    metrics = profiler.calculate_energy_metrics(duration / 3600)
    
    click.echo(f"\n📊 Energy Profiling Results:")
    click.echo(f"├─ Duration: {duration}s")
    click.echo(f"├─ Total Energy: {metrics.total_energy_wh:.2f} Wh")
    click.echo(f"├─ Average Power: {metrics.avg_power_w:.1f} W")
    click.echo(f"├─ Peak Power: {metrics.peak_power_w:.1f} W")
    click.echo(f"├─ CPU Energy: {metrics.cpu_energy_wh:.2f} Wh")
    click.echo(f"├─ Memory Energy: {metrics.memory_energy_wh:.2f} Wh")
    click.echo(f"└─ CO2 Emissions: {metrics.co2_emission_g:.1f} g")
    
    if output:
        profiler.export_measurements(output, duration / 3600)
        click.echo(f"\n💾 Results exported to {output}")

@profile.command()
def current():
    """Show current energy metrics"""
    profiler = get_global_profiler()
    profiler.start_monitoring()
    
    import time
    time.sleep(2)  # Collect some data
    
    current_metrics = profiler.get_current_metrics()
    efficiency_metrics = profiler.get_efficiency_metrics()
    
    profiler.stop_monitoring()
    
    click.echo("⚡ Current Energy Metrics:")
    if current_metrics:
        click.echo(f"├─ Current Power: {current_metrics.get('current_power_w', 0):.1f} W")
        click.echo(f"├─ CPU Power: {current_metrics.get('cpu_power_w', 0):.1f} W")
        click.echo(f"├─ Memory Power: {current_metrics.get('memory_power_w', 0):.1f} W")
        click.echo(f"├─ CPU Utilization: {current_metrics.get('cpu_utilization', 0):.1f}%")
        click.echo(f"├─ Memory Utilization: {current_metrics.get('memory_utilization', 0):.1f}%")
        click.echo(f"└─ Active Agents: {current_metrics.get('active_agents', 0)}")
    
    if efficiency_metrics:
        click.echo(f"\n📈 Efficiency Metrics:")
        click.echo(f"├─ Power per CPU%: {efficiency_metrics.get('power_per_cpu_percent', 0):.2f} W")
        click.echo(f"├─ Power per Agent: {efficiency_metrics.get('power_per_agent', 0):.2f} W")
        click.echo(f"└─ Efficiency Score: {efficiency_metrics.get('utilization_efficiency_score', 0):.0f}/100")

# Power Optimization Commands
@cli.group()
def optimize():
    """Power optimization commands"""
    pass

@optimize.command()
@click.option('--strategy', '-s', 
              type=click.Choice(['aggressive', 'balanced', 'conservative']),
              default='balanced',
              help='Optimization strategy')
@click.option('--duration', '-d', default=3600, help='Optimization duration in seconds')
def start(strategy: str, duration: int):
    """Start power optimization"""
    click.echo(f"🔧 Starting power optimization with {strategy} strategy...")
    
    optimizer = get_global_optimizer(OptimizationStrategy(strategy))
    optimizer.start_optimization()
    
    try:
        import time
        start_time = time.time()
        while time.time() - start_time < duration:
            time.sleep(10)
            stats = optimizer.get_optimization_stats()
            
            # Show periodic updates
            elapsed = int(time.time() - start_time)
            if elapsed % 60 == 0:  # Every minute
                click.echo(f"⏱️  {elapsed//60}m - Power saved: {stats.get('total_power_saved_w', 0):.1f}W")
    
    except KeyboardInterrupt:
        click.echo("\n⏹️  Optimization stopped by user")
    finally:
        stats = optimizer.get_optimization_stats()
        optimizer.stop_optimization()
    
    # Show final results
    click.echo(f"\n📊 Optimization Results:")
    click.echo(f"├─ Total Optimizations: {stats.get('total_optimizations', 0)}")
    click.echo(f"├─ Successful: {stats.get('successful_optimizations', 0)}")
    click.echo(f"├─ Success Rate: {stats.get('success_rate', 0)*100:.1f}%")
    click.echo(f"├─ Total Power Saved: {stats.get('total_power_saved_w', 0):.1f} W")
    click.echo(f"└─ Average Power Saved: {stats.get('avg_power_saved_w', 0):.1f} W")

@optimize.command()
def status():
    """Show optimization status"""
    optimizer = get_global_optimizer()
    stats = optimizer.get_optimization_stats()
    
    click.echo("🔧 Power Optimization Status:")
    click.echo(f"├─ Strategy: {stats.get('current_strategy', 'unknown')}")
    click.echo(f"├─ Total Optimizations: {stats.get('total_optimizations', 0)}")
    click.echo(f"├─ Success Rate: {stats.get('success_rate', 0)*100:.1f}%")
    click.echo(f"└─ Power Saved: {stats.get('total_power_saved_w', 0):.1f} W")

# Agent Hibernation Commands
@cli.group()
def hibernation():
    """Agent hibernation management commands"""
    pass

@hibernation.command()
@click.option('--duration', '-d', default=3600, help='Monitoring duration in seconds')
def start(duration: int):
    """Start agent hibernation monitoring"""
    click.echo("😴 Starting agent hibernation monitoring...")
    
    hibernation_manager = get_hibernation_manager()
    hibernation_manager.start_monitoring()
    
    try:
        import time
        start_time = time.time()
        while time.time() - start_time < duration:
            time.sleep(30)
            stats = hibernation_manager.get_hibernation_stats()
            
            # Show periodic updates
            elapsed = int(time.time() - start_time)
            if elapsed % 300 == 0:  # Every 5 minutes
                hibernated = stats.get('currently_hibernated', 0)
                total = stats.get('total_agents', 0)
                click.echo(f"⏱️  {elapsed//60}m - Hibernated: {hibernated}/{total} agents")
    
    except KeyboardInterrupt:
        click.echo("\n⏹️  Hibernation monitoring stopped by user")
    finally:
        stats = hibernation_manager.get_hibernation_stats()
        hibernation_manager.stop_monitoring()
    
    # Show final results
    click.echo(f"\n📊 Hibernation Results:")
    click.echo(f"├─ Total Agents: {stats.get('total_agents', 0)}")
    click.echo(f"├─ Currently Hibernated: {stats.get('currently_hibernated', 0)}")
    click.echo(f"├─ Hibernation Ratio: {stats.get('hibernation_ratio', 0)*100:.1f}%")
    click.echo(f"├─ Total Power Saved: {stats.get('total_power_saved_w', 0):.1f} W")
    click.echo(f"└─ Success Rate: {stats.get('hibernation_success_rate', 0)*100:.1f}%")

@hibernation.command()
def wake_all():
    """Wake all hibernated agents"""
    hibernation_manager = get_hibernation_manager()
    woken_count = hibernation_manager.force_wake_all()
    click.echo(f"🌅 Woke {woken_count} hibernated agents")

@hibernation.command()
def status():
    """Show hibernation status"""
    hibernation_manager = get_hibernation_manager()
    stats = hibernation_manager.get_hibernation_stats()
    
    click.echo("😴 Agent Hibernation Status:")
    click.echo(f"├─ Total Agents: {stats.get('total_agents', 0)}")
    click.echo(f"├─ Currently Hibernated: {stats.get('currently_hibernated', 0)}")
    click.echo(f"├─ Hibernation Ratio: {stats.get('hibernation_ratio', 0)*100:.1f}%")
    click.echo(f"├─ Power Saved: {stats.get('total_power_saved_w', 0):.1f} W")
    click.echo(f"└─ Monitoring Active: {stats.get('monitoring_active', False)}")

# Scheduling Commands
@cli.group()
def schedule():
    """Energy-aware workload scheduling commands"""
    pass

@schedule.command()
@click.option('--policy', '-p',
              type=click.Choice(['energy_first', 'performance_first', 'balanced', 'carbon_aware']),
              default='balanced',
              help='Scheduling policy')
def start(policy: str):
    """Start energy-aware scheduling"""
    click.echo(f"📅 Starting energy-aware scheduling with {policy} policy...")
    
    scheduler = get_global_scheduler(SchedulingPolicy(policy))
    scheduler.start_scheduling()
    
    click.echo("✅ Scheduler started. Use 'schedule status' to monitor.")

@schedule.command()
def stop():
    """Stop energy-aware scheduling"""
    scheduler = get_global_scheduler()
    scheduler.stop_scheduling()
    click.echo("⏹️  Scheduler stopped")

@schedule.command()
def status():
    """Show scheduling status"""
    scheduler = get_global_scheduler()
    stats = scheduler.get_scheduling_stats()
    
    click.echo("📅 Scheduling Status:")
    click.echo(f"├─ Policy: {stats.get('policy', 'unknown')}")
    click.echo(f"├─ Pending Tasks: {stats.get('tasks_pending', 0)}")
    click.echo(f"├─ Running Tasks: {stats.get('tasks_running', 0)}")
    click.echo(f"├─ Completed Tasks: {stats.get('tasks_completed', 0)}")
    click.echo(f"├─ Energy Budget Used: {stats.get('energy_budget_utilization', 0)*100:.1f}%")
    click.echo(f"└─ Carbon Budget Used: {stats.get('carbon_budget_utilization', 0)*100:.1f}%")

# Resource Allocation Commands
@cli.group()
def allocate():
    """Resource allocation commands"""
    pass

@allocate.command()
@click.option('--strategy', '-s',
              type=click.Choice(['energy_proportional', 'workload_aware', 'thermal_balanced', 'carbon_optimized']),
              default='energy_proportional',
              help='Allocation strategy')
def start(strategy: str):
    """Start resource allocation monitoring"""
    click.echo(f"🎯 Starting resource allocation with {strategy} strategy...")
    
    allocator = get_global_allocator(AllocationStrategy(strategy))
    allocator.start_monitoring()
    
    click.echo("✅ Resource allocator started")

@allocate.command()
def status():
    """Show allocation status"""
    allocator = get_global_allocator()
    stats = allocator.get_allocation_stats()
    
    click.echo("🎯 Resource Allocation Status:")
    click.echo(f"├─ Strategy: {stats.get('strategy', 'unknown')}")
    click.echo(f"├─ Total Allocations: {stats.get('total_allocations', 0)}")
    click.echo(f"├─ CPU Utilization: {stats.get('resource_utilization', {}).get('cpu_cores', 0)*100:.1f}%")
    click.echo(f"├─ Memory Utilization: {stats.get('resource_utilization', {}).get('memory', 0)*100:.1f}%")
    click.echo(f"├─ Total Power Budget: {stats.get('total_power_budget_w', 0):.1f} W")
    click.echo(f"└─ Avg Efficiency Score: {stats.get('avg_efficiency_score', 0):.1f}/100")

# Sustainability Commands
@cli.group()
def sustainability():
    """Sustainability and carbon footprint commands"""
    pass

@sustainability.command()
@click.option('--hours', '-h', default=24.0, help='Hours of data to analyze')
def metrics(hours: float):
    """Show sustainability metrics"""
    sustainability_manager = get_global_sustainability_manager()
    sustainability_manager.start_monitoring()
    
    import time
    time.sleep(5)  # Collect some data
    
    metrics = sustainability_manager.calculate_sustainability_metrics(hours)
    budget_status = sustainability_manager.get_daily_budget_status()
    
    click.echo("🌱 Sustainability Metrics:")
    click.echo(f"├─ Period: {hours} hours")
    click.echo(f"├─ Total Energy: {metrics.total_energy_kwh:.3f} kWh")
    click.echo(f"├─ Total CO2: {metrics.total_co2_kg:.3f} kg")
    click.echo(f"├─ Carbon Intensity: {metrics.avg_carbon_intensity:.3f} kg/kWh")
    click.echo(f"├─ Renewable Energy: {metrics.renewable_energy_kwh:.3f} kWh")
    click.echo(f"├─ Efficiency Score: {metrics.efficiency_score:.1f}/100")
    click.echo(f"├─ Sustainability Grade: {metrics.sustainability_grade}")
    click.echo(f"├─ Carbon Saved: {metrics.carbon_saved_kg:.3f} kg")
    click.echo(f"└─ Energy Saved: {metrics.energy_saved_kwh:.3f} kWh")
    
    click.echo(f"\n💰 Daily Budget Status ({budget_status['status'].upper()}):")
    click.echo(f"├─ Energy: {budget_status['energy']['utilization_pct']:.1f}% ({budget_status['energy']['consumed_kwh']:.2f}/{budget_status['energy']['budget_kwh']:.1f} kWh)")
    click.echo(f"├─ Carbon: {budget_status['carbon']['utilization_pct']:.1f}% ({budget_status['carbon']['emitted_kg']:.2f}/{budget_status['carbon']['budget_kg']:.1f} kg)")
    click.echo(f"└─ Cost: {budget_status['cost']['utilization_pct']:.1f}% (${budget_status['cost']['spent_usd']:.2f}/${budget_status['cost']['budget_usd']:.2f})")

@sustainability.command()
def recommendations():
    """Get sustainability recommendations"""
    sustainability_manager = get_global_sustainability_manager()
    recommendations = sustainability_manager.get_sustainability_recommendations()
    
    if not recommendations:
        click.echo("🌟 No recommendations at this time - system is performing optimally!")
        return
    
    click.echo("💡 Sustainability Recommendations:")
    for i, rec in enumerate(recommendations, 1):
        priority_icon = "🔴" if rec['priority'] == 'high' else "🟡" if rec['priority'] == 'medium' else "🟢"
        click.echo(f"\n{priority_icon} {i}. {rec['title']} ({rec['priority'].upper()})")
        click.echo(f"   Category: {rec['category']}")
        click.echo(f"   Description: {rec['description']}")
        click.echo(f"   Estimated Savings: {rec['estimated_savings_kwh']:.1f} kWh, {rec['estimated_co2_reduction_kg']:.1f} kg CO2")

@sustainability.command()
@click.option('--hours', '-h', default=24, help='Hours ahead to forecast')
def forecast(hours: int):
    """Show carbon intensity forecast"""
    sustainability_manager = get_global_sustainability_manager()
    forecast_data = sustainability_manager.get_carbon_forecast(hours)
    
    click.echo(f"🔮 Carbon Intensity Forecast ({hours} hours):")
    
    optimal_hours = []
    suboptimal_hours = []
    
    for i, period in enumerate(forecast_data):
        time_str = datetime.fromisoformat(period['timestamp']).strftime('%H:%M')
        intensity = period['carbon_intensity_kg_per_kwh']
        renewable_pct = period['renewable_percentage']
        
        if period['recommendation'] == 'optimal':
            optimal_hours.append(time_str)
        else:
            suboptimal_hours.append(time_str)
        
        if i < 8:  # Show first 8 hours in detail
            status_icon = "🟢" if period['recommendation'] == 'optimal' else "🟡"
            click.echo(f"{status_icon} {time_str}: {intensity:.3f} kg/kWh ({renewable_pct:.0f}% renewable)")
    
    if len(forecast_data) > 8:
        click.echo(f"... and {len(forecast_data) - 8} more hours")
    
    click.echo(f"\n📊 Summary:")
    click.echo(f"├─ Optimal hours: {len(optimal_hours)} ({', '.join(optimal_hours[:5])}{'...' if len(optimal_hours) > 5 else ''})")
    click.echo(f"└─ Suboptimal hours: {len(suboptimal_hours)}")

# Dashboard Commands
@cli.group()
def dashboard():
    """Monitoring dashboard commands"""
    pass

@dashboard.command()
@click.option('--host', '-h', default='0.0.0.0', help='Host to bind dashboard to')
@click.option('--port', '-p', default=8080, help='Port to bind dashboard to')
def start(host: str, port: int):
    """Start the energy monitoring dashboard"""
    click.echo(f"🚀 Starting energy monitoring dashboard at http://{host}:{port}")
    click.echo("Press Ctrl+C to stop")
    
    try:
        asyncio.run(run_dashboard(host, port))
    except KeyboardInterrupt:
        click.echo("\n⏹️  Dashboard stopped")

# System Commands
@cli.group()
def system():
    """System-wide energy management commands"""
    pass

@system.command()
def start_all():
    """Start all energy management systems"""
    click.echo("🚀 Starting all energy management systems...")
    
    # Start all components
    start_global_monitoring()
    get_global_optimizer().start_optimization()
    get_hibernation_manager().start_monitoring()
    get_global_scheduler().start_scheduling()
    get_global_allocator().start_monitoring()
    get_global_sustainability_manager().start_monitoring()
    
    click.echo("✅ All systems started:")
    click.echo("├─ ⚡ Energy Monitoring")
    click.echo("├─ 🔧 Power Optimization")
    click.echo("├─ 😴 Agent Hibernation")
    click.echo("├─ 📅 Workload Scheduling")
    click.echo("├─ 🎯 Resource Allocation")
    click.echo("└─ 🌱 Sustainability Tracking")

@system.command()
def stop_all():
    """Stop all energy management systems"""
    click.echo("⏹️  Stopping all energy management systems...")
    
    # Stop all components
    stop_global_monitoring()
    get_global_optimizer().stop_optimization()
    get_hibernation_manager().stop_monitoring()
    get_global_scheduler().stop_scheduling()
    get_global_allocator().stop_monitoring()
    get_global_sustainability_manager().stop_monitoring()
    
    click.echo("✅ All systems stopped")

@system.command()
def status():
    """Show overall system status"""
    click.echo("📊 SutazAI Energy Management System Status\n")
    
    # Energy profiling
    profiler = get_global_profiler()
    current_metrics = profiler.get_current_metrics()
    if current_metrics:
        click.echo("⚡ Energy Monitoring: ACTIVE")
        click.echo(f"├─ Current Power: {current_metrics.get('current_power_w', 0):.1f} W")
        click.echo(f"└─ Active Agents: {current_metrics.get('active_agents', 0)}")
    else:
        click.echo("⚡ Energy Monitoring: INACTIVE")
    
    # Power optimization
    optimizer = get_global_optimizer()
    opt_stats = optimizer.get_optimization_stats()
    click.echo(f"\n🔧 Power Optimization: {'ACTIVE' if opt_stats.get('total_optimizations', 0) > 0 else 'INACTIVE'}")
    click.echo(f"├─ Strategy: {opt_stats.get('current_strategy', 'none')}")
    click.echo(f"└─ Power Saved: {opt_stats.get('total_power_saved_w', 0):.1f} W")
    
    # Hibernation
    hibernation_manager = get_hibernation_manager()
    hib_stats = hibernation_manager.get_hibernation_stats()
    click.echo(f"\n😴 Agent Hibernation: {'ACTIVE' if hib_stats.get('monitoring_active') else 'INACTIVE'}")
    click.echo(f"├─ Hibernated: {hib_stats.get('currently_hibernated', 0)}/{hib_stats.get('total_agents', 0)}")
    click.echo(f"└─ Power Saved: {hib_stats.get('total_power_saved_w', 0):.1f} W")
    
    # Scheduling
    scheduler = get_global_scheduler()
    sched_stats = scheduler.get_scheduling_stats()
    click.echo(f"\n📅 Workload Scheduling: {'ACTIVE' if sched_stats.get('scheduling_active') else 'INACTIVE'}")
    click.echo(f"├─ Policy: {sched_stats.get('policy', 'none')}")
    click.echo(f"└─ Tasks: {sched_stats.get('tasks_pending', 0)} pending, {sched_stats.get('tasks_running', 0)} running")
    
    # Resource allocation
    allocator = get_global_allocator()
    alloc_stats = allocator.get_allocation_stats()
    click.echo(f"\n🎯 Resource Allocation: {'ACTIVE' if alloc_stats.get('monitoring_active') else 'INACTIVE'}")
    click.echo(f"├─ Strategy: {alloc_stats.get('strategy', 'none')}")
    click.echo(f"└─ Allocations: {alloc_stats.get('total_allocations', 0)}")
    
    # Sustainability
    sustainability_manager = get_global_sustainability_manager()
    sustainability_manager.start_monitoring()
    import time
    time.sleep(1)
    sust_metrics = sustainability_manager.calculate_sustainability_metrics(1.0)
    click.echo(f"\n🌱 Sustainability Tracking: MONITORING")
    click.echo(f"├─ Grade: {sust_metrics.sustainability_grade}")
    click.echo(f"└─ Efficiency Score: {sust_metrics.efficiency_score:.1f}/100")

# Export Commands
@cli.group()
def export():
    """Data export commands"""
    pass

@export.command()
@click.option('--hours', '-h', default=24.0, help='Hours of data to export')
@click.option('--output', '-o', help='Output file path')
def energy(hours: float, output: Optional[str]):
    """Export energy consumption data"""
    if not output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = f"energy_export_{timestamp}.json"
    
    profiler = get_global_profiler()
    profiler.export_measurements(output, hours)
    
    click.echo(f"💾 Energy data exported to {output}")

@export.command()
@click.option('--days', '-d', default=30, help='Days of data to include')
@click.option('--output', '-o', help='Output file path')
def sustainability(days: int, output: Optional[str]):
    """Export sustainability report"""
    if not output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = f"sustainability_report_{timestamp}.json"
    
    sustainability_manager = get_global_sustainability_manager()
    sustainability_manager.export_sustainability_report(output, days)
    
    click.echo(f"💾 Sustainability report exported to {output}")

if __name__ == '__main__':
    cli()