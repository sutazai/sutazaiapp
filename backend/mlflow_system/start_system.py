#!/usr/bin/env python3
"""
SutazAI MLflow System Startup Script
Initialize and start the complete MLflow experiment tracking system
"""

import asyncio
import logging
import signal
import sys
import os
from pathlib import Path
from typing import Optional
import click

# Add the backend directory to Python path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from mlflow_system import initialize_system, mlflow_system, shutdown_system
from mlflow_system.config import MLflowConfig, config_manager
from mlflow_system.dashboard import run_dashboard


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/opt/sutazaiapp/backend/logs/mlflow_system.log')
    ]
)

logger = logging.getLogger(__name__)


class MLflowSystemLauncher:
    """Main launcher for the MLflow system"""
    
    def __init__(self):
        self.shutdown_event = asyncio.Event()
        self.system_running = False
        
    async def start_system(self, dashboard: bool = False, config_file: Optional[str] = None):
        """Start the complete MLflow system"""
        try:
            logger.info("🚀 Starting SutazAI MLflow System...")
            
            # Load custom config if provided
            if config_file and os.path.exists(config_file):
                logger.info(f"Loading configuration from {config_file}")
                # Custom config loading would go here
            
            # Initialize the system
            success = await initialize_system()
            
            if not success:
                logger.error("❌ Failed to initialize MLflow system")
                return False
            
            self.system_running = True
            
            # Set up signal handlers for graceful shutdown
            self._setup_signal_handlers()
            
            # Start dashboard if requested
            if dashboard:
                logger.info("🖥️ Starting MLflow dashboard...")
                await self._start_dashboard()
            
            # Print system information
            self._print_system_info()
            
            # Wait for shutdown signal
            logger.info("✅ SutazAI MLflow System is now running")
            logger.info("Press Ctrl+C to shutdown gracefully")
            
            await self.shutdown_event.wait()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ System startup failed: {e}")
            return False
        finally:
            await self._shutdown()
    
    def _setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info(f"🛑 Received signal {signum}, initiating shutdown...")
            self.shutdown_event.set()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def _start_dashboard(self):
        """Start the Streamlit dashboard in a separate process"""
        try:
            import subprocess
            import sys
            
            # Start dashboard as subprocess
            dashboard_script = Path(__file__).parent / "dashboard.py"
            
            # Create dashboard startup command
            cmd = [
                sys.executable, "-m", "streamlit", "run",
                str(dashboard_script),
                "--server.port", "8501",
                "--server.address", "0.0.0.0",
                "--server.headless", "true"
            ]
            
            # Start dashboard process
            subprocess.Popen(cmd, cwd=str(Path(__file__).parent))
            
            logger.info("🖥️ Dashboard starting at http://localhost:8501")
            
        except Exception as e:
            logger.error(f"Failed to start dashboard: {e}")
    
    def _print_system_info(self):
        """Print system information and URLs"""
        status = mlflow_system.get_system_status()
        
        logger.info("=" * 60)
        logger.info("🧪 SutazAI MLflow System - READY")
        logger.info("=" * 60)
        logger.info(f"📊 MLflow Tracking UI: http://localhost:5000")
        logger.info(f"🖥️ System Dashboard: http://localhost:8501")
        logger.info(f"📈 Prometheus Metrics: http://localhost:8080/metrics")
        logger.info(f"🤖 Agents Tracked: {status.get('components', {}).get('agent_tracking', {}).get('total_agents', 0)}")
        logger.info(f"⚙️ Pipelines Available: {len(status.get('components', {}).get('pipelines', []))}")
        logger.info("=" * 60)
    
    async def _shutdown(self):
        """Shutdown the system gracefully"""
        if self.system_running:
            logger.info("🛑 Shutting down SutazAI MLflow System...")
            
            try:
                await shutdown_system()
                logger.info("✅ System shutdown complete")
            except Exception as e:
                logger.error(f"❌ Shutdown error: {e}")
            
            self.system_running = False


@click.group()
def cli():
    """SutazAI MLflow System - Comprehensive ML Experiment Tracking"""
    pass


@cli.command()
@click.option('--config', '-c', help='Path to configuration file')
@click.option('--dashboard/--no-dashboard', default=True, help='Start web dashboard')
@click.option('--debug', is_flag=True, help='Enable debug logging')
def start(config: Optional[str], dashboard: bool, debug: bool):
    """Start the complete MLflow system"""
    
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    launcher = MLflowSystemLauncher()
    
    try:
        asyncio.run(launcher.start_system(dashboard=dashboard, config_file=config))
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")
    except Exception as e:
        logger.error(f"System error: {e}")
        sys.exit(1)


@cli.command()
def status():
    """Check system status"""
    try:
        status = mlflow_system.get_system_status()
        
        print("🧪 SutazAI MLflow System Status")
        print("=" * 40)
        print(f"Running: {'✅ Yes' if status['system']['running'] else '❌ No'}")
        print(f"Initialized: {'✅ Yes' if status['system']['initialized'] else '❌ No'}")
        print(f"Version: {status['system'].get('version', 'Unknown')}")
        print()
        print(f"📊 Statistics:")
        stats = status.get('statistics', {})
        print(f"  Experiments Created: {stats.get('total_experiments_created', 0)}")
        print(f"  Runs Executed: {stats.get('total_runs_executed', 0)}")
        print(f"  Models Registered: {stats.get('total_models_registered', 0)}")
        print()
        print(f"🤖 Agent Tracking:")
        agent_stats = status.get('components', {}).get('agent_tracking', {})
        print(f"  Total Agents: {agent_stats.get('total_agents', 0)}")
        print(f"  Tracking Enabled: {'✅ Yes' if agent_stats.get('tracking_enabled', False) else '❌ No'}")
        
    except Exception as e:
        print(f"❌ Failed to get status: {e}")
        sys.exit(1)


@cli.command()
def dashboard():
    """Start only the web dashboard"""
    try:
        logger.info("🖥️ Starting MLflow Dashboard...")
        run_dashboard()
    except KeyboardInterrupt:
        logger.info("Dashboard shutdown requested")
    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        sys.exit(1)


@cli.command()
@click.argument('agent_id')
@click.option('--agent-type', help='Type of agent')
def track_agent(agent_id: str, agent_type: Optional[str]):
    """Start tracking for a specific agent"""
    async def _track():
        try:
            from mlflow_system import start_agent_tracking
            
            tracker = await start_agent_tracking(agent_id, agent_type)
            
            if tracker:
                print(f"✅ Started tracking for agent {agent_id}")
                print(f"Experiment ID: {tracker.experiment_id}")
            else:
                print(f"❌ Failed to start tracking for agent {agent_id}")
                sys.exit(1)
                
        except Exception as e:
            print(f"❌ Tracking error: {e}")
            sys.exit(1)
    
    asyncio.run(_track())


@cli.command()
@click.argument('experiment_ids', nargs=-1, required=True)
@click.option('--metrics', help='Comma-separated list of metrics to compare')
def compare(experiment_ids: tuple, metrics: Optional[str]):
    """Compare multiple experiments"""
    async def _compare():
        try:
            from mlflow_system import compare_experiments_simple
            
            exp_ids = list(experiment_ids)
            metrics_list = metrics.split(',') if metrics else None
            
            result = await compare_experiments_simple(exp_ids, metrics_list)
            
            if result:
                print("✅ Experiment comparison completed")
                print(f"Total runs analyzed: {result['total_runs']}")
                print(f"Experiments compared: {len(result['experiment_ids'])}")
                
                if result['recommendations']:
                    print("\n💡 Recommendations:")
                    for rec in result['recommendations']:
                        print(f"  • {rec}")
            else:
                print("❌ Comparison failed")
                sys.exit(1)
                
        except Exception as e:
            print(f"❌ Comparison error: {e}")
            sys.exit(1)
    
    asyncio.run(_compare())


@cli.command()
def config():
    """Show current configuration"""
    try:
        config = config_manager.config
        
        print("🔧 SutazAI MLflow Configuration")
        print("=" * 40)
        print(f"Tracking URI: {config.tracking_uri}")
        print(f"Artifact Root: {config.artifact_root}")
        print(f"Agent Tracking: {'✅ Enabled' if config.agent_tracking_enabled else '❌ Disabled'}")
        print(f"Auto Log Models: {'✅ Yes' if config.auto_log_models else '❌ No'}")
        print(f"Max Concurrent: {config.max_concurrent_experiments}")
        print(f"Batch Size: {config.batch_logging_size}")
        print(f"Compression: {'✅ Enabled' if config.enable_compression else '❌ Disabled'}")
        print(f"Prometheus: {'✅ Enabled' if config.enable_prometheus_metrics else '❌ Disabled'}")
        
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    cli()