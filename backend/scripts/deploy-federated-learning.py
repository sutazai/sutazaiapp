#!/usr/bin/env python3
"""
SutazAI Federated Learning Deployment Script
============================================

Deploys federated learning capabilities across the SutazAI distributed system.
Optimized for CPU-only environments with 12-core constraint.

Usage:
    python deploy-federated-learning.py [--config CONFIG_FILE] [--dashboard-only]

Features:
- Complete system deployment and configuration
- Agent capability enhancement
- Privacy-preserving distributed training
- Real-time monitoring and analytics
- Web-based dashboard interface
"""

import asyncio
import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from federated_learning.integration import (
    deploy_federated_learning_system, 
    FederatedSystemConfig,
    FederatedSystemIntegrator
)


def setup_logging(level: str = "INFO"):
    """Setup logging configuration"""
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('/opt/sutazaiapp/backend/logs/federated_deployment.log')
        ]
    )


def load_config(config_file: str = None) -> FederatedSystemConfig:
    """Load system configuration from file or use defaults"""
    config = FederatedSystemConfig()
    
    if config_file and Path(config_file).exists():
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            # Update config with loaded values
            for key, value in config_data.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            
            print(f"✅ Loaded configuration from {config_file}")
            
        except Exception as e:
            print(f"⚠️  Failed to load config file: {e}")
            print("Using default configuration")
    
    return config


def print_banner():
    """Print deployment banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║              SutazAI Federated Learning System              ║
    ║                     Deployment Script                       ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  🤖 Distributed AI Training Across 69 Agents               ║
    ║  🔒 Privacy-Preserving Machine Learning                     ║
    ║  ⚡ CPU-Optimized for 12-Core Constraint                   ║
    ║  📊 Real-time Monitoring & Analytics                        ║
    ║  🌐 Web-based Dashboard Interface                           ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_system_requirements():
    """Print system requirements"""
    requirements = """
    📋 System Requirements:
    ├── CPU: 12 cores (CPU-only constraint)
    ├── Memory: 8GB+ RAM recommended
    ├── Storage: 10GB+ free space
    ├── Network: Redis server accessible
    ├── Python: 3.8+ with required packages
    └── Agents: 3+ SutazAI agents with learning capability
    """
    print(requirements)


def print_deployment_summary(integrator: FederatedSystemIntegrator, config: FederatedSystemConfig):
    """Print deployment summary"""
    status = integrator.get_system_status()
    
    summary = f"""
    ✅ Deployment Summary:
    ┌─────────────────────────────────────────────────────────────┐
    │ System Status:                                              │
    │   • Initialized: {'✅' if status['initialized'] else '❌'}                                │
    │   • Components Started: {'✅' if status['components_started'] else '❌'}                    │
    │   • Enhanced Agents: {status['enhanced_agents']:>2} agents                          │
    │                                                             │
    │ Core Components:                                            │
    │   • Coordinator: {status['coordinator_status']:>8}                          │
    │   • Monitor: {status['monitor_status']:>11}                             │
    │   • Version Manager: {status['version_manager_status']:>7}                       │
    │   • Dashboard: {status['dashboard_status']:>9}                            │
    │                                                             │
    │ Access Points:                                              │
    │   • Dashboard: http://{config.dashboard_host}:{config.dashboard_port}                       │
    │   • Redis: {config.redis_url:>15}                             │
    │   • Storage: {config.model_storage_path:>10}...                         │
    └─────────────────────────────────────────────────────────────┘
    """
    print(summary)


def print_next_steps():
    """Print next steps for users"""
    next_steps = """
    🚀 Next Steps:
    
    1. Access the Dashboard:
       Open http://localhost:8080 in your browser
    
    2. Start a Training Session:
       • Click "Start New Training"
       • Configure training parameters
       • Select federated algorithm (FedAvg, FedProx, FedOpt)
       • Set privacy level and target accuracy
    
    3. Monitor Progress:
       • View real-time training metrics
       • Monitor client performance
       • Track privacy budget consumption
       • Receive alerts for anomalies
    
    4. Manage Models:
       • View model version history
       • Rollback to previous versions
       • Compare model performance
    
    5. System Administration:
       • Monitor system health
       • Configure privacy settings
       • Manage client selection strategies
    
    📚 Documentation:
       - API Reference: /api/docs (when dashboard is running)
       - System Logs: /opt/sutazaiapp/backend/logs/
       - Model Storage: /opt/sutazaiapp/backend/federated_models/
    
    🆘 Support:
       - Check logs for troubleshooting
       - Monitor system health in dashboard
       - Review deployment report for configuration details
    """
    print(next_steps)


async def run_system_diagnostics(integrator: FederatedSystemIntegrator) -> bool:
    """Run system diagnostics to verify deployment"""
    logger = logging.getLogger("diagnostics")
    
    print("\n🔧 Running System Diagnostics...")
    
    diagnostics = {
        "coordinator_responsive": False,
        "monitor_collecting_metrics": False,
        "version_manager_ready": False,
        "agents_connected": False,
        "redis_accessible": False,
        "storage_writable": False
    }
    
    try:
        # Test coordinator
        if integrator.coordinator:
            stats = integrator.coordinator.get_coordinator_stats()
            diagnostics["coordinator_responsive"] = stats is not None
        
        # Test monitor
        if integrator.monitor:
            health = integrator.monitor.get_system_health()
            diagnostics["monitor_collecting_metrics"] = health is not None
        
        # Test version manager
        if integrator.version_manager:
            version_stats = integrator.version_manager.get_version_stats()
            diagnostics["version_manager_ready"] = version_stats is not None
        
        # Test agent connectivity
        enhanced_agents = integrator.get_enhanced_agents()
        diagnostics["agents_connected"] = len(enhanced_agents) >= 3
        
        # Test Redis connectivity
        try:
            import aioredis
            redis = aioredis.from_url(integrator.config.redis_url)
            await redis.ping()
            await redis.close()
            diagnostics["redis_accessible"] = True
        except:
            pass
        
        # Test storage accessibility
        storage_path = Path(integrator.config.model_storage_path)
        try:
            storage_path.mkdir(parents=True, exist_ok=True)
            test_file = storage_path / "test_write.txt"
            test_file.write_text("test")
            test_file.unlink()
            diagnostics["storage_writable"] = True
        except:
            pass
        
        # Print results
        print("\n📊 Diagnostic Results:")
        for test, result in diagnostics.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"   • {test.replace('_', ' ').title()}: {status}")
        
        # Overall health
        passed_tests = sum(diagnostics.values())
        total_tests = len(diagnostics)
        health_percentage = (passed_tests / total_tests) * 100
        
        print(f"\n🏥 Overall System Health: {health_percentage:.1f}% ({passed_tests}/{total_tests} tests passed)")
        
        if health_percentage >= 80:
            print("✅ System is ready for federated learning!")
            return True
        else:
            print("⚠️  System has issues that may affect performance")
            return False
    
    except Exception as e:
        logger.error(f"Diagnostics failed: {e}")
        return False


async def deploy_system(config: FederatedSystemConfig, dashboard_only: bool = False) -> FederatedSystemIntegrator:
    """Deploy the federated learning system"""
    logger = logging.getLogger("deployment")
    
    try:
        if dashboard_only:
            print("🌐 Starting dashboard-only mode...")
            # In dashboard-only mode, assume components are already running
            from federated_learning.coordinator import get_federated_coordinator
            from federated_learning.monitoring import FederatedMonitor
            from federated_learning.versioning import ModelVersionManager
            from federated_learning.dashboard import FederatedDashboard
            
            coordinator = get_federated_coordinator()
            monitor = FederatedMonitor(config.redis_url)
            version_manager = ModelVersionManager()
            
            if coordinator:
                await monitor.initialize()
                await version_manager.initialize()
            
            dashboard = FederatedDashboard(coordinator, monitor, version_manager, config.dashboard_host, config.dashboard_port)
            
            # Create minimal integrator for dashboard
            integrator = FederatedSystemIntegrator(config)
            integrator.coordinator = coordinator
            integrator.monitor = monitor
            integrator.version_manager = version_manager
            integrator.dashboard = dashboard
            integrator.system_initialized = True
            integrator.components_started = True
            
            return integrator
        
        else:
            print("🚀 Starting full system deployment...")
            
            # Deploy complete system
            integrator = await deploy_federated_learning_system(config)
            
            # Run diagnostics
            await run_system_diagnostics(integrator)
            
            return integrator
    
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        raise


async def main():
    """Main deployment function"""
    parser = argparse.ArgumentParser(
        description="Deploy SutazAI Federated Learning System",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        help="Configuration file path"
    )
    
    parser.add_argument(
        "--dashboard-only", 
        action="store_true",
        help="Start only the dashboard (assume other components are running)"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    parser.add_argument(
        "--no-banner",
        action="store_true",
        help="Skip banner and requirements display"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger("main")
    
    try:
        if not args.no_banner:
            print_banner()
            print_system_requirements()
        
        # Load configuration
        config = load_config(args.config)
        
        # Deploy system
        print("🔄 Initializing deployment...")
        start_time = time.time()
        
        integrator = await deploy_system(config, args.dashboard_only)
        
        deployment_time = time.time() - start_time
        print(f"⏱️  Deployment completed in {deployment_time:.2f} seconds")
        
        # Print summary
        print_deployment_summary(integrator, config)
        print_next_steps()
        
        # Start dashboard
        print("🌐 Starting web dashboard...")
        print(f"Dashboard will be available at: http://{config.dashboard_host}:{config.dashboard_port}")
        print("\nPress Ctrl+C to shutdown the system")
        
        await integrator.start_dashboard()
        
    except KeyboardInterrupt:
        print("\n🛑 Shutdown requested by user")
        if 'integrator' in locals():
            await integrator.shutdown()
        print("✅ System shutdown complete")
    
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        print(f"\n❌ Deployment Error: {e}")
        print("\n🔍 Troubleshooting Tips:")
        print("   • Check Redis server is running")
        print("   • Verify Python dependencies are installed")
        print("   • Ensure sufficient system resources")
        print("   • Check logs in /opt/sutazaiapp/backend/logs/")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())