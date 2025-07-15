#!/usr/bin/env python3
"""
SutazAI Enterprise System Launcher
Production-ready launch script with comprehensive system initialization
"""

import os
import sys
import json
import time
import signal
import logging
import asyncio
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
import subprocess
import threading

# Add current directory to Python path
sys.path.insert(0, '/opt/sutazaiapp')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/opt/sutazaiapp/logs/enterprise_launch.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnterpriseSystemLauncher:
    """Enterprise system launcher with comprehensive initialization"""
    
    def __init__(self):
        self.base_dir = Path("/opt/sutazaiapp")
        self.running = False
        self.components = {}
        self.start_time = None
        
    def print_banner(self):
        """Print system banner"""
        banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                         SutazAI Enterprise AGI/ASI System                   ║
║                                                                              ║
║                     Advanced Self-Improving Intelligence                     ║
║                                                                              ║
║  🧠 Neural Link Networks    🔐 Enterprise Security    🚀 Auto-Scaling      ║
║  🤖 Local Model Management  📊 Real-time Monitoring   🎯 100% Offline       ║
║                                                                              ║
║                              v1.0.0 - Production Ready                      ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
        """
        print(banner)
        print(f"🚀 Starting SutazAI Enterprise System at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📍 Installation Directory: {self.base_dir}")
        print(f"👤 Authorized User: chrissuta01@gmail.com")
        print()
    
    def check_prerequisites(self):
        """Check system prerequisites"""
        logger.info("Checking system prerequisites...")
        
        # Check Python version
        if sys.version_info < (3, 8):
            logger.error("Python 3.8+ required")
            sys.exit(1)
        
        # Check required directories
        required_dirs = ['core', 'api', 'models', 'monitoring', 'nln', 'data', 'logs']
        for dir_name in required_dirs:
            dir_path = self.base_dir / dir_name
            if not dir_path.exists():
                logger.error(f"Required directory missing: {dir_name}")
                sys.exit(1)
        
        # Check disk space
        import shutil
        free_space = shutil.disk_usage(str(self.base_dir)).free / (1024**3)
        if free_space < 1.0:
            logger.error(f"Insufficient disk space: {free_space:.2f}GB available")
            sys.exit(1)
        
        logger.info("✅ Prerequisites check passed")
    
    def initialize_system_components(self):
        """Initialize all system components"""
        logger.info("Initializing system components...")
        
        try:
            # Initialize AGI System
            logger.info("🧠 Initializing AGI System...")
            from core.agi_system import IntegratedAGISystem
            self.components['agi_system'] = IntegratedAGISystem()
            logger.info("✅ AGI System initialized")
            
            # Initialize Security Manager
            logger.info("🔐 Initializing Security Manager...")
            from core.security import SecurityManager
            self.components['security'] = SecurityManager()
            logger.info("✅ Security Manager initialized")
            
            # Initialize Model Manager
            logger.info("🤖 Initializing Model Manager...")
            from models.local_model_manager_simple import LocalModelManager
            self.components['models'] = LocalModelManager()
            logger.info("✅ Model Manager initialized")
            
            # Initialize Monitoring System
            logger.info("📊 Initializing Monitoring System...")
            from monitoring.observability_simple import ObservabilitySystem
            self.components['monitoring'] = ObservabilitySystem()
            self.components['monitoring'].start_monitoring()
            logger.info("✅ Monitoring System initialized")
            
            # Initialize API System
            logger.info("🌐 Initializing API System...")
            from api.agi_api_simple import AGIAPISystem
            self.components['api'] = AGIAPISystem()
            logger.info("✅ API System initialized")
            
            logger.info("🎉 All system components initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Component initialization failed: {e}")
            sys.exit(1)
    
    def start_background_services(self):
        """Start background services"""
        logger.info("Starting background services...")
        
        # Start monitoring thread
        monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        monitoring_thread.start()
        
        # Start health check thread
        health_thread = threading.Thread(
            target=self._health_check_loop,
            daemon=True
        )
        health_thread.start()
        
        # Start performance tracking thread
        performance_thread = threading.Thread(
            target=self._performance_tracking_loop,
            daemon=True
        )
        performance_thread.start()
        
        logger.info("✅ Background services started")
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.running:
            try:
                # Collect system metrics
                metrics = self.components['monitoring'].collect_metrics()
                
                # Log system status every 5 minutes
                if int(time.time()) % 300 == 0:
                    logger.info(f"📊 System Status - CPU: {metrics['cpu_usage']:.1f}%, Memory: {metrics['memory_usage']:.1f}%, Uptime: {metrics['uptime']:.1f}s")
                
                time.sleep(30)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(60)
    
    def _health_check_loop(self):
        """Background health check loop"""
        while self.running:
            try:
                # Check AGI system health
                if 'agi_system' in self.components:
                    status = self.components['agi_system'].get_system_status()
                    if status['metrics']['system_health'] != 'healthy':
                        logger.warning(f"⚠️ AGI System health: {status['metrics']['system_health']}")
                
                time.sleep(60)
                
            except Exception as e:
                logger.error(f"Health check loop error: {e}")
                time.sleep(120)
    
    def _performance_tracking_loop(self):
        """Background performance tracking loop"""
        while self.running:
            try:
                # Track performance metrics
                if 'agi_system' in self.components:
                    status = self.components['agi_system'].get_system_status()
                    
                    # Log performance issues
                    if status['metrics']['cpu_usage'] > 80:
                        logger.warning(f"⚠️ High CPU usage: {status['metrics']['cpu_usage']:.1f}%")
                    
                    if status['metrics']['memory_usage'] > 85:
                        logger.warning(f"⚠️ High memory usage: {status['metrics']['memory_usage']:.1f}%")
                
                time.sleep(120)
                
            except Exception as e:
                logger.error(f"Performance tracking loop error: {e}")
                time.sleep(180)
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info(f"📶 Received signal {signum}, initiating graceful shutdown...")
            self.shutdown_system()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        logger.info("✅ Signal handlers configured")
    
    def create_system_status_file(self):
        """Create system status file for monitoring"""
        try:
            status_data = {
                "system_info": {
                    "status": "running",
                    "start_time": self.start_time.isoformat(),
                    "uptime": (datetime.now() - self.start_time).total_seconds(),
                    "components": {
                        "agi_system": "active" if 'agi_system' in self.components else "inactive",
                        "security": "active" if 'security' in self.components else "inactive",
                        "models": "active" if 'models' in self.components else "inactive",
                        "monitoring": "active" if 'monitoring' in self.components else "inactive",
                        "api": "active" if 'api' in self.components else "inactive"
                    }
                },
                "api_endpoints": {
                    "health": "http://localhost:8000/health",
                    "api_docs": "http://localhost:8000/api/docs",
                    "system_status": "http://localhost:8000/orchestrator/status"
                },
                "timestamp": datetime.now().isoformat()
            }
            
            status_file = self.base_dir / "SYSTEM_STATUS.json"
            with open(status_file, 'w') as f:
                json.dump(status_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to create system status file: {e}")
    
    def display_system_info(self):
        """Display system information"""
        print("\n" + "="*80)
        print("🎯 SYSTEM READY - SutazAI Enterprise AGI/ASI System")
        print("="*80)
        
        if 'agi_system' in self.components:
            status = self.components['agi_system'].get_system_status()
            print(f"🧠 AGI System Status: {status['state']}")
            print(f"🔗 Neural Network: {status['neural_network']['total_nodes']} nodes, {status['neural_network']['total_connections']} connections")
            print(f"📊 System Health: {status['metrics']['system_health']}")
        
        if 'models' in self.components:
            models_status = self.components['models'].get_system_status()
            print(f"🤖 Model Manager: {models_status['total_models']} models available")
            print(f"🔄 Ollama Status: {models_status['ollama_status']['status']}")
        
        print(f"🔐 Security: Enterprise-grade with hardcoded authorization")
        print(f"📡 Monitoring: Real-time metrics and alerting active")
        print(f"⏱️  Uptime: {(datetime.now() - self.start_time).total_seconds():.1f} seconds")
        
        print("\n🌐 API Endpoints:")
        print(f"   • Health Check: http://localhost:8000/health")
        print(f"   • API Docs: http://localhost:8000/api/docs")
        print(f"   • System Status: http://localhost:8000/orchestrator/status")
        
        print("\n📝 Logs:")
        print(f"   • System Log: {self.base_dir}/logs/enterprise_launch.log")
        print(f"   • AGI Log: {self.base_dir}/logs/agi_system.log")
        
        print("\n🎯 System Capabilities:")
        print("   • 🧠 Advanced Neural Network Processing")
        print("   • 🔐 Enterprise-grade Security Framework")
        print("   • 🤖 Local Model Management (100% Offline)")
        print("   • 📊 Real-time Monitoring and Alerting")
        print("   • 🚀 Auto-scaling and Load Balancing")
        print("   • 🎯 Self-improving AI Capabilities")
        
        print("\n" + "="*80)
        print("💡 System is ready for production use!")
        print("   Press Ctrl+C to gracefully shutdown the system")
        print("="*80)
    
    def run_system_loop(self):
        """Main system loop"""
        logger.info("Starting main system loop...")
        
        try:
            while self.running:
                # Update system status file
                self.create_system_status_file()
                
                # Check for system health
                if 'agi_system' in self.components:
                    status = self.components['agi_system'].get_system_status()
                    if status['state'] == 'emergency_shutdown':
                        logger.warning("Emergency shutdown detected, stopping system...")
                        break
                
                time.sleep(10)
                
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received, shutting down...")
        except Exception as e:
            logger.error(f"System loop error: {e}")
        finally:
            self.shutdown_system()
    
    def shutdown_system(self):
        """Graceful system shutdown"""
        logger.info("🔄 Initiating graceful system shutdown...")
        
        self.running = False
        
        # Stop monitoring
        if 'monitoring' in self.components:
            try:
                self.components['monitoring'].stop_monitoring()
                logger.info("✅ Monitoring system stopped")
            except Exception as e:
                logger.error(f"Error stopping monitoring: {e}")
        
        # Create shutdown status file
        try:
            shutdown_data = {
                "shutdown_info": {
                    "timestamp": datetime.now().isoformat(),
                    "uptime": (datetime.now() - self.start_time).total_seconds(),
                    "reason": "graceful_shutdown",
                    "status": "completed"
                }
            }
            
            shutdown_file = self.base_dir / "SHUTDOWN_STATUS.json"
            with open(shutdown_file, 'w') as f:
                json.dump(shutdown_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to create shutdown status file: {e}")
        
        logger.info("🎉 System shutdown completed successfully")
        print("\n" + "="*80)
        print("🎯 SutazAI Enterprise System Shutdown Complete")
        print("="*80)
        print("Thank you for using SutazAI Enterprise AGI/ASI System!")
        print("="*80)
    
    def launch(self):
        """Main launch method"""
        try:
            # Print banner
            self.print_banner()
            
            # Initialize system
            self.start_time = datetime.now()
            self.running = True
            
            # Check prerequisites
            self.check_prerequisites()
            
            # Initialize components
            self.initialize_system_components()
            
            # Start background services
            self.start_background_services()
            
            # Setup signal handlers
            self.setup_signal_handlers()
            
            # Display system info
            self.display_system_info()
            
            # Run main loop
            self.run_system_loop()
            
        except Exception as e:
            logger.error(f"❌ System launch failed: {e}")
            sys.exit(1)

def main():
    """Main entry point"""
    launcher = EnterpriseSystemLauncher()
    launcher.launch()

if __name__ == "__main__":
    main()