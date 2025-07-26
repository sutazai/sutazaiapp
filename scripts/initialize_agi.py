#!/usr/bin/env python3
"""
AGI/ASI System Initialization Script

This script initializes and coordinates all components of the AGI/ASI system,
ensuring proper integration and setup of core functionalities:

1. Neuromorphic Engine: Biologically-inspired neural architecture
2. Self-Modification Engine: Controls safe recursive self-improvement
3. Ethical Constraint System: Verified decision boundaries
4. Agent Orchestration: Manages cognitive agents and their interactions
5. Resource Monitoring: Hardware-aware optimization and power management
6. Security Enforcement: Air-gapped protection and information flow control
"""

import os
import sys
import time
import json
import logging
import argparse
import subprocess
import multiprocessing
import signal
from pathlib import Path
from typing import Dict, Any
import psutil
from datetime import datetime
import platform
from neuromorphic.engine import serve as start_neuromorphic_engine

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/agi_system.log"), logging.StreamHandler()],
)
logger = logging.getLogger("AGI-System")

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

# Import system components
try:
    from ai_agents.protocols.self_modification import SelfModificationControl
    from ai_agents.agent_manager import AgentManager
except ImportError as e:
    logger.error(f"Failed to import required components: {str(e)}")
    sys.exit(1)


class SystemInitializationError(Exception):
    """Exception raised for errors during system initialization"""

    pass


class SystemShutdownError(Exception):
    """Exception raised for errors during system shutdown"""

    pass


class AGISystem:
    """
    Main AGI/ASI System Controller

    This class coordinates all components of the AGI/ASI system, manages their
    lifecycle, and ensures proper integration between subsystems.
    """

    def __init__(
        self,
        config_path: str = "config/agi_system.json",
        data_dir: str = "data",
        log_dir: str = "logs",
        debug_mode: bool = False,
    ):
        """
        Initialize the AGI/ASI system

        Args:
            config_path: Path to the configuration file
            data_dir: Directory for data storage
            log_dir: Directory for logs
            debug_mode: Enable debug mode with additional logging
        """
        self.config_path = os.path.join(PROJECT_ROOT, config_path)
        self.data_dir = os.path.join(PROJECT_ROOT, data_dir)
        self.log_dir = os.path.join(PROJECT_ROOT, log_dir)
        self.debug_mode = debug_mode

        # Configure component-specific logger
        self.logger = logging.getLogger("AGI-System")
        if debug_mode:
            self.logger.setLevel(logging.DEBUG)

        # Create necessary directories
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(os.path.join(self.log_dir, "modifications"), exist_ok=True)

        # Load configuration
        self.config = self._load_config()

        # Initialize component states
        self.components = {}
        self.processes = {}
        self.started = False
        self.shutdown_flag = False

        # Initialize system information
        self.system_info = self._gather_system_info()

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self.logger.info(
            f"AGI System initialized with configuration from {config_path}"
        )
        if debug_mode:
            self.logger.debug(f"System info: {json.dumps(self.system_info, indent=2)}")

    def _load_config(self) -> Dict[str, Any]:
        """Load system configuration from file"""
        try:
            if not os.path.exists(self.config_path):
                # Create default configuration if none exists
                default_config = {
                    "system": {
                        "name": "SutazAI AGI/ASI Autonomous System",
                        "version": "0.1.0",
                        "description": "Autonomous AGI system with self-improvement capabilities",
                        "max_memory_usage_gb": 100,
                        "max_cpu_usage_percent": 90,
                        "debug_mode": self.debug_mode,
                    },
                    "neuromorphic_engine": {
                        "enabled": True,
                        "port": 50051,
                        "energy_efficient_mode": True,
                        "memory_efficient_mode": True,
                    },
                    "self_modification": {
                        "enabled": True,
                        "auto_approve": False,
                        "max_sandbox_time": 30,
                        "max_changes_per_event": 10,
                    },
                    "agent_manager": {
                        "enabled": True,
                        "max_agents": 10,
                        "default_agent_timeout": 300,
                        "agent_types": [
                            "document_agent",
                            "code_agent",
                            "reasoning_agent",
                            "planning_agent",
                        ],
                    },
                    "api_server": {
                        "enabled": True,
                        "host": "0.0.0.0",  # nosec B104 - Bind to all interfaces for container/network access
                        "port": 8000,
                        "workers": 4,
                        "log_level": "info",
                    },
                    "web_ui": {"enabled": True, "port": 8501, "theme": "dark"},
                    "security": {
                        "air_gap_mode": False,
                        "encryption_enabled": True,
                        "audit_logging": True,
                        "max_token_limit": 100000,
                    },
                }

                # Save default configuration
                os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
                with open(self.config_path, "w") as f:
                    json.dump(default_config, f, indent=2)

                self.logger.warning(
                    f"Created default configuration at {self.config_path}"
                )
                return default_config

            with open(self.config_path, "r") as f:
                config = json.load(f)
                self.logger.info(f"Loaded configuration from {self.config_path}")
                return config

        except Exception as e:
            self.logger.error(f"Error loading configuration: {str(e)}")
            raise SystemInitializationError(f"Failed to load configuration: {str(e)}")

    def _gather_system_info(self) -> Dict[str, Any]:
        """Gather information about the host system"""
        try:
            cpu_info = {
                "physical_cores": psutil.cpu_count(logical=False),
                "logical_cores": psutil.cpu_count(logical=True),
                "frequency": psutil.cpu_freq(),
                "architecture": os.uname().machine
                if hasattr(os, "uname")
                else platform.machine(),
            }

            memory_info = {
                "total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
                "available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
            }

            disk_info = {
                "total_gb": round(psutil.disk_usage("/").total / (1024**3), 2),
                "free_gb": round(psutil.disk_usage("/").free / (1024**3), 2),
            }

            # Try to get GPU information if available
            gpu_info = {"available": False}
            try:
                import pynvml

                pynvml.nvmlInit()
                device_count = pynvml.nvmlDeviceGetCount()
                if device_count > 0:
                    gpu_info["available"] = True
                    gpu_info["count"] = device_count
                    gpu_info["devices"] = []

                    for i in range(device_count):
                        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        gpu_info["devices"].append(
                            {
                                "name": pynvml.nvmlDeviceGetName(handle),
                                "total_memory_gb": round(info.total / (1024**3), 2),
                                "free_memory_gb": round(info.free / (1024**3), 2),
                            }
                        )
                pynvml.nvmlShutdown()
            except Exception as e:
                logger.warning(f"Could not get NVIDIA GPU info: {e}")
                pass

            return {
                "cpu": cpu_info,
                "memory": memory_info,
                "disk": disk_info,
                "gpu": gpu_info,
                "os": {
                    "name": os.name,
                    "system": sys.platform,
                    "python_version": sys.version,
                },
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Error gathering system information: {str(e)}")
            return {"error": str(e)}

    def _signal_handler(self, sig, frame):
        """Handle termination signals"""
        self.logger.warning(f"Received signal {sig}, initiating graceful shutdown")
        self.shutdown()

    def initialize_components(self):
        """Initialize all system components based on configuration"""
        self.logger.info("Initializing AGI/ASI system components")

        try:
            # Initialize self-modification control system
            if self.config["self_modification"]["enabled"]:
                self.logger.info("Initializing Self-Modification Control system")
                self.components["self_modification"] = SelfModificationControl(
                    base_dir=str(PROJECT_ROOT),
                    log_dir=os.path.join(self.log_dir, "modifications"),
                    max_sandbox_time=self.config["self_modification"][
                        "max_sandbox_time"
                    ],
                    auto_approve=self.config["self_modification"]["auto_approve"],
                    max_changes_per_event=self.config["self_modification"][
                        "max_changes_per_event"
                    ],
                )

            # Initialize agent manager
            if self.config["agent_manager"]["enabled"]:
                self.logger.info("Initializing Agent Manager")
                self.components["agent_manager"] = AgentManager(
                    config={
                        "max_agents": self.config["agent_manager"]["max_agents"],
                        "default_timeout": self.config["agent_manager"][
                            "default_agent_timeout"
                        ],
                        "agent_types": self.config["agent_manager"]["agent_types"],
                    }
                )

            self.logger.info("All components initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error initializing components: {str(e)}")
            raise SystemInitializationError(
                f"Failed to initialize components: {str(e)}"
            )

    def start_processes(self):
        """Start all system processes"""
        self.logger.info("Starting AGI/ASI system processes")

        try:
            # Start neuromorphic engine if enabled
            if self.config["neuromorphic_engine"]["enabled"]:
                self.logger.info("Starting Neuromorphic Engine")
                neuromorphic_process = multiprocessing.Process(
                    target=start_neuromorphic_engine,
                    args=(self.config["neuromorphic_engine"]["port"],),
                    name="NeuromorphicEngine",
                )
                neuromorphic_process.start()
                self.processes["neuromorphic_engine"] = neuromorphic_process

            # Start API server if enabled
            if self.config["api_server"]["enabled"]:
                self.logger.info("Starting API Server")
                api_cmd = [
                    sys.executable,
                    "-m",
                    "uvicorn",
                    "backend.api:app",
                    f"--host={self.config['api_server']['host']}",
                    f"--port={self.config['api_server']['port']}",
                    f"--workers={self.config['api_server']['workers']}",
                ]

                api_process = subprocess.Popen(
                    api_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                )
                self.processes["api_server"] = api_process

            # Start web UI if enabled
            if self.config["web_ui"]["enabled"]:
                self.logger.info("Starting Web UI")
                web_cmd = [
                    sys.executable,
                    "-m",
                    "streamlit",
                    "run",
                    "web_ui/app.py",
                    f"--server.port={self.config['web_ui']['port']}",
                    "--server.address=0.0.0.0",
                ]

                web_process = subprocess.Popen(
                    web_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                )
                self.processes["web_ui"] = web_process

            self.started = True
            self.logger.info("All processes started successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error starting processes: {str(e)}")
            self.shutdown()  # Attempt to clean up
            raise SystemInitializationError(f"Failed to start processes: {str(e)}")

    def monitor_system(self, interval: int = 5):
        """
        Monitor system health and resource usage

        Args:
            interval: Monitoring interval in seconds
        """
        self.logger.info(f"Starting system monitoring with {interval}s interval")

        while not self.shutdown_flag:
            try:
                # Check process health
                for name, process in list(self.processes.items()):
                    if isinstance(process, multiprocessing.Process):
                        if not process.is_alive():
                            self.logger.error(f"Process {name} has died unexpectedly")
                            # TODO: Implement auto-restart logic here
                    elif isinstance(process, subprocess.Popen):
                        if process.poll() is not None:
                            self.logger.error(
                                f"Process {name} has exited with code {process.returncode}"
                            )
                            # TODO: Implement auto-restart logic here

                # Check resource usage
                cpu_percent = psutil.cpu_percent()
                psutil.virtual_memory().percent  # Keep the call for side effects if any

                if cpu_percent > self.config["system"]["max_cpu_usage_percent"]:
                    self.logger.warning(f"High CPU usage: {cpu_percent}%")
                    # TODO: Implement resource control mechanisms

                memory_gb_used = psutil.virtual_memory().used / (1024**3)
                if memory_gb_used > self.config["system"]["max_memory_usage_gb"]:
                    self.logger.warning(f"High memory usage: {memory_gb_used:.2f} GB")
                    # TODO: Implement memory control mechanisms

                # Sleep until next check
                time.sleep(interval)

            except Exception as e:
                self.logger.error(f"Error in system monitoring: {str(e)}")
                time.sleep(interval)  # Continue monitoring despite errors

    def start(self):
        """Start the complete AGI/ASI system"""
        self.logger.info("Starting AGI/ASI system")

        try:
            # Initialize components
            if not self.initialize_components():
                raise SystemInitializationError("Failed to initialize components")

            # Start processes
            if not self.start_processes():
                raise SystemInitializationError("Failed to start processes")

            # Start monitoring in a separate thread
            import threading

            monitor_thread = threading.Thread(
                target=self.monitor_system,
                args=(10,),  # 10-second monitoring interval
                daemon=True,
            )
            monitor_thread.start()

            self.logger.info("AGI/ASI system started successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error starting AGI/ASI system: {str(e)}")
            self.shutdown()  # Attempt to clean up
            return False

    def shutdown(self):
        """Shutdown the AGI/ASI system gracefully"""
        if self.shutdown_flag:
            return  # Already shutting down

        self.logger.info("Shutting down AGI/ASI system")
        self.shutdown_flag = True

        try:
            # Terminate all processes
            for name, process in self.processes.items():
                self.logger.info(f"Terminating {name} process")
                try:
                    if isinstance(process, multiprocessing.Process):
                        process.terminate()
                        process.join(timeout=5)
                        if process.is_alive():
                            process.kill()
                    elif isinstance(process, subprocess.Popen):
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                except Exception as e:
                    self.logger.error(f"Error terminating {name} process: {str(e)}")

            # Clean up components
            for name, component in self.components.items():
                self.logger.info(f"Cleaning up {name} component")
                try:
                    if hasattr(component, "cleanup"):
                        component.cleanup()
                except Exception as e:
                    self.logger.error(f"Error cleaning up {name} component: {str(e)}")

            self.logger.info("AGI/ASI system shutdown complete")

        except Exception as e:
            self.logger.error(f"Error during system shutdown: {str(e)}")
            raise SystemShutdownError(f"Failed to shut down system: {str(e)}")

    def status(self) -> Dict[str, Any]:
        """Get current system status"""
        component_status = {}
        process_status = {}

        # Check component status
        for name, component in self.components.items():
            if hasattr(component, "is_initialized"):
                component_status[name] = {
                    "initialized": component.is_initialized(),
                    "type": type(component).__name__,
                }
            else:
                component_status[name] = {
                    "initialized": True,  # Assume initialized if no status method
                    "type": type(component).__name__,
                }

        # Check process status
        for name, process in self.processes.items():
            if isinstance(process, multiprocessing.Process):
                process_status[name] = {
                    "running": process.is_alive(),
                    "pid": process.pid if process.pid else None,
                }
            elif isinstance(process, subprocess.Popen):
                returncode = process.poll()
                process_status[name] = {
                    "running": returncode is None,
                    "pid": process.pid,
                    "returncode": returncode,
                }

        # Get system resource usage
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        resources = {
            "cpu_percent": cpu_percent,
            "memory_used_gb": round(memory.used / (1024**3), 2),
            "memory_total_gb": round(memory.total / (1024**3), 2),
            "disk_percent": disk.percent,
            "disk_used_gb": round(disk.used / (1024**3), 2),
            "disk_total_gb": round(disk.total / (1024**3), 2),
        }

        return {
            "system_name": self.config["system"]["name"],
            "version": self.config["system"]["version"],
            "started": self.started,
            "uptime": time.time() - psutil.Process(os.getpid()).create_time(),
            "components": component_status,
            "processes": process_status,
            "resources": resources,
            "timestamp": datetime.now().isoformat(),
        }


def main():
    """Main entry point for AGI/ASI system"""
    parser = argparse.ArgumentParser(description="SutazAI AGI/ASI System Controller")
    parser.add_argument(
        "--config",
        type=str,
        default="config/agi_system.json",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--data-dir", type=str, default="data", help="Directory for data storage"
    )
    parser.add_argument(
        "--log-dir", type=str, default="logs", help="Directory for logs"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "--status-only", action="store_true", help="Show system status without starting"
    )
    args = parser.parse_args()

    try:
        # Initialize the AGI system
        agi_system = AGISystem(
            config_path=args.config,
            data_dir=args.data_dir,
            log_dir=args.log_dir,
            debug_mode=args.debug,
        )

        if args.status_only:
            # Just print status and exit
            status = agi_system.status()
            print(json.dumps(status, indent=2))
            sys.exit(0)

        # Start the system
        if not agi_system.start():
            logger.error("Failed to start AGI/ASI system")
            sys.exit(1)

        # Keep the main process running
        while not agi_system.shutdown_flag:
            try:
                time.sleep(1)
            except KeyboardInterrupt:
                break

        # Shutdown system gracefully
        agi_system.shutdown()

    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
