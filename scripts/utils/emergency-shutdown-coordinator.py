#!/usr/bin/env python3
"""
SutazAI Emergency Shutdown Coordinator
Comprehensive emergency shutdown system with state preservation and graceful degradation.
"""

import os
import sys
import json
import time
import signal
import threading
import subprocess
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
import sqlite3
import psutil
import docker
import redis
from contextlib import contextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/opt/sutazaiapp/logs/emergency-shutdown.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ShutdownTrigger(Enum):
    """Emergency shutdown trigger types"""
    MANUAL = "manual"
    DEADMAN_SWITCH = "deadman_switch"
    SYSTEM_FAILURE = "system_failure"
    SECURITY_BREACH = "security_breach"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    NETWORK_PARTITION = "network_partition"
    DATA_CORRUPTION = "data_corruption"
    EXTERNAL_SIGNAL = "external_signal"

class ShutdownPhase(Enum):
    """Shutdown sequence phases"""
    INITIATED = "initiated"
    STOPPING_SERVICES = "stopping_services"
    DRAINING_QUEUES = "draining_queues"
    PERSISTING_STATE = "persisting_state"
    GRACEFUL_TERMINATION = "graceful_termination"
    FORCED_TERMINATION = "forced_termination"
    CLEANUP = "cleanup"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class ServiceShutdownConfig:
    """Configuration for service shutdown behavior"""
    name: str
    priority: int  # 1=highest, 10=lowest
    graceful_timeout: int  # seconds
    force_timeout: int  # seconds
    pre_shutdown_commands: List[str] = None
    post_shutdown_commands: List[str] = None
    health_check_command: str = None
    dependency_services: List[str] = None
    preserve_data: bool = True
    allow_restart: bool = True
    critical: bool = False

@dataclass
class ShutdownState:
    """Current shutdown state tracking"""
    trigger: ShutdownTrigger
    phase: ShutdownPhase
    started_at: datetime
    estimated_completion: Optional[datetime]
    services_shutdown: List[str]
    services_failed: List[str]
    data_preserved: bool
    error_messages: List[str]
    recovery_possible: bool

class EmergencyShutdownCoordinator:
    """Centralized emergency shutdown coordination"""
    
    def __init__(self, config_path: str = "/opt/sutazaiapp/config/emergency-shutdown.json"):
        self.config_path = config_path
        self.state_db_path = "/opt/sutazaiapp/data/shutdown-state.db"
        self.deadman_switch_file = "/tmp/sutazai-deadman-switch"
        self.shutdown_lock_file = "/tmp/sutazai-shutdown.lock"
        
        # Initialize components
        self._init_database()
        self._load_service_configs()
        self._setup_signal_handlers()
        
        # State tracking
        self.shutdown_active = False
        self.shutdown_state: Optional[ShutdownState] = None
        self.abort_shutdown = threading.Event()
        self.force_shutdown = threading.Event()
        
        # Docker client for container management
        try:
            self.docker_client = docker.from_env()
        except Exception as e:
            logger.error(f"Failed to initialize Docker client: {e}")
            self.docker_client = None
        
        # Redis client for coordination
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
            self.redis_client.ping()
        except Exception as e:
            logger.warning(f"Redis not available for coordination: {e}")
            self.redis_client = None
        
        # Monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_system, daemon=True)
        self.monitor_thread.start()
        
        logger.info("Emergency Shutdown Coordinator initialized")

    def _init_database(self):
        """Initialize shutdown state database"""
        os.makedirs(os.path.dirname(self.state_db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.state_db_path, check_same_thread=False)
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS shutdown_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trigger TEXT NOT NULL,
                phase TEXT NOT NULL,
                started_at DATETIME NOT NULL,
                completed_at DATETIME,
                success BOOLEAN,
                error_message TEXT,
                state_data TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS service_shutdown_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                shutdown_id INTEGER,
                service_name TEXT NOT NULL,
                action TEXT NOT NULL,
                status TEXT NOT NULL,
                duration_seconds REAL,
                error_message TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        self.conn.commit()

    def _load_service_configs(self):
        """Load service shutdown configurations"""
        default_configs = [
            ServiceShutdownConfig(
                name="frontend",
                priority=1,
                graceful_timeout=30,
                force_timeout=60,
                health_check_command="curl -f http://localhost:8501/health || exit 1",
                preserve_data=False,
                allow_restart=True
            ),
            ServiceShutdownConfig(
                name="backend",
                priority=2,
                graceful_timeout=60,
                force_timeout=120,
                pre_shutdown_commands=["pkill -TERM -f 'python.*app.py'"],
                health_check_command="curl -f http://localhost:8000/health || exit 1",
                preserve_data=True,
                critical=True
            ),
            ServiceShutdownConfig(
                name="agent-orchestrator",
                priority=3,
                graceful_timeout=90,
                force_timeout=180,
                dependency_services=["backend"],
                preserve_data=True,
                critical=True
            ),
            ServiceShutdownConfig(
                name="monitoring",
                priority=4,
                graceful_timeout=30,
                force_timeout=60,
                preserve_data=True
            ),
            ServiceShutdownConfig(
                name="databases",
                priority=9,  # Shutdown last
                graceful_timeout=120,
                force_timeout=300,
                pre_shutdown_commands=[
                    "sqlite3 /opt/sutazaiapp/data/*.db '.backup /opt/sutazaiapp/backups/emergency-$(date +%s).db'"
                ],
                preserve_data=True,
                critical=True
            ),
            ServiceShutdownConfig(
                name="ollama",
                priority=8,
                graceful_timeout=60,
                force_timeout=180,
                pre_shutdown_commands=["ollama stop"],
                preserve_data=True
            ),
            ServiceShutdownConfig(
                name="docker-containers",
                priority=10,  # Very last
                graceful_timeout=300,
                force_timeout=600,
                preserve_data=True,
                critical=True
            )
        ]
        
        self.service_configs = {}
        for config in default_configs:
            self.service_configs[config.name] = config

    def _setup_signal_handlers(self):
        """Setup signal handlers for emergency shutdown"""
        def signal_handler(signum, frame):
            trigger_map = {
                signal.SIGTERM: ShutdownTrigger.EXTERNAL_SIGNAL,
                signal.SIGINT: ShutdownTrigger.MANUAL,
                signal.SIGUSR1: ShutdownTrigger.DEADMAN_SWITCH,
                signal.SIGUSR2: ShutdownTrigger.SYSTEM_FAILURE
            }
            
            trigger = trigger_map.get(signum, ShutdownTrigger.EXTERNAL_SIGNAL)
            logger.info(f"Received signal {signum}, initiating emergency shutdown: {trigger}")
            self.initiate_emergency_shutdown(trigger)
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGUSR1, signal_handler)
        signal.signal(signal.SIGUSR2, signal_handler)

    def _monitor_system(self):
        """Monitor system conditions for automatic shutdown triggers"""
        while True:
            try:
                # Check deadman switch
                if not os.path.exists(self.deadman_switch_file):
                    # Deadman switch file missing
                    last_touch = getattr(self, '_last_deadman_touch', 0)
                    if time.time() - last_touch > 300:  # 5 minutes
                        logger.warning("Deadman switch timeout - initiating emergency shutdown")
                        self.initiate_emergency_shutdown(ShutdownTrigger.DEADMAN_SWITCH)
                        break
                else:
                    self._last_deadman_touch = os.path.getmtime(self.deadman_switch_file)
                
                # Check system resources
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                if memory.percent > 95:
                    logger.warning("Critical memory usage detected")
                    self.initiate_emergency_shutdown(ShutdownTrigger.RESOURCE_EXHAUSTION)
                    break
                
                if disk.percent > 98:
                    logger.warning("Critical disk usage detected")
                    self.initiate_emergency_shutdown(ShutdownTrigger.RESOURCE_EXHAUSTION)
                    break
                
                # Check for running processes
                sutazai_processes = []
                for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                    try:
                        cmdline = ' '.join(proc.info['cmdline'] or [])
                        if 'sutazai' in cmdline.lower() or 'agent' in cmdline.lower():
                            sutazai_processes.append(proc)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                
                # Check for data corruption indicators
                critical_files = [
                    "/opt/sutazaiapp/data/backup-metadata.db",
                    "/opt/sutazaiapp/config/services.yaml"
                ]
                
                for file_path in critical_files:
                    if os.path.exists(file_path):
                        try:
                            # Basic file integrity check
                            with open(file_path, 'rb') as f:
                                f.read(1024)  # Try to read first 1KB
                        except Exception as e:
                            logger.error(f"Data corruption detected in {file_path}: {e}")
                            self.initiate_emergency_shutdown(ShutdownTrigger.DATA_CORRUPTION)
                            return
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"System monitoring error: {e}")
                time.sleep(60)

    def initiate_emergency_shutdown(self, trigger: ShutdownTrigger, force: bool = False) -> bool:
        """Initiate emergency shutdown sequence"""
        if self.shutdown_active:
            logger.warning("Emergency shutdown already in progress")
            return False
        
        # Create shutdown lock
        with open(self.shutdown_lock_file, 'w') as f:
            f.write(f"{os.getpid()}\n{datetime.now().isoformat()}\n{trigger.value}")
        
        try:
            self.shutdown_active = True
            
            # Initialize shutdown state
            self.shutdown_state = ShutdownState(
                trigger=trigger,
                phase=ShutdownPhase.INITIATED,
                started_at=datetime.now(),
                estimated_completion=None,
                services_shutdown=[],
                services_failed=[],
                data_preserved=False,
                error_messages=[],
                recovery_possible=True
            )
            
            # Log shutdown event
            shutdown_id = self._log_shutdown_event(self.shutdown_state)
            
            logger.critical(f"EMERGENCY SHUTDOWN INITIATED - Trigger: {trigger.value}")
            
            # Broadcast shutdown to other systems
            self._broadcast_shutdown_notification(trigger)
            
            # Execute shutdown sequence
            success = self._execute_shutdown_sequence(shutdown_id, force)
            
            # Update final state
            self.shutdown_state.phase = ShutdownPhase.COMPLETED if success else ShutdownPhase.FAILED
            self._update_shutdown_event(shutdown_id, self.shutdown_state, success)
            
            return success
            
        except Exception as e:
            logger.error(f"Emergency shutdown failed: {e}")
            if self.shutdown_state:
                self.shutdown_state.error_messages.append(str(e))
                self.shutdown_state.phase = ShutdownPhase.FAILED
            return False
        finally:
            self.shutdown_active = False
            if os.path.exists(self.shutdown_lock_file):
                os.remove(self.shutdown_lock_file)

    def _execute_shutdown_sequence(self, shutdown_id: int, force: bool = False) -> bool:
        """Execute the complete shutdown sequence"""
        try:
            # Phase 1: Stop services in priority order
            self.shutdown_state.phase = ShutdownPhase.STOPPING_SERVICES
            if not self._shutdown_services(shutdown_id, force):
                return False
            
            # Phase 2: Drain message queues and connections
            self.shutdown_state.phase = ShutdownPhase.DRAINING_QUEUES
            self._drain_queues_and_connections()
            
            # Phase 3: Persist critical state
            self.shutdown_state.phase = ShutdownPhase.PERSISTING_STATE
            self.shutdown_state.data_preserved = self._persist_critical_state()
            
            # Phase 4: Graceful termination
            self.shutdown_state.phase = ShutdownPhase.GRACEFUL_TERMINATION
            self._graceful_process_termination()
            
            # Phase 5: Force termination if needed
            if force or self.force_shutdown.is_set():
                self.shutdown_state.phase = ShutdownPhase.FORCED_TERMINATION
                self._force_process_termination()
            
            # Phase 6: Cleanup
            self.shutdown_state.phase = ShutdownPhase.CLEANUP
            self._cleanup_resources()
            
            return True
            
        except Exception as e:
            logger.error(f"Shutdown sequence failed: {e}")
            self.shutdown_state.error_messages.append(str(e))
            return False

    def _shutdown_services(self, shutdown_id: int, force: bool = False) -> bool:
        """Shutdown services in priority order"""
        # Sort services by priority (1 = highest priority, shutdown first)
        sorted_services = sorted(self.service_configs.items(), key=lambda x: x[1].priority)
        
        for service_name, config in sorted_services:
            if self.abort_shutdown.is_set():
                logger.info("Shutdown aborted by user")
                return False
            
            start_time = time.time()
            logger.info(f"Shutting down service: {service_name}")
            
            try:
                # Check dependencies
                if config.dependency_services:
                    for dep in config.dependency_services:
                        if dep not in self.shutdown_state.services_shutdown:
                            logger.warning(f"Dependency {dep} not yet shutdown for {service_name}")
                
                # Execute pre-shutdown commands
                if config.pre_shutdown_commands:
                    for cmd in config.pre_shutdown_commands:
                        try:
                            subprocess.run(cmd, shell=True, timeout=30, check=True)
                        except subprocess.TimeoutExpired:
                            logger.warning(f"Pre-shutdown command timeout: {cmd}")
                        except subprocess.CalledProcessError as e:
                            logger.warning(f"Pre-shutdown command failed: {cmd} - {e}")
                
                # Shutdown service based on type
                success = False
                if service_name == "docker-containers":
                    success = self._shutdown_docker_containers(config, force)
                elif service_name == "databases":
                    success = self._shutdown_databases(config)
                elif service_name in ["frontend", "backend", "agent-orchestrator", "monitoring"]:
                    success = self._shutdown_application_service(service_name, config, force)
                else:
                    success = self._shutdown_generic_service(service_name, config, force)
                
                duration = time.time() - start_time
                
                if success:
                    self.shutdown_state.services_shutdown.append(service_name)
                    self._log_service_action(shutdown_id, service_name, "shutdown", "success", duration)
                    logger.info(f"Service shutdown successful: {service_name} ({duration:.1f}s)")
                else:
                    self.shutdown_state.services_failed.append(service_name)
                    self._log_service_action(shutdown_id, service_name, "shutdown", "failed", duration)
                    logger.error(f"Service shutdown failed: {service_name}")
                    
                    if config.critical and not force:
                        logger.error(f"Critical service shutdown failed: {service_name}")
                        return False
                
                # Execute post-shutdown commands
                if config.post_shutdown_commands:
                    for cmd in config.post_shutdown_commands:
                        try:
                            subprocess.run(cmd, shell=True, timeout=30, check=True)
                        except Exception as e:
                            logger.warning(f"Post-shutdown command failed: {cmd} - {e}")
                
            except Exception as e:
                logger.error(f"Service shutdown error: {service_name} - {e}")
                self.shutdown_state.services_failed.append(service_name)
                self.shutdown_state.error_messages.append(f"{service_name}: {str(e)}")
                
                if config.critical and not force:
                    return False
        
        return True

    def _shutdown_docker_containers(self, config: ServiceShutdownConfig, force: bool = False) -> bool:
        """Shutdown Docker containers gracefully"""
        if not self.docker_client:
            logger.warning("Docker client not available")
            return True
        
        try:
            containers = self.docker_client.containers.list()
            
            for container in containers:
                try:
                    logger.info(f"Stopping container: {container.name}")
                    
                    if force:
                        container.kill()
                    else:
                        container.stop(timeout=config.graceful_timeout)
                    
                    # Wait for container to stop
                    container.wait(timeout=config.force_timeout)
                    
                    logger.info(f"Container stopped: {container.name}")
                    
                except Exception as e:
                    logger.error(f"Failed to stop container {container.name}: {e}")
                    if config.critical:
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Docker shutdown failed: {e}")
            return False

    def _shutdown_databases(self, config: ServiceShutdownConfig) -> bool:
        """Shutdown databases with data preservation"""
        try:
            # Create emergency backup first
            backup_dir = "/opt/sutazaiapp/backups/emergency"
            os.makedirs(backup_dir, exist_ok=True)
            
            timestamp = int(time.time())
            
            # Backup SQLite databases
            for root, dirs, files in os.walk("/opt/sutazaiapp/data"):
                for file in files:
                    if file.endswith('.db'):
                        source = os.path.join(root, file)
                        backup = os.path.join(backup_dir, f"{file}.{timestamp}.backup")
                        try:
                            # Use SQLite backup for consistency
                            subprocess.run([
                                "sqlite3", source, f".backup {backup}"
                            ], check=True, timeout=60)
                            logger.info(f"Database backed up: {source} -> {backup}")
                        except Exception as e:
                            logger.error(f"Database backup failed: {source} - {e}")
                            return False
            
            # Stop database connections
            if self.redis_client:
                try:
                    self.redis_client.shutdown(nosave=False)
                except Exception as e:
                    logger.warning(f"Redis shutdown warning: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"Database shutdown failed: {e}")
            return False

    def _shutdown_application_service(self, service_name: str, config: ServiceShutdownConfig, force: bool = False) -> bool:
        """Shutdown application services"""
        try:
            # Find processes for this service
            service_processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = ' '.join(proc.info['cmdline'] or [])
                    if service_name.replace('-', '_') in cmdline or service_name in cmdline:
                        service_processes.append(proc)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            if not service_processes:
                logger.info(f"No processes found for service: {service_name}")
                return True
            
            # Send SIGTERM first
            for proc in service_processes:
                try:
                    if not force:
                        proc.terminate()
                        logger.info(f"Sent SIGTERM to {service_name} process {proc.pid}")
                    else:
                        proc.kill()
                        logger.info(f"Sent SIGKILL to {service_name} process {proc.pid}")
                except psutil.NoSuchProcess:
                    continue
                except Exception as e:
                    logger.error(f"Failed to terminate process {proc.pid}: {e}")
            
            # Wait for graceful shutdown
            if not force:
                timeout = config.graceful_timeout
                start_time = time.time()
                
                while time.time() - start_time < timeout:
                    alive_processes = [p for p in service_processes if p.is_running()]
                    if not alive_processes:
                        break
                    time.sleep(1)
                
                # Force kill if still running
                remaining_processes = [p for p in service_processes if p.is_running()]
                if remaining_processes:
                    logger.warning(f"Force killing {len(remaining_processes)} processes for {service_name}")
                    for proc in remaining_processes:
                        try:
                            proc.kill()
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            continue
            
            return True
            
        except Exception as e:
            logger.error(f"Application service shutdown failed: {service_name} - {e}")
            return False

    def _shutdown_generic_service(self, service_name: str, config: ServiceShutdownConfig, force: bool = False) -> bool:
        """Shutdown generic system service"""
        try:
            # Try systemctl first
            try:
                cmd = ["systemctl", "stop", service_name]
                result = subprocess.run(cmd, timeout=config.graceful_timeout, 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    logger.info(f"Systemd service stopped: {service_name}")
                    return True
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
                pass
            
            # Try pkill
            try:
                subprocess.run(["pkill", "-f", service_name], timeout=30)
                time.sleep(5)
                
                # Check if still running
                result = subprocess.run(["pgrep", "-f", service_name], 
                                      capture_output=True, timeout=10)
                if result.returncode != 0:  # No processes found
                    return True
                
                # Force kill if still running
                if force:
                    subprocess.run(["pkill", "-9", "-f", service_name], timeout=30)
                
            except subprocess.TimeoutExpired:
                logger.warning(f"Timeout killing processes for {service_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Generic service shutdown failed: {service_name} - {e}")
            return False

    def _drain_queues_and_connections(self):
        """Drain message queues and close connections"""
        logger.info("Draining queues and connections")
        
        try:
            # Redis queue draining
            if self.redis_client:
                try:
                    # Get all keys that look like queues
                    queue_keys = self.redis_client.keys("queue:*") + self.redis_client.keys("*:queue")
                    
                    for key in queue_keys:
                        queue_length = self.redis_client.llen(key)
                        if queue_length > 0:
                            logger.info(f"Draining queue {key} with {queue_length} items")
                            # Save queue contents for recovery
                            items = self.redis_client.lrange(key, 0, -1)
                            queue_backup_file = f"/opt/sutazaiapp/backups/emergency/queue_{key.decode()}_{int(time.time())}.json"
                            with open(queue_backup_file, 'w') as f:
                                json.dump([item.decode() for item in items], f)
                            
                            # Clear the queue
                            self.redis_client.delete(key)
                
                except Exception as e:
                    logger.error(f"Redis queue draining failed: {e}")
            
            # Close active network connections
            try:
                # Find and close open sockets
                connections = psutil.net_connections()
                sutazai_connections = []
                
                for conn in connections:
                    if conn.pid:
                        try:
                            proc = psutil.Process(conn.pid)
                            if 'sutazai' in proc.name().lower() or any('agent' in cmd for cmd in proc.cmdline()):
                                sutazai_connections.append(conn)
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            continue
                
                logger.info(f"Found {len(sutazai_connections)} SutazAI connections to close")
                
            except Exception as e:
                logger.error(f"Connection draining failed: {e}")
        
        except Exception as e:
            logger.error(f"Queue and connection draining failed: {e}")

    def _persist_critical_state(self) -> bool:
        """Persist critical system state for recovery"""
        logger.info("Persisting critical state")
        
        try:
            state_backup_dir = "/opt/sutazaiapp/backups/emergency/state"
            os.makedirs(state_backup_dir, exist_ok=True)
            timestamp = int(time.time())
            
            # System state
            system_state = {
                "shutdown_trigger": self.shutdown_state.trigger.value,
                "shutdown_time": self.shutdown_state.started_at.isoformat(),
                "services_shutdown": self.shutdown_state.services_shutdown,
                "services_failed": self.shutdown_state.services_failed,
                "system_info": {
                    "hostname": os.uname().nodename,
                    "pid": os.getpid(),
                    "working_directory": os.getcwd(),
                    "environment": dict(os.environ)
                }
            }
            
            state_file = os.path.join(state_backup_dir, f"system_state_{timestamp}.json")
            with open(state_file, 'w') as f:
                json.dump(system_state, f, indent=2)
            
            # Running processes state
            processes_state = []
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'status', 'cpu_percent', 'memory_percent']):
                try:
                    processes_state.append(proc.info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            processes_file = os.path.join(state_backup_dir, f"processes_{timestamp}.json")
            with open(processes_file, 'w') as f:
                json.dump(processes_state, f, indent=2)
            
            # Network state
            network_state = {
                "interfaces": dict(psutil.net_if_addrs()),
                "connections": [conn._asdict() for conn in psutil.net_connections()],
                "stats": dict(psutil.net_if_stats())
            }
            
            network_file = os.path.join(state_backup_dir, f"network_{timestamp}.json")
            with open(network_file, 'w') as f:
                json.dump(network_state, f, indent=2, default=str)
            
            # Docker state if available
            if self.docker_client:
                try:
                    containers_state = []
                    for container in self.docker_client.containers.list(all=True):
                        containers_state.append({
                            "id": container.id,
                            "name": container.name,
                            "status": container.status,
                            "image": container.image.tags,
                            "ports": container.ports,
                            "labels": container.labels
                        })
                    
                    docker_file = os.path.join(state_backup_dir, f"docker_{timestamp}.json")
                    with open(docker_file, 'w') as f:
                        json.dump(containers_state, f, indent=2)
                        
                except Exception as e:
                    logger.error(f"Docker state persistence failed: {e}")
            
            logger.info(f"Critical state persisted to {state_backup_dir}")
            return True
            
        except Exception as e:
            logger.error(f"State persistence failed: {e}")
            return False

    def _graceful_process_termination(self):
        """Gracefully terminate remaining processes"""
        logger.info("Graceful process termination")
        
        try:
            # Find all SutazAI related processes
            sutazai_processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = ' '.join(proc.info['cmdline'] or [])
                    if ('sutazai' in cmdline.lower() or 
                        'agent' in cmdline.lower() or
                        '/opt/sutazaiapp' in cmdline):
                        sutazai_processes.append(proc)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            if not sutazai_processes:
                logger.info("No remaining processes to terminate")
                return
            
            logger.info(f"Gracefully terminating {len(sutazai_processes)} processes")
            
            # Send SIGTERM to all processes
            for proc in sutazai_processes:
                try:
                    proc.terminate()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            # Wait for processes to exit
            timeout = 60  # 1 minute
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                alive_processes = [p for p in sutazai_processes if p.is_running()]
                if not alive_processes:
                    break
                time.sleep(1)
            
            remaining = [p for p in sutazai_processes if p.is_running()]
            if remaining:
                logger.warning(f"{len(remaining)} processes did not terminate gracefully")
                self.force_shutdown.set()
        
        except Exception as e:
            logger.error(f"Graceful termination failed: {e}")

    def _force_process_termination(self):
        """Force terminate remaining processes"""
        logger.info("Force process termination")
        
        try:
            # Find all remaining SutazAI processes
            sutazai_processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = ' '.join(proc.info['cmdline'] or [])
                    if ('sutazai' in cmdline.lower() or 
                        'agent' in cmdline.lower() or
                        '/opt/sutazaiapp' in cmdline):
                        sutazai_processes.append(proc)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            if not sutazai_processes:
                logger.info("No processes require force termination")
                return
            
            logger.warning(f"Force killing {len(sutazai_processes)} processes")
            
            for proc in sutazai_processes:
                try:
                    proc.kill()
                    logger.info(f"Force killed process: {proc.pid} ({proc.name()})")
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
                except Exception as e:
                    logger.error(f"Failed to kill process {proc.pid}: {e}")
        
        except Exception as e:
            logger.error(f"Force termination failed: {e}")

    def _cleanup_resources(self):
        """Clean up system resources"""
        logger.info("Cleaning up resources")
        
        try:
            # Clean up temporary files
            temp_patterns = [
                "/tmp/sutazai-*",
                "/tmp/agent-*",
                "/tmp/backup-*"
            ]
            
            for pattern in temp_patterns:
                try:
                    subprocess.run(f"rm -rf {pattern}", shell=True, timeout=30)
                except subprocess.TimeoutExpired:
                    logger.warning(f"Cleanup timeout for pattern: {pattern}")
            
            # Clean up shared memory
            try:
                subprocess.run("ipcrm -a", shell=True, timeout=30)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                pass  # Not critical
            
            # Sync filesystem
            try:
                subprocess.run("sync", timeout=30)
            except subprocess.TimeoutExpired:
                logger.warning("Filesystem sync timeout")
            
            logger.info("Resource cleanup completed")
            
        except Exception as e:
            logger.error(f"Resource cleanup failed: {e}")

    def _broadcast_shutdown_notification(self, trigger: ShutdownTrigger):
        """Broadcast shutdown notification to other systems"""
        try:
            notification = {
                "event": "emergency_shutdown",
                "trigger": trigger.value,
                "timestamp": datetime.now().isoformat(),
                "hostname": os.uname().nodename,
                "pid": os.getpid()
            }
            
            # Redis notification
            if self.redis_client:
                try:
                    self.redis_client.publish("sutazai:emergency", json.dumps(notification))
                except Exception as e:
                    logger.warning(f"Redis notification failed: {e}")
            
            # File-based notification for other monitoring systems
            notification_file = "/tmp/sutazai-emergency-shutdown.json"
            with open(notification_file, 'w') as f:
                json.dump(notification, f, indent=2)
            
            # Log to system
            subprocess.run([
                "logger", "-t", "sutazai-emergency", 
                f"Emergency shutdown initiated: {trigger.value}"
            ])
            
        except Exception as e:
            logger.error(f"Shutdown notification failed: {e}")

    def _log_shutdown_event(self, state: ShutdownState) -> int:
        """Log shutdown event to database"""
        cursor = self.conn.execute('''
            INSERT INTO shutdown_events (
                trigger, phase, started_at, state_data
            ) VALUES (?, ?, ?, ?)
        ''', (
            state.trigger.value,
            state.phase.value,
            state.started_at.isoformat(),
            json.dumps(asdict(state), default=str)
        ))
        self.conn.commit()
        return cursor.lastrowid

    def _update_shutdown_event(self, shutdown_id: int, state: ShutdownState, success: bool):
        """Update shutdown event in database"""
        self.conn.execute('''
            UPDATE shutdown_events 
            SET phase = ?, completed_at = ?, success = ?, state_data = ?
            WHERE id = ?
        ''', (
            state.phase.value,
            datetime.now().isoformat(),
            success,
            json.dumps(asdict(state), default=str),
            shutdown_id
        ))
        self.conn.commit()

    def _log_service_action(self, shutdown_id: int, service_name: str, action: str, 
                           status: str, duration: float, error_message: str = None):
        """Log service action to database"""
        self.conn.execute('''
            INSERT INTO service_shutdown_log (
                shutdown_id, service_name, action, status, 
                duration_seconds, error_message
            ) VALUES (?, ?, ?, ?, ?, ?)
        ''', (shutdown_id, service_name, action, status, duration, error_message))
        self.conn.commit()

    def abort_emergency_shutdown(self) -> bool:
        """Abort an ongoing emergency shutdown if possible"""
        if not self.shutdown_active:
            logger.info("No active shutdown to abort")
            return False
        
        if self.shutdown_state.phase in [ShutdownPhase.FORCED_TERMINATION, ShutdownPhase.CLEANUP]:
            logger.warning("Shutdown too advanced to abort safely")
            return False
        
        logger.info("Aborting emergency shutdown")
        self.abort_shutdown.set()
        
        # Try to restart critical services
        critical_services = [name for name, config in self.service_configs.items() if config.critical]
        
        for service_name in critical_services:
            if service_name in self.shutdown_state.services_shutdown:
                try:
                    logger.info(f"Attempting to restart service: {service_name}")
                    # This would need service-specific restart logic
                    subprocess.run(["systemctl", "start", service_name], timeout=30)
                except Exception as e:
                    logger.error(f"Failed to restart {service_name}: {e}")
        
        return True

    def get_shutdown_status(self) -> Optional[Dict[str, Any]]:
        """Get current shutdown status"""
        if not self.shutdown_active or not self.shutdown_state:
            return None
        
        return {
            "active": self.shutdown_active,
            "trigger": self.shutdown_state.trigger.value,
            "phase": self.shutdown_state.phase.value,
            "started_at": self.shutdown_state.started_at.isoformat(),
            "services_shutdown": self.shutdown_state.services_shutdown,
            "services_failed": self.shutdown_state.services_failed,
            "data_preserved": self.shutdown_state.data_preserved,
            "error_messages": self.shutdown_state.error_messages,
            "recovery_possible": self.shutdown_state.recovery_possible,
            "estimated_completion": self.shutdown_state.estimated_completion.isoformat() if self.shutdown_state.estimated_completion else None
        }

    def touch_deadman_switch(self):
        """Touch the deadman switch file to prevent automatic shutdown"""
        try:
            with open(self.deadman_switch_file, 'w') as f:
                f.write(f"{datetime.now().isoformat()}\n{os.getpid()}")
            self._last_deadman_touch = time.time()
        except Exception as e:
            logger.error(f"Failed to touch deadman switch: {e}")

    def setup_deadman_switch(self, interval: int = 60):
        """Setup automatic deadman switch touching"""
        def touch_periodically():
            while not self.shutdown_active:
                self.touch_deadman_switch()
                time.sleep(interval)
        
        deadman_thread = threading.Thread(target=touch_periodically, daemon=True)
        deadman_thread.start()
        logger.info(f"Deadman switch setup with {interval}s interval")

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="SutazAI Emergency Shutdown Coordinator")
    parser.add_argument("command", choices=["start", "shutdown", "abort", "status", "deadman"])
    parser.add_argument("--trigger", choices=[t.value for t in ShutdownTrigger], 
                       default=ShutdownTrigger.MANUAL.value)
    parser.add_argument("--force", action="store_true", help="Force immediate shutdown")
    
    args = parser.parse_args()
    
    coordinator = EmergencyShutdownCoordinator()
    
    try:
        if args.command == "start":
            coordinator.setup_deadman_switch()
            logger.info("Emergency shutdown coordinator started - monitoring system")
            # Keep the process running
            while True:
                time.sleep(60)
                
        elif args.command == "shutdown":
            trigger = ShutdownTrigger(args.trigger)
            success = coordinator.initiate_emergency_shutdown(trigger, args.force)
            if success:
                logger.info("Emergency shutdown completed successfully")
            else:
                logger.error("Emergency shutdown failed")
                sys.exit(1)
                
        elif args.command == "abort":
            success = coordinator.abort_emergency_shutdown()
            if success:
                logger.info("Emergency shutdown aborted")
            else:
                logger.info("Could not abort shutdown")
                sys.exit(1)
                
        elif args.command == "status":
            status = coordinator.get_shutdown_status()
            if status:
                logger.info(json.dumps(status, indent=2))
            else:
                logger.info("No active shutdown")
                
        elif args.command == "deadman":
            coordinator.touch_deadman_switch()
            logger.info("Deadman switch touched")
    
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
        coordinator.initiate_emergency_shutdown(ShutdownTrigger.MANUAL)
    except Exception as e:
        logger.error(f"Emergency shutdown coordinator error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()