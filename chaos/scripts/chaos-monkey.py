#!/usr/bin/env python3
"""
SutazAI Chaos Monkey Implementation
Automated, scheduled chaos experiments with safe mode and rollback capabilities
"""

import sys
import os
import json
import time
import random
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import yaml
import docker
import schedule
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import signal

# Add chaos directory to path
sys.path.append('/opt/sutazaiapp/chaos')

class ChaosMonkeyMode(Enum):
    SAFE = "safe"
    PRODUCTION = "production"
    AGGRESSIVE = "aggressive"
    DISABLED = "disabled"

class ExperimentFrequency(Enum):
    CONTINUOUS = "continuous"
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    CUSTOM = "custom"

@dataclass
class ChaosExperimentSchedule:
    """Scheduled chaos experiment configuration"""
    name: str
    experiment_type: str
    frequency: ExperimentFrequency
    schedule_pattern: str  # Cron-like pattern
    target_services: List[str]
    safe_mode: bool
    enabled: bool
    max_duration: int  # seconds
    probability: float  # 0.0 to 1.0
    maintenance_window_only: bool

@dataclass
class ChaosMonkeyState:
    """Current state of Chaos Monkey"""
    mode: ChaosMonkeyMode
    active_experiments: List[str]
    last_experiment_time: Optional[datetime]
    total_experiments_run: int
    successful_experiments: int
    failed_experiments: int
    emergency_stop_active: bool
    system_health_score: float

class SafetyController:
    """Controls safety mechanisms for Chaos Monkey"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.docker_client = docker.from_env()
        
        # Safety thresholds
        self.min_healthy_services = 0.8  # 80% of services must be healthy
        self.max_cpu_usage = 85          # Max system CPU usage
        self.max_memory_usage = 90       # Max system memory usage
        self.max_concurrent_experiments = 2
        
        # Critical services that should never be touched
        self.protected_services = [
            'sutazai-postgres',
            'sutazai-prometheus',
            'sutazai-grafana',
            'sutazai-health-monitor'
        ]
    
    async def is_system_safe_for_chaos(self) -> Tuple[bool, str]:
        """Check if system is safe for chaos experiments"""
        try:
            # Check service health
            healthy_count = 0
            total_count = 0
            
            for container in self.docker_client.containers.list():
                if container.name.startswith('sutazai-'):
                    total_count += 1
                    if container.status == 'running':
                        # Check health status if available
                        try:
                            if hasattr(container, 'attrs') and 'State' in container.attrs:
                                state = container.attrs['State']
                                if 'Health' in state and state['Health']['Status'] == 'healthy':
                                    healthy_count += 1
                                elif 'Health' not in state:
                                    # No health check, assume healthy if running
                                    healthy_count += 1
                        except Exception:
                            pass
            
            if total_count == 0:
                return False, "No services found"
            
            health_ratio = healthy_count / total_count
            if health_ratio < self.min_healthy_services:
                return False, f"Only {health_ratio:.1%} of services are healthy (minimum: {self.min_healthy_services:.1%})"
            
            # Check system resources
            cpu_usage = self._get_system_cpu_usage()
            memory_usage = self._get_system_memory_usage()
            
            if cpu_usage > self.max_cpu_usage:
                return False, f"CPU usage too high: {cpu_usage}% (max: {self.max_cpu_usage}%)"
            
            if memory_usage > self.max_memory_usage:
                return False, f"Memory usage too high: {memory_usage}% (max: {self.max_memory_usage}%)"
            
            # Check if we're in maintenance window
            if not self._is_maintenance_window():
                return False, "Not in maintenance window"
            
            return True, "System is safe for chaos experiments"
            
        except Exception as e:
            self.logger.error(f"Error checking system safety: {e}")
            return False, f"Safety check failed: {e}"
    
    def is_service_protected(self, service_name: str) -> bool:
        """Check if a service is protected from chaos"""
        return service_name in self.protected_services
    
    def get_safe_target_services(self, requested_services: List[str] = None) -> List[str]:
        """Get list of services safe for chaos experiments"""
        all_services = [container.name for container in self.docker_client.containers.list() 
                       if container.name.startswith('sutazai-')]
        
        safe_services = [s for s in all_services if not self.is_service_protected(s)]
        
        if requested_services:
            # Filter to only requested services that are also safe
            safe_services = [s for s in safe_services if s in requested_services]
        
        return safe_services
    
    def _get_system_cpu_usage(self) -> float:
        """Get system CPU usage percentage"""
        try:
            import psutil
            return psutil.cpu_percent(interval=1)
        except ImportError:
            # Fallback using /proc/loadavg
            try:
                with open('/proc/loadavg', 'r') as f:
                    load_avg = float(f.read().split()[0])
                # Rough conversion assuming 4 cores
                return min(load_avg * 25, 100)
            except Exception:
                return 0
    
    def _get_system_memory_usage(self) -> float:
        """Get system memory usage percentage"""
        try:
            import psutil
            return psutil.virtual_memory().percent
        except ImportError:
            # Fallback using /proc/meminfo
            try:
                with open('/proc/meminfo', 'r') as f:
                    lines = f.readlines()
                meminfo = {}
                for line in lines:
                    key, value = line.split(':')
                    meminfo[key.strip()] = int(value.strip().split()[0])
                
                total = meminfo['MemTotal']
                available = meminfo.get('MemAvailable', meminfo['MemFree'])
                used = total - available
                return (used / total) * 100
            except Exception:
                return 0
    
    def _is_maintenance_window(self) -> bool:
        """Check if current time is within maintenance window"""
        now = datetime.now()
        current_hour = now.hour
        current_day = now.strftime('%A').lower()
        
        # Maintenance windows: Mon/Wed/Fri 2-4 AM
        maintenance_days = ['monday', 'wednesday', 'friday']
        maintenance_hours = range(2, 4)
        
        return current_day in maintenance_days and current_hour in maintenance_hours

class ExperimentScheduler:
    """Manages scheduling of chaos experiments"""
    
    def __init__(self, safety_controller: SafetyController, logger: logging.Logger):
        self.safety_controller = safety_controller
        self.logger = logger
        self.schedules: List[ChaosExperimentSchedule] = []
        self.scheduler_thread = None
        self.running = False
    
    def load_schedules(self, config_path: str):
        """Load experiment schedules from configuration"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            schedule_configs = config.get('chaos_monkey', {}).get('schedules', [])
            
            for schedule_config in schedule_configs:
                schedule_obj = ChaosExperimentSchedule(
                    name=schedule_config['name'],
                    experiment_type=schedule_config['experiment_type'],
                    frequency=ExperimentFrequency(schedule_config['frequency']),
                    schedule_pattern=schedule_config['schedule_pattern'],
                    target_services=schedule_config.get('target_services', []),
                    safe_mode=schedule_config.get('safe_mode', True),
                    enabled=schedule_config.get('enabled', True),
                    max_duration=schedule_config.get('max_duration', 600),
                    probability=schedule_config.get('probability', 0.3),
                    maintenance_window_only=schedule_config.get('maintenance_window_only', True)
                )
                self.schedules.append(schedule_obj)
            
            self.logger.info(f"Loaded {len(self.schedules)} experiment schedules")
            
        except Exception as e:
            self.logger.error(f"Failed to load schedules: {e}")
    
    def start_scheduler(self):
        """Start the experiment scheduler"""
        if self.running:
            return
        
        self.running = True
        
        # Clear any existing schedules
        schedule.clear()
        
        # Setup schedules
        for exp_schedule in self.schedules:
            if not exp_schedule.enabled:
                continue
            
            if exp_schedule.frequency == ExperimentFrequency.DAILY:
                schedule.every().day.at(exp_schedule.schedule_pattern).do(
                    self._schedule_experiment, exp_schedule
                )
            elif exp_schedule.frequency == ExperimentFrequency.HOURLY:
                schedule.every().hour.do(self._schedule_experiment, exp_schedule)
            elif exp_schedule.frequency == ExperimentFrequency.WEEKLY:
                day, time_str = exp_schedule.schedule_pattern.split(' ')
                getattr(schedule.every(), day.lower()).at(time_str).do(
                    self._schedule_experiment, exp_schedule
                )
        
        # Start scheduler thread
        self.scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.scheduler_thread.start()
        
        self.logger.info("Chaos Monkey scheduler started")
    
    def stop_scheduler(self):
        """Stop the experiment scheduler"""
        self.running = False
        schedule.clear()
        self.logger.info("Chaos Monkey scheduler stopped")
    
    def _run_scheduler(self):
        """Run the scheduler loop"""
        while self.running:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    def _schedule_experiment(self, exp_schedule: ChaosExperimentSchedule):
        """Execute a scheduled experiment"""
        # Check probability
        if random.random() > exp_schedule.probability:
            self.logger.info(f"Skipping experiment {exp_schedule.name} due to probability")
            return
        
        # Check maintenance window if required
        if exp_schedule.maintenance_window_only and not self.safety_controller._is_maintenance_window():
            self.logger.info(f"Skipping experiment {exp_schedule.name} - not in maintenance window")
            return
        
        self.logger.info(f"Executing scheduled experiment: {exp_schedule.name}")
        
        # Run experiment asynchronously
        asyncio.create_task(self._execute_experiment(exp_schedule))
    
    async def _execute_experiment(self, exp_schedule: ChaosExperimentSchedule):
        """Execute a chaos experiment"""
        try:
            # Safety check
            is_safe, reason = await self.safety_controller.is_system_safe_for_chaos()
            if not is_safe:
                self.logger.warning(f"Skipping experiment {exp_schedule.name}: {reason}")
                return
            
            # Get safe targets
            target_services = self.safety_controller.get_safe_target_services(
                exp_schedule.target_services
            )
            
            if not target_services:
                self.logger.warning(f"No safe target services for experiment {exp_schedule.name}")
                return
            
            # Select random target
            target_service = random.choice(target_services)
            
            # Execute experiment using chaos engine
            from chaos_engine import ChaosEngine
            
            engine = ChaosEngine()
            result = await engine.run_experiment(
                exp_schedule.experiment_type, 
                safe_mode=exp_schedule.safe_mode
            )
            
            self.logger.info(f"Experiment {exp_schedule.name} completed with status: {result.status.value}")
            
        except Exception as e:
            self.logger.error(f"Failed to execute experiment {exp_schedule.name}: {e}")

class ChaosMonkey:
    """Main Chaos Monkey orchestrator"""
    
    def __init__(self, config_path: str = "/opt/sutazaiapp/chaos/config/chaos-config.yaml"):
        self.config_path = config_path
        self.logger = self._setup_logger()
        self.safety_controller = SafetyController(self.logger)
        self.scheduler = ExperimentScheduler(self.safety_controller, self.logger)
        
        self.state = ChaosMonkeyState(
            mode=ChaosMonkeyMode.SAFE,
            active_experiments=[],
            last_experiment_time=None,
            total_experiments_run=0,
            successful_experiments=0,
            failed_experiments=0,
            emergency_stop_active=False,
            system_health_score=100.0
        )
        
        self.state_file = "/opt/sutazaiapp/chaos/chaos_monkey_state.json"
        self.running = False
        
        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging for Chaos Monkey"""
        logger = logging.getLogger("chaos_monkey")
        logger.setLevel(logging.INFO)
        
        # File handler
        log_file = "/opt/sutazaiapp/logs/chaos_monkey.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}, shutting down Chaos Monkey")
        self.stop()
        sys.exit(0)
    
    def load_config(self):
        """Load Chaos Monkey configuration"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            chaos_config = config.get('chaos_monkey', {})
            
            # Set mode
            mode_str = chaos_config.get('mode', 'safe')
            self.state.mode = ChaosMonkeyMode(mode_str)
            
            # Load schedules
            self.scheduler.load_schedules(self.config_path)
            
            self.logger.info(f"Chaos Monkey configured in {self.state.mode.value} mode")
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            raise
    
    def start(self):
        """Start Chaos Monkey"""
        if self.running:
            return
        
        self.logger.info("Starting Chaos Monkey")
        
        # Load state
        self._load_state()
        
        # Load configuration
        self.load_config()
        
        # Check if disabled
        if self.state.mode == ChaosMonkeyMode.DISABLED:
            self.logger.info("Chaos Monkey is disabled")
            return
        
        # Start scheduler
        self.scheduler.start_scheduler()
        
        self.running = True
        
        # Start main loop
        asyncio.create_task(self._main_loop())
        
        self.logger.info("Chaos Monkey started successfully")
    
    def stop(self):
        """Stop Chaos Monkey"""
        if not self.running:
            return
        
        self.logger.info("Stopping Chaos Monkey")
        
        self.running = False
        self.scheduler.stop_scheduler()
        
        # Save state
        self._save_state()
        
        self.logger.info("Chaos Monkey stopped")
    
    def emergency_stop(self):
        """Emergency stop all chaos activities"""
        self.logger.warning("EMERGENCY STOP activated")
        
        self.state.emergency_stop_active = True
        self.stop()
        
        # Kill any running experiments
        try:
            # This would need to be implemented to track and kill active experiments
            pass
        except Exception as e:
            self.logger.error(f"Error during emergency stop: {e}")
        
        self.logger.warning("Emergency stop completed")
    
    async def _main_loop(self):
        """Main Chaos Monkey monitoring loop"""
        while self.running:
            try:
                # Update system health
                await self._update_system_health()
                
                # Check for emergency conditions
                if self.state.system_health_score < 50:
                    self.logger.warning(f"Low system health detected: {self.state.system_health_score}%")
                    if self.state.mode != ChaosMonkeyMode.SAFE:
                        self.logger.warning("Switching to safe mode due to low health")
                        self.state.mode = ChaosMonkeyMode.SAFE
                
                # Save state periodically
                self._save_state()
                
                # Sleep for monitoring interval
                await asyncio.sleep(300)  # 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(60)
    
    async def _update_system_health(self):
        """Update system health score"""
        try:
            is_safe, reason = await self.safety_controller.is_system_safe_for_chaos()
            
            # Calculate health score based on various factors
            healthy_services = 0
            total_services = 0
            
            docker_client = docker.from_env()
            for container in docker_client.containers.list():
                if container.name.startswith('sutazai-'):
                    total_services += 1
                    if container.status == 'running':
                        healthy_services += 1
            
            if total_services > 0:
                health_ratio = healthy_services / total_services
                self.state.system_health_score = health_ratio * 100
            else:
                self.state.system_health_score = 0
            
        except Exception as e:
            self.logger.error(f"Error updating system health: {e}")
            self.state.system_health_score = 0
    
    def _load_state(self):
        """Load Chaos Monkey state from file"""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    state_data = json.load(f)
                
                # Restore state
                self.state.mode = ChaosMonkeyMode(state_data.get('mode', 'safe'))
                self.state.total_experiments_run = state_data.get('total_experiments_run', 0)
                self.state.successful_experiments = state_data.get('successful_experiments', 0)
                self.state.failed_experiments = state_data.get('failed_experiments', 0)
                
                last_exp_str = state_data.get('last_experiment_time')
                if last_exp_str:
                    self.state.last_experiment_time = datetime.fromisoformat(last_exp_str)
                
                self.logger.info("Chaos Monkey state loaded")
        except Exception as e:
            self.logger.error(f"Failed to load state: {e}")
    
    def _save_state(self):
        """Save Chaos Monkey state to file"""
        try:
            state_data = {
                'mode': self.state.mode.value,
                'active_experiments': self.state.active_experiments,
                'last_experiment_time': self.state.last_experiment_time.isoformat() if self.state.last_experiment_time else None,
                'total_experiments_run': self.state.total_experiments_run,
                'successful_experiments': self.state.successful_experiments,
                'failed_experiments': self.state.failed_experiments,
                'emergency_stop_active': self.state.emergency_stop_active,
                'system_health_score': self.state.system_health_score,
                'last_updated': datetime.now().isoformat()
            }
            
            os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
            with open(self.state_file, 'w') as f:
                json.dump(state_data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current Chaos Monkey status"""
        return {
            'mode': self.state.mode.value,
            'running': self.running,
            'active_experiments': self.state.active_experiments,
            'last_experiment_time': self.state.last_experiment_time.isoformat() if self.state.last_experiment_time else None,
            'total_experiments_run': self.state.total_experiments_run,
            'successful_experiments': self.state.successful_experiments,
            'failed_experiments': self.state.failed_experiments,
            'success_rate': (self.state.successful_experiments / self.state.total_experiments_run * 100) if self.state.total_experiments_run > 0 else 0,
            'emergency_stop_active': self.state.emergency_stop_active,
            'system_health_score': self.state.system_health_score,
            'scheduled_experiments': len(self.scheduler.schedules)
        }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SutazAI Chaos Monkey")
    parser.add_argument("--config", default="/opt/sutazaiapp/chaos/config/chaos-config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--mode", choices=['safe', 'production', 'aggressive', 'disabled'],
                       help="Override mode setting")
    parser.add_argument("--daemon", action="store_true", help="Run as daemon")
    parser.add_argument("--status", action="store_true", help="Show status and exit")
    parser.add_argument("--stop", action="store_true", help="Stop running Chaos Monkey")
    parser.add_argument("--emergency-stop", action="store_true", help="Emergency stop")
    
    args = parser.parse_args()
    
    async def main():
        monkey = ChaosMonkey(args.config)
        
        if args.status:
            # Load state and show status
            monkey._load_state()
            status = monkey.get_status()
            print(json.dumps(status, indent=2))
            return
        
        if args.stop:
            # Stop running instance (would need process management)
            print("Stop functionality requires process management implementation")
            return
        
        if args.emergency_stop:
            monkey.emergency_stop()
            return
        
        # Override mode if specified
        if args.mode:
            monkey.state.mode = ChaosMonkeyMode(args.mode)
        
        # Start Chaos Monkey
        monkey.start()
        
        if args.daemon:
            # Run as daemon
            try:
                while True:
                    await asyncio.sleep(3600)  # Sleep for 1 hour
            except KeyboardInterrupt:
                pass
        else:
            # Run for a limited time (for testing)
            await asyncio.sleep(600)  # Run for 10 minutes
        
        monkey.stop()
    
    asyncio.run(main())