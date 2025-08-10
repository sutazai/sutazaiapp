#!/usr/bin/env python3
"""
SutazAI Chaos Engineering Framework - Main Engine
Implements automated failure injection and resilience testing
"""

import sys
import os
import json
import time
import yaml
import random
import logging
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import docker
import requests
import subprocess
from dataclasses import dataclass
from enum import Enum

# Add chaos directory to path
sys.path.append('/opt/sutazaiapp/chaos')

class ExperimentStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    EMERGENCY_STOPPED = "emergency_stopped"

class ChaosType(Enum):
    CONTAINER = "container"
    NETWORK = "network"
    RESOURCE = "resource"
    DEPENDENCY = "dependency"

@dataclass
class ChaosTarget:
    """Represents a target for chaos experiments"""
    name: str
    container_id: str
    criticality: str
    chaos_allowed: bool
    max_downtime: str
    service_type: str

@dataclass
class ExperimentResult:
    """Stores experiment execution results"""
    experiment_name: str
    start_time: datetime
    end_time: Optional[datetime]
    status: ExperimentStatus
    targets_affected: List[str]
    recovery_time: Optional[float]
    errors: List[str]
    metrics: Dict[str, Any]

class ChaosLogger:
    """Centralized logging for chaos experiments"""
    
    def __init__(self, log_file: str = "/opt/sutazaiapp/logs/chaos.log"):
        self.logger = logging.getLogger("chaos_engine")
        self.logger.setLevel(logging.INFO)
        
        # File handler
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
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def info(self, message: str):
        self.logger.info(message)
    
    def error(self, message: str):
        self.logger.error(message)
    
    def warning(self, message: str):
        self.logger.warning(message)

class HealthMonitor:
    """Monitors service health during chaos experiments"""
    
    def __init__(self, logger: ChaosLogger):
        self.logger = logger
        self.docker_client = docker.from_env()
        
    async def check_container_health(self, container_name: str) -> Dict[str, Any]:
        """Check health of a specific container"""
        try:
            container = self.docker_client.containers.get(container_name)
            
            health_status = {
                'name': container_name,
                'status': container.status,
                'health': 'unknown',
                'running': container.status == 'running',
                'timestamp': datetime.now().isoformat()
            }
            
            # Check health status if available
            if hasattr(container, 'attrs') and 'State' in container.attrs:
                state = container.attrs['State']
                if 'Health' in state:
                    health_status['health'] = state['Health']['Status']
                
            return health_status
            
        except docker.errors.NotFound:
            return {
                'name': container_name,
                'status': 'not_found',
                'health': 'unhealthy',
                'running': False,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Error checking health for {container_name}: {e}")
            return {
                'name': container_name,
                'status': 'error',
                'health': 'unknown',
                'running': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def check_service_endpoint(self, url: str, timeout: int = 10) -> bool:
        """Check if a service endpoint is responsive"""
        try:
            response = requests.get(url, timeout=timeout)
            return response.status_code == 200
        except Exception:
            return False
    
    async def get_system_health(self, targets: List[ChaosTarget]) -> Dict[str, Any]:
        """Get comprehensive system health status"""
        health_results = {}
        
        for target in targets:
            container_health = await self.check_container_health(target.container_id)
            health_results[target.name] = container_health
        
        # Calculate overall health score
        healthy_count = sum(1 for result in health_results.values() 
                          if result.get('running', False))
        total_count = len(health_results)
        health_score = (healthy_count / total_count) * 100 if total_count > 0 else 0
        
        return {
            'timestamp': datetime.now().isoformat(),
            'health_score': health_score,
            'healthy_services': healthy_count,
            'total_services': total_count,
            'services': health_results
        }

class ContainerChaos:
    """Implements container-level chaos experiments"""
    
    def __init__(self, logger: ChaosLogger):
        self.logger = logger
        self.docker_client = docker.from_env()
    
    async def kill_container(self, target: ChaosTarget) -> Dict[str, Any]:
        """Kill a container and measure recovery time"""
        start_time = time.time()
        
        try:
            container = self.docker_client.containers.get(target.container_id)
            self.logger.info(f"Killing container: {target.name}")
            
            container.kill()
            
            # Wait for container to restart (if it has restart policy)
            recovery_time = await self._wait_for_recovery(target, timeout=300)
            
            return {
                'action': 'kill_container',
                'target': target.name,
                'start_time': start_time,
                'recovery_time': recovery_time,
                'success': recovery_time is not None
            }
            
        except Exception as e:
            self.logger.error(f"Failed to kill container {target.name}: {e}")
            return {
                'action': 'kill_container',
                'target': target.name,
                'start_time': start_time,
                'error': str(e),
                'success': False
            }
    
    async def restart_container(self, target: ChaosTarget) -> Dict[str, Any]:
        """Restart a container and measure recovery time"""
        start_time = time.time()
        
        try:
            container = self.docker_client.containers.get(target.container_id)
            self.logger.info(f"Restarting container: {target.name}")
            
            container.restart()
            
            # Wait for container to be healthy
            recovery_time = await self._wait_for_recovery(target, timeout=300)
            
            return {
                'action': 'restart_container',
                'target': target.name,
                'start_time': start_time,
                'recovery_time': recovery_time,
                'success': recovery_time is not None
            }
            
        except Exception as e:
            self.logger.error(f"Failed to restart container {target.name}: {e}")
            return {
                'action': 'restart_container',
                'target': target.name,
                'start_time': start_time,
                'error': str(e),
                'success': False
            }
    
    async def pause_container(self, target: ChaosTarget, duration: int = 60) -> Dict[str, Any]:
        """Pause a container for specified duration"""
        start_time = time.time()
        
        try:
            container = self.docker_client.containers.get(target.container_id)
            self.logger.info(f"Pausing container: {target.name} for {duration}s")
            
            container.pause()
            
            # Wait for duration
            await asyncio.sleep(duration)
            
            # Unpause container
            container.unpause()
            
            # Wait for recovery
            recovery_time = await self._wait_for_recovery(target, timeout=120)
            
            return {
                'action': 'pause_container',
                'target': target.name,
                'start_time': start_time,
                'duration': duration,
                'recovery_time': recovery_time,
                'success': recovery_time is not None
            }
            
        except Exception as e:
            self.logger.error(f"Failed to pause container {target.name}: {e}")
            return {
                'action': 'pause_container',
                'target': target.name,
                'start_time': start_time,
                'error': str(e),
                'success': False
            }
    
    async def _wait_for_recovery(self, target: ChaosTarget, timeout: int = 300) -> Optional[float]:
        """Wait for container to recover and return recovery time"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                container = self.docker_client.containers.get(target.container_id)
                if container.status == 'running':
                    # Additional health check if available
                    if hasattr(container, 'attrs') and 'State' in container.attrs:
                        state = container.attrs['State']
                        if 'Health' in state:
                            if state['Health']['Status'] == 'healthy':
                                return time.time() - start_time
                        else:
                            # No health check, assume healthy if running
                            return time.time() - start_time
                    else:
                        return time.time() - start_time
                        
            except docker.errors.NotFound:
                pass
            
            await asyncio.sleep(5)
        
        return None

class NetworkChaos:
    """Implements network-level chaos experiments"""
    
    def __init__(self, logger: ChaosLogger):
        self.logger = logger
    
    async def inject_latency(self, target: ChaosTarget, latency: str = "100ms", 
                           jitter: str = "10ms", duration: int = 300) -> Dict[str, Any]:
        """Inject network latency using tc (traffic control)"""
        start_time = time.time()
        
        try:
            # Get network interface of container
            interface = await self._get_container_interface(target)
            if not interface:
                raise Exception("Could not determine container network interface")
            
            self.logger.info(f"Injecting {latency} latency on {target.name}")
            
            # Add latency using tc
            cmd = [
                "docker", "exec", target.container_id,
                "tc", "qdisc", "add", "dev", interface, "root", "netem",
                "delay", latency, jitter
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise Exception(f"Failed to add latency: {result.stderr}")
            
            # Wait for duration
            await asyncio.sleep(duration)
            
            # Remove latency
            cleanup_cmd = [
                "docker", "exec", target.container_id,
                "tc", "qdisc", "del", "dev", interface, "root"
            ]
            
            subprocess.run(cleanup_cmd, capture_output=True)
            
            return {
                'action': 'inject_latency',
                'target': target.name,
                'latency': latency,
                'jitter': jitter,
                'duration': duration,
                'start_time': start_time,
                'success': True
            }
            
        except Exception as e:
            self.logger.error(f"Failed to inject latency on {target.name}: {e}")
            return {
                'action': 'inject_latency',
                'target': target.name,
                'start_time': start_time,
                'error': str(e),
                'success': False
            }
    
    async def inject_packet_loss(self, target: ChaosTarget, loss_rate: str = "5%",
                               duration: int = 180) -> Dict[str, Any]:
        """Inject packet loss using tc"""
        start_time = time.time()
        
        try:
            interface = await self._get_container_interface(target)
            if not interface:
                raise Exception("Could not determine container network interface")
            
            self.logger.info(f"Injecting {loss_rate} packet loss on {target.name}")
            
            # Add packet loss using tc
            cmd = [
                "docker", "exec", target.container_id,
                "tc", "qdisc", "add", "dev", interface, "root", "netem",
                "loss", loss_rate
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise Exception(f"Failed to add packet loss: {result.stderr}")
            
            # Wait for duration
            await asyncio.sleep(duration)
            
            # Remove packet loss
            cleanup_cmd = [
                "docker", "exec", target.container_id,
                "tc", "qdisc", "del", "dev", interface, "root"
            ]
            
            subprocess.run(cleanup_cmd, capture_output=True)
            
            return {
                'action': 'inject_packet_loss',
                'target': target.name,
                'loss_rate': loss_rate,
                'duration': duration,
                'start_time': start_time,
                'success': True
            }
            
        except Exception as e:
            self.logger.error(f"Failed to inject packet loss on {target.name}: {e}")
            return {
                'action': 'inject_packet_loss',
                'target': target.name,
                'start_time': start_time,
                'error': str(e),
                'success': False
            }
    
    async def _get_container_interface(self, target: ChaosTarget) -> Optional[str]:
        """Get the network interface for a container"""
        try:
            # Most containers use eth0 as primary interface
            # This could be enhanced to detect actual interface
            return "eth0"
        except Exception:
            return None

class ResourceChaos:
    """Implements resource-level chaos experiments"""
    
    def __init__(self, logger: ChaosLogger):
        self.logger = logger
    
    async def stress_cpu(self, target: ChaosTarget, cpu_percent: int = 80,
                        duration: int = 300) -> Dict[str, Any]:
        """Stress CPU using stress-ng"""
        start_time = time.time()
        
        try:
            self.logger.info(f"Stressing CPU on {target.name} to {cpu_percent}%")
            
            # Run stress-ng in container
            cmd = [
                "docker", "exec", "-d", target.container_id,
                "stress-ng", "--cpu", "0", "--cpu-load", str(cpu_percent),
                "--timeout", f"{duration}s"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise Exception(f"Failed to start CPU stress: {result.stderr}")
            
            # Monitor for duration
            await asyncio.sleep(duration)
            
            return {
                'action': 'stress_cpu',
                'target': target.name,
                'cpu_percent': cpu_percent,
                'duration': duration,
                'start_time': start_time,
                'success': True
            }
            
        except Exception as e:
            self.logger.error(f"Failed to stress CPU on {target.name}: {e}")
            return {
                'action': 'stress_cpu',
                'target': target.name,
                'start_time': start_time,
                'error': str(e),
                'success': False
            }
    
    async def stress_memory(self, target: ChaosTarget, memory_percent: int = 75,
                          duration: int = 180) -> Dict[str, Any]:
        """Stress memory using stress-ng"""
        start_time = time.time()
        
        try:
            self.logger.info(f"Stressing memory on {target.name} to {memory_percent}%")
            
            # Run memory stress in container
            cmd = [
                "docker", "exec", "-d", target.container_id,
                "stress-ng", "--vm", "1", "--vm-bytes", f"{memory_percent}%",
                "--timeout", f"{duration}s"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise Exception(f"Failed to start memory stress: {result.stderr}")
            
            # Monitor for duration
            await asyncio.sleep(duration)
            
            return {
                'action': 'stress_memory',
                'target': target.name,
                'memory_percent': memory_percent,
                'duration': duration,
                'start_time': start_time,
                'success': True
            }
            
        except Exception as e:
            self.logger.error(f"Failed to stress memory on {target.name}: {e}")
            return {
                'action': 'stress_memory',
                'target': target.name,
                'start_time': start_time,
                'error': str(e),
                'success': False
            }

class ChaosEngine:
    """Main chaos engineering orchestrator"""
    
    def __init__(self, config_path: str = "/opt/sutazaiapp/chaos/config/chaos-config.yaml"):
        self.config = self._load_config(config_path)
        self.logger = ChaosLogger()
        self.health_monitor = HealthMonitor(self.logger)
        self.container_chaos = ContainerChaos(self.logger)
        self.network_chaos = NetworkChaos(self.logger)
        self.resource_chaos = ResourceChaos(self.logger)
        
        self.current_experiment: Optional[ExperimentResult] = None
        self.emergency_stop = False
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load chaos configuration"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Failed to load config: {e}")
            sys.exit(1)
    
    async def get_targets(self) -> List[ChaosTarget]:
        """Get available chaos targets from configuration"""
        targets = []
        docker_client = docker.from_env()
        
        # Get all target categories from config
        target_categories = [
            'core_services', 'vector_services', 'app_services', 
            'agent_services', 'monitoring_services'
        ]
        
        for category in target_categories:
            if category in self.config.get('targets', {}):
                for service_config in self.config['targets'][category]:
                    if service_config.get('chaos_allowed', False):
                        try:
                            container = docker_client.containers.get(service_config['container'])
                            
                            target = ChaosTarget(
                                name=service_config['name'],
                                container_id=container.id,
                                criticality=service_config['criticality'],
                                chaos_allowed=service_config['chaos_allowed'],
                                max_downtime=service_config.get('max_downtime', '300s'),
                                service_type=category
                            )
                            targets.append(target)
                            
                        except docker.errors.NotFound:
                            self.logger.warning(f"Container {service_config['container']} not found")
        
        return targets
    
    async def run_experiment(self, experiment_name: str, safe_mode: bool = True) -> ExperimentResult:
        """Run a chaos experiment"""
        self.logger.info(f"Starting chaos experiment: {experiment_name}")
        
        # Load experiment configuration
        experiment_path = f"/opt/sutazaiapp/chaos/experiments/{experiment_name}.yaml"
        
        try:
            with open(experiment_path, 'r') as f:
                experiment_config = yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"Failed to load experiment {experiment_name}: {e}")
            return ExperimentResult(
                experiment_name=experiment_name,
                start_time=datetime.now(),
                end_time=datetime.now(),
                status=ExperimentStatus.FAILED,
                targets_affected=[],
                recovery_time=None,
                errors=[str(e)],
                metrics={}
            )
        
        # Initialize experiment result
        experiment_result = ExperimentResult(
            experiment_name=experiment_name,
            start_time=datetime.now(),
            end_time=None,
            status=ExperimentStatus.RUNNING,
            targets_affected=[],
            recovery_time=None,
            errors=[],
            metrics={}
        )
        
        self.current_experiment = experiment_result
        
        try:
            # Pre-experiment health check
            targets = await self.get_targets()
            initial_health = await self.health_monitor.get_system_health(targets)
            
            if safe_mode and initial_health['health_score'] < 80:
                raise Exception("System health too low for chaos experiment")
            
            # Select targets based on experiment config
            selected_targets = self._select_targets(targets, experiment_config)
            experiment_result.targets_affected = [t.name for t in selected_targets]
            
            # Execute experiment scenarios
            scenario_results = []
            for scenario in experiment_config['spec']['scenarios']:
                if self.emergency_stop:
                    break
                    
                result = await self._execute_scenario(scenario, selected_targets)
                scenario_results.append(result)
            
            # Post-experiment health check and recovery measurement
            recovery_start = time.time()
            await self._wait_for_system_recovery(targets, timeout=300)
            recovery_time = time.time() - recovery_start
            
            experiment_result.recovery_time = recovery_time
            experiment_result.status = ExperimentStatus.COMPLETED
            experiment_result.metrics = {
                'initial_health': initial_health,
                'scenarios_executed': len(scenario_results),
                'successful_scenarios': sum(1 for r in scenario_results if r.get('success', False)),
                'recovery_time': recovery_time
            }
            
        except Exception as e:
            self.logger.error(f"Experiment {experiment_name} failed: {e}")
            experiment_result.status = ExperimentStatus.FAILED
            experiment_result.errors.append(str(e))
        
        finally:
            experiment_result.end_time = datetime.now()
            self.current_experiment = None
            
            # Save experiment results
            await self._save_experiment_result(experiment_result)
        
        return experiment_result
    
    def _select_targets(self, targets: List[ChaosTarget], 
                       experiment_config: Dict[str, Any]) -> List[ChaosTarget]:
        """Select targets for experiment based on configuration"""
        target_specs = experiment_config['spec'].get('targets', [])
        selected = []
        
        for target_spec in target_specs:
            for target in targets:
                if target.name == target_spec.get('service') or \
                   target.container_id.startswith(target_spec.get('service', '')):
                    selected.append(target)
                    break
        
        # Apply safety limits
        max_targets = experiment_config['spec']['safety'].get('max_affected_services', 3)
        return selected[:max_targets]
    
    async def _execute_scenario(self, scenario: Dict[str, Any], 
                               targets: List[ChaosTarget]) -> Dict[str, Any]:
        """Execute a single chaos scenario"""
        scenario_name = scenario['name']
        probability = scenario.get('probability', 1.0)
        
        # Check if scenario should run based on probability
        if random.random() > probability:
            return {'scenario': scenario_name, 'skipped': True, 'reason': 'probability'}
        
        self.logger.info(f"Executing scenario: {scenario_name}")
        
        # Select random target
        target = random.choice(targets)
        
        # Execute actions based on scenario type
        for action in scenario.get('actions', []):
            action_type = action['type']
            
            if scenario_name.startswith('container_'):
                if action_type == 'kill':
                    return await self.container_chaos.kill_container(target)
                elif action_type == 'restart':
                    return await self.container_chaos.restart_container(target)
                elif action_type == 'pause':
                    duration = self._parse_duration(action.get('duration', '60s'))
                    return await self.container_chaos.pause_container(target, duration)
            
            elif scenario_name.startswith('network_'):
                if action_type == 'delay':
                    latency = action.get('latency', '100ms')
                    jitter = action.get('jitter', '10ms')
                    duration = self._parse_duration(action.get('duration', '5m'))
                    return await self.network_chaos.inject_latency(target, latency, jitter, duration)
                elif action_type == 'loss':
                    loss_rate = action.get('percentage', '5%')
                    duration = self._parse_duration(action.get('duration', '3m'))
                    return await self.network_chaos.inject_packet_loss(target, loss_rate, duration)
            
            elif scenario_name.startswith('cpu_'):
                cpu_percent = action.get('percentage', 80)
                duration = self._parse_duration(action.get('duration', '5m'))
                return await self.resource_chaos.stress_cpu(target, cpu_percent, duration)
            
            elif scenario_name.startswith('memory_'):
                memory_percent = action.get('percentage', 75)
                duration = self._parse_duration(action.get('duration', '3m'))
                return await self.resource_chaos.stress_memory(target, memory_percent, duration)
        
        return {'scenario': scenario_name, 'success': False, 'error': 'Unknown scenario type'}
    
    def _parse_duration(self, duration_str: str) -> int:
        """Parse duration string to seconds"""
        if duration_str.endswith('s'):
            return int(duration_str[:-1])
        elif duration_str.endswith('m'):
            return int(duration_str[:-1]) * 60
        elif duration_str.endswith('h'):
            return int(duration_str[:-1]) * 3600
        else:
            return int(duration_str)
    
    async def _wait_for_system_recovery(self, targets: List[ChaosTarget], timeout: int = 300):
        """Wait for system to recover after chaos experiment"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            health = await self.health_monitor.get_system_health(targets)
            if health['health_score'] >= 80:
                self.logger.info("System recovered successfully")
                return
                
            await asyncio.sleep(10)
        
        self.logger.warning("System did not fully recover within timeout")
    
    async def _save_experiment_result(self, result: ExperimentResult):
        """Save experiment results to file"""
        results_dir = "/opt/sutazaiapp/chaos/reports"
        os.makedirs(results_dir, exist_ok=True)
        
        timestamp = result.start_time.strftime("%Y%m%d_%H%M%S")
        filename = f"{results_dir}/experiment_{result.experiment_name}_{timestamp}.json"
        
        # Convert result to dictionary
        result_dict = {
            'experiment_name': result.experiment_name,
            'start_time': result.start_time.isoformat(),
            'end_time': result.end_time.isoformat() if result.end_time else None,
            'status': result.status.value,
            'targets_affected': result.targets_affected,
            'recovery_time': result.recovery_time,
            'errors': result.errors,
            'metrics': result.metrics
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(result_dict, f, indent=2)
            self.logger.info(f"Experiment results saved to {filename}")
        except Exception as e:
            self.logger.error(f"Failed to save experiment results: {e}")
    
    def emergency_stop_experiment(self):
        """Emergency stop current experiment"""
        self.emergency_stop = True
        self.logger.warning("Emergency stop triggered")
        
        if self.current_experiment:
            self.current_experiment.status = ExperimentStatus.EMERGENCY_STOPPED
            self.current_experiment.end_time = datetime.now()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SutazAI Chaos Engineering Engine")
    parser.add_argument("--experiment", required=True, help="Experiment name to run")
    parser.add_argument("--safe-mode", action="store_true", help="Run in safe mode")
    parser.add_argument("--config", default="/opt/sutazaiapp/chaos/config/chaos-config.yaml",
                       help="Path to configuration file")
    
    args = parser.parse_args()
    
    async def main():
        engine = ChaosEngine(args.config)
        result = await engine.run_experiment(args.experiment, args.safe_mode)
        
        print(f"Experiment completed with status: {result.status.value}")
        if result.recovery_time:
            print(f"Recovery time: {result.recovery_time:.2f} seconds")
        if result.errors:
            print(f"Errors: {result.errors}")
    
    asyncio.run(main())