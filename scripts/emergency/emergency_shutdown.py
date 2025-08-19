#!/usr/bin/env python3
"""
Emergency Shutdown Coordinator
Provides controlled emergency shutdown and disaster recovery for the SutazAI system.

This module implements:
- Multi-level severity classification
- Graceful service shutdown sequences
- Data integrity protection
- System state preservation
- Recovery coordination
"""

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/opt/sutazaiapp/logs/emergency_shutdown.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class EmergencySeverity(Enum):
    """Emergency severity levels with response priorities."""
    CRITICAL = 1  # Immediate shutdown required
    HIGH = 2      # Graceful shutdown with data preservation
    MEDIUM = 3    # Selective service shutdown
    LOW = 4       # Monitoring and alerting only


class ServicePriority(Enum):
    """Service shutdown priority ordering."""
    FRONTEND = 1      # User-facing services
    API = 2           # API services
    PROCESSING = 3    # Background processing
    MESSAGING = 4     # Message queues
    CACHE = 5        # Cache services
    DATABASE = 6     # Database services (last to shutdown)
    MONITORING = 7   # Monitoring (keeps running longest)


class EmergencyShutdownCoordinator:
    """Coordinates emergency shutdown and recovery procedures."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the emergency coordinator."""
        self.config_path = config_path or '/opt/sutazaiapp/config/emergency.json'
        self.state_file = Path('/opt/sutazaiapp/logs/emergency_state.json')
        self.docker_compose_file = '/opt/sutazaiapp/docker/docker-compose.consolidated.yml'
        self.backup_dir = Path('/opt/sutazaiapp/backups/emergency')
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Service mapping with priorities
        self.service_map = {
            ServicePriority.FRONTEND: ['sutazai-frontend', 'sutazai-streamlit'],
            ServicePriority.API: ['sutazai-backend', 'sutazai-unified-dev'],
            ServicePriority.PROCESSING: ['sutazai-ai-agent-orchestrator', 'sutazai-task-coordinator'],
            ServicePriority.MESSAGING: ['sutazai-rabbitmq'],
            ServicePriority.CACHE: ['sutazai-redis'],
            ServicePriority.DATABASE: ['sutazai-postgres', 'sutazai-neo4j'],
            ServicePriority.MONITORING: ['sutazai-prometheus', 'sutazai-grafana', 'sutazai-consul']
        }
        
        # MCP container patterns
        self.mcp_patterns = [
            'mcp-', 'claude-flow', 'ruv-swarm', 'files-mcp',
            'context7', 'extended-memory', 'ultimatecoder'
        ]
        
        # Track shutdown state
        self.shutdown_in_progress = False
        self.start_time = None
        self.shutdown_log = []
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle interrupt signals during emergency procedures."""
        logger.warning(f"Received signal {signum} during emergency procedure")
        if not self.shutdown_in_progress:
            self.initiate_emergency_shutdown(EmergencySeverity.CRITICAL, "Signal interrupt received")
        sys.exit(1)
    
    def assess_system_state(self) -> Dict:
        """Assess current system state and identify issues."""
        logger.info("Assessing system state...")
        state = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'containers': self._check_containers(),
            'services': self._check_services(),
            'resources': self._check_resources(),
            'networks': self._check_networks(),
            'volumes': self._check_volumes()
        }
        
        # Save state for recovery
        self._save_state(state)
        return state
    
    def _check_containers(self) -> Dict:
        """Check Docker container status."""
        try:
            result = subprocess.run(
                ['docker', 'ps', '-a', '--format', 'json'],
                capture_output=True, text=True, timeout=10
            )
            containers = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    containers.append(json.loads(line))
            
            running = len([c for c in containers if 'Up' in c.get('Status', '')])
            stopped = len(containers) - running
            
            # Identify problematic containers
            problematic = []
            for container in containers:
                if 'Exited' in container.get('Status', '') and '(0)' not in container.get('Status', ''):
                    problematic.append(container.get('Names', 'unknown'))
            
            return {
                'total': len(containers),
                'running': running,
                'stopped': stopped,
                'problematic': problematic,
                'details': containers[:10]  # First 10 for brevity
            }
        except Exception as e:
            logger.error(f"Failed to check containers: {e}")
            return {'error': str(e)}
    
    def _check_services(self) -> Dict:
        """Check service health via API endpoints."""
        services = {}
        
        # Check backend API
        try:
            result = subprocess.run(
                ['curl', '-s', '-o', '/dev/null', '-w', '%{http_code}', 
                 'http://localhost:10010/health'],
                capture_output=True, text=True, timeout=5
            )
            services['backend'] = {'status': result.stdout.strip(), 'healthy': result.stdout.strip() == '200'}
        except:
            services['backend'] = {'status': 'unreachable', 'healthy': False}
        
        # Check other critical services
        critical_ports = {
            'postgres': 10000,
            'redis': 10001,
            'consul': 10006,
            'prometheus': 10200
        }
        
        for service, port in critical_ports.items():
            try:
                result = subprocess.run(
                    ['nc', '-zv', 'localhost', str(port)],
                    capture_output=True, text=True, timeout=2
                )
                services[service] = {'port': port, 'listening': result.returncode == 0}
            except:
                services[service] = {'port': port, 'listening': False}
        
        return services
    
    def _check_resources(self) -> Dict:
        """Check system resource usage."""
        try:
            # Check disk usage
            df_result = subprocess.run(
                ['df', '-h', '/'],
                capture_output=True, text=True
            )
            
            # Check memory
            mem_result = subprocess.run(
                ['free', '-h'],
                capture_output=True, text=True
            )
            
            # Check Docker resources
            docker_result = subprocess.run(
                ['docker', 'system', 'df'],
                capture_output=True, text=True
            )
            
            return {
                'disk': df_result.stdout,
                'memory': mem_result.stdout,
                'docker': docker_result.stdout
            }
        except Exception as e:
            logger.error(f"Failed to check resources: {e}")
            return {'error': str(e)}
    
    def _check_networks(self) -> List[str]:
        """Check Docker networks."""
        try:
            result = subprocess.run(
                ['docker', 'network', 'ls', '--format', '{{.Name}}'],
                capture_output=True, text=True
            )
            return result.stdout.strip().split('\n')
        except:
            return []
    
    def _check_volumes(self) -> List[str]:
        """Check Docker volumes."""
        try:
            result = subprocess.run(
                ['docker', 'volume', 'ls', '--format', '{{.Name}}'],
                capture_output=True, text=True
            )
            return result.stdout.strip().split('\n')
        except:
            return []
    
    def initiate_emergency_shutdown(self, severity: EmergencySeverity, reason: str) -> bool:
        """Initiate emergency shutdown procedure."""
        logger.critical(f"EMERGENCY SHUTDOWN INITIATED - Severity: {severity.name}, Reason: {reason}")
        
        self.shutdown_in_progress = True
        self.start_time = datetime.now(timezone.utc)
        
        # Record emergency event
        self._record_event('shutdown_initiated', {
            'severity': severity.name,
            'reason': reason,
            'timestamp': self.start_time.isoformat()
        })
        
        # Execute shutdown based on severity
        if severity == EmergencySeverity.CRITICAL:
            return self._critical_shutdown()
        elif severity == EmergencySeverity.HIGH:
            return self._graceful_shutdown()
        elif severity == EmergencySeverity.MEDIUM:
            return self._selective_shutdown()
        else:
            return self._monitoring_mode()
    
    def _critical_shutdown(self) -> bool:
        """Execute immediate critical shutdown."""
        logger.critical("Executing CRITICAL shutdown - immediate termination")
        
        try:
            # Stop all containers immediately
            logger.info("Stopping all containers...")
            subprocess.run(['docker', 'stop', '-t', '5', '$(docker ps -q)'], 
                         shell=True, timeout=30)
            
            # Save critical state
            self._save_critical_state()
            
            # Clean up
            self._emergency_cleanup()
            
            self._record_event('critical_shutdown_complete', {
                'duration': (datetime.now(timezone.utc) - self.start_time).total_seconds()
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Critical shutdown failed: {e}")
            return False
    
    def _graceful_shutdown(self) -> bool:
        """Execute graceful shutdown with data preservation."""
        logger.info("Executing graceful shutdown with data preservation")
        
        try:
            # Shutdown services in priority order
            for priority in ServicePriority:
                services = self.service_map.get(priority, [])
                for service in services:
                    logger.info(f"Stopping {service} (Priority: {priority.name})")
                    self._stop_service(service)
                    time.sleep(2)  # Brief pause between services
            
            # Stop MCP containers
            self._stop_mcp_containers()
            
            # Preserve data
            self._backup_critical_data()
            
            # Final cleanup
            self._cleanup_resources()
            
            self._record_event('graceful_shutdown_complete', {
                'duration': (datetime.now(timezone.utc) - self.start_time).total_seconds()
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Graceful shutdown failed: {e}")
            return False
    
    def _selective_shutdown(self) -> bool:
        """Shutdown specific problematic services."""
        logger.info("Executing selective shutdown of problematic services")
        
        state = self.assess_system_state()
        problematic = state.get('containers', {}).get('problematic', [])
        
        for container in problematic:
            logger.info(f"Stopping problematic container: {container}")
            self._stop_service(container)
        
        return True
    
    def _monitoring_mode(self) -> bool:
        """Enter monitoring mode without shutdown."""
        logger.info("Entering monitoring mode - no shutdown required")
        
        # Just monitor and log
        state = self.assess_system_state()
        logger.info(f"System state: {json.dumps(state, indent=2)}")
        
        return True
    
    def _stop_service(self, service_name: str):
        """Stop a specific service."""
        try:
            subprocess.run(['docker', 'stop', service_name], timeout=30)
            self._record_event('service_stopped', {'service': service_name})
        except Exception as e:
            logger.error(f"Failed to stop {service_name}: {e}")
    
    def _stop_mcp_containers(self):
        """Stop all MCP containers."""
        logger.info("Stopping MCP containers...")
        
        try:
            result = subprocess.run(
                ['docker', 'ps', '--format', '{{.Names}}'],
                capture_output=True, text=True
            )
            
            for container in result.stdout.strip().split('\n'):
                if any(pattern in container.lower() for pattern in self.mcp_patterns):
                    logger.info(f"Stopping MCP container: {container}")
                    self._stop_service(container)
        except Exception as e:
            logger.error(f"Failed to stop MCP containers: {e}")
    
    def _backup_critical_data(self):
        """Backup critical data before shutdown."""
        logger.info("Backing up critical data...")
        
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        backup_path = self.backup_dir / f"emergency_backup_{timestamp}"
        backup_path.mkdir(parents=True, exist_ok=True)
        
        # Backup Docker volumes
        critical_volumes = ['postgres_data', 'redis_data', 'neo4j_data']
        for volume in critical_volumes:
            try:
                logger.info(f"Backing up volume: {volume}")
                subprocess.run([
                    'docker', 'run', '--rm',
                    '-v', f'{volume}:/data',
                    '-v', f'{backup_path}:/backup',
                    'alpine', 'tar', '-czf', f'/backup/{volume}.tar.gz', '/data'
                ], timeout=300)
            except Exception as e:
                logger.error(f"Failed to backup {volume}: {e}")
    
    def _cleanup_resources(self):
        """Clean up system resources."""
        logger.info("Cleaning up resources...")
        
        try:
            # Prune stopped containers
            subprocess.run(['docker', 'container', 'prune', '-f'], timeout=30)
            
            # Clean up networks
            subprocess.run(['docker', 'network', 'prune', '-f'], timeout=30)
            
            # Clean up dangling images
            subprocess.run(['docker', 'image', 'prune', '-f'], timeout=30)
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
    
    def _emergency_cleanup(self):
        """Emergency cleanup for critical situations."""
        logger.info("Performing emergency cleanup...")
        
        try:
            # Kill all containers
            subprocess.run(['docker', 'kill', '$(docker ps -q)'], shell=True, timeout=10)
            
            # Remove all containers
            subprocess.run(['docker', 'rm', '-f', '$(docker ps -aq)'], shell=True, timeout=30)
            
        except Exception as e:
            logger.error(f"Emergency cleanup failed: {e}")
    
    def _save_state(self, state: Dict):
        """Save system state for recovery."""
        try:
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    def _save_critical_state(self):
        """Save critical state during emergency."""
        state = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'type': 'emergency',
            'containers': self._get_container_list()
        }
        self._save_state(state)
    
    def _get_container_list(self) -> List[str]:
        """Get list of running containers."""
        try:
            result = subprocess.run(
                ['docker', 'ps', '--format', '{{.Names}}'],
                capture_output=True, text=True
            )
            return result.stdout.strip().split('\n')
        except:
            return []
    
    def _record_event(self, event_type: str, data: Dict):
        """Record emergency event."""
        event = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'type': event_type,
            'data': data
        }
        self.shutdown_log.append(event)
        logger.info(f"Event recorded: {event_type}")
    
    def initiate_recovery(self) -> bool:
        """Initiate system recovery procedure."""
        logger.info("Initiating system recovery...")
        
        try:
            # Load saved state
            if self.state_file.exists():
                with open(self.state_file, 'r') as f:
                    saved_state = json.load(f)
                logger.info(f"Loaded saved state from {saved_state['timestamp']}")
            
            # Start core services first
            logger.info("Starting core services...")
            subprocess.run([
                'docker-compose', '-f', self.docker_compose_file,
                'up', '-d', 'postgres', 'redis', 'consul'
            ], timeout=60)
            
            time.sleep(10)  # Wait for core services
            
            # Start remaining services
            logger.info("Starting all services...")
            subprocess.run([
                'docker-compose', '-f', self.docker_compose_file,
                'up', '-d'
            ], timeout=300)
            
            # Verify recovery
            time.sleep(30)
            recovery_state = self.assess_system_state()
            
            logger.info("Recovery complete")
            return True
            
        except Exception as e:
            logger.error(f"Recovery failed: {e}")
            return False
    
    def generate_report(self) -> str:
        """Generate emergency response report."""
        report = []
        report.append("=" * 60)
        report.append("EMERGENCY RESPONSE REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
        report.append("")
        
        if self.shutdown_log:
            report.append("Event Log:")
            for event in self.shutdown_log:
                report.append(f"  - {event['timestamp']}: {event['type']}")
        
        # Current state
        state = self.assess_system_state()
        report.append("")
        report.append("Current System State:")
        report.append(f"  Containers: {state.get('containers', {}).get('running', 0)} running")
        report.append(f"  Services: {json.dumps(state.get('services', {}), indent=4)}")
        
        return '\n'.join(report)


def main():
    """Main entry point for emergency shutdown coordinator."""
    parser = argparse.ArgumentParser(description='Emergency Shutdown Coordinator')
    parser.add_argument('action', choices=['assess', 'shutdown', 'recover', 'report'],
                       help='Action to perform')
    parser.add_argument('--severity', choices=['critical', 'high', 'medium', 'low'],
                       default='high', help='Shutdown severity level')
    parser.add_argument('--reason', type=str, default='Manual intervention',
                       help='Reason for emergency action')
    
    args = parser.parse_args()
    
    coordinator = EmergencyShutdownCoordinator()
    
    if args.action == 'assess':
        state = coordinator.assess_system_state()
        print(json.dumps(state, indent=2))
    
    elif args.action == 'shutdown':
        severity_map = {
            'critical': EmergencySeverity.CRITICAL,
            'high': EmergencySeverity.HIGH,
            'medium': EmergencySeverity.MEDIUM,
            'low': EmergencySeverity.LOW
        }
        severity = severity_map[args.severity]
        success = coordinator.initiate_emergency_shutdown(severity, args.reason)
        sys.exit(0 if success else 1)
    
    elif args.action == 'recover':
        success = coordinator.initiate_recovery()
        sys.exit(0 if success else 1)
    
    elif args.action == 'report':
        print(coordinator.generate_report())


if __name__ == '__main__':
    main()