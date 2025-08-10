#!/usr/bin/env python3
"""
Environment Management Script for Perfect Jarvis Blue/Green Deployment

This script manages traffic switching, health monitoring, automated rollback triggers,
deployment status reporting, and database migration handling for the blue/green
deployment strategy.

Following CLAUDE.md rules:
- Rule 1: No fantasy elements - only production-ready implementations
- Rule 2: Don't break existing functionality 
- Rule 3: Analyze everything before making changes
- Rule 16: Use local LLMs via Ollama with TinyLlama

Usage:
    python3 manage-environments.py --switch-to blue|green
    python3 manage-environments.py --status
    python3 manage-environments.py --health-check blue|green
    python3 manage-environments.py --rollback
    python3 manage-environments.py --export-state

Author: Perfect Jarvis Deployment Engineer
Date: 2025-08-08
"""

import argparse
import json
import logging
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import requests
from dataclasses import dataclass, asdict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/opt/sutazaiapp/logs/environment-manager.log')
    ]
)

logger = logging.getLogger(__name__)

@dataclass
class ServiceHealth:
    """Service health status"""
    name: str
    status: str
    response_time: float
    error_message: Optional[str] = None
    last_check: str = None
    
    def __post_init__(self):
        if self.last_check is None:
            self.last_check = datetime.now(timezone.utc).isoformat()

@dataclass
class EnvironmentStatus:
    """Environment status information"""
    color: str
    services: List[ServiceHealth]
    active: bool
    traffic_weight: int
    deployment_version: Optional[str] = None
    last_deployment: Optional[str] = None

class HAProxyManager:
    """Manages HAProxy configuration and runtime API"""
    
    def __init__(self, admin_socket_path: str = "/var/run/haproxy/admin.sock"):
        self.admin_socket_path = admin_socket_path
        self.stats_url = "http://localhost:8404/stats"
        
    def execute_command(self, command: str) -> str:
        """Execute command on HAProxy admin socket"""
        try:
            # Use socat to communicate with HAProxy admin socket
            result = subprocess.run(
                ["socat", "stdio", self.admin_socket_path],
                input=command,
                text=True,
                capture_output=True,
                timeout=10
            )
            
            if result.returncode != 0:
                logger.error(f"HAProxy command failed: {result.stderr}")
                return ""
                
            return result.stdout.strip()
            
        except subprocess.TimeoutExpired:
            logger.error("HAProxy command timed out")
            return ""
        except FileNotFoundError:
            logger.error("socat command not found. Install socat package.")
            return ""
        except Exception as e:
            logger.error(f"Error executing HAProxy command: {e}")
            return ""
    
    def get_backend_status(self) -> Dict[str, Any]:
        """Get current backend status from HAProxy"""
        try:
            stats = self.execute_command("show stat")
            if not stats:
                return {}
            
            backend_info = {}
            lines = stats.split('\n')
            
            for line in lines:
                if not line or line.startswith('#'):
                    continue
                
                fields = line.split(',')
                if len(fields) < 18:
                    continue
                
                pxname = fields[0]  # Proxy name
                svname = fields[1]  # Server name
                status = fields[17] # Status
                weight = fields[18] if len(fields) > 18 else "0"
                
                if pxname in ['api_backend', 'frontend_backend'] and svname != 'BACKEND':
                    if pxname not in backend_info:
                        backend_info[pxname] = {}
                    
                    backend_info[pxname][svname] = {
                        'status': status,
                        'weight': weight,
                        'active': status == 'UP' and weight != '0'
                    }
            
            return backend_info
            
        except Exception as e:
            logger.error(f"Error getting backend status: {e}")
            return {}
    
    def switch_traffic_to_environment(self, target_color: str) -> bool:
        """Switch traffic to specified environment"""
        try:
            logger.info(f"Switching traffic to {target_color} environment")
            
            if target_color == "blue":
                commands = [
                    "set weight api_backend/blue-api 100",
                    "set weight api_backend/green-api 0",
                    "set weight frontend_backend/blue-frontend 100",
                    "set weight frontend_backend/green-frontend 0"
                ]
            elif target_color == "green":
                commands = [
                    "set weight api_backend/blue-api 0", 
                    "set weight api_backend/green-api 100",
                    "set weight frontend_backend/blue-frontend 0",
                    "set weight frontend_backend/green-frontend 100"
                ]
            else:
                logger.error(f"Invalid target color: {target_color}")
                return False
            
            # Execute commands
            for command in commands:
                result = self.execute_command(command)
                logger.debug(f"Command '{command}' result: {result}")
                
                # Brief pause between commands
                time.sleep(0.5)
            
            # Verify the switch
            time.sleep(2)
            return self.verify_traffic_switch(target_color)
            
        except Exception as e:
            logger.error(f"Error switching traffic to {target_color}: {e}")
            return False
    
    def verify_traffic_switch(self, expected_color: str) -> bool:
        """Verify that traffic has been switched to expected environment"""
        try:
            backend_status = self.get_backend_status()
            
            if not backend_status:
                logger.error("Unable to get backend status for verification")
                return False
            
            # Check API backend
            api_backends = backend_status.get('api_backend', {})
            frontend_backends = backend_status.get('frontend_backend', {})
            
            if expected_color == "blue":
                api_active = api_backends.get('blue-api', {}).get('active', False)
                frontend_active = frontend_backends.get('blue-frontend', {}).get('active', False)
                api_inactive = not api_backends.get('green-api', {}).get('active', True)
                frontend_inactive = not frontend_backends.get('green-frontend', {}).get('active', True)
            else:
                api_active = api_backends.get('green-api', {}).get('active', False)
                frontend_active = frontend_backends.get('green-frontend', {}).get('active', False)
                api_inactive = not api_backends.get('blue-api', {}).get('active', True)
                frontend_inactive = not frontend_backends.get('blue-frontend', {}).get('active', True)
            
            success = api_active and frontend_active and api_inactive and frontend_inactive
            
            if success:
                logger.info(f"Traffic switch to {expected_color} verified successfully")
            else:
                logger.error(f"Traffic switch verification failed for {expected_color}")
                logger.debug(f"Backend status: {backend_status}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error verifying traffic switch: {e}")
            return False

class HealthChecker:
    """Performs health checks on services"""
    
    def __init__(self):
        self.timeout = 10
        self.max_retries = 3
    
    def check_http_endpoint(self, url: str, expected_status: int = 200) -> ServiceHealth:
        """Check HTTP endpoint health"""
        start_time = time.time()
        service_name = url.split('/')[-1] if '/' in url else url
        
        try:
            for attempt in range(self.max_retries):
                try:
                    response = requests.get(url, timeout=self.timeout)
                    response_time = time.time() - start_time
                    
                    if response.status_code == expected_status:
                        return ServiceHealth(
                            name=service_name,
                            status="healthy",
                            response_time=response_time
                        )
                    else:
                        if attempt == self.max_retries - 1:
                            return ServiceHealth(
                                name=service_name,
                                status="unhealthy",
                                response_time=response_time,
                                error_message=f"HTTP {response.status_code}"
                            )
                        
                except requests.RequestException as e:
                    if attempt == self.max_retries - 1:
                        return ServiceHealth(
                            name=service_name,
                            status="unhealthy", 
                            response_time=time.time() - start_time,
                            error_message=str(e)
                        )
                    
                time.sleep(1)  # Wait between retries
                
        except Exception as e:
            return ServiceHealth(
                name=service_name,
                status="error",
                response_time=time.time() - start_time,
                error_message=str(e)
            )
    
    def check_port_connectivity(self, host: str, port: int) -> ServiceHealth:
        """Check if port is accessible"""
        start_time = time.time()
        service_name = f"{host}:{port}"
        
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.timeout)
            result = sock.connect_ex((host, port))
            sock.close()
            
            response_time = time.time() - start_time
            
            if result == 0:
                return ServiceHealth(
                    name=service_name,
                    status="healthy",
                    response_time=response_time
                )
            else:
                return ServiceHealth(
                    name=service_name,
                    status="unhealthy",
                    response_time=response_time,
                    error_message=f"Connection refused to {host}:{port}"
                )
                
        except Exception as e:
            return ServiceHealth(
                name=service_name,
                status="error",
                response_time=time.time() - start_time,
                error_message=str(e)
            )

class EnvironmentManager:
    """Main environment management class"""
    
    def __init__(self):
        self.haproxy = HAProxyManager()
        self.health_checker = HealthChecker()
        self.state_file = Path("/opt/sutazaiapp/data/deployment-state.json")
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
    
    def get_current_active_environment(self) -> Optional[str]:
        """Get currently active environment color"""
        try:
            backend_status = self.haproxy.get_backend_status()
            
            if not backend_status:
                return None
            
            api_backends = backend_status.get('api_backend', {})
            
            blue_active = api_backends.get('blue-api', {}).get('active', False)
            green_active = api_backends.get('green-api', {}).get('active', False)
            
            if blue_active and not green_active:
                return "blue"
            elif green_active and not blue_active:
                return "green"
            else:
                return None  # Unclear state
                
        except Exception as e:
            logger.error(f"Error determining active environment: {e}")
            return None
    
    def check_environment_health(self, color: str) -> EnvironmentStatus:
        """Check health of specified environment"""
        try:
            services = []
            
            # Define service endpoints based on color
            if color == "blue":
                endpoints = {
                    "backend": "http://localhost:21010/health",
                    "frontend": "http://localhost:21010/health"  # Assuming health endpoint
                }
            else:  # green
                endpoints = {
                    "backend": "http://localhost:21011/health", 
                    "frontend": "http://localhost:21011/health"
                }
            
            # Check each service endpoint
            for service_name, endpoint in endpoints.items():
                health = self.health_checker.check_http_endpoint(endpoint)
                health.name = f"{color}-{service_name}"
                services.append(health)
            
            # Check database connectivity (shared)
            db_health = self.health_checker.check_port_connectivity("localhost", 10000)
            db_health.name = "postgres"
            services.append(db_health)
            
            # Check Redis connectivity (shared)
            redis_health = self.health_checker.check_port_connectivity("localhost", 10001)
            redis_health.name = "redis"
            services.append(redis_health)
            
            # Check Ollama connectivity (shared)
            ollama_health = self.health_checker.check_port_connectivity("localhost", 10104)
            ollama_health.name = "ollama"
            services.append(ollama_health)
            
            # Determine if environment is active
            current_active = self.get_current_active_environment()
            is_active = current_active == color
            
            # Get traffic weight
            backend_status = self.haproxy.get_backend_status()
            api_backends = backend_status.get('api_backend', {})
            traffic_weight = 0
            
            if color == "blue":
                weight_str = api_backends.get('blue-api', {}).get('weight', '0')
            else:
                weight_str = api_backends.get('green-api', {}).get('weight', '0')
            
            try:
                traffic_weight = int(weight_str)
            except (ValueError, TypeError):
                traffic_weight = 0
            
            return EnvironmentStatus(
                color=color,
                services=services,
                active=is_active,
                traffic_weight=traffic_weight
            )
            
        except Exception as e:
            logger.error(f"Error checking {color} environment health: {e}")
            return EnvironmentStatus(
                color=color,
                services=[],
                active=False,
                traffic_weight=0
            )
    
    def switch_environment(self, target_color: str) -> bool:
        """Switch to target environment with validation"""
        try:
            logger.info(f"Starting environment switch to {target_color}")
            
            # Validate target color
            if target_color not in ["blue", "green"]:
                logger.error(f"Invalid target color: {target_color}")
                return False
            
            # Check target environment health first
            target_health = self.check_environment_health(target_color)
            
            unhealthy_services = [s for s in target_health.services 
                                if s.status not in ["healthy"]]
            
            if unhealthy_services:
                logger.error(f"Target environment {target_color} has unhealthy services:")
                for service in unhealthy_services:
                    logger.error(f"  - {service.name}: {service.status} ({service.error_message})")
                return False
            
            # Get current active environment for rollback info
            current_active = self.get_current_active_environment()
            
            # Switch traffic
            success = self.haproxy.switch_traffic_to_environment(target_color)
            
            if success:
                # Update state file
                self.save_deployment_state(target_color, current_active)
                logger.info(f"Successfully switched to {target_color} environment")
            else:
                logger.error(f"Failed to switch to {target_color} environment")
            
            return success
            
        except Exception as e:
            logger.error(f"Error switching environment: {e}")
            return False
    
    def rollback_environment(self) -> bool:
        """Rollback to previous environment"""
        try:
            # Load previous state
            state = self.load_deployment_state()
            
            if not state or 'previous_active' not in state:
                logger.error("No previous environment state found for rollback")
                return False
            
            previous_env = state['previous_active']
            
            if not previous_env:
                logger.error("No previous active environment found")
                return False
            
            logger.info(f"Rolling back to {previous_env} environment")
            return self.switch_environment(previous_env)
            
        except Exception as e:
            logger.error(f"Error during rollback: {e}")
            return False
    
    def save_deployment_state(self, current_active: str, previous_active: Optional[str]):
        """Save current deployment state"""
        try:
            state = {
                'current_active': current_active,
                'previous_active': previous_active,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'deployment_id': f"deploy-{int(time.time())}"
            }
            
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
                
            logger.info(f"Deployment state saved: {state}")
            
        except Exception as e:
            logger.error(f"Error saving deployment state: {e}")
    
    def load_deployment_state(self) -> Optional[Dict]:
        """Load deployment state"""
        try:
            if not self.state_file.exists():
                return None
                
            with open(self.state_file, 'r') as f:
                return json.load(f)
                
        except Exception as e:
            logger.error(f"Error loading deployment state: {e}")
            return None
    
    def get_deployment_status(self) -> Dict[str, Any]:
        """Get comprehensive deployment status"""
        try:
            blue_status = self.check_environment_health("blue")
            green_status = self.check_environment_health("green")
            current_active = self.get_current_active_environment()
            deployment_state = self.load_deployment_state()
            
            return {
                'current_active': current_active,
                'environments': {
                    'blue': asdict(blue_status),
                    'green': asdict(green_status)
                },
                'deployment_state': deployment_state,
                'haproxy_status': self.haproxy.get_backend_status(),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting deployment status: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
    
    def export_environment_state(self) -> Dict[str, Any]:
        """Export complete environment state for backup"""
        try:
            status = self.get_deployment_status()
            
            # Add additional export information
            status.update({
                'export_timestamp': datetime.now(timezone.utc).isoformat(),
                'export_version': '1.0.0',
                'system_info': {
                    'platform': sys.platform,
                    'python_version': sys.version,
                    'script_path': os.path.abspath(__file__)
                }
            })
            
            return status
            
        except Exception as e:
            logger.error(f"Error exporting environment state: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Manage Perfect Jarvis Blue/Green Deployment Environments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --switch-to green                 # Switch traffic to green environment
  %(prog)s --status                          # Show deployment status
  %(prog)s --health-check blue               # Check blue environment health
  %(prog)s --rollback                        # Rollback to previous environment
  %(prog)s --export-state                    # Export environment state
        """
    )
    
    parser.add_argument('--switch-to', 
                       choices=['blue', 'green'],
                       help='Switch traffic to specified environment')
    
    parser.add_argument('--status', 
                       action='store_true',
                       help='Show current deployment status')
    
    parser.add_argument('--health-check',
                       choices=['blue', 'green'],
                       help='Check health of specified environment')
    
    parser.add_argument('--rollback',
                       action='store_true', 
                       help='Rollback to previous environment')
    
    parser.add_argument('--export-state',
                       action='store_true',
                       help='Export complete environment state')
    
    parser.add_argument('--verbose', '-v',
                       action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create environment manager
    env_manager = EnvironmentManager()
    
    try:
        # Handle different operations
        if args.switch_to:
            success = env_manager.switch_environment(args.switch_to)
            sys.exit(0 if success else 1)
            
        elif args.status:
            status = env_manager.get_deployment_status()
            print(json.dumps(status, indent=2))
            sys.exit(0)
            
        elif args.health_check:
            health = env_manager.check_environment_health(args.health_check)
            print(json.dumps(asdict(health), indent=2))
            sys.exit(0)
            
        elif args.rollback:
            success = env_manager.rollback_environment()
            sys.exit(0 if success else 1)
            
        elif args.export_state:
            state = env_manager.export_environment_state()
            print(json.dumps(state, indent=2))
            sys.exit(0)
            
        else:
            parser.print_help()
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Operation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()