#!/usr/bin/env python3
"""
Deployment State Validation and Log Analysis
"""

import json
import subprocess
import time
import os
from datetime import datetime
from typing import Dict, List, Any
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DeploymentValidator:
    """Comprehensive deployment validation and log analysis"""
    
    def __init__(self):
        self.containers = ['sutazai-postgres', 'sutazai-redis', 'sutazai-neo4j', 'sutazai-chromadb']
        self.test_results = {
            'timestamp': datetime.now().isoformat(),
            'deployment_status': {},
            'container_health': {},
            'log_analysis': {},
            'resource_usage': {},
            'overall_status': 'pending'
        }
    
    def check_deployment_status(self) -> Dict[str, Any]:
        """Check overall deployment status"""
        logger.info("Checking deployment status...")
        
        status = {}
        
        try:
            # Check Docker service status
            result = subprocess.run(['docker', 'version'], capture_output=True, text=True)
            status['docker_available'] = result.returncode == 0
            
            # Check container states
            result = subprocess.run(['docker', 'ps', '-a'], capture_output=True, text=True)
            if result.returncode == 0:
                running_containers = []
                stopped_containers = []
                
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                for line in lines:
                    if any(container in line for container in self.containers):
                        parts = line.split()
                        container_name = parts[-1]
                        if 'Up' in line:
                            running_containers.append(container_name)
                        else:
                            stopped_containers.append(container_name)
                
                status['running_containers'] = running_containers
                status['stopped_containers'] = stopped_containers
                status['total_containers'] = len(running_containers) + len(stopped_containers)
                status['expected_containers'] = len(self.containers)
                status['all_containers_running'] = len(running_containers) == len(self.containers)
            
            # Check Docker Compose status
            if os.path.exists('/opt/sutazaiapp/docker-compose.yml'):
                result = subprocess.run(['docker-compose', '-f', '/opt/sutazaiapp/docker-compose.yml', 'ps'], 
                                      capture_output=True, text=True, cwd='/opt/sutazaiapp')
                status['compose_file_exists'] = True
                status['compose_status'] = 'accessible' if result.returncode == 0 else 'error'
            else:
                status['compose_file_exists'] = False
                status['compose_status'] = 'not_found'
            
        except Exception as e:
            status['error'] = str(e)
        
        self.test_results['deployment_status'] = status
        return status
    
    def check_container_health(self) -> Dict[str, Any]:
        """Check detailed health status of each container"""
        logger.info("Checking container health...")
        
        health_status = {}
        
        for container in self.containers:
            try:
                # Get container details
                result = subprocess.run([
                    'docker', 'inspect', container,
                    '--format={{.State.Status}},{{.State.Health.Status}},{{.State.StartedAt}},{{.RestartCount}}'
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    state_info = result.stdout.strip().split(',')
                    
                    # Get resource usage
                    stats_result = subprocess.run([
                        'docker', 'stats', container, '--no-stream', '--format',
                        '{{.CPUPerc}},{{.MemUsage}},{{.NetIO}},{{.BlockIO}}'
                    ], capture_output=True, text=True)
                    
                    stats_info = stats_result.stdout.strip().split(',') if stats_result.returncode == 0 else ['N/A', 'N/A', 'N/A', 'N/A']
                    
                    health_status[container] = {
                        'status': state_info[0] if len(state_info) > 0 else 'unknown',
                        'health': state_info[1] if len(state_info) > 1 and state_info[1] != '<no value>' else 'no_healthcheck',
                        'started_at': state_info[2] if len(state_info) > 2 else 'unknown',
                        'restart_count': int(state_info[3]) if len(state_info) > 3 and state_info[3].isdigit() else 0,
                        'cpu_usage': stats_info[0] if len(stats_info) > 0 else 'N/A',
                        'memory_usage': stats_info[1] if len(stats_info) > 1 else 'N/A',
                        'network_io': stats_info[2] if len(stats_info) > 2 else 'N/A',
                        'block_io': stats_info[3] if len(stats_info) > 3 else 'N/A',
                        'accessible': True
                    }
                else:
                    health_status[container] = {
                        'status': 'not_found',
                        'health': 'not_found',
                        'accessible': False,
                        'error': result.stderr.strip()
                    }
                    
            except Exception as e:
                health_status[container] = {
                    'status': 'error',
                    'health': 'error',
                    'accessible': False,
                    'error': str(e)
                }
        
        self.test_results['container_health'] = health_status
        return health_status
    
    def analyze_logs(self) -> Dict[str, Any]:
        """Analyze container logs for issues and patterns"""
        logger.info("Analyzing container logs...")
        
        log_analysis = {}
        
        for container in self.containers:
            try:
                # Get recent logs
                result = subprocess.run([
                    'docker', 'logs', container, '--tail', '50', '--timestamps'
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    logs = result.stdout
                    analysis = self.parse_logs(logs, container)
                    log_analysis[container] = analysis
                else:
                    log_analysis[container] = {
                        'status': 'error',
                        'error': result.stderr.strip(),
                        'log_lines': 0
                    }
                    
            except Exception as e:
                log_analysis[container] = {
                    'status': 'error',
                    'error': str(e),
                    'log_lines': 0
                }
        
        self.test_results['log_analysis'] = log_analysis
        return log_analysis
    
    def parse_logs(self, logs: str, container: str) -> Dict[str, Any]:
        """Parse and analyze log content for specific patterns"""
        lines = logs.strip().split('\n')
        
        # Count different log levels
        error_count = 0
        warning_count = 0
        info_count = 0
        recent_errors = []
        recent_warnings = []
        startup_events = []
        
        # Common error/warning patterns
        error_patterns = ['ERROR', 'FATAL', 'CRITICAL', 'Exception', 'Failed', 'failed']
        warning_patterns = ['WARN', 'WARNING', 'deprecated', 'slow query']
        startup_patterns = ['started', 'ready', 'listening', 'initialized', 'accepting connections']
        
        for line in lines:
            if not line.strip():
                continue
                
            line_lower = line.lower()
            
            # Check for errors
            if any(pattern.lower() in line_lower for pattern in error_patterns):
                error_count += 1
                if len(recent_errors) < 5:  # Keep last 5 errors
                    recent_errors.append(line.strip())
            
            # Check for warnings
            elif any(pattern.lower() in line_lower for pattern in warning_patterns):
                warning_count += 1
                if len(recent_warnings) < 5:  # Keep last 5 warnings
                    recent_warnings.append(line.strip())
            
            # Check for startup events
            if any(pattern.lower() in line_lower for pattern in startup_patterns):
                startup_events.append(line.strip())
            
            # Count info messages
            if 'info' in line_lower:
                info_count += 1
        
        # Determine log health status
        if error_count > 0:
            log_status = 'errors_present'
        elif warning_count > 5:  # Allow some warnings
            log_status = 'warnings_present'
        elif len(startup_events) > 0:
            log_status = 'healthy'
        else:
            log_status = 'unknown'
        
        return {
            'status': log_status,
            'total_lines': len(lines),
            'error_count': error_count,
            'warning_count': warning_count,
            'info_count': info_count,
            'recent_errors': recent_errors,
            'recent_warnings': recent_warnings,
            'startup_events': startup_events[-5:],  # Last 5 startup events
            'last_log_time': self.extract_last_timestamp(lines)
        }
    
    def extract_last_timestamp(self, lines: List[str]) -> str:
        """Extract the timestamp from the last log line"""
        if not lines:
            return 'unknown'
        
        last_line = lines[-1].strip()
        if 'T' in last_line and 'Z' in last_line:
            # Look for ISO timestamp
            try:
                parts = last_line.split()
                if len(parts) > 0:
                    timestamp = parts[0]
                    if 'T' in timestamp:
                        return timestamp
            except:
                pass
        
        return 'unknown'
    
    def check_resource_usage(self) -> Dict[str, Any]:
        """Check system resource usage"""
        logger.info("Checking resource usage...")
        
        resources = {}
        
        try:
            # Get system memory info
            with open('/proc/meminfo', 'r') as f:
                meminfo = f.read()
            
            mem_total = None
            mem_available = None
            
            for line in meminfo.split('\n'):
                if 'MemTotal:' in line:
                    mem_total = int(line.split()[1]) * 1024  # Convert to bytes
                elif 'MemAvailable:' in line:
                    mem_available = int(line.split()[1]) * 1024  # Convert to bytes
            
            if mem_total and mem_available:
                mem_used = mem_total - mem_available
                mem_usage_percent = (mem_used / mem_total) * 100
                
                resources['memory'] = {
                    'total_bytes': mem_total,
                    'used_bytes': mem_used,
                    'available_bytes': mem_available,
                    'usage_percent': round(mem_usage_percent, 2)
                }
            
            # Get CPU load
            with open('/proc/loadavg', 'r') as f:
                loadavg = f.read().strip().split()
            
            resources['cpu'] = {
                'load_1min': float(loadavg[0]),
                'load_5min': float(loadavg[1]),
                'load_15min': float(loadavg[2])
            }
            
            # Get disk usage for /opt/sutazaiapp
            result = subprocess.run(['df', '/opt/sutazaiapp'], capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:
                    parts = lines[1].split()
                    if len(parts) >= 5:
                        resources['disk'] = {
                            'total_kb': int(parts[1]),
                            'used_kb': int(parts[2]),
                            'available_kb': int(parts[3]),
                            'usage_percent': int(parts[4].rstrip('%'))
                        }
            
        except Exception as e:
            resources['error'] = str(e)
        
        self.test_results['resource_usage'] = resources
        return resources
    
    def run_all_validations(self) -> Dict[str, Any]:
        """Run all deployment validation tests"""
        logger.info("Starting comprehensive deployment validation...")
        
        # Run all validation tests
        self.check_deployment_status()
        self.check_container_health()
        self.analyze_logs()
        self.check_resource_usage()
        
        # Determine overall status
        deployment_ok = self.test_results['deployment_status'].get('all_containers_running', False)
        
        health_issues = sum(1 for container, health in self.test_results['container_health'].items()
                           if health.get('status') != 'running' or health.get('health') not in ['healthy', 'no_healthcheck'])
        
        log_errors = sum(analysis.get('error_count', 0) for analysis in self.test_results['log_analysis'].values()
                        if isinstance(analysis, dict))
        
        if deployment_ok and health_issues == 0 and log_errors == 0:
            self.test_results['overall_status'] = 'passed'
        elif deployment_ok and health_issues == 0 and log_errors < 5:
            self.test_results['overall_status'] = 'warning'
        else:
            self.test_results['overall_status'] = 'failed'
        
        return self.test_results
    
    def generate_report(self) -> str:
        """Generate comprehensive deployment validation report"""
        report = []
        report.append("=" * 80)
        report.append("SUTAZAI DEPLOYMENT VALIDATION REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {self.test_results['timestamp']}")
        report.append(f"Overall Status: {self.test_results['overall_status'].upper()}")
        report.append("")
        
        # Deployment Status
        report.append("DEPLOYMENT STATUS:")
        report.append("-" * 40)
        deployment = self.test_results['deployment_status']
        
        docker_status = "✓" if deployment.get('docker_available') else "✗"
        report.append(f"  {docker_status} Docker Available: {deployment.get('docker_available', False)}")
        
        if 'running_containers' in deployment:
            report.append(f"  Running Containers: {len(deployment['running_containers'])}/{deployment['expected_containers']}")
            for container in deployment['running_containers']:
                report.append(f"    ✓ {container}")
            
            if deployment['stopped_containers']:
                report.append("  Stopped Containers:")
                for container in deployment['stopped_containers']:
                    report.append(f"    ✗ {container}")
        
        compose_status = "✓" if deployment.get('compose_status') == 'accessible' else "✗"
        report.append(f"  {compose_status} Docker Compose: {deployment.get('compose_status', 'unknown')}")
        report.append("")
        
        # Container Health
        report.append("CONTAINER HEALTH:")
        report.append("-" * 40)
        for container, health in self.test_results['container_health'].items():
            if health.get('accessible', False):
                health_symbol = "✓" if health.get('status') == 'running' else "✗"
                report.append(f"  {health_symbol} {container}:")
                report.append(f"    Status: {health.get('status', 'unknown')}")
                report.append(f"    Health: {health.get('health', 'unknown')}")
                report.append(f"    CPU: {health.get('cpu_usage', 'N/A')}")
                report.append(f"    Memory: {health.get('memory_usage', 'N/A')}")
                report.append(f"    Restarts: {health.get('restart_count', 0)}")
            else:
                report.append(f"  ✗ {container}: Not accessible")
                if 'error' in health:
                    report.append(f"    Error: {health['error']}")
        report.append("")
        
        # Log Analysis
        report.append("LOG ANALYSIS:")
        report.append("-" * 40)
        for container, analysis in self.test_results['log_analysis'].items():
            if isinstance(analysis, dict) and 'status' in analysis:
                status_symbol = "✓" if analysis['status'] == 'healthy' else "⚠" if analysis['status'] in ['warnings_present'] else "✗"
                report.append(f"  {status_symbol} {container}:")
                report.append(f"    Status: {analysis.get('status', 'unknown')}")
                report.append(f"    Log Lines: {analysis.get('total_lines', 0)}")
                report.append(f"    Errors: {analysis.get('error_count', 0)}")
                report.append(f"    Warnings: {analysis.get('warning_count', 0)}")
                
                if analysis.get('recent_errors'):
                    report.append("    Recent Errors:")
                    for error in analysis['recent_errors']:
                        report.append(f"      - {error[:100]}...")
        report.append("")
        
        # Resource Usage
        report.append("RESOURCE USAGE:")
        report.append("-" * 40)
        resources = self.test_results['resource_usage']
        
        if 'memory' in resources:
            mem = resources['memory']
            mem_gb_total = mem['total_bytes'] / (1024**3)
            mem_gb_used = mem['used_bytes'] / (1024**3)
            report.append(f"  Memory: {mem_gb_used:.1f}GB / {mem_gb_total:.1f}GB ({mem['usage_percent']}%)")
        
        if 'cpu' in resources:
            cpu = resources['cpu']
            report.append(f"  CPU Load: {cpu['load_1min']} (1m), {cpu['load_5min']} (5m), {cpu['load_15min']} (15m)")
        
        if 'disk' in resources:
            disk = resources['disk']
            disk_gb_total = disk['total_kb'] / (1024**2)
            disk_gb_used = disk['used_kb'] / (1024**2)
            report.append(f"  Disk: {disk_gb_used:.1f}GB / {disk_gb_total:.1f}GB ({disk['usage_percent']}%)")
        
        report.append("")
        return "\n".join(report)

def main():
    """Main execution function"""
    validator = DeploymentValidator()
    
    try:
        # Run all validation tests
        results = validator.run_all_validations()
        
        # Generate and display report
        report = validator.generate_report()
        print(report)
        
        # Save results
        os.makedirs('/opt/sutazaiapp/backend/tests/reports', exist_ok=True)
        
        # Save JSON results
        with open('/opt/sutazaiapp/backend/tests/reports/deployment_validation.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save text report
        with open('/opt/sutazaiapp/backend/tests/reports/deployment_validation_report.txt', 'w') as f:
            f.write(report)
        
        logger.info("Deployment validation results saved to /opt/sutazaiapp/backend/tests/reports/")
        
        # Exit with appropriate code
        if results['overall_status'] == 'passed':
            logger.info("All deployment validation tests passed!")
            return 0
        elif results['overall_status'] == 'warning':
            logger.warning("Deployment validation completed with warnings!")
            return 0
        else:
            logger.error("Deployment validation failed!")
            return 1
            
    except Exception as e:
        logger.error(f"Deployment validation execution failed: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())