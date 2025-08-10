#!/usr/bin/env python3
"""
Resource Enforcement Script for Sutazai 69-Agent System
Implements automated resource limit enforcement and monitoring
"""

import docker
import json
import time
import logging
import subprocess
import yaml
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/opt/sutazaiapp/logs/resource-enforcer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ResourceEnforcer:
    """Enforces resource limits and manages container resource allocation"""
    
    def __init__(self, config_path: str = "/opt/sutazaiapp/config/agent-resource-allocation.yml"):
        self.client = docker.from_env()
        self.config_path = config_path
        self.load_configuration()
        
        # System constraints
        self.max_cpu_cores = 10.0  # Reserve 2 cores for system
        self.max_memory_gb = 25.0  # Reserve 4GB for system
        self.violations = []
        
    def load_configuration(self):
        """Load resource allocation configuration"""
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {self.config_path}")
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Invalid YAML configuration: {e}")
            raise
    
    def get_container_tier(self, container_name: str) -> str:
        """Determine container tier based on agent name"""
        # Tier 1: Critical Core Agents
        tier1_agents = [
            'ai-system-architect', 'hardware-resource-optimizer', 
            'agent-orchestrator', 'deep-learning-brain-architect',
            'ollama-integration-specialist'
        ]
        
        # Tier 2: Performance Agents
        tier2_agents = [
            'ai-senior-backend-developer', 'ai-senior-frontend-developer',
            'cicd-pipeline-orchestrator', 'ai-qa-team-lead',
            'deployment-automation-master'
        ]
        
        # Check agent name against tiers
        for agent in tier1_agents:
            if agent in container_name:
                return 'critical'
                
        for agent in tier2_agents:
            if agent in container_name:
                return 'performance'
                
        return 'specialized'  # Default to Tier 3
        
    def get_resource_limits_for_tier(self, tier: str) -> Dict[str, str]:
        """Get resource limits based on tier"""
        tier_mapping = {
            'critical': 'critical_pool',
            'performance': 'performance_pool', 
            'specialized': 'specialized_pool'
        }
        
        pool_name = tier_mapping.get(tier, 'specialized_pool')
        pool_config = self.config['resource_pools'][pool_name]
        
        return {
            'cpu': pool_config['agent_limits']['cpu'],
            'memory': pool_config['agent_limits']['memory'],
            'storage': pool_config['agent_limits']['storage']
        }
    
    def check_container_compliance(self, container) -> List[Dict]:
        """Check if container meets resource compliance requirements"""
        violations = []
        container_name = container.name
        
        try:
            # Get container configuration
            config = self.client.api.inspect_container(container.id)
            host_config = config['HostConfig']
            
            # Determine expected limits based on tier
            tier = self.get_container_tier(container_name)
            expected_limits = self.get_resource_limits_for_tier(tier)
            
            # Check CPU limits
            cpu_quota = host_config.get('CpuQuota', 0)
            cpu_period = host_config.get('CpuPeriod', 100000)
            
            if cpu_quota == 0:
                violations.append({
                    'container': container_name,
                    'violation_type': 'missing_cpu_limit',
                    'severity': 'high',
                    'expected': expected_limits['cpu'],
                    'actual': 'unlimited',
                    'timestamp': datetime.now().isoformat()
                })
            else:
                actual_cpu = cpu_quota / cpu_period
                expected_cpu = float(expected_limits['cpu'])
                
                if actual_cpu > expected_cpu * 1.1:  # 10% tolerance
                    violations.append({
                        'container': container_name,
                        'violation_type': 'cpu_limit_exceeded',
                        'severity': 'medium',
                        'expected': expected_limits['cpu'],
                        'actual': str(actual_cpu),
                        'timestamp': datetime.now().isoformat()
                    })
            
            # Check memory limits
            memory_limit = host_config.get('Memory', 0)
            
            if memory_limit == 0:
                violations.append({
                    'container': container_name,
                    'violation_type': 'missing_memory_limit',
                    'severity': 'high',
                    'expected': expected_limits['memory'],
                    'actual': 'unlimited',
                    'timestamp': datetime.now().isoformat()
                })
            else:
                actual_memory_gb = memory_limit / (1024**3)
                expected_memory_gb = float(expected_limits['memory'].replace('Gi', ''))
                
                if actual_memory_gb > expected_memory_gb * 1.1:  # 10% tolerance
                    violations.append({
                        'container': container_name,
                        'violation_type': 'memory_limit_exceeded',
                        'severity': 'medium',
                        'expected': expected_limits['memory'],
                        'actual': f"{actual_memory_gb:.2f}Gi",
                        'timestamp': datetime.now().isoformat()
                    })
            
        except Exception as e:
            logger.error(f"Error checking container {container_name}: {e}")
            violations.append({
                'container': container_name,
                'violation_type': 'inspection_error',
                'severity': 'low',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
        
        return violations
    
    def enforce_resource_limits(self, container_name: str, violation: Dict):
        """Automatically enforce resource limits for non-compliant containers"""
        try:
            tier = self.get_container_tier(container_name)
            expected_limits = self.get_resource_limits_for_tier(tier)
            
            logger.info(f"Enforcing resource limits for {container_name} (tier: {tier})")
            
            if violation['violation_type'] in ['missing_cpu_limit', 'cpu_limit_exceeded']:
                # Apply CPU limit using docker update
                cpu_limit = expected_limits['cpu']
                cmd = f"docker update --cpus='{cpu_limit}' {container_name}"
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                
                if result.returncode == 0:
                    logger.info(f"Applied CPU limit {cpu_limit} to {container_name}")
                else:
                    logger.error(f"Failed to apply CPU limit: {result.stderr}")
            
            if violation['violation_type'] in ['missing_memory_limit', 'memory_limit_exceeded']:
                # Apply memory limit using docker update
                memory_limit = expected_limits['memory']
                cmd = f"docker update --memory='{memory_limit}' {container_name}"
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                
                if result.returncode == 0:
                    logger.info(f"Applied memory limit {memory_limit} to {container_name}")
                else:
                    logger.error(f"Failed to apply memory limit: {result.stderr}")
                    
        except Exception as e:
            logger.error(f"Error enforcing limits for {container_name}: {e}")
    
    def calculate_system_utilization(self) -> Dict[str, float]:
        """Calculate current system resource utilization"""
        try:
            # Get system stats
            result = subprocess.run(['docker', 'stats', '--no-stream', '--format', 
                                   'table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}'], 
                                   capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Failed to get docker stats: {result.stderr}")
                return {'cpu_usage': 0.0, 'memory_usage': 0.0}
            
            lines = result.stdout.strip().split('\n')[1:]  # Skip header
            total_cpu = 0.0
            total_memory_used = 0.0
            
            for line in lines:
                if 'sutazai' in line.lower():
                    parts = line.split('\t')
                    if len(parts) >= 3:
                        # Parse CPU percentage
                        cpu_str = parts[1].replace('%', '')
                        if cpu_str != '--':
                            total_cpu += float(cpu_str)
                        
                        # Parse memory usage (format: "used / limit")
                        memory_str = parts[2].split('/')[0].strip()
                        if 'GiB' in memory_str:
                            memory_val = float(memory_str.replace('GiB', ''))
                            total_memory_used += memory_val
                        elif 'MiB' in memory_str:
                            memory_val = float(memory_str.replace('MiB', '')) / 1024
                            total_memory_used += memory_val
            
            return {
                'cpu_usage': total_cpu,
                'memory_usage': total_memory_used,
                'cpu_percentage': (total_cpu / (self.max_cpu_cores * 100)) * 100,
                'memory_percentage': (total_memory_used / self.max_memory_gb) * 100
            }
            
        except Exception as e:
            logger.error(f"Error calculating system utilization: {e}")
            return {'cpu_usage': 0.0, 'memory_usage': 0.0}
    
    def generate_compliance_report(self) -> Dict:
        """Generate comprehensive compliance report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'system_utilization': self.calculate_system_utilization(),
            'total_violations': len(self.violations),
            'violations_by_severity': {
                'high': len([v for v in self.violations if v['severity'] == 'high']),
                'medium': len([v for v in self.violations if v['severity'] == 'medium']),
                'low': len([v for v in self.violations if v['severity'] == 'low'])
            },
            'violations': self.violations,
            'enforcement_actions': []
        }
        
        return report
    
    def run_enforcement_cycle(self, auto_fix: bool = True):
        """Run a complete enforcement cycle"""
        logger.info("Starting resource enforcement cycle")
        
        # Reset violations list
        self.violations = []
        
        # Get all sutazai containers
        containers = self.client.containers.list(filters={'name': 'sutazai'})
        logger.info(f"Found {len(containers)} Sutazai containers")
        
        # Check compliance for each container
        for container in containers:
            violations = self.check_container_compliance(container)
            self.violations.extend(violations)
            
            # Auto-fix violations if enabled
            if auto_fix:
                for violation in violations:
                    if violation['severity'] in ['high', 'medium']:
                        self.enforce_resource_limits(container.name, violation)
        
        # Generate and save report
        report = self.generate_compliance_report()
        
        # Save report to file
        report_file = f"/opt/sutazaiapp/reports/resource_enforcement_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Enforcement cycle completed. Report saved to {report_file}")
        logger.info(f"Total violations: {len(self.violations)}")
        
        # Alert on critical system utilization
        sys_util = report['system_utilization']
        if sys_util.get('cpu_percentage', 0) > 85:
            logger.warning(f"Critical CPU utilization: {sys_util['cpu_percentage']:.1f}%")
        if sys_util.get('memory_percentage', 0) > 90:
            logger.warning(f"Critical memory utilization: {sys_util['memory_percentage']:.1f}%")
        
        return report

def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Sutazai Resource Enforcer')
    parser.add_argument('--config', default='/opt/sutazaiapp/config/agent-resource-allocation.yml',
                       help='Path to resource allocation configuration')
    parser.add_argument('--no-auto-fix', action='store_true',
                       help='Disable automatic violation fixing')
    parser.add_argument('--continuous', action='store_true',
                       help='Run in continuous monitoring mode')
    parser.add_argument('--interval', type=int, default=300,
                       help='Monitoring interval in seconds (default: 300)')
    
    args = parser.parse_args()
    
    # Initialize enforcer
    enforcer = ResourceEnforcer(args.config)
    
    if args.continuous:
        logger.info(f"Starting continuous monitoring (interval: {args.interval}s)")
        try:
            while True:
                enforcer.run_enforcement_cycle(auto_fix=not args.no_auto_fix)
                time.sleep(args.interval)
        except KeyboardInterrupt:
            logger.info("Continuous monitoring stopped")
    else:
        # Single enforcement cycle
        report = enforcer.run_enforcement_cycle(auto_fix=not args.no_auto_fix)
        print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()