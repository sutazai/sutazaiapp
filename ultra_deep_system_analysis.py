#!/usr/bin/env python3
"""
ULTRA-DEEP SYSTEM ANALYSIS & REAL-TIME MONITORING
Agent: ARCH-001 (Master System Architect)
Purpose: Continuous system analysis and 200-agent coordination monitoring
"""

import logging

# Configure logger for exception handling
logger = logging.getLogger(__name__)

import json
import subprocess
import time
import os
import sys
from datetime import datetime
from typing import Dict, List, Any, Tuple
import requests
from pathlib import Path
import re
import shlex

class UltraSystemAnalyzer:
    """Ultra-comprehensive system analysis and monitoring"""
    
    def __init__(self):
        self.timestamp = datetime.now().isoformat()
        self.results = {
            "timestamp": self.timestamp,
            "containers": {},
            "services": {},
            "code_quality": {},
            "security": {},
            "infrastructure": {},
            "agent_progress": {}
        }
        
    def analyze_containers(self) -> Dict[str, Any]:
        """Analyze Docker container status"""
        try:
            # Get all containers - SECURE: Using parameterized command
            cmd = ["docker", "ps", "-a", "--format", "{{json .}}"]
            result = subprocess.run(cmd, shell=False, capture_output=True, text=True)
            containers = []
            
            for line in result.stdout.strip().split('\n'):
                if line:
                    containers.append(json.loads(line))
            
            # Count running vs total
            running = [c for c in containers if 'Up' in c.get('Status', '')]
            
            self.results['containers'] = {
                'total': len(containers),
                'running': len(running),
                'stopped': len(containers) - len(running),
                'health_status': {}
            }
            
            # Check health status
            for container in running:
                name = container.get('Names', 'unknown')
                status = container.get('Status', 'unknown')
                if 'healthy' in status.lower():
                    self.results['containers']['health_status'][name] = 'healthy'
                elif 'unhealthy' in status.lower():
                    self.results['containers']['health_status'][name] = 'unhealthy'
                else:
                    self.results['containers']['health_status'][name] = 'no_health_check'
                    
            return self.results['containers']
            
        except Exception as e:
            return {"error": str(e)}
    
    def check_service_endpoints(self) -> Dict[str, Any]:
        """Check all service endpoints"""
        endpoints = {
            'backend': 'http://localhost:10010/health',
            'frontend': 'http://localhost:10011',
            'ollama': 'http://localhost:10104/api/tags',
            'prometheus': 'http://localhost:10200/-/healthy',
            'grafana': 'http://localhost:10201/api/health',
            'rabbitmq': 'http://localhost:10008/api/overview',
            'neo4j': 'http://localhost:10002',
            'postgres': 'localhost:10000',
            'redis': 'localhost:10001'
        }
        
        for service, url in endpoints.items():
            try:
                if service in ['postgres', 'redis']:
                    # Use netcat for TCP services - SECURE: Validated parameters
                    host, port = url.split(':')
                    # Input validation to prevent command injection
                    if not re.match(r'^[a-zA-Z0-9.-]+$', host) or not re.match(r'^[0-9]+$', port):
                        self.results['services'][service] = {'status': 'invalid_endpoint'}
                        continue
                    cmd = ["nc", "-zv", host, port]
                    result = subprocess.run(cmd, shell=False, capture_output=True, text=True, timeout=2, stderr=subprocess.STDOUT)
                    self.results['services'][service] = {
                        'status': 'reachable' if result.returncode == 0 else 'unreachable'
                    }
                else:
                    # HTTP services
                    response = requests.get(url, timeout=2, auth=('guest', 'guest') if service == 'rabbitmq' else None)
                    self.results['services'][service] = {
                        'status': 'healthy' if response.status_code == 200 else 'unhealthy',
                        'status_code': response.status_code
                    }
            except Exception as e:
                # TODO: Review this exception handling
                logger.error(f"Unexpected exception: {e}", exc_info=True)
                self.results['services'][service] = {'status': 'unreachable'}
                
        return self.results['services']
    
    def analyze_code_quality(self) -> Dict[str, Any]:
        """Analyze code quality metrics"""
        metrics = {
            'python_files': 0,
            'total_lines': 0,
            'hardcoded_credentials': [],
            'fantasy_elements': [],
            'unused_imports': [],
            'bare_excepts': [],
            'duplicate_patterns': []
        }
        
        # Count Python files
        python_files = list(Path('/opt/sutazaiapp').rglob('*.py'))
        metrics['python_files'] = len(python_files)
        
        # Analyze patterns
        patterns = {
            'hardcoded_credentials': [
                r'password\s*=\s*["\'][^"\']+["\']',
                r'api_key\s*=\s*["\'][^"\']+["\']',
                r'secret\s*=\s*["\'][^"\']+["\']'
            ],
            'fantasy_elements': [
                r'\bquantum\b', r'\bAGI\b', r'\bASI\b', 
                r'\btelepathy\b', r'\bconsciousness\b',
                r'\bmagic\b', r'\bwizard\b', r'\bteleport\b'
            ],
            'bare_excepts': [r'except\s*:'],
            'unused_imports': [r'^import\s+\w+$', r'^from\s+\w+\s+import\s+\w+$']
        }
        
        for file_path in python_files[:100]:  # Sample first 100 files
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                    lines = content.split('\n')
                    metrics['total_lines'] += len(lines)
                    
                    # Check patterns
                    for pattern_type, pattern_list in patterns.items():
                        for pattern in pattern_list:
                            matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
                            if matches and pattern_type in metrics:
                                metrics[pattern_type].extend([
                                    f"{file_path}:{i+1}" 
                                    for i, line in enumerate(lines) 
                                    if re.search(pattern, line, re.IGNORECASE)
                                ])
            except Exception as e:
                # Suppressed exception (was bare except)
                logger.debug(f"Suppressed exception: {e}")
                pass
        
        # Summarize findings
        self.results['code_quality'] = {
            'files_analyzed': len(python_files[:100]),
            'total_files': metrics['python_files'],
            'total_lines': metrics['total_lines'],
            'critical_issues': {
                'hardcoded_credentials': len(metrics['hardcoded_credentials']),
                'fantasy_elements': len(metrics['fantasy_elements']),
                'bare_excepts': len(metrics['bare_excepts'])
            },
            'compliance_score': self.calculate_compliance_score(metrics)
        }
        
        return self.results['code_quality']
    
    def calculate_compliance_score(self, metrics: Dict) -> float:
        """Calculate overall compliance score"""
        issues = sum([
            len(metrics.get('hardcoded_credentials', [])),
            len(metrics.get('fantasy_elements', [])),
            len(metrics.get('bare_excepts', [])),
            len(metrics.get('unused_imports', [])) / 10  # Weight unused imports less
        ])
        
        total_lines = max(metrics.get('total_lines', 1), 1)
        score = max(0, 100 - (issues / total_lines * 1000))
        return round(score, 2)
    
    def check_security_status(self) -> Dict[str, Any]:
        """Check security configuration"""
        security_checks = {
            'root_containers': 0,
            'exposed_ports': [],
            'ssl_enabled': False,
            'credentials_in_env': False,
            'security_scanning': False
        }
        
        # Check for root containers - SECURE: Individual container checks
        try:
            # First get container names securely
            cmd = ["docker", "ps", "--format", "{{.Names}}"]
            result = subprocess.run(cmd, shell=False, capture_output=True, text=True)
            container_names = [name.strip() for name in result.stdout.strip().split('\n') if name.strip()]
            
            root_count = 0
            for container_name in container_names:
                # Validate container name to prevent injection
                if re.match(r'^[a-zA-Z0-9_.-]+$', container_name):
                    try:
                        user_cmd = ["docker", "exec", container_name, "whoami"]
                        user_result = subprocess.run(user_cmd, shell=False, capture_output=True, text=True, timeout=5)
                        if user_result.stdout.strip() == 'root':
                            root_count += 1
                    except Exception as e:
                        # TODO: Review this exception handling
                        logger.error(f"Unexpected exception: {e}", exc_info=True)
                        pass  # Skip containers that can't be queried
            security_checks['root_containers'] = root_count
        except Exception as e:
            # Suppressed exception (was bare except)
            logger.debug(f"Suppressed exception: {e}")
            pass
        
        # Check for .env file
        env_file = Path('/opt/sutazaiapp/.env')
        security_checks['credentials_in_env'] = env_file.exists()
        
        # Check for SSL certificates
        ssl_dir = Path('/opt/sutazaiapp/config/ssl')
        security_checks['ssl_enabled'] = ssl_dir.exists() and any(ssl_dir.glob('*.crt'))
        
        self.results['security'] = security_checks
        return self.results['security']
    
    def monitor_agent_progress(self) -> Dict[str, Any]:
        """Monitor 200-agent coordination progress"""
        # This would connect to RabbitMQ to get real agent progress
        # For now, we'll create a template structure
        phases = {
            'phase_1_security': {
                'agents': list(range(7, 36)),
                'status': 'pending',
                'progress': 0,
                'tasks_completed': 0,
                'tasks_total': 29
            },
            'phase_2_organization': {
                'agents': list(range(36, 76)),
                'status': 'pending',
                'progress': 0,
                'tasks_completed': 0,
                'tasks_total': 40
            },
            'phase_3_quality': {
                'agents': list(range(76, 136)),
                'status': 'pending',
                'progress': 0,
                'tasks_completed': 0,
                'tasks_total': 60
            },
            'phase_4_architecture': {
                'agents': list(range(136, 176)),
                'status': 'pending',
                'progress': 0,
                'tasks_completed': 0,
                'tasks_total': 40
            },
            'phase_5_testing': {
                'agents': list(range(176, 196)),
                'status': 'pending',
                'progress': 0,
                'tasks_completed': 0,
                'tasks_total': 20
            },
            'phase_6_validation': {
                'agents': list(range(196, 201)),
                'status': 'pending',
                'progress': 0,
                'tasks_completed': 0,
                'tasks_total': 5
            }
        }
        
        self.results['agent_progress'] = phases
        return self.results['agent_progress']
    
    def generate_report(self) -> str:
        """Generate comprehensive analysis report"""
        report = f"""
╔══════════════════════════════════════════════════════════════════════╗
║           ULTRA-DEEP SYSTEM ANALYSIS REPORT                         ║
║           Timestamp: {self.timestamp}                               ║
╚══════════════════════════════════════════════════════════════════════╝

📊 CONTAINER STATUS
├─ Total Containers: {self.results['containers'].get('total', 0)}
├─ Running: {self.results['containers'].get('running', 0)}
├─ Stopped: {self.results['containers'].get('stopped', 0)}
└─ Health Summary:
   ├─ Healthy: {sum(1 for v in self.results['containers'].get('health_status', {}).values() if v == 'healthy')}
   ├─ Unhealthy: {sum(1 for v in self.results['containers'].get('health_status', {}).values() if v == 'unhealthy')}
   └─ No Check: {sum(1 for v in self.results['containers'].get('health_status', {}).values() if v == 'no_health_check')}

🔌 SERVICE ENDPOINTS
"""
        for service, status in self.results.get('services', {}).items():
            status_icon = '✅' if status.get('status') in ['healthy', 'reachable'] else '❌'
            report += f"├─ {status_icon} {service}: {status.get('status', 'unknown')}\n"
        
        report += f"""
📝 CODE QUALITY
├─ Files Analyzed: {self.results['code_quality'].get('files_analyzed', 0)}/{self.results['code_quality'].get('total_files', 0)}
├─ Total Lines: {self.results['code_quality'].get('total_lines', 0):,}
├─ Compliance Score: {self.results['code_quality'].get('compliance_score', 0)}%
└─ Critical Issues:
   ├─ Hardcoded Credentials: {self.results['code_quality'].get('critical_issues', {}).get('hardcoded_credentials', 0)}
   ├─ Fantasy Elements: {self.results['code_quality'].get('critical_issues', {}).get('fantasy_elements', 0)}
   └─ Bare Excepts: {self.results['code_quality'].get('critical_issues', {}).get('bare_excepts', 0)}

🔐 SECURITY STATUS
├─ Root Containers: {self.results['security'].get('root_containers', 0)}
├─ Credentials in .env: {'✅' if self.results['security'].get('credentials_in_env') else '❌'}
└─ SSL Enabled: {'✅' if self.results['security'].get('ssl_enabled') else '❌'}

🤖 200-AGENT PROGRESS
"""
        for phase_name, phase_data in self.results.get('agent_progress', {}).items():
            progress_bar = '█' * (phase_data['progress'] // 10) + '░' * (10 - phase_data['progress'] // 10)
            report += f"├─ {phase_name.replace('_', ' ').title()}\n"
            report += f"│  ├─ Status: {phase_data['status']}\n"
            report += f"│  ├─ Progress: [{progress_bar}] {phase_data['progress']}%\n"
            report += f"│  └─ Tasks: {phase_data['tasks_completed']}/{phase_data['tasks_total']}\n"
        
        report += """
╔══════════════════════════════════════════════════════════════════════╗
║                    IMMEDIATE ACTIONS REQUIRED                        ║
╚══════════════════════════════════════════════════════════════════════╝
"""
        
        # Determine critical actions
        if self.results['security'].get('root_containers', 0) > 0:
            report += "🔴 CRITICAL: Remove root user from containers\n"
        if self.results['code_quality'].get('critical_issues', {}).get('hardcoded_credentials', 0) > 0:
            report += "🔴 CRITICAL: Remove hardcoded credentials\n"
        if self.results['containers'].get('running', 0) < 20:
            report += "🟠 HIGH: Start missing critical services\n"
        if not self.results['security'].get('credentials_in_env'):
            report += "🟠 HIGH: Create .env file with secure credentials\n"
            
        return report
    
    def _clear_screen(self):
        """Secure screen clearing without shell injection"""
        try:
            # Use appropriate clear command for the platform
            if os.name == 'nt':  # Windows
                subprocess.run(["cls"], shell=False)
            else:  # Unix/Linux/macOS
                subprocess.run(["clear"], shell=False)
        except Exception as e:
            # TODO: Review this exception handling
            logger.error(f"Unexpected exception: {e}", exc_info=True)
            # Fallback: print newlines if clear command fails
            print("\n" * 50)
    
    def save_results(self):
        """Save analysis results to file"""
        with open('/opt/sutazaiapp/ultra_deep_analysis_report.json', 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        with open('/opt/sutazaiapp/ultra_deep_analysis_report.txt', 'w') as f:
            f.write(self.generate_report())
    
    def run_continuous_monitoring(self, interval=60):
        """Run continuous monitoring"""
        print("🚀 Starting ULTRA-DEEP System Monitoring...")
        print(f"📊 Monitoring interval: {interval} seconds")
        print("Press Ctrl+C to stop\n")
        
        try:
            while True:
                # Run all analyses
                self.analyze_containers()
                self.check_service_endpoints()
                self.analyze_code_quality()
                self.check_security_status()
                self.monitor_agent_progress()
                
                # Generate and display report
                report = self.generate_report()
                # SECURE: Safe screen clear without shell injection
                self._clear_screen()
                print(report)
                
                # Save results
                self.save_results()
                
                # Wait for next cycle
                print(f"\n⏱️  Next update in {interval} seconds...")
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n\n✋ Monitoring stopped by user")
            print(f"📁 Final report saved to:")
            print("   - /opt/sutazaiapp/ultra_deep_analysis_report.json")
            print("   - /opt/sutazaiapp/ultra_deep_analysis_report.txt")

def main():
    """Main execution"""
    analyzer = UltraSystemAnalyzer()
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == '--once':
            # Run once and exit
            analyzer.analyze_containers()
            analyzer.check_service_endpoints()
            analyzer.analyze_code_quality()
            analyzer.check_security_status()
            analyzer.monitor_agent_progress()
            print(analyzer.generate_report())
            analyzer.save_results()
        elif sys.argv[1] == '--interval':
            # Custom interval
            interval = int(sys.argv[2]) if len(sys.argv) > 2 else 60
            analyzer.run_continuous_monitoring(interval)
        else:
            print("Usage: python ultra_deep_system_analysis.py [--once | --interval <seconds>]")
    else:
        # Default: continuous monitoring every 60 seconds
        analyzer.run_continuous_monitoring(60)

if __name__ == "__main__":
    main()