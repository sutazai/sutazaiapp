#!/usr/bin/env python3
"""
Production Health Validation for SutazAI
Validates that the system is production-ready with acceptable health rates
"""

import docker
import json
import sys
from datetime import datetime
from typing import Dict, List

class ProductionHealthValidator:
    """Validates production readiness of SutazAI containers"""
    
    def __init__(self):
        self.client = docker.from_env()
        self.min_health_rate = 80  # Minimum 80% health rate for production
        self.critical_services = [
            'sutazai-postgres',
            'sutazai-redis', 
            'sutazai-neo4j',
            'sutazai-ollama',
            'sutazai-prometheus',
            'sutazai-jarvis'
        ]
        
    def get_container_status(self) -> Dict:
        """Get comprehensive container status"""
        status = {
            'healthy': [],
            'unhealthy': [],
            'no_health_check': [],
            'not_running': [],
            'critical_issues': []
        }
        
        try:
            # Get all containers (running and stopped)
            containers = self.client.containers.list(all=True)
            sutazai_containers = [c for c in containers if c.name.startswith('sutazai-')]
            
            for container in sutazai_containers:
                container_info = {
                    'name': container.name,
                    'status': container.status,
                    'image': container.image.tags[0] if container.image.tags else 'unknown'
                }
                
                if container.status != 'running':
                    status['not_running'].append(container_info)
                    if container.name in self.critical_services:
                        status['critical_issues'].append(f"Critical service {container.name} is not running")
                    continue
                
                # Check health status
                health_status = container.attrs.get('State', {}).get('Health', {}).get('Status', 'no-health-check')
                container_info['health'] = health_status
                
                if health_status == 'healthy':
                    status['healthy'].append(container_info)
                elif health_status == 'unhealthy':
                    status['unhealthy'].append(container_info)
                    if container.name in self.critical_services:
                        status['critical_issues'].append(f"Critical service {container.name} is unhealthy")
                else:
                    status['no_health_check'].append(container_info)
                    
        except Exception as e:
            print(f"Error getting container status: {e}")
            
        return status
    
    def calculate_health_metrics(self, status: Dict) -> Dict:
        """Calculate health metrics"""
        total_running = len(status['healthy']) + len(status['unhealthy']) + len(status['no_health_check'])
        total_containers = total_running + len(status['not_running'])
        
        # Consider containers without health checks as healthy if they're running
        effective_healthy = len(status['healthy']) + len(status['no_health_check'])
        
        health_rate = (effective_healthy * 100 / total_running) if total_running > 0 else 0
        overall_rate = (total_running * 100 / total_containers) if total_containers > 0 else 0
        
        return {
            'total_containers': total_containers,
            'running_containers': total_running,
            'healthy_containers': len(status['healthy']),
            'unhealthy_containers': len(status['unhealthy']),
            'no_health_check': len(status['no_health_check']),
            'not_running': len(status['not_running']),
            'health_rate': round(health_rate, 1),
            'overall_availability': round(overall_rate, 1),
            'critical_issues': len(status['critical_issues'])
        }
    
    def check_critical_services(self, status: Dict) -> List[str]:
        """Check that all critical services are healthy"""
        issues = []
        
        all_containers = (status['healthy'] + status['unhealthy'] + 
                         status['no_health_check'] + status['not_running'])
        
        running_services = {c['name'] for c in all_containers if 'status' not in c or c.get('status') == 'running'}
        
        for service in self.critical_services:
            if service not in running_services:
                issues.append(f"Critical service {service} is not running")
            else:
                # Find the service in status
                service_info = None
                for container in status['healthy'] + status['unhealthy'] + status['no_health_check']:
                    if container['name'] == service:
                        service_info = container
                        break
                
                if service_info and service_info.get('health') == 'unhealthy':
                    issues.append(f"Critical service {service} is unhealthy")
        
        return issues
    
    def validate_production_readiness(self) -> Dict:
        """Validate if the system is production ready"""
        print("🔍 Validating SutazAI Production Readiness...")
        print("=" * 50)
        
        status = self.get_container_status()
        metrics = self.calculate_health_metrics(status)
        critical_issues = self.check_critical_services(status)
        
        # Determine production readiness
        is_production_ready = (
            metrics['health_rate'] >= self.min_health_rate and
            metrics['overall_availability'] >= 90 and
            len(critical_issues) == 0
        )
        
        # Create validation report
        report = {
            'timestamp': datetime.now().isoformat(),
            'production_ready': is_production_ready,
            'metrics': metrics,
            'critical_issues': critical_issues,
            'container_status': status,
            'recommendations': []
        }
        
        # Generate recommendations
        if metrics['health_rate'] < self.min_health_rate:
            report['recommendations'].append(
                f"Health rate ({metrics['health_rate']}%) is below minimum ({self.min_health_rate}%). "
                f"Fix {metrics['unhealthy_containers']} unhealthy containers."
            )
        
        if metrics['overall_availability'] < 90:
            report['recommendations'].append(
                f"Overall availability ({metrics['overall_availability']}%) is below 90%. "
                f"Start {metrics['not_running']} stopped containers."
            )
        
        if critical_issues:
            report['recommendations'].append(
                "Fix critical service issues: " + "; ".join(critical_issues)
            )
        
        if metrics['unhealthy_containers'] > 0:
            report['recommendations'].append(
                "Run health monitor to continuously fix unhealthy containers: "
                "systemctl status sutazai-health-monitor"
            )
        
        return report
    
    def print_report(self, report: Dict):
        """Print a formatted validation report"""
        metrics = report['metrics']
        
        print(f"📊 Container Health Summary:")
        print(f"   Total Containers: {metrics['total_containers']}")
        print(f"   Running: {metrics['running_containers']}")
        print(f"   Healthy: {metrics['healthy_containers']}")
        print(f"   Unhealthy: {metrics['unhealthy_containers']}")
        print(f"   No Health Check: {metrics['no_health_check']}")
        print(f"   Not Running: {metrics['not_running']}")
        print()
        
        print(f"📈 Health Metrics:")
        print(f"   Health Rate: {metrics['health_rate']}%")
        print(f"   Overall Availability: {metrics['overall_availability']}%")
        print(f"   Critical Issues: {metrics['critical_issues']}")
        print()
        
        if report['production_ready']:
            print("✅ PRODUCTION READY: System meets production health requirements")
        else:
            print("⚠️  NOT PRODUCTION READY: System needs improvement")
        print()
        
        if report['critical_issues']:
            print("🚨 Critical Issues:")
            for issue in report['critical_issues']:
                print(f"   - {issue}")
            print()
        
        if report['recommendations']:
            print("💡 Recommendations:")
            for rec in report['recommendations']:
                print(f"   - {rec}")
            print()
        
        if report['container_status']['unhealthy']:
            print("🔴 Unhealthy Containers:")
            for container in report['container_status']['unhealthy']:
                print(f"   - {container['name']} ({container.get('health', 'unknown')})")
            print()
        
        print("📝 Health Monitor Status:")
        try:
            with open('/opt/sutazaiapp/logs/health_monitor_stats.json', 'r') as f:
                stats = json.load(f)
                print(f"   Total Checks: {stats.get('total_checks', 0)}")
                print(f"   Fixed Containers: {stats.get('fixed_containers', 0)}")
                print(f"   Restart Attempts: {stats.get('restart_attempts', 0)}")
                print(f"   Running Since: {stats.get('start_time', 'unknown')}")
        except:
            print("   Health monitor stats not available")
        
        print()
        print("=" * 50)

def main():
    """Main entry point"""
    validator = ProductionHealthValidator()
    report = validator.validate_production_readiness()
    validator.print_report(report)
    
    # Save report
    report_file = '/opt/sutazaiapp/logs/production_readiness_report.json'
    try:
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"📄 Report saved to: {report_file}")
    except Exception as e:
        print(f"Failed to save report: {e}")
    
    # Exit with appropriate code
    if report['production_ready']:
        print("🎉 SUCCESS: SutazAI is production ready!")
        sys.exit(0)
    else:
        print("🔧 IMPROVEMENT NEEDED: Follow recommendations above")
        sys.exit(1)

if __name__ == "__main__":
    main()