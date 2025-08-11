#!/usr/bin/env python3
"""
Comprehensive Health Check Script using HealthMaster

This script provides a complete health check system for all 25 SutazAI services
using the centralized HealthMaster class infrastructure.

Author: COORDINATED ARCHITECT TEAM
Created: 2025-08-11
Purpose: Restore missing health check functionality and provide comprehensive coverage
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional

# Add scripts directory to path for imports
SCRIPT_DIR = Path(__file__).parent
sys.path.append(str(SCRIPT_DIR))

try:
    from lib.master_health_controller import HealthMaster
    from lib.logging_utils import setup_logging, ScriptTimer
except ImportError as e:
    print(f"❌ Failed to import required modules: {e}")
    print("🔍 Checking lib directory structure...")
    lib_dir = SCRIPT_DIR / "lib"
    if lib_dir.exists():
        print(f"✅ lib directory exists at: {lib_dir}")
        for file in lib_dir.glob("*.py"):
            print(f"   📁 {file.name}")
    else:
        print(f"❌ lib directory not found at: {lib_dir}")
    sys.exit(1)


class ComprehensiveHealthChecker(HealthMaster):
    """
    Extended health checker with additional service coverage and reporting.
    """
    
    def __init__(self):
        """Initialize comprehensive health checker."""
        super().__init__("comprehensive_health_check")
        
        # Override RabbitMQ URL to use management interface
        self.core_services['rabbitmq']['url'] = 'http://localhost:10008'
        
        # Add Kong and Consul services (corrected URLs)
        self.service_mesh_services = {
            'kong': {
                'url': 'http://localhost:10015',  # Admin port for health check
                'check_method': self._check_http_endpoint,
                'critical': False,
                'description': 'Kong API Gateway'
            },
            'consul': {
                'url': 'http://localhost:10006/v1/status/leader',
                'check_method': self._check_http_endpoint,
                'critical': False,
                'description': 'Consul Service Discovery'
            }
        }
        
        # Add additional agent services that might be missing
        self.additional_agent_services = {
            'jarvis_automation': {
                'url': 'http://localhost:11102/health',
                'check_method': self._check_http_endpoint,
                'critical': False,
                'description': 'Jarvis Automation Agent'
            },
            'jarvis_hardware_optimizer_alt': {
                'url': 'http://localhost:11104/health',
                'check_method': self._check_http_endpoint,
                'critical': False,
                'description': 'Jarvis Hardware Optimizer Alternative'
            }
        }
        
        # Monitoring exporters (corrected port mappings from docker-compose.yml)
        self.exporter_services = {
            'blackbox_exporter': {
                'url': 'http://localhost:10204',
                'check_method': self._check_http_endpoint,
                'critical': False,
                'description': 'Blackbox Exporter'
            },
            'postgres_exporter': {
                'url': 'http://localhost:10207/metrics',
                'check_method': self._check_http_endpoint,
                'critical': False,
                'description': 'PostgreSQL Exporter'
            },
            'redis_exporter': {
                'url': 'http://localhost:10208/metrics',
                'check_method': self._check_http_endpoint,
                'critical': False,
                'description': 'Redis Exporter'
            },
            'node_exporter': {
                'url': 'http://localhost:10205/metrics',
                'check_method': self._check_http_endpoint,
                'critical': False,
                'description': 'Node Exporter'
            },
            'cadvisor': {
                'url': 'http://localhost:10206/metrics',
                'check_method': self._check_http_endpoint,
                'critical': False,
                'description': 'Container Advisor'
            },
            'alertmanager': {
                'url': 'http://localhost:10203/-/ready',
                'check_method': self._check_http_endpoint,
                'critical': False,
                'description': 'Alert Manager'
            }
        }
        
        # Additional services found in docker-compose.yml
        self.additional_services = {
            'jaeger': {
                'url': 'http://localhost:10210',
                'check_method': self._check_http_endpoint,
                'critical': False,
                'description': 'Jaeger Distributed Tracing'
            },
            'promtail': {
                'url': 'http://localhost:9080/targets',  # Promtail doesn't expose HTTP endpoint typically
                'check_method': self._check_docker_container,
                'critical': False,
                'description': 'Promtail Log Shipper'
            }
        }
    
    def check_all_services_extended(self, critical_only: bool = False) -> Dict:
        """
        Extended health check that includes all services from the platform.
        
        Args:
            critical_only: Only check critical services
            
        Returns:
            Dictionary with comprehensive health check results
        """
        with ScriptTimer(self.logger, "comprehensive_health_check"):
            results = {}
            summary = {
                'healthy': 0,
                'degraded': 0,
                'failed': 0,
                'total': 0
            }
            
            # Combine all services
            all_services = {
                **self.core_services,
                **self.agent_services,
                **self.service_mesh_services,
                **self.additional_agent_services,
                **self.exporter_services,
                **self.additional_services
            }
            
            for service_name, service_config in all_services.items():
                if critical_only and not service_config.get('critical', False):
                    continue
                
                self.logger.info(f"Checking {service_name} ({service_config['description']})")
                
                try:
                    check_method = service_config.get('check_method', self._check_http_endpoint)
                    result = check_method(service_config['url'], service_config)
                    
                    # Add service metadata
                    result.update({
                        'critical': service_config.get('critical', False),
                        'description': service_config['description'],
                        'service_type': self._get_service_type(service_name),
                        'timestamp': time.time()
                    })
                    
                    results[service_name] = result
                    
                    # Update summary
                    summary['total'] += 1
                    status = result.get('status', 'failed')
                    if status == 'healthy':
                        summary['healthy'] += 1
                    elif status == 'degraded':
                        summary['degraded'] += 1
                    else:
                        summary['failed'] += 1
                    
                    # Log the result with appropriate icon
                    status_icon = {'healthy': '✅', 'degraded': '⚠️', 'failed': '❌'}.get(status, '❓')
                    self.logger.info(f"{status_icon} {service_name}: {status.upper()}")
                    
                except Exception as e:
                    self.logger.error(f"Error checking {service_name}: {str(e)}")
                    results[service_name] = {
                        'status': 'failed',
                        'error': str(e),
                        'critical': service_config.get('critical', False),
                        'description': service_config['description'],
                        'service_type': self._get_service_type(service_name),
                        'timestamp': time.time()
                    }
                    summary['failed'] += 1
                    summary['total'] += 1
            
            # Calculate overall health
            overall_status = 'healthy'
            critical_failures = sum(1 for r in results.values() 
                                  if r.get('critical') and r['status'] == 'failed')
            
            if critical_failures > 0:
                overall_status = 'critical_failure'
            elif summary['failed'] > 0:
                overall_status = 'degraded'
            elif summary['degraded'] > 0:
                overall_status = 'warnings'
            
            return {
                'overall_status': overall_status,
                'summary': summary,
                'results': results,
                'critical_failures': critical_failures,
                'timestamp': time.time(),
                'service_breakdown': self._get_service_breakdown(results)
            }
    
    def _get_service_type(self, service_name: str) -> str:
        """Categorize services by type."""
        if service_name in self.core_services:
            if service_name in ['postgres', 'redis', 'neo4j']:
                return 'database'
            elif service_name in ['backend', 'frontend']:
                return 'application'
            elif service_name in ['ollama']:
                return 'ai_model'
            elif service_name in ['qdrant', 'chromadb', 'faiss']:
                return 'vector_db'
            elif service_name in ['prometheus', 'grafana', 'loki']:
                return 'monitoring'
            elif service_name in ['rabbitmq']:
                return 'messaging'
        elif service_name in self.agent_services or service_name in self.additional_agent_services:
            return 'agent'
        elif service_name in self.service_mesh_services:
            return 'service_mesh'
        elif service_name in self.exporter_services:
            return 'exporter'
        elif service_name in self.additional_services:
            return 'additional'
        return 'unknown'
    
    def _check_docker_container(self, url: str, config: Dict) -> Dict:
        """Check if Docker container is running (for services without HTTP endpoints)."""
        import subprocess
        start_time = time.time()
        
        try:
            # Extract container name from service description
            service_name = config.get('description', '').lower().replace(' ', '-')
            container_name = f"sutazai-{service_name.replace(' ', '-')}"
            
            # Check if container is running
            result = subprocess.run(
                ['docker', 'ps', '--filter', f'name={container_name}', '--format', '{{.Names}}'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            response_time = time.time() - start_time
            
            if result.returncode == 0 and container_name in result.stdout:
                return {
                    'status': 'healthy',
                    'response_time': response_time,
                    'details': f'Container {container_name} running'
                }
            else:
                return {
                    'status': 'failed',
                    'response_time': response_time,
                    'error': f'Container {container_name} not found or not running'
                }
                
        except subprocess.TimeoutExpired:
            return {
                'status': 'failed',
                'error': 'Docker command timeout',
                'response_time': time.time() - start_time
            }
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e),
                'response_time': time.time() - start_time
            }
    
    def _get_service_breakdown(self, results: Dict) -> Dict:
        """Get breakdown of services by type and status."""
        breakdown = {}
        for service_name, result in results.items():
            service_type = result.get('service_type', 'unknown')
            status = result.get('status', 'failed')
            
            if service_type not in breakdown:
                breakdown[service_type] = {'healthy': 0, 'degraded': 0, 'failed': 0, 'total': 0}
            
            breakdown[service_type][status] += 1
            breakdown[service_type]['total'] += 1
        
        return breakdown
    
    def generate_detailed_report(self, results: Dict) -> str:
        """Generate a detailed health report."""
        overall_status = results['overall_status']
        summary = results['summary']
        service_breakdown = results['service_breakdown']
        
        # Status icon mapping
        status_icons = {
            'healthy': '✅',
            'warnings': '⚠️',
            'degraded': '🔸',
            'critical_failure': '❌'
        }
        
        report = []
        report.append("=" * 80)
        report.append("🏥 SUTAZAI COMPREHENSIVE SYSTEM HEALTH REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Overall status
        icon = status_icons.get(overall_status, '❓')
        report.append(f"{icon} OVERALL STATUS: {overall_status.upper()}")
        report.append("")
        
        # Summary statistics
        report.append("📊 SUMMARY STATISTICS:")
        report.append(f"   Total Services: {summary['total']}")
        report.append(f"   ✅ Healthy: {summary['healthy']}")
        report.append(f"   ⚠️ Degraded: {summary['degraded']}")
        report.append(f"   ❌ Failed: {summary['failed']}")
        
        health_percentage = (summary['healthy'] * 100) // summary['total'] if summary['total'] > 0 else 0
        report.append(f"   📈 Health Score: {health_percentage}%")
        report.append("")
        
        # Service breakdown by type
        report.append("🔍 SERVICE BREAKDOWN BY TYPE:")
        for service_type, stats in service_breakdown.items():
            type_health = (stats['healthy'] * 100) // stats['total'] if stats['total'] > 0 else 0
            report.append(f"   {service_type.upper()}: {stats['healthy']}/{stats['total']} healthy ({type_health}%)")
        report.append("")
        
        # Critical failures (if any)
        if results['critical_failures'] > 0:
            report.append("🚨 CRITICAL FAILURES:")
            for service_name, result in results['results'].items():
                if result.get('critical') and result['status'] == 'failed':
                    error = result.get('error', 'Unknown error')
                    report.append(f"   ❌ {service_name}: {error}")
            report.append("")
        
        # Detailed service status
        report.append("📋 DETAILED SERVICE STATUS:")
        for service_type in sorted(service_breakdown.keys()):
            report.append(f"\n🔸 {service_type.upper()} SERVICES:")
            type_services = {name: result for name, result in results['results'].items() 
                           if result.get('service_type') == service_type}
            
            for service_name, result in sorted(type_services.items()):
                status = result['status']
                icon = {'healthy': '✅', 'degraded': '⚠️', 'failed': '❌'}.get(status, '❓')
                description = result.get('description', 'Unknown service')
                
                line = f"   {icon} {service_name}: {description}"
                
                if 'response_time' in result:
                    line += f" ({result['response_time']:.0f}ms)"
                
                if 'error' in result:
                    line += f" - ERROR: {result['error']}"
                elif 'details' in result:
                    line += f" - {result['details']}"
                
                report.append(line)
        
        report.append("")
        report.append("=" * 80)
        report.append(f"Report generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def save_json_report(self, results: Dict, filepath: str = None) -> str:
        """Save results as JSON for CI/CD integration."""
        if filepath is None:
            timestamp = int(time.time())
            filepath = f"/opt/sutazaiapp/logs/health_report_comprehensive_{timestamp}.json"
        
        # Ensure logs directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"JSON health report saved to: {filepath}")
        return filepath


def main():
    """Main entry point for comprehensive health checking."""
    parser = argparse.ArgumentParser(
        description="Comprehensive SutazAI System Health Check",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 scripts/health_check_comprehensive.py                # Full health check
  python3 scripts/health_check_comprehensive.py --critical     # Critical services only
  python3 scripts/health_check_comprehensive.py --json         # JSON output
  python3 scripts/health_check_comprehensive.py --verbose      # Verbose logging
        """
    )
    
    parser.add_argument('--critical', action='store_true',
                       help='Check only critical services')
    parser.add_argument('--json', action='store_true',
                       help='Output results in JSON format')
    parser.add_argument('--save-json', type=str, metavar='FILE',
                       help='Save JSON report to specific file')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress console output (useful with --save-json)')
    
    args = parser.parse_args()
    
    # Initialize health checker
    health_checker = ComprehensiveHealthChecker()
    
    if args.verbose:
        health_checker.logger.setLevel("DEBUG")
    
    if not args.quiet:
        print("🏥 SutazAI Comprehensive System Health Check")
        print("=" * 60)
        print("🔍 Checking all 25+ system services...")
        print("")
    
    try:
        # Run comprehensive health checks
        results = health_checker.check_all_services_extended(critical_only=args.critical)
        
        if args.json:
            # JSON output
            print(json.dumps(results, indent=2, default=str))
        elif not args.quiet:
            # Human-readable output
            report = health_checker.generate_detailed_report(results)
            print(report)
        
        # Save JSON report if requested
        if args.save_json:
            health_checker.save_json_report(results, args.save_json)
        
        # Determine exit code
        overall_status = results['overall_status']
        if overall_status == 'critical_failure':
            exit_code = 2  # Critical failure
        elif overall_status in ['degraded', 'warnings']:
            exit_code = 1  # Non-critical issues
        else:
            exit_code = 0  # All healthy
        
        if not args.quiet:
            if exit_code == 0:
                print("\n🎉 System is healthy and fully operational!")
            elif exit_code == 1:
                print("\n⚠️ System has non-critical issues but is operational")
            else:
                print("\n🚨 System has critical failures requiring immediate attention")
        
        return exit_code
        
    except Exception as e:
        health_checker.logger.error(f"Health check failed with exception: {str(e)}")
        if not args.quiet:
            print(f"❌ Health check failed: {str(e)}")
        return 3  # Internal error


if __name__ == '__main__':
    import time
    exit(main())