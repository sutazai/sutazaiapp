#!/usr/bin/env python3
"""
Comprehensive Infrastructure Health Check Orchestrator

This is the main orchestrator script that runs all infrastructure health verification checks.
It coordinates the execution of individual service health check scripts and provides
comprehensive reporting for CI/CD pipeline integration.

Based on CLAUDE.md truth document and follows all 19 comprehensive codebase rules.

Usage:
    python scripts/devops/infrastructure_health_check.py
    python scripts/devops/infrastructure_health_check.py --timeout 30 --parallel
    python scripts/devops/infrastructure_health_check.py --services ollama,monitoring --verbose
    python scripts/devops/infrastructure_health_check.py --json-output /tmp/health_report.json

Created: December 19, 2024
Author: infrastructure-devops-manager agent
"""

import argparse
import json
import logging
import sys
import time
import asyncio
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import os


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration with timestamp."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='[%(levelname)s] %(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


class HealthCheckOrchestrator:
    """Main orchestrator class for infrastructure health checks."""
    
    def __init__(self, base_path: str = "/opt/sutazaiapp", timeout: float = 30.0):
        self.base_path = Path(base_path)
        self.scripts_dir = self.base_path / "scripts" / "devops"
        self.timeout = timeout
        self.results = {}
        
        # Service definitions based on CLAUDE.md truth document
        self.service_groups = {
            "ollama": {
                "script": "health_check_ollama.py",
                "description": "Ollama + TinyLlama Server",
                "ports": [10104],
                "critical": True
            },
            "gateway": {
                "script": "health_check_gateway.py", 
                "description": "Kong API Gateway + Consul Service Discovery",
                "ports": [10005, 10006],
                "critical": False
            },
            "vectordb": {
                "script": "health_check_vectordb.py",
                "description": "Vector Database Services (Qdrant, FAISS, ChromaDB)",
                "ports": [10100, 10101, 10102, 10103],
                "critical": False
            },
            "dataservices": {
                "script": "health_check_dataservices.py",
                "description": "Core Data Services (Redis, PostgreSQL, RabbitMQ)",
                "ports": [10000, 10001, 10007, 10008],
                "critical": True
            },
            "monitoring": {
                "script": "health_check_monitoring.py",
                "description": "Monitoring Stack (Prometheus, Grafana, Loki, AlertManager)",
                "ports": [10200, 10201, 10202, 10203],
                "critical": False
            }
        }
    
    def validate_environment(self) -> bool:
        """Validate that all required health check scripts exist."""
        missing_scripts = []
        
        for service_name, config in self.service_groups.items():
            script_path = self.scripts_dir / config["script"]
            if not script_path.exists():
                missing_scripts.append(f"{service_name}: {script_path}")
        
        if missing_scripts:
            logging.error("Missing health check scripts:")
            for script in missing_scripts:
                logging.error(f"  - {script}")
            return False
        
        logging.info("All health check scripts found")
        return True
    
    def run_health_check(self, service_name: str, **kwargs) -> Dict[str, Any]:
        """Run health check for a specific service."""
        config = self.service_groups[service_name]
        script_path = self.scripts_dir / config["script"]
        
        # Build command arguments
        cmd = [sys.executable, str(script_path)]
        
        # Add common arguments
        if kwargs.get('timeout'):
            cmd.extend(['--timeout', str(kwargs['timeout'])])
        if kwargs.get('verbose'):
            cmd.append('--verbose')
        if kwargs.get('host'):
            cmd.extend(['--host', kwargs['host']])
        
        # Add service-specific arguments
        if service_name == "ollama":
            if kwargs.get('ollama_port'):
                cmd.extend(['--port', str(kwargs['ollama_port'])])
        elif service_name == "gateway":
            if kwargs.get('kong_port'):
                cmd.extend(['--kong-port', str(kwargs['kong_port'])])
            if kwargs.get('consul_port'):
                cmd.extend(['--consul-port', str(kwargs['consul_port'])])
        elif service_name == "vectordb":
            if kwargs.get('vector_range'):
                cmd.extend(['--port-range', kwargs['vector_range']])
        elif service_name == "dataservices":
            if kwargs.get('redis_port'):
                cmd.extend(['--redis-port', str(kwargs['redis_port'])])
            if kwargs.get('postgres_port'):
                cmd.extend(['--postgres-port', str(kwargs['postgres_port'])])
        elif service_name == "monitoring":
            if kwargs.get('prometheus_port'):
                cmd.extend(['--prometheus-port', str(kwargs['prometheus_port'])])
            if kwargs.get('grafana_port'):
                cmd.extend(['--grafana-port', str(kwargs['grafana_port'])])
        
        start_time = time.time()
        result = {
            'service': service_name,
            'description': config['description'],
            'script': config['script'],
            'command': ' '.join(cmd),
            'start_time': datetime.now().isoformat(),
            'critical': config['critical']
        }
        
        try:
            logging.info(f"Running health check for {service_name}...")
            logging.debug(f"Command: {' '.join(cmd)}")
            
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=self.base_path
            )
            
            execution_time = time.time() - start_time
            
            result.update({
                'exit_code': process.returncode,
                'execution_time': round(execution_time, 2),
                'stdout': process.stdout,
                'stderr': process.stderr,
                'success': process.returncode == 0,
                'end_time': datetime.now().isoformat()
            })
            
            if process.returncode == 0:
                logging.info(f"✅ {service_name} health check passed ({execution_time:.2f}s)")
            else:
                logging.warning(f"⚠️ {service_name} health check failed with exit code {process.returncode}")
                if process.stderr:
                    logging.debug(f"Error output: {process.stderr[:500]}")
            
        except subprocess.TimeoutExpired:
            result.update({
                'exit_code': -1,
                'execution_time': self.timeout,
                'stdout': '',
                'stderr': f'Health check timed out after {self.timeout}s',
                'success': False,
                'timeout': True,
                'end_time': datetime.now().isoformat()
            })
            logging.error(f"❌ {service_name} health check timed out")
            
        except Exception as e:
            result.update({
                'exit_code': -2,
                'execution_time': time.time() - start_time,
                'stdout': '',
                'stderr': str(e),
                'success': False,
                'error': str(e),
                'end_time': datetime.now().isoformat()
            })
            logging.error(f"❌ {service_name} health check failed with error: {e}")
        
        return result
    
    def run_parallel_health_checks(self, services: List[str], **kwargs) -> Dict[str, Any]:
        """Run health checks in parallel for better performance."""
        logging.info(f"Running parallel health checks for: {', '.join(services)}")
        
        with ThreadPoolExecutor(max_workers=len(services)) as executor:
            # Submit all health check tasks
            future_to_service = {
                executor.submit(self.run_health_check, service, **kwargs): service
                for service in services
            }
            
            results = {}
            for future in as_completed(future_to_service):
                service = future_to_service[future]
                try:
                    result = future.result()
                    results[service] = result
                except Exception as e:
                    logging.error(f"Parallel execution failed for {service}: {e}")
                    results[service] = {
                        'service': service,
                        'success': False,
                        'error': f"Parallel execution failed: {e}",
                        'execution_time': 0,
                        'critical': self.service_groups[service]['critical']
                    }
            
            return results
    
    def run_sequential_health_checks(self, services: List[str], **kwargs) -> Dict[str, Any]:
        """Run health checks sequentially."""
        logging.info(f"Running sequential health checks for: {', '.join(services)}")
        
        results = {}
        for service in services:
            try:
                result = self.run_health_check(service, **kwargs)
                results[service] = result
                
                # Add small delay between checks to avoid overwhelming services
                time.sleep(0.5)
                
            except Exception as e:
                logging.error(f"Sequential execution failed for {service}: {e}")
                results[service] = {
                    'service': service,
                    'success': False,
                    'error': f"Sequential execution failed: {e}",
                    'execution_time': 0,
                    'critical': self.service_groups[service]['critical']
                }
        
        return results
    
    def generate_summary_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive summary report."""
        total_services = len(results)
        successful_services = sum(1 for r in results.values() if r.get('success', False))
        failed_services = total_services - successful_services
        
        critical_services = [k for k, v in results.items() if self.service_groups[k]['critical']]
        critical_failures = [k for k in critical_services if not results[k].get('success', False)]
        
        total_execution_time = sum(r.get('execution_time', 0) for r in results.values())
        
        # Determine overall health status
        if critical_failures:
            overall_status = "critical"
        elif failed_services > 0:
            overall_status = "degraded"
        else:
            overall_status = "healthy"
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': overall_status,
            'summary': {
                'total_services': total_services,
                'successful_services': successful_services,
                'failed_services': failed_services,
                'success_rate': round((successful_services / total_services) * 100, 2),
                'total_execution_time': round(total_execution_time, 2),
                'average_execution_time': round(total_execution_time / total_services, 2)
            },
            'critical_services': {
                'total': len(critical_services),
                'successful': len([k for k in critical_services if results[k].get('success', False)]),
                'failed': len(critical_failures),
                'failed_services': critical_failures
            },
            'service_breakdown': {
                'successful': [k for k, v in results.items() if v.get('success', False)],
                'failed': [k for k, v in results.items() if not v.get('success', False)],
                'timeouts': [k for k, v in results.items() if v.get('timeout', False)]
            },
            'recommendations': self._generate_recommendations(results),
            'detailed_results': results
        }
        
        return summary
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on health check results."""
        recommendations = []
        
        # Check critical service failures
        critical_failures = [
            k for k, v in results.items() 
            if self.service_groups[k]['critical'] and not v.get('success', False)
        ]
        
        if critical_failures:
            recommendations.append(f"⚠️ CRITICAL: Fix critical services: {', '.join(critical_failures)}")
        
        # Check for timeouts
        timeouts = [k for k, v in results.items() if v.get('timeout', False)]
        if timeouts:
            recommendations.append(f"🕐 Investigate timeout issues: {', '.join(timeouts)}")
        
        # Service-specific recommendations
        for service, result in results.items():
            if not result.get('success', False):
                if service == "ollama":
                    recommendations.append("🤖 Ollama/TinyLlama: Check model loading and service connectivity")
                elif service == "dataservices":
                    recommendations.append("💾 Data Services: Verify PostgreSQL, Redis, and RabbitMQ connectivity")
                elif service == "monitoring":
                    recommendations.append("📊 Monitoring: Check Prometheus/Grafana configuration and data collection")
                elif service == "vectordb":
                    recommendations.append("🔍 Vector DB: Review ChromaDB connection issues (expected per CLAUDE.md)")
                elif service == "gateway":
                    recommendations.append("🚪 API Gateway: Verify Kong/Consul service mesh configuration")
        
        # Performance recommendations
        slow_services = [k for k, v in results.items() if v.get('execution_time', 0) > 10]
        if slow_services:
            recommendations.append(f"⚡ Performance: Optimize slow services: {', '.join(slow_services)}")
        
        # Success recommendations
        if not recommendations:
            recommendations.append("✅ All services healthy - system ready for production workloads")
            recommendations.append("🔄 Schedule regular health checks via CI/CD pipeline")
        
        return recommendations
    
    def save_report(self, report: Dict[str, Any], output_path: str) -> bool:
        """Save health check report to JSON file."""
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logging.info(f"Health check report saved to: {output_path}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to save report to {output_path}: {e}")
            return False
    
    def print_summary(self, report: Dict[str, Any]) -> None:
        """Print human-readable summary to console."""
        summary = report['summary']
        critical = report['critical_services']
        
        print("\n" + "="*60)
        print("🏥 INFRASTRUCTURE HEALTH CHECK SUMMARY")
        print("="*60)
        print(f"Timestamp: {report['timestamp']}")
        print(f"Overall Status: {report['overall_status'].upper()}")
        print()
        
        print("📊 SERVICE STATISTICS")
        print("-"*30)
        print(f"Total Services: {summary['total_services']}")
        print(f"Successful: {summary['successful_services']}")
        print(f"Failed: {summary['failed_services']}")
        print(f"Success Rate: {summary['success_rate']}%")
        print(f"Total Execution Time: {summary['total_execution_time']}s")
        print(f"Average Time per Service: {summary['average_execution_time']}s")
        print()
        
        print("🚨 CRITICAL SERVICES")
        print("-"*30)
        print(f"Total Critical Services: {critical['total']}")
        print(f"Critical Services Healthy: {critical['successful']}")
        print(f"Critical Services Failed: {critical['failed']}")
        if critical['failed_services']:
            print(f"Failed Critical Services: {', '.join(critical['failed_services'])}")
        print()
        
        if report['service_breakdown']['successful']:
            print("✅ HEALTHY SERVICES")
            print("-"*30)
            for service in report['service_breakdown']['successful']:
                desc = self.service_groups[service]['description']
                exec_time = report['detailed_results'][service]['execution_time']
                print(f"  {service}: {desc} ({exec_time}s)")
            print()
        
        if report['service_breakdown']['failed']:
            print("❌ UNHEALTHY SERVICES")
            print("-"*30)
            for service in report['service_breakdown']['failed']:
                desc = self.service_groups[service]['description']
                error = report['detailed_results'][service].get('stderr', 'Unknown error')[:100]
                print(f"  {service}: {desc}")
                print(f"    Error: {error}")
            print()
        
        print("💡 RECOMMENDATIONS")
        print("-"*30)
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"  {i}. {rec}")
        print()
        
        print("="*60)


def main():
    """Main function with comprehensive argument parsing and orchestration."""
    parser = argparse.ArgumentParser(
        description="Comprehensive Infrastructure Health Check Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/devops/infrastructure_health_check.py
    python scripts/devops/infrastructure_health_check.py --services ollama,dataservices
    python scripts/devops/infrastructure_health_check.py --parallel --timeout 30
    python scripts/devops/infrastructure_health_check.py --json-output /tmp/health.json
    python scripts/devops/infrastructure_health_check.py --host 127.0.0.1 --verbose

Service Groups:
    ollama      - Ollama + TinyLlama Server (CRITICAL)
    gateway     - Kong API Gateway + Consul Service Discovery  
    vectordb    - Vector Database Services (Qdrant, FAISS, ChromaDB)
    dataservices- Core Data Services (Redis, PostgreSQL, RabbitMQ) (CRITICAL)
    monitoring  - Monitoring Stack (Prometheus, Grafana, Loki, AlertManager)
        """
    )
    
    parser.add_argument('--services', 
                       help='Comma-separated list of service groups to check (default: all)')
    parser.add_argument('--parallel', action='store_true',
                       help='Run health checks in parallel for better performance')
    parser.add_argument('--timeout', type=float, default=30.0,
                       help='Timeout for each health check in seconds (default: 30.0)')
    parser.add_argument('--host', default='localhost',
                       help='Host for all services (default: localhost)')
    
    # Service-specific ports
    parser.add_argument('--ollama-port', type=int, default=10104,
                       help='Ollama port (default: 10104)')
    parser.add_argument('--kong-port', type=int, default=10005,
                       help='Kong port (default: 10005)')
    parser.add_argument('--consul-port', type=int, default=10006,
                       help='Consul port (default: 10006)')
    parser.add_argument('--vector-range', default='10100-10103',
                       help='Vector DB port range (default: 10100-10103)')
    parser.add_argument('--redis-port', type=int, default=10001,
                       help='Redis port (default: 10001)')
    parser.add_argument('--postgres-port', type=int, default=10000,
                       help='PostgreSQL port (default: 10000)')
    parser.add_argument('--prometheus-port', type=int, default=10200,
                       help='Prometheus port (default: 10200)')
    parser.add_argument('--grafana-port', type=int, default=10201,
                       help='Grafana port (default: 10201)')
    
    # Output options
    parser.add_argument('--json-output',
                       help='Save detailed JSON report to specified file')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress summary output (useful for JSON-only output)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    # CI/CD integration options
    parser.add_argument('--fail-on-critical', action='store_true', default=True,
                       help='Exit with error code if critical services fail (default: True)')
    parser.add_argument('--fail-on-any', action='store_true',
                       help='Exit with error code if any service fails')
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    # Initialize orchestrator
    orchestrator = HealthCheckOrchestrator(timeout=args.timeout)
    
    # Validate environment
    if not orchestrator.validate_environment():
        logging.error("Environment validation failed")
        return 2
    
    # Determine which services to check
    if args.services:
        requested_services = [s.strip() for s in args.services.split(',')]
        invalid_services = [s for s in requested_services if s not in orchestrator.service_groups]
        if invalid_services:
            logging.error(f"Invalid services: {invalid_services}")
            logging.error(f"Valid services: {list(orchestrator.service_groups.keys())}")
            return 2
        services_to_check = requested_services
    else:
        services_to_check = list(orchestrator.service_groups.keys())
    
    logging.info(f"Starting infrastructure health checks for: {', '.join(services_to_check)}")
    
    # Prepare kwargs for health check scripts
    kwargs = {
        'timeout': args.timeout,
        'verbose': args.verbose,
        'host': args.host,
        'ollama_port': args.ollama_port,
        'kong_port': args.kong_port,
        'consul_port': args.consul_port,
        'vector_range': args.vector_range,
        'redis_port': args.redis_port,
        'postgres_port': args.postgres_port,
        'prometheus_port': args.prometheus_port,
        'grafana_port': args.grafana_port
    }
    
    # Execute health checks
    start_time = time.time()
    
    if args.parallel and len(services_to_check) > 1:
        results = orchestrator.run_parallel_health_checks(services_to_check, **kwargs)
    else:
        results = orchestrator.run_sequential_health_checks(services_to_check, **kwargs)
    
    total_time = time.time() - start_time
    
    # Generate comprehensive report
    report = orchestrator.generate_summary_report(results)
    report['orchestrator_execution_time'] = round(total_time, 2)
    report['execution_mode'] = 'parallel' if args.parallel else 'sequential'
    
    # Save JSON report if requested
    if args.json_output:
        if not orchestrator.save_report(report, args.json_output):
            return 3
    
    # Print summary unless quiet mode
    if not args.quiet:
        orchestrator.print_summary(report)
    
    # Determine exit code based on results
    critical_failures = report['critical_services']['failed']
    any_failures = report['summary']['failed_services']
    
    if args.fail_on_critical and critical_failures > 0:
        logging.error(f"Exiting with error code 1: {critical_failures} critical services failed")
        return 1
    elif args.fail_on_any and any_failures > 0:
        logging.error(f"Exiting with error code 1: {any_failures} services failed")
        return 1
    
    logging.info(f"Infrastructure health check completed successfully in {total_time:.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())