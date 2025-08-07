#!/usr/bin/env python3
"""
Purpose: Monitor health and status of all integrated AI services
Usage: python monitor-ai-services.py [--service SERVICE_NAME] [--category CATEGORY]
Requirements: aiohttp, pyyaml, tabulate
"""

import asyncio
import argparse
import yaml
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import aiohttp
from tabulate import tabulate
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
SERVICES_CONFIG = PROJECT_ROOT / "config" / "services.yaml"


class ServiceMonitor:
    """Monitor for AI services health and status"""
    
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.services = {}
        self.load_config()
        
    def load_config(self):
        """Load services configuration"""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Parse services from config
        for category, services in config.get('services', {}).items():
            for service_name, service_config in services.items():
                if service_config.get('enabled', False):
                    self.services[service_name] = {
                        'category': category,
                        'config': service_config.get('config', {}),
                        'health_check': service_config.get('health_check', {}),
                        'resources': service_config.get('resources', {})
                    }
                    
    async def check_service_health(self, service_name: str, service_info: Dict) -> Dict[str, Any]:
        """Check health of a single service"""
        result = {
            'service': service_name,
            'category': service_info['category'],
            'status': 'unknown',
            'response_time': None,
            'error': None,
            'details': {}
        }
        
        try:
            # Get health check configuration
            health_config = service_info.get('health_check', {})
            base_url = service_info['config'].get('base_url', '')
            health_endpoint = health_config.get('endpoint', '/health')
            
            if not base_url:
                # Try to construct from host and port
                host = service_info['config'].get('host', service_name)
                port = service_info['config'].get('port', 8000)
                base_url = f"http://{host}:{port}"
                
            url = f"{base_url}{health_endpoint}"
            
            # Make health check request
            start_time = datetime.utcnow()
            timeout = aiohttp.ClientTimeout(total=10)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as response:
                    end_time = datetime.utcnow()
                    response_time = (end_time - start_time).total_seconds()
                    
                    result['response_time'] = response_time
                    
                    if response.status == 200:
                        result['status'] = 'healthy'
                        try:
                            result['details'] = await response.json()
                        except:
                            result['details'] = {'message': 'Service is healthy'}
                    else:
                        result['status'] = 'unhealthy'
                        result['error'] = f"HTTP {response.status}"
                        
        except asyncio.TimeoutError:
            result['status'] = 'timeout'
            result['error'] = 'Health check timed out'
        except aiohttp.ClientError as e:
            result['status'] = 'error'
            result['error'] = f"Connection error: {str(e)}"
        except Exception as e:
            result['status'] = 'error'
            result['error'] = str(e)
            
        return result
        
    async def check_all_services(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Check health of all services or services in a category"""
        tasks = []
        
        for service_name, service_info in self.services.items():
            if category and service_info['category'] != category:
                continue
                
            task = self.check_service_health(service_name, service_info)
            tasks.append(task)
            
        results = await asyncio.gather(*tasks)
        return results
        
    async def get_resource_usage(self, service_name: str) -> Dict[str, Any]:
        """Get resource usage for a service (via Docker stats)"""
        import subprocess
        
        try:
            container_name = f"sutazai-{service_name}"
            cmd = f"docker stats {container_name} --no-stream --format json"
            
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0 and result.stdout:
                stats = json.loads(result.stdout)
                return {
                    'cpu_percent': stats.get('CPUPerc', '0%').strip('%'),
                    'memory_usage': stats.get('MemUsage', 'N/A'),
                    'memory_percent': stats.get('MemPerc', '0%').strip('%'),
                    'network_io': stats.get('NetIO', 'N/A')
                }
        except:
            pass
            
        return {
            'cpu_percent': 'N/A',
            'memory_usage': 'N/A',
            'memory_percent': 'N/A',
            'network_io': 'N/A'
        }
        
    def format_results(self, results: List[Dict[str, Any]], show_details: bool = False) -> str:
        """Format results for display"""
        # Prepare table data
        table_data = []
        
        for result in results:
            status_icon = {
                'healthy': '✅',
                'unhealthy': '❌',
                'timeout': '⏱️',
                'error': '🔥',
                'unknown': '❓'
            }.get(result['status'], '❓')
            
            response_time = f"{result['response_time']:.2f}s" if result['response_time'] else 'N/A'
            
            row = [
                result['service'],
                result['category'],
                f"{status_icon} {result['status']}",
                response_time,
                result['error'] or 'OK'
            ]
            
            table_data.append(row)
            
        # Create table
        headers = ['Service', 'Category', 'Status', 'Response Time', 'Details']
        table = tabulate(table_data, headers=headers, tablefmt='grid')
        
        # Add summary
        total = len(results)
        healthy = sum(1 for r in results if r['status'] == 'healthy')
        unhealthy = total - healthy
        
        summary = f"\nSummary: {healthy}/{total} services healthy"
        if unhealthy > 0:
            summary += f" ({unhealthy} issues detected)"
            
        return table + summary
        
    async def monitor_continuous(self, interval: int = 30):
        """Continuously monitor services"""
        logger.info(f"Starting continuous monitoring (interval: {interval}s)")
        
        while True:
            try:
                print("\033[2J\033[H")  # Clear screen
                print(f"=== SutazAI Service Health Monitor ===")
                print(f"Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
                print()
                
                results = await self.check_all_services()
                print(self.format_results(results))
                
                await asyncio.sleep(interval)
                
            except KeyboardInterrupt:
                logger.info("Monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"Error during monitoring: {str(e)}")
                await asyncio.sleep(interval)


async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Monitor AI services health')
    parser.add_argument('--service', help='Monitor specific service')
    parser.add_argument('--category', help='Monitor services in category')
    parser.add_argument('--continuous', action='store_true', help='Continuous monitoring')
    parser.add_argument('--interval', type=int, default=30, help='Monitoring interval (seconds)')
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    
    args = parser.parse_args()
    
    # Create monitor
    monitor = ServiceMonitor(SERVICES_CONFIG)
    
    if args.continuous:
        await monitor.monitor_continuous(args.interval)
    else:
        # Single check
        if args.service:
            # Check single service
            if args.service not in monitor.services:
                print(f"Error: Service '{args.service}' not found")
                sys.exit(1)
                
            results = [await monitor.check_service_health(
                args.service, 
                monitor.services[args.service]
            )]
        else:
            # Check all or category
            results = await monitor.check_all_services(args.category)
            
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            print(monitor.format_results(results))


if __name__ == "__main__":
    asyncio.run(main())