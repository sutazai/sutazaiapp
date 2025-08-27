#!/usr/bin/env python3
"""
SutazAI Hardware Memory Optimization Monitor
Real-time monitoring of memory usage and container efficiency
"""

import subprocess
import time
import json
import sys
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional
import re

class MemoryOptimizationMonitor:
    """Monitor system memory usage and container efficiency"""
    
    def __init__(self):
        self.start_time = datetime.now(timezone.utc)
        self.baseline_memory = self.get_system_memory_usage()
        self.baseline_containers = self.get_container_count()
        
    def get_system_memory_usage(self) -> Dict[str, float]:
        """Get system memory usage in MB and percentage"""
        try:
            result = subprocess.run(['free', '-m'], capture_output=True, text=True, check=True)
            lines = result.stdout.strip().split('\n')
            memory_line = lines[1]  # Mem: line
            
            parts = memory_line.split()
            total = float(parts[1])
            used = float(parts[2])
            available = float(parts[6])
            
            return {
                'total_mb': total,
                'used_mb': used,
                'available_mb': available,
                'used_percent': (used / total) * 100,
                'available_percent': (available / total) * 100
            }
        except Exception as e:
            return {'error': str(e)}
    
    def get_container_count(self) -> int:
        """Get current running container count"""
        try:
            result = subprocess.run(['docker', 'ps', '-q'], capture_output=True, text=True, check=True)
            return len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0
        except Exception:
            return 0
    
    def get_container_memory_usage(self) -> List[Dict[str, str]]:
        """Get memory usage for all containers"""
        try:
            result = subprocess.run([
                'docker', 'stats', '--no-stream', '--format',
                'table {{.Name}}\t{{.MemUsage}}\t{{.MemPerc}}\t{{.CPUPerc}}'
            ], capture_output=True, text=True, check=True)
            
            lines = result.stdout.strip().split('\n')
            containers = []
            
            for line in lines[1:]:  # Skip header
                parts = line.split('\t')
                if len(parts) >= 4:
                    name = parts[0].strip()
                    mem_usage = parts[1].strip()
                    mem_percent = parts[2].strip()
                    cpu_percent = parts[3].strip()
                    
                    # Extract memory in MB
                    mem_match = re.search(r'(\d+\.?\d*)\s*MiB', mem_usage)
                    mem_mb = float(mem_match.group(1)) if mem_match else 0.0
                    
                    containers.append({
                        'name': name,
                        'memory_usage': mem_usage,
                        'memory_mb': mem_mb,
                        'memory_percent': mem_percent,
                        'cpu_percent': cpu_percent
                    })
            
            return sorted(containers, key=lambda x: x['memory_mb'], reverse=True)
        except Exception as e:
            return [{'error': str(e)}]
    
    def identify_mcp_pollution(self) -> Dict[str, any]:
        """Identify MCP container pollution"""
        try:
            result = subprocess.run([
                'docker', 'ps', '--format', 'table {{.Names}}\t{{.Image}}'
            ], capture_output=True, text=True, check=True)
            
            lines = result.stdout.strip().split('\n')
            mcp_containers = []
            total_mcp_memory = 0.0
            
            for line in lines[1:]:  # Skip header
                parts = line.split('\t')
                if len(parts) >= 2:
                    name = parts[0].strip()
                    image = parts[1].strip()
                    
                    # Check for MCP pollution patterns
                    if any(pattern in image.lower() for pattern in ['mcp/', 'fetch', 'duckduckgo', 'sequential']) or \
                        
                        # Get memory usage for this container
                        try:
                            mem_result = subprocess.run([
                                'docker', 'stats', '--no-stream', '--format', '{{.MemUsage}}', name
                            ], capture_output=True, text=True, check=True)
                            
                            mem_match = re.search(r'(\d+\.?\d*)\s*MiB', mem_result.stdout)
                            mem_mb = float(mem_match.group(1)) if mem_match else 0.0
                            total_mcp_memory += mem_mb
                            
                            mcp_containers.append({
                                'name': name,
                                'image': image,
                                'memory_mb': mem_mb,
                                'category': self._categorize_mcp_container(name, image)
                            })
                        except Exception:
                            mcp_containers.append({
                                'name': name,
                                'image': image,
                                'memory_mb': 0.0,
                                'category': 'unknown'
                            })
            
            return {
                'total_containers': len(mcp_containers),
                'total_memory_mb': total_mcp_memory,
                'containers': mcp_containers,
                'categories': self._group_by_category(mcp_containers)
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _categorize_mcp_container(self, name: str, image: str) -> str:
        """Categorize MCP container type"""
        elif 'fetch' in image:
            return 'fetch-duplicate'
        elif 'duckduckgo' in image:
            return 'duckduckgo-duplicate'
        elif 'sequential' in image:
            return 'sequentialthinking-duplicate'
        else:
            return 'other-mcp'
    
    def _group_by_category(self, containers: List[Dict]) -> Dict[str, Dict]:
        """Group containers by category"""
        categories = {}
        for container in containers:
            category = container['category']
            if category not in categories:
                categories[category] = {'count': 0, 'total_memory_mb': 0.0, 'containers': []}
            
            categories[category]['count'] += 1
            categories[category]['total_memory_mb'] += container['memory_mb']
            categories[category]['containers'].append(container['name'])
        
        return categories
    
    def get_optimization_metrics(self) -> Dict[str, any]:
        """Calculate optimization metrics vs baseline"""
        current_memory = self.get_system_memory_usage()
        current_containers = self.get_container_count()
        
        memory_saved_mb = self.baseline_memory['used_mb'] - current_memory['used_mb']
        memory_saved_percent = self.baseline_memory['used_percent'] - current_memory['used_percent']
        containers_removed = self.baseline_containers - current_containers
        
        return {
            'baseline': {
                'memory_used_mb': self.baseline_memory['used_mb'],
                'memory_used_percent': self.baseline_memory['used_percent'],
                'container_count': self.baseline_containers
            },
            'current': {
                'memory_used_mb': current_memory['used_mb'],
                'memory_used_percent': current_memory['used_percent'],
                'container_count': current_containers
            },
            'optimization': {
                'memory_saved_mb': memory_saved_mb,
                'memory_saved_percent': memory_saved_percent,
                'containers_removed': containers_removed,
                'target_achieved': current_memory['used_percent'] < 50.0
            }
        }
    
    def generate_report(self) -> str:
        """Generate comprehensive monitoring report"""
        system_memory = self.get_system_memory_usage()
        container_memory = self.get_container_memory_usage()
        mcp_pollution = self.identify_mcp_pollution()
        optimization_metrics = self.get_optimization_metrics()
        
        report = f"""
🔧 SutazAI Hardware Memory Optimization Monitor
═══════════════════════════════════════════════

📊 SYSTEM MEMORY STATUS
├─ Total Memory: {system_memory['total_mb']:.1f} MB
├─ Used Memory: {system_memory['used_mb']:.1f} MB ({system_memory['used_percent']:.1f}%)
├─ Available: {system_memory['available_mb']:.1f} MB ({system_memory['available_percent']:.1f}%)
└─ Target Met: {'✅ YES' if system_memory['used_percent'] < 50.0 else '❌ NO'} (Target: <50%)

🐳 CONTAINER ANALYSIS
├─ Total Containers: {self.get_container_count()}
├─ Top Memory Consumers:"""

        # Top 10 memory consumers
        for i, container in enumerate(container_memory[:10], 1):
            if 'error' not in container:
                report += f"\n│  {i:2d}. {container['name']:<30} {container['memory_mb']:>6.1f} MB"
        
        report += f"""

🚨 MCP CONTAINER POLLUTION
├─ Total MCP Containers: {mcp_pollution.get('total_containers', 0)}
├─ Total Memory Used: {mcp_pollution.get('total_memory_mb', 0):.1f} MB
└─ Categories:"""

        if 'categories' in mcp_pollution:
            for category, data in mcp_pollution['categories'].items():
                report += f"\n   ├─ {category}: {data['count']} containers, {data['total_memory_mb']:.1f} MB"

        report += f"""

📈 OPTIMIZATION METRICS (vs Baseline)
├─ Memory Saved: {optimization_metrics['optimization']['memory_saved_mb']:+.1f} MB
├─ Percentage Change: {optimization_metrics['optimization']['memory_saved_percent']:+.1f}%
├─ Containers Removed: {optimization_metrics['optimization']['containers_removed']:+d}
└─ Target Achievement: {'✅ SUCCESS' if optimization_metrics['optimization']['target_achieved'] else '⏳ IN PROGRESS'}

⏰ Monitor Duration: {(datetime.now(timezone.utc) - self.start_time).total_seconds():.0f} seconds
📅 Report Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}
"""
        return report
    
    def save_metrics_json(self, filepath: str):
        """Save detailed metrics to JSON file"""
        metrics = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'system_memory': self.get_system_memory_usage(),
            'container_memory': self.get_container_memory_usage(),
            'mcp_pollution': self.identify_mcp_pollution(),
            'optimization_metrics': self.get_optimization_metrics(),
            'container_count': self.get_container_count()
        }
        
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def continuous_monitor(self, duration_minutes: int = 30, interval_seconds: int = 30):
        """Run continuous monitoring for specified duration"""
        print(f"🔧 Starting {duration_minutes}-minute memory optimization monitor...")
        print(f"📊 Monitoring interval: {interval_seconds} seconds")
        print("=" * 60)
        
        end_time = time.time() + (duration_minutes * 60)
        
        while time.time() < end_time:
            # Clear screen (works on most terminals)
            print("\033[2J\033[H")
            
            # Generate and display report
            report = self.generate_report()
            print(report)
            
            # Save metrics snapshot
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            metrics_file = f"/opt/sutazaiapp/logs/memory_metrics_{timestamp}.json"
            try:
                self.save_metrics_json(metrics_file)
                print(f"📁 Metrics saved: {metrics_file}")
            except Exception as e:
                print(f"⚠️  Failed to save metrics: {e}")
            
            # Wait for next interval
            remaining_time = end_time - time.time()
            if remaining_time > 0:
                sleep_time = min(interval_seconds, remaining_time)
                print(f"\n⏳ Next update in {sleep_time:.0f} seconds...")
                time.sleep(sleep_time)
        
        print("\n🏁 Monitoring completed!")

def main():
    """Main execution function"""
    if len(sys.argv) > 1:
        if sys.argv[1] == '--continuous':
            duration = int(sys.argv[2]) if len(sys.argv) > 2 else 30
            monitor = MemoryOptimizationMonitor()
            monitor.continuous_monitor(duration_minutes=duration)
        elif sys.argv[1] == '--json':
            filepath = sys.argv[2] if len(sys.argv) > 2 else '/opt/sutazaiapp/logs/current_metrics.json'
            monitor = MemoryOptimizationMonitor()
            monitor.save_metrics_json(filepath)
            print(f"Metrics saved to: {filepath}")
        else:
            print("Usage: memory_optimization_monitor.py [--continuous [minutes] | --json [filepath]]")
    else:
        # Single report
        monitor = MemoryOptimizationMonitor()
        report = monitor.generate_report()
        print(report)

if __name__ == "__main__":
    main()