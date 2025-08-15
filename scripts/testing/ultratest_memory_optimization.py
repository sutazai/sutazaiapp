#!/usr/bin/env python3
"""
import logging

logger = logging.getLogger(__name__)
ULTRATEST Memory Optimization Validation
Tests system memory usage reduction from 15GB to 8GB target.
"""

import subprocess
import json
import time
from datetime import datetime
from typing import Dict, List, Any

class UltratestMemoryValidator:
    def __init__(self):
        self.memory_data = {}
        
    def get_system_memory_usage(self) -> Dict[str, Any]:
        """Get overall system memory usage"""
        try:
            # Get total system memory info
            result = subprocess.run(['free', '-m'], capture_output=True, text=True, check=True)
            lines = result.stdout.strip().split('\n')
            
            mem_line = lines[1].split()  # Memory line
            swap_line = lines[2].split() if len(lines) > 2 else None  # Swap line
            
            return {
                'total_memory_mb': int(mem_line[1]),
                'used_memory_mb': int(mem_line[2]),
                'free_memory_mb': int(mem_line[3]),
                'available_memory_mb': int(mem_line[6]) if len(mem_line) > 6 else int(mem_line[3]),
                'memory_usage_percentage': (int(mem_line[2]) / int(mem_line[1])) * 100,
                'swap_total_mb': int(swap_line[1]) if swap_line else 0,
                'swap_used_mb': int(swap_line[2]) if swap_line else 0,
            }
        except Exception as e:
            logger.error(f"❌ Failed to get system memory: {e}")
            return {}
    
    def get_docker_memory_usage(self) -> Dict[str, Any]:
        """Get Docker container memory usage"""
        try:
            # Get container memory stats
            result = subprocess.run(
                ['docker', 'stats', '--no-stream', '--format', 'table {{.Name}}\t{{.MemUsage}}\t{{.MemPerc}}'],
                capture_output=True, text=True, check=True
            )
            
            lines = result.stdout.strip().split('\n')[1:]  # Skip header
            containers = []
            total_docker_memory_mb = 0
            
            for line in lines:
                if line.strip():
                    parts = line.split('\t')
                    if len(parts) >= 3:
                        name = parts[0]
                        mem_usage = parts[1]  # e.g., "123.4MiB / 2GiB"
                        mem_perc = parts[2].replace('%', '')  # e.g., "6.17%"
                        
                        # Parse memory usage (extract the used part)
                        if '/' in mem_usage:
                            used_part = mem_usage.split('/')[0].strip()
                            if 'MiB' in used_part:
                                used_mb = float(used_part.replace('MiB', ''))
                            elif 'GiB' in used_part:
                                used_mb = float(used_part.replace('GiB', '')) * 1024
                            else:
                                used_mb = 0
                        else:
                            used_mb = 0
                        
                        containers.append({
                            'name': name,
                            'memory_usage_mb': used_mb,
                            'memory_percentage': float(mem_perc) if mem_perc != 'N/A' else 0
                        })
                        
                        total_docker_memory_mb += used_mb
            
            return {
                'containers': containers,
                'total_containers': len(containers),
                'total_docker_memory_mb': total_docker_memory_mb,
                'total_docker_memory_gb': total_docker_memory_mb / 1024
            }
        except Exception as e:
            logger.error(f"❌ Failed to get Docker memory stats: {e}")
            return {'containers': [], 'total_docker_memory_mb': 0}
    
    def get_process_memory_usage(self) -> Dict[str, Any]:
        """Get top memory-consuming processes"""
        try:
            # Get top 10 memory-consuming processes
            result = subprocess.run(
                ['ps', 'aux', '--sort=-%mem', '--no-headers'],
                capture_output=True, text=True, check=True
            )
            
            lines = result.stdout.strip().split('\n')[:10]  # Top 10 processes
            processes = []
            
            for line in lines:
                parts = line.split()
                if len(parts) >= 11:
                    processes.append({
                        'user': parts[0],
                        'pid': parts[1],
                        'cpu_percent': float(parts[2]),
                        'memory_percent': float(parts[3]),
                        'vsz_kb': int(parts[4]),
                        'rss_kb': int(parts[5]),
                        'memory_mb': int(parts[5]) / 1024,  # Convert KB to MB
                        'command': ' '.join(parts[10:])[:50] + '...' if len(' '.join(parts[10:])) > 50 else ' '.join(parts[10:])
                    })
            
            return {
                'top_processes': processes,
                'total_top_processes_memory_mb': sum(p['memory_mb'] for p in processes)
            }
        except Exception as e:
            logger.error(f"❌ Failed to get process memory: {e}")
            return {'top_processes': [], 'total_top_processes_memory_mb': 0}
    
    def analyze_memory_efficiency(self) -> Dict[str, Any]:
        """Analyze memory usage efficiency and optimization"""
        logger.info("💾 Analyzing memory usage patterns...")
        
        # Get comprehensive memory data
        system_memory = self.get_system_memory_usage()
        docker_memory = self.get_docker_memory_usage()
        process_memory = self.get_process_memory_usage()
        
        # Calculate efficiency metrics
        if system_memory and docker_memory:
            system_total_gb = system_memory.get('total_memory_mb', 0) / 1024
            system_used_gb = system_memory.get('used_memory_mb', 0) / 1024
            docker_used_gb = docker_memory.get('total_docker_memory_gb', 0)
            
            # Calculate Docker memory efficiency (how much of used memory is Docker)
            docker_efficiency = (docker_used_gb / system_used_gb * 100) if system_used_gb > 0 else 0
            
            return {
                'system_memory': system_memory,
                'docker_memory': docker_memory,
                'process_memory': process_memory,
                'analysis': {
                    'system_total_gb': system_total_gb,
                    'system_used_gb': system_used_gb,
                    'docker_used_gb': docker_used_gb,
                    'non_docker_memory_gb': system_used_gb - docker_used_gb,
                    'docker_efficiency_percentage': docker_efficiency,
                    'memory_target_8gb_met': system_used_gb <= 8.0,
                    'previous_usage_15gb': 15.0,  # Claimed previous usage
                    'optimization_achieved_gb': 15.0 - system_used_gb if system_used_gb < 15.0 else 0,
                    'optimization_percentage': ((15.0 - system_used_gb) / 15.0 * 100) if system_used_gb < 15.0 else 0
                }
            }
        else:
            return {'error': 'Failed to collect memory data'}
    
    def run_memory_stress_test(self) -> Dict[str, Any]:
        """Run a brief memory stress test to see system behavior"""
        logger.info("🔬 Running memory stress test...")
        
        # Get baseline memory
        baseline = self.get_system_memory_usage()
        baseline_used_mb = baseline.get('used_memory_mb', 0)
        
        # Create some memory pressure (small test)
        test_data = []
        try:
            # Allocate 100MB of data in chunks
            for i in range(10):
                chunk = 'x' * (10 * 1024 * 1024)  # 10MB chunks
                test_data.append(chunk)
                time.sleep(0.1)
            
            # Measure peak memory
            peak = self.get_system_memory_usage()
            peak_used_mb = peak.get('used_memory_mb', 0)
            
            # Clean up
            del test_data
            
            # Measure recovery
            time.sleep(1)
            recovery = self.get_system_memory_usage()
            recovery_used_mb = recovery.get('used_memory_mb', 0)
            
            return {
                'baseline_memory_mb': baseline_used_mb,
                'peak_memory_mb': peak_used_mb,
                'recovery_memory_mb': recovery_used_mb,
                'memory_increase_mb': peak_used_mb - baseline_used_mb,
                'memory_recovery_mb': peak_used_mb - recovery_used_mb,
                'recovery_percentage': ((peak_used_mb - recovery_used_mb) / (peak_used_mb - baseline_used_mb) * 100) if peak_used_mb > baseline_used_mb else 100
            }
        except Exception as e:
            return {'error': f'Stress test failed: {e}'}
    
    def generate_memory_report(self, analysis: Dict[str, Any]):
        """Generate comprehensive memory optimization report"""
        logger.info("\n" + "=" * 80)
        logger.info("💾 ULTRATEST MEMORY OPTIMIZATION REPORT")
        logger.info("=" * 80)
        logger.info(f"Test Execution Time: {datetime.now().isoformat()}")
        
        if 'error' in analysis:
            logger.error(f"❌ Error: {analysis['error']}")
            return False
        
        system_analysis = analysis.get('analysis', {})
        system_total_gb = system_analysis.get('system_total_gb', 0)
        system_used_gb = system_analysis.get('system_used_gb', 0)
        docker_used_gb = system_analysis.get('docker_used_gb', 0)
        
        logger.info(f"\n🖥️  SYSTEM MEMORY OVERVIEW:")
        logger.info("-" * 50)
        logger.info(f"Total System Memory: {system_total_gb:.2f} GB")
        logger.info(f"Used Memory: {system_used_gb:.2f} GB")
        logger.info(f"Free Memory: {(system_total_gb - system_used_gb):.2f} GB")
        logger.info(f"Memory Usage: {(system_used_gb / system_total_gb * 100):.1f}%")
        
        logger.info(f"\n🐳 DOCKER MEMORY USAGE:")
        logger.info("-" * 50)
        logger.info(f"Docker Memory: {docker_used_gb:.2f} GB")
        logger.info(f"Docker Containers: {analysis.get('docker_memory', {}).get('total_containers', 0)}")
        logger.info(f"Non-Docker Memory: {system_analysis.get('non_docker_memory_gb', 0):.2f} GB")
        logger.info(f"Docker Efficiency: {system_analysis.get('docker_efficiency_percentage', 0):.1f}%")
        
        # Target analysis
        target_met = system_analysis.get('memory_target_8gb_met', False)
        optimization_gb = system_analysis.get('optimization_achieved_gb', 0)
        optimization_pct = system_analysis.get('optimization_percentage', 0)
        
        logger.info(f"\n🎯 OPTIMIZATION TARGET ANALYSIS:")
        logger.info("-" * 50)
        logger.info(f"Current Usage: {system_used_gb:.2f} GB")
        logger.info(f"Target Usage: 8.00 GB")
        logger.info(f"Previous Usage: 15.00 GB (claimed)")
        
        if target_met:
            logger.info("✅ MEMORY TARGET ACHIEVED (<8GB)")
        else:
            logger.info(f"❌ Above target by {system_used_gb - 8.0:.2f} GB")
        
        if optimization_gb > 0:
            logger.info(f"✅ Optimization achieved: {optimization_gb:.2f} GB reduction ({optimization_pct:.1f}%)")
        else:
            logger.info("❌ No optimization from claimed 15GB usage")
        
        # Container breakdown
        docker_containers = analysis.get('docker_memory', {}).get('containers', [])
        if docker_containers:
            logger.info(f"\n📊 TOP MEMORY-CONSUMING CONTAINERS:")
            logger.info("-" * 50)
            
            # Sort by memory usage and show top 10
            sorted_containers = sorted(docker_containers, key=lambda x: x['memory_usage_mb'], reverse=True)[:10]
            for container in sorted_containers:
                name = container['name'][:30] + "..." if len(container['name']) > 30 else container['name']
                memory_mb = container['memory_usage_mb']
                logger.info(f"{name:35} {memory_mb:6.1f} MB")
        
        # Top system processes
        top_processes = analysis.get('process_memory', {}).get('top_processes', [])
        if top_processes:
            logger.info(f"\n🔍 TOP MEMORY-CONSUMING PROCESSES:")
            logger.info("-" * 50)
            for proc in top_processes[:5]:  # Show top 5
                logger.info(f"{proc['memory_mb']:6.1f} MB - {proc['command']}")
        
        # Overall assessment
        logger.info(f"\n🏆 MEMORY OPTIMIZATION ASSESSMENT:")
        logger.info("-" * 50)
        
        achievements = []
        issues = []
        
        if target_met:
            achievements.append(f"Memory usage under 8GB target ({system_used_gb:.2f} GB)")
        else:
            issues.append(f"Memory usage exceeds 8GB target ({system_used_gb:.2f} GB)")
        
        if docker_used_gb < 6.0:
            achievements.append(f"Efficient Docker memory usage ({docker_used_gb:.2f} GB)")
        elif docker_used_gb < 8.0:
            achievements.append(f"Reasonable Docker memory usage ({docker_used_gb:.2f} GB)")
        else:
            issues.append(f"High Docker memory usage ({docker_used_gb:.2f} GB)")
        
        if optimization_gb > 5.0:
            achievements.append(f"Significant optimization achieved ({optimization_gb:.2f} GB saved)")
        elif optimization_gb > 2.0:
            achievements.append(f"Good optimization achieved ({optimization_gb:.2f} GB saved)")
        
        if len(docker_containers) <= 30:
            achievements.append(f"Reasonable container count ({len(docker_containers)} containers)")
        
        logger.info("🎉 ACHIEVEMENTS:")
        for achievement in achievements:
            logger.info(f"   ✅ {achievement}")
        
        if issues:
            logger.info("\n⚠️  AREAS FOR IMPROVEMENT:")
            for issue in issues:
                logger.info(f"   ❌ {issue}")
        
        success_rate = (len(achievements) / (len(achievements) + len(issues))) * 100 if (achievements or issues) else 0
        logger.info(f"\n📈 Memory Optimization Success Rate: {success_rate:.1f}%")
        
        return target_met and success_rate >= 75

def main():
    """Run comprehensive memory optimization validation"""
    logger.info("🚀 Starting ULTRATEST Memory Optimization Validation")
    
    validator = UltratestMemoryValidator()
    
    # Run comprehensive memory analysis
    analysis = validator.analyze_memory_efficiency()
    
    # Run stress test
    stress_results = validator.run_memory_stress_test()
    analysis['stress_test'] = stress_results
    
    # Generate report
    success = validator.generate_memory_report(analysis)
    
    # Save results
    with open('/opt/sutazaiapp/tests/ultratest_memory_report.json', 'w') as f:
        json.dump(analysis, f, indent=2, default=str)
    
    logger.info(f"\n📄 Full report saved to: /opt/sutazaiapp/tests/ultratest_memory_report.json")
    
    if success:
        logger.info("\n🎉 MEMORY OPTIMIZATION VALIDATION SUCCESSFUL!")
        return 0
    else:
        logger.info("\n⚠️  MEMORY OPTIMIZATION NEEDS IMPROVEMENT")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())