#!/usr/bin/env python3
"""
MCP Server Performance Analysis and Optimization Tool
Analyzes resource consumption, performance metrics, and provides optimization recommendations
for 21 MCP servers deployed in Docker-in-Docker architecture
"""

import json
import subprocess
import time
import statistics
import os
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Any
import psutil
import docker

class MCPPerformanceAnalyzer:
    """Comprehensive performance analyzer for MCP services"""
    
    def __init__(self):
        self.mcp_services = [
            'claude-flow', 'ruv-swarm', 'claude-task-runner', 'files', 
            'context7', 'http_fetch', 'ddg', 'sequentialthinking',
            'nx-mcp', 'extended-memory', 'mcp_ssh', 'ultimatecoder',
            'postgres', 'playwright-mcp', 'memory-bank-mcp', 'puppeteer-mcp (no longer in use)',
            'knowledge-graph-mcp', 'compass-mcp', 'github', 'http', 
            'language-server'
        ]
        
        self.performance_metrics = {}
        self.resource_baselines = {}
        self.optimization_opportunities = []
        self.docker_client = None
        
        try:
            self.docker_client = docker.from_env()
        except:
            print("Warning: Docker client initialization failed")
    
    def collect_container_metrics(self) -> Dict[str, Any]:
        """Collect resource metrics for all containers"""
        metrics = {}
        
        try:
            # Get container stats using docker stats
            cmd = 'docker stats --no-stream --format "{{json .}}"'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line:
                        try:
                            container_stat = json.loads(line)
                            name = container_stat.get('Name', '')
                            
                            # Process CPU percentage
                            cpu_str = container_stat.get('CPUPerc', '0%').replace('%', '')
                            cpu_percent = float(cpu_str) if cpu_str else 0.0
                            
                            # Process memory usage
                            mem_usage_str = container_stat.get('MemUsage', '0MiB / 0MiB')
                            mem_parts = mem_usage_str.split(' / ')
                            mem_current = self._parse_memory_size(mem_parts[0])
                            mem_limit = self._parse_memory_size(mem_parts[1]) if len(mem_parts) > 1 else 0
                            
                            # Process network I/O
                            net_io = container_stat.get('NetIO', '0B / 0B')
                            net_parts = net_io.split(' / ')
                            net_rx = self._parse_memory_size(net_parts[0])
                            net_tx = self._parse_memory_size(net_parts[1]) if len(net_parts) > 1 else 0
                            
                            # Process block I/O
                            block_io = container_stat.get('BlockIO', '0B / 0B')
                            block_parts = block_io.split(' / ')
                            block_read = self._parse_memory_size(block_parts[0])
                            block_write = self._parse_memory_size(block_parts[1]) if len(block_parts) > 1 else 0
                            
                            metrics[name] = {
                                'cpu_percent': cpu_percent,
                                'memory_mb': mem_current / (1024 * 1024),
                                'memory_limit_mb': mem_limit / (1024 * 1024),
                                'memory_percent': (mem_current / mem_limit * 100) if mem_limit > 0 else 0,
                                'network_rx_mb': net_rx / (1024 * 1024),
                                'network_tx_mb': net_tx / (1024 * 1024),
                                'block_read_mb': block_read / (1024 * 1024),
                                'block_write_mb': block_write / (1024 * 1024)
                            }
                        except Exception as e:
                            print(f"Error parsing container stats: {e}")
        except Exception as e:
            print(f"Error collecting container metrics: {e}")
        
        return metrics
    
    def _parse_memory_size(self, size_str: str) -> float:
        """Parse memory size string to bytes"""
        size_str = size_str.strip()
        if not size_str:
            return 0
        
        units = {
            'B': 1, 'KB': 1024, 'KiB': 1024,
            'MB': 1024**2, 'MiB': 1024**2,
            'GB': 1024**3, 'GiB': 1024**3,
            'TB': 1024**4, 'TiB': 1024**4
        }
        
        for unit, multiplier in units.items():
            if unit in size_str:
                try:
                    number = float(size_str.replace(unit, '').strip())
                    return number * multiplier
                except:
                    return 0
        
        try:
            return float(size_str)
        except:
            return 0
    
    def analyze_mcp_services(self) -> Dict[str, Any]:
        """Analyze performance characteristics of MCP services"""
        analysis = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'services': {},
            'summary': {},
            'resource_totals': {
                'total_cpu_percent': 0,
                'total_memory_mb': 0,
                'total_network_io_mb': 0,
                'total_disk_io_mb': 0
            }
        }
        
        # Collect container metrics
        container_metrics = self.collect_container_metrics()
        
        # Analyze MCP-specific containers
        mcp_containers_found = 0
        mcp_resource_usage = {
            'cpu': [],
            'memory': [],
            'network': [],
            'disk': []
        }
        
        for container_name, metrics in container_metrics.items():
            # Check if this is an MCP container
            is_mcp = any(svc in container_name.lower() for svc in self.mcp_services)
            
            if is_mcp or 'mcp' in container_name.lower():
                mcp_containers_found += 1
                
                service_analysis = {
                    'container_name': container_name,
                    'cpu_percent': metrics['cpu_percent'],
                    'memory_mb': metrics['memory_mb'],
                    'memory_percent': metrics['memory_percent'],
                    'network_io_mb': metrics['network_rx_mb'] + metrics['network_tx_mb'],
                    'disk_io_mb': metrics['block_read_mb'] + metrics['block_write_mb'],
                    'resource_efficiency_score': self._calculate_efficiency_score(metrics),
                    'optimization_potential': self._assess_optimization_potential(metrics)
                }
                
                analysis['services'][container_name] = service_analysis
                
                # Track resource usage
                mcp_resource_usage['cpu'].append(metrics['cpu_percent'])
                mcp_resource_usage['memory'].append(metrics['memory_mb'])
                mcp_resource_usage['network'].append(metrics['network_rx_mb'] + metrics['network_tx_mb'])
                mcp_resource_usage['disk'].append(metrics['block_read_mb'] + metrics['block_write_mb'])
        
        # Calculate summary statistics
        if mcp_containers_found > 0:
            analysis['summary'] = {
                'total_mcp_containers': mcp_containers_found,
                'average_cpu_percent': statistics.mean(mcp_resource_usage['cpu']) if mcp_resource_usage['cpu'] else 0,
                'total_memory_mb': sum(mcp_resource_usage['memory']),
                'average_memory_mb': statistics.mean(mcp_resource_usage['memory']) if mcp_resource_usage['memory'] else 0,
                'total_network_io_mb': sum(mcp_resource_usage['network']),
                'total_disk_io_mb': sum(mcp_resource_usage['disk'])
            }
            
            # Update resource totals
            analysis['resource_totals']['total_cpu_percent'] = sum(mcp_resource_usage['cpu'])
            analysis['resource_totals']['total_memory_mb'] = sum(mcp_resource_usage['memory'])
            analysis['resource_totals']['total_network_io_mb'] = sum(mcp_resource_usage['network'])
            analysis['resource_totals']['total_disk_io_mb'] = sum(mcp_resource_usage['disk'])
        
        return analysis
    
    def _calculate_efficiency_score(self, metrics: Dict[str, float]) -> float:
        """Calculate resource efficiency score (0-100)"""
        # Lower resource usage = higher efficiency
        cpu_efficiency = max(0, 100 - metrics['cpu_percent'] * 10)
        memory_efficiency = max(0, 100 - metrics['memory_percent'])
        
        # Weight CPU more heavily as it's often the bottleneck
        efficiency_score = (cpu_efficiency * 0.6 + memory_efficiency * 0.4)
        
        return round(efficiency_score, 2)
    
    def _assess_optimization_potential(self, metrics: Dict[str, float]) -> str:
        """Assess optimization potential based on resource usage"""
        cpu = metrics['cpu_percent']
        memory = metrics['memory_percent']
        
        if cpu < 1 and memory < 10:
            return "HIGH - Very low utilization, candidate for consolidation"
        elif cpu < 5 and memory < 20:
            return "MEDIUM - Low utilization, consider resource reduction"
        elif cpu > 50 or memory > 70:
            return "SCALING - High utilization, may need more resources"
        else:
            return "LOW - Acceptable resource utilization"
    
    def generate_optimization_recommendations(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate specific optimization recommendations"""
        recommendations = []
        
        # Analyze services for consolidation opportunities
        low_usage_services = []
        high_usage_services = []
        idle_services = []
        
        for service_name, service_data in analysis['services'].items():
            if service_data['cpu_percent'] < 0.1 and service_data['memory_mb'] < 50:
                idle_services.append(service_name)
            elif service_data['cpu_percent'] < 1 and service_data['memory_mb'] < 100:
                low_usage_services.append(service_name)
            elif service_data['cpu_percent'] > 50 or service_data['memory_percent'] > 70:
                high_usage_services.append(service_name)
        
        # Generate recommendations
        if idle_services:
            recommendations.append({
                'priority': 'HIGH',
                'category': 'CONSOLIDATION',
                'services': idle_services,
                'recommendation': f"Consider removing or consolidating {len(idle_services)} idle services",
                'potential_savings': f"~{len(idle_services) * 50}MB memory, reduce container overhead",
                'implementation': "Merge functionality into multi-purpose service containers"
            })
        
        if low_usage_services:
            recommendations.append({
                'priority': 'MEDIUM',
                'category': 'RESOURCE_OPTIMIZATION',
                'services': low_usage_services,
                'recommendation': f"Optimize resource allocation for {len(low_usage_services)} low-usage services",
                'potential_savings': f"~{len(low_usage_services) * 100}MB memory reduction possible",
                'implementation': "Reduce memory limits, implement resource pooling"
            })
        
        if high_usage_services:
            recommendations.append({
                'priority': 'HIGH',
                'category': 'SCALING',
                'services': high_usage_services,
                'recommendation': f"Scale or optimize {len(high_usage_services)} high-usage services",
                'potential_impact': "Improve performance and prevent bottlenecks",
                'implementation': "Increase resources or implement horizontal scaling"
            })
        
        # Service-specific recommendations based on functionality overlap
        functionality_groups = {
            'browser_automation': ['playwright-mcp', 'puppeteer-mcp (no longer in use)'],
            'memory_management': ['extended-memory', 'memory-bank-mcp'],
            'http_operations': ['http', 'http_fetch'],
            'task_execution': ['claude-task-runner', 'claude-flow', 'ruv-swarm']
        }
        
        for group_name, services in functionality_groups.items():
            overlapping = [s for s in services if any(s in svc for svc in analysis['services'].keys())]
            if len(overlapping) > 1:
                recommendations.append({
                    'priority': 'MEDIUM',
                    'category': 'FUNCTIONALITY_CONSOLIDATION',
                    'services': overlapping,
                    'recommendation': f"Consolidate {group_name} services",
                    'potential_savings': f"Reduce {len(overlapping)-1} container(s)",
                    'implementation': f"Merge {', '.join(overlapping)} into single service"
                })
        
        return recommendations
    
    def calculate_cost_benefit_analysis(self, analysis: Dict[str, Any], recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate cost-benefit analysis for optimization"""
        cost_benefit = {
            'current_state': {
                'total_containers': len(analysis['services']),
                'total_memory_mb': analysis['resource_totals']['total_memory_mb'],
                'total_cpu_percent': analysis['resource_totals']['total_cpu_percent'],
                'monthly_cost_estimate': self._estimate_monthly_cost(analysis['resource_totals'])
            },
            'optimized_state': {},
            'potential_savings': {},
            'implementation_effort': {}
        }
        
        # Calculate potential optimized state
        containers_to_remove = 0
        memory_savings_mb = 0
        
        for rec in recommendations:
            if rec['category'] == 'CONSOLIDATION':
                containers_to_remove += len(rec['services']) - 1
                memory_savings_mb += len(rec['services']) * 50
            elif rec['category'] == 'RESOURCE_OPTIMIZATION':
                memory_savings_mb += len(rec['services']) * 50
            elif rec['category'] == 'FUNCTIONALITY_CONSOLIDATION':
                containers_to_remove += len(rec['services']) - 1
                memory_savings_mb += (len(rec['services']) - 1) * 100
        
        cost_benefit['optimized_state'] = {
            'total_containers': cost_benefit['current_state']['total_containers'] - containers_to_remove,
            'total_memory_mb': max(0, cost_benefit['current_state']['total_memory_mb'] - memory_savings_mb),
            'total_cpu_percent': cost_benefit['current_state']['total_cpu_percent'] * 0.8,  # Estimate 20% CPU reduction
            'monthly_cost_estimate': self._estimate_monthly_cost({
                'total_memory_mb': max(0, cost_benefit['current_state']['total_memory_mb'] - memory_savings_mb),
                'total_cpu_percent': cost_benefit['current_state']['total_cpu_percent'] * 0.8
            })
        }
        
        cost_benefit['potential_savings'] = {
            'containers_reduced': containers_to_remove,
            'memory_saved_mb': memory_savings_mb,
            'percentage_reduction': round((containers_to_remove / cost_benefit['current_state']['total_containers']) * 100, 1),
            'monthly_cost_savings': cost_benefit['current_state']['monthly_cost_estimate'] - cost_benefit['optimized_state']['monthly_cost_estimate']
        }
        
        cost_benefit['implementation_effort'] = {
            'estimated_hours': containers_to_remove * 2 + len(recommendations) * 1,
            'risk_level': 'MEDIUM' if containers_to_remove > 5 else 'LOW',
            'testing_required': 'Comprehensive integration testing required',
            'rollback_plan': 'Container snapshots before consolidation'
        }
        
        return cost_benefit
    
    def _estimate_monthly_cost(self, resources: Dict[str, float]) -> float:
        """Estimate monthly cost based on resource usage (simplified model)"""
        # Simplified cost model: $0.01 per GB-hour memory, $0.02 per vCPU-hour
        memory_gb = resources.get('total_memory_mb', 0) / 1024
        cpu_cores = resources.get('total_cpu_percent', 0) / 100
        
        memory_cost = memory_gb * 0.01 * 24 * 30  # Monthly
        cpu_cost = cpu_cores * 0.02 * 24 * 30  # Monthly
        
        return round(memory_cost + cpu_cost, 2)
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance analysis report"""
        print("Analyzing MCP server performance...")
        
        # Collect and analyze metrics
        analysis = self.analyze_mcp_services()
        
        # Generate recommendations
        recommendations = self.generate_optimization_recommendations(analysis)
        
        # Calculate cost-benefit
        cost_benefit = self.calculate_cost_benefit_analysis(analysis, recommendations)
        
        # Build complete report
        report = {
            'report_metadata': {
                'generated_at': datetime.now(timezone.utc).isoformat(),
                'analyzer_version': '1.0.0',
                'analysis_type': 'MCP Server Performance Analysis'
            },
            'executive_summary': {
                'total_mcp_services': len(self.mcp_services),
                'active_containers': len(analysis['services']),
                'resource_efficiency': self._calculate_overall_efficiency(analysis),
                'optimization_opportunities': len(recommendations),
                'potential_container_reduction': cost_benefit['potential_savings']['containers_reduced'],
                'potential_monthly_savings': f"${cost_benefit['potential_savings']['monthly_cost_savings']}"
            },
            'performance_analysis': analysis,
            'optimization_recommendations': recommendations,
            'cost_benefit_analysis': cost_benefit,
            'implementation_roadmap': self._generate_implementation_roadmap(recommendations)
        }
        
        return report
    
    def _calculate_overall_efficiency(self, analysis: Dict[str, Any]) -> str:
        """Calculate overall system efficiency rating"""
        if not analysis['services']:
            return "UNKNOWN"
        
        efficiency_scores = [s['resource_efficiency_score'] for s in analysis['services'].values()]
        avg_efficiency = statistics.mean(efficiency_scores) if efficiency_scores else 0
        
        if avg_efficiency > 80:
            return "EXCELLENT"
        elif avg_efficiency > 60:
            return "GOOD"
        elif avg_efficiency > 40:
            return "FAIR"
        else:
            return "POOR"
    
    def _generate_implementation_roadmap(self, recommendations: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Generate implementation roadmap for optimizations"""
        roadmap = []
        
        # Phase 1: Quick wins
        phase1_recs = [r for r in recommendations if r['priority'] == 'HIGH' and r['category'] == 'CONSOLIDATION']
        if phase1_recs:
            roadmap.append({
                'phase': 'Phase 1: Quick Wins (Week 1)',
                'actions': f"Remove/consolidate {sum(len(r['services']) for r in phase1_recs)} idle services",
                'expected_impact': 'Immediate resource reduction, simplified architecture',
                'risk': 'LOW'
            })
        
        # Phase 2: Service consolidation
        phase2_recs = [r for r in recommendations if r['category'] == 'FUNCTIONALITY_CONSOLIDATION']
        if phase2_recs:
            roadmap.append({
                'phase': 'Phase 2: Service Consolidation (Week 2-3)',
                'actions': f"Merge overlapping functionality in {len(phase2_recs)} service groups",
                'expected_impact': 'Reduced complexity, improved maintainability',
                'risk': 'MEDIUM'
            })
        
        # Phase 3: Resource optimization
        phase3_recs = [r for r in recommendations if r['category'] == 'RESOURCE_OPTIMIZATION']
        if phase3_recs:
            roadmap.append({
                'phase': 'Phase 3: Resource Optimization (Week 4)',
                'actions': f"Optimize resource allocation for {sum(len(r['services']) for r in phase3_recs)} services",
                'expected_impact': 'Improved resource utilization, cost reduction',
                'risk': 'LOW'
            })
        
        # Phase 4: Scaling adjustments
        phase4_recs = [r for r in recommendations if r['category'] == 'SCALING']
        if phase4_recs:
            roadmap.append({
                'phase': 'Phase 4: Scaling Adjustments (Ongoing)',
                'actions': f"Scale {sum(len(r['services']) for r in phase4_recs)} high-usage services",
                'expected_impact': 'Better performance, prevent bottlenecks',
                'risk': 'LOW'
            })
        
        return roadmap


def main():
    """Main execution function"""
    analyzer = MCPPerformanceAnalyzer()
    
    # Generate comprehensive report
    report = analyzer.generate_performance_report()
    
    # Save report
    report_path = f"/opt/sutazaiapp/docs/reports/MCP_PERFORMANCE_ANALYSIS_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    # Print summary
    print("\n" + "="*80)
    print("MCP SERVER PERFORMANCE ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nExecutive Summary:")
    print(f"  - Total MCP Services Defined: {report['executive_summary']['total_mcp_services']}")
    print(f"  - Active Containers Found: {report['executive_summary']['active_containers']}")
    print(f"  - Resource Efficiency: {report['executive_summary']['resource_efficiency']}")
    print(f"  - Optimization Opportunities: {report['executive_summary']['optimization_opportunities']}")
    print(f"  - Potential Container Reduction: {report['executive_summary']['potential_container_reduction']}")
    print(f"  - Potential Monthly Savings: {report['executive_summary']['potential_monthly_savings']}")
    
    print(f"\nFull report saved to: {report_path}")
    
    # Print top recommendations
    if report['optimization_recommendations']:
        print("\nTop Optimization Recommendations:")
        for i, rec in enumerate(report['optimization_recommendations'][:3], 1):
            print(f"\n{i}. [{rec['priority']}] {rec['recommendation']}")
            print(f"   Category: {rec['category']}")
            print(f"   Services: {', '.join(rec['services'][:3])}...")
            print(f"   Impact: {rec.get('potential_savings', rec.get('potential_impact', 'N/A'))}")
    
    return report


if __name__ == "__main__":
    main()