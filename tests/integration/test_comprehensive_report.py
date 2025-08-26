#!/usr/bin/env python3
"""
Comprehensive SutazAI Testing Report Generator
Aggregates all test results and creates a master report
"""

import json
import os
import time
from datetime import datetime
from typing import Dict, List, Any
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveReportGenerator:
    """Generates comprehensive testing and validation reports"""
    
    def __init__(self):
        self.report_data = {
            'timestamp': datetime.now().isoformat(),
            'test_suite': 'SutazAI Testing QA Validator',
            'summary': {},
            'detailed_results': {},
            'recommendations': [],
            'overall_assessment': 'pending'
        }
        
        self.report_dir = '/opt/sutazaiapp/backend/tests/reports'
        os.makedirs(self.report_dir, exist_ok=True)
    
    def load_test_results(self) -> None:
        """Load all available test results"""
        logger.info("Loading test results from all test suites...")
        
        # Test result files to look for
        test_files = [
            'database_tests.json',
            'network_validation.json',
            'deployment_validation.json'
        ]
        
        for test_file in test_files:
            file_path = os.path.join(self.report_dir, test_file)
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    test_name = test_file.replace('.json', '')
                    self.report_data['detailed_results'][test_name] = data
                    logger.info(f"Loaded {test_name} results")
                except Exception as e:
                    logger.error(f"Failed to load {test_file}: {str(e)}")
            else:
                logger.warning(f"Test result file not found: {test_file}")
    
    def analyze_results(self) -> None:
        """Analyze all test results and generate summary"""
        logger.info("Analyzing comprehensive test results...")
        
        summary = {
            'total_test_suites': len(self.report_data['detailed_results']),
            'passed_suites': 0,
            'failed_suites': 0,
            'warning_suites': 0,
            'total_individual_tests': 0,
            'passed_individual_tests': 0,
            'failed_individual_tests': 0,
            'critical_issues': [],
            'warnings': [],
            'success_rate': 0.0
        }
        
        # Analyze each test suite
        for suite_name, suite_data in self.report_data['detailed_results'].items():
            logger.info(f"Analyzing {suite_name}...")
            
            if suite_name == 'database_tests':
                self._analyze_database_tests(suite_data, summary)
            elif suite_name == 'network_validation':
                self._analyze_network_tests(suite_data, summary)
            elif suite_name == 'deployment_validation':
                self._analyze_deployment_tests(suite_data, summary)
        
        # Calculate overall success rate
        if summary['total_individual_tests'] > 0:
            summary['success_rate'] = (summary['passed_individual_tests'] / summary['total_individual_tests']) * 100
        
        # Determine overall assessment
        if summary['failed_suites'] == 0 and len(summary['critical_issues']) == 0:
            if summary['warning_suites'] == 0:
                self.report_data['overall_assessment'] = 'excellent'
            else:
                self.report_data['overall_assessment'] = 'good'
        elif summary['failed_suites'] <= 1 and len(summary['critical_issues']) <= 2:
            self.report_data['overall_assessment'] = 'fair'
        else:
            self.report_data['overall_assessment'] = 'poor'
        
        self.report_data['summary'] = summary
    
    def _analyze_database_tests(self, data: Dict, summary: Dict) -> None:
        """Analyze database test results"""
        overall_status = data.get('overall', {}).get('status', 'unknown')
        
        if overall_status == 'passed':
            summary['passed_suites'] += 1
        else:
            summary['failed_suites'] += 1
            summary['critical_issues'].append('Database connectivity issues detected')
        
        # Count individual tests
        for db_name in ['postgres', 'redis', 'neo4j', 'chromadb']:
            if db_name in data:
                db_tests = data[db_name].get('tests', [])
                for test in db_tests:
                    summary['total_individual_tests'] += 1
                    if test.get('status') == 'passed':
                        summary['passed_individual_tests'] += 1
                    else:
                        summary['failed_individual_tests'] += 1
                        summary['critical_issues'].append(f"{db_name}: {test.get('message', 'Unknown error')}")
    
    def _analyze_network_tests(self, data: Dict, summary: Dict) -> None:
        """Analyze network validation results"""
        overall_status = data.get('overall_status', 'unknown')
        
        if overall_status == 'passed':
            summary['passed_suites'] += 1
        elif overall_status == 'warning':
            summary['warning_suites'] += 1
        else:
            summary['failed_suites'] += 1
        
        # Check port availability
        port_tests = data.get('port_availability', {})
        for port, info in port_tests.items():
            summary['total_individual_tests'] += 1
            if info.get('accessible'):
                summary['passed_individual_tests'] += 1
            else:
                summary['failed_individual_tests'] += 1
                summary['critical_issues'].append(f"Port {port} not accessible")
        
        # Check for port conflicts
        conflicts = data.get('port_conflicts', [])
        for conflict in conflicts:
            if 'port' in conflict and conflict['port'] in [5432, 6379, 7474, 7687, 8001]:
                summary['warnings'].append(f"Port conflict detected: {conflict.get('message', '')}")
    
    def _analyze_deployment_tests(self, data: Dict, summary: Dict) -> None:
        """Analyze deployment validation results"""
        overall_status = data.get('overall_status', 'unknown')
        
        if overall_status == 'passed':
            summary['passed_suites'] += 1
        elif overall_status == 'warning':
            summary['warning_suites'] += 1
        else:
            summary['failed_suites'] += 1
        
        # Check container health
        container_health = data.get('container_health', {})
        for container, health in container_health.items():
            summary['total_individual_tests'] += 1
            if health.get('accessible') and health.get('status') == 'running':
                summary['passed_individual_tests'] += 1
            else:
                summary['failed_individual_tests'] += 1
                summary['critical_issues'].append(f"Container {container} health issue")
        
        # Check log analysis
        log_analysis = data.get('log_analysis', {})
        for container, analysis in log_analysis.items():
            if isinstance(analysis, dict):
                error_count = analysis.get('error_count', 0)
                if error_count > 0:
                    summary['warnings'].append(f"{container}: {error_count} errors in logs")
    
    def generate_recommendations(self) -> None:
        """Generate recommendations based on test results"""
        logger.info("Generating recommendations...")
        
        recommendations = []
        summary = self.report_data['summary']
        
        # Database recommendations
        if 'database_tests' in self.report_data['detailed_results']:
            db_data = self.report_data['detailed_results']['database_tests']
            for db_name in ['postgres', 'redis', 'neo4j', 'chromadb']:
                if db_name in db_data and db_data[db_name].get('status') != 'passed':
                    recommendations.append({
                        'category': 'Database',
                        'priority': 'High',
                        'issue': f'{db_name.capitalize()} connection failed',
                        'recommendation': f'Check {db_name} service status, credentials, and network connectivity',
                        'action_items': [
                            f'Verify {db_name} container is running and healthy',
                            f'Check {db_name} logs for error messages',
                            f'Validate {db_name} configuration and credentials',
                            f'Test {db_name} connectivity from host system'
                        ]
                    })
        
        # Network recommendations
        if summary['failed_individual_tests'] > 0:
            recommendations.append({
                'category': 'Network',
                'priority': 'Medium',
                'issue': 'Network connectivity issues detected',
                'recommendation': 'Review Docker networking and port configurations',
                'action_items': [
                    'Check Docker network configuration',
                    'Verify port mappings are correct',
                    'Review firewall settings',
                    'Test inter-container communication'
                ]
            })
        
        # Performance recommendations
        if 'deployment_validation' in self.report_data['detailed_results']:
            deployment_data = self.report_data['detailed_results']['deployment_validation']
            resources = deployment_data.get('resource_usage', {})
            
            if 'memory' in resources:
                mem_usage = resources['memory'].get('usage_percent', 0)
                if mem_usage > 80:
                    recommendations.append({
                        'category': 'Performance',
                        'priority': 'Medium',
                        'issue': f'High memory usage detected ({mem_usage:.1f}%)',
                        'recommendation': 'Monitor and optimize memory usage',
                        'action_items': [
                            'Review container memory limits',
                            'Identify memory-intensive processes',
                            'Consider scaling horizontally',
                            'Implement memory monitoring alerts'
                        ]
                    })
        
        # Security recommendations
        recommendations.append({
            'category': 'Security',
            'priority': 'High',
            'issue': 'Security hardening recommendations',
            'recommendation': 'Implement additional security measures',
            'action_items': [
                'Rotate default passwords and API keys',
                'Enable SSL/TLS for all database connections',
                'Implement network segmentation',
                'Regular security vulnerability scans',
                'Enable audit logging for all services'
            ]
        })
        
        # Monitoring recommendations
        recommendations.append({
            'category': 'Monitoring',
            'priority': 'Medium',
            'issue': 'Comprehensive monitoring setup',
            'recommendation': 'Implement full observability stack',
            'action_items': [
                'Deploy Prometheus metrics collection',
                'Set up Grafana dashboards',
                'Configure alerting rules',
                'Implement distributed tracing',
                'Set up log aggregation and analysis'
            ]
        })
        
        self.report_data['recommendations'] = recommendations
    
    def generate_report(self) -> str:
        """Generate comprehensive test report"""
        logger.info("Generating comprehensive test report...")
        
        report = []
        report.append("=" * 100)
        report.append("SUTAZAI TESTING QA VALIDATOR - COMPREHENSIVE REPORT")
        report.append("=" * 100)
        report.append(f"Generated: {self.report_data['timestamp']}")
        report.append(f"Test Suite: {self.report_data['test_suite']}")
        report.append(f"Overall Assessment: {self.report_data['overall_assessment'].upper()}")
        report.append("")
        
        # Executive Summary
        summary = self.report_data['summary']
        report.append("EXECUTIVE SUMMARY:")
        report.append("-" * 50)
        report.append(f"Total Test Suites: {summary['total_test_suites']}")
        report.append(f"Passed Suites: {summary['passed_suites']}")
        report.append(f"Failed Suites: {summary['failed_suites']}")
        report.append(f"Warning Suites: {summary['warning_suites']}")
        report.append(f"Total Individual Tests: {summary['total_individual_tests']}")
        report.append(f"Success Rate: {summary['success_rate']:.1f}%")
        report.append("")
        
        # Critical Issues
        if summary['critical_issues']:
            report.append("CRITICAL ISSUES:")
            report.append("-" * 50)
            for issue in summary['critical_issues'][:10]:  # Show top 10
                report.append(f"  🔴 {issue}")
            if len(summary['critical_issues']) > 10:
                report.append(f"  ... and {len(summary['critical_issues']) - 10} more issues")
            report.append("")
        
        # Warnings
        if summary['warnings']:
            report.append("WARNINGS:")
            report.append("-" * 50)
            for warning in summary['warnings'][:10]:  # Show top 10
                report.append(f"  🟡 {warning}")
            if len(summary['warnings']) > 10:
                report.append(f"  ... and {len(summary['warnings']) - 10} more warnings")
            report.append("")
        
        # Test Suite Details
        report.append("DETAILED TEST RESULTS:")
        report.append("-" * 50)
        
        for suite_name, suite_data in self.report_data['detailed_results'].items():
            suite_title = suite_name.replace('_', ' ').title()
            report.append(f"\n{suite_title}:")
            report.append("~" * (len(suite_title) + 1))
            
            if suite_name == 'database_tests':
                self._format_database_results(suite_data, report)
            elif suite_name == 'network_validation':
                self._format_network_results(suite_data, report)
            elif suite_name == 'deployment_validation':
                self._format_deployment_results(suite_data, report)
        
        # Recommendations
        report.append("\nRECOMMendations:")
        report.append("-" * 50)
        
        for rec in self.report_data['recommendations']:
            priority_symbol = "🔴" if rec['priority'] == 'High' else "🟡" if rec['priority'] == 'Medium' else "🟢"
            report.append(f"\n{priority_symbol} {rec['category']} - {rec['priority']} Priority")
            report.append(f"Issue: {rec['issue']}")
            report.append(f"Recommendation: {rec['recommendation']}")
            report.append("Action Items:")
            for action in rec['action_items']:
                report.append(f"  • {action}")
        
        # Footer
        report.append("\n" + "=" * 100)
        report.append("Generated by SutazAI Testing QA Validator")
        report.append("For questions or support, check the project documentation")
        report.append("=" * 100)
        
        return "\n".join(report)
    
    def _format_database_results(self, data: Dict, report: List[str]) -> None:
        """Format database test results for report"""
        overall = data.get('overall', {})
        report.append(f"Status: {overall.get('status', 'unknown').upper()}")
        report.append(f"Success Rate: {overall.get('success_rate', 0):.1f}%")
        
        for db_name in ['postgres', 'redis', 'neo4j', 'chromadb']:
            if db_name in data:
                db_status = data[db_name].get('status', 'unknown')
                symbol = "✅" if db_status == 'passed' else "❌"
                report.append(f"  {symbol} {db_name.capitalize()}: {db_status}")
    
    def _format_network_results(self, data: Dict, report: List[str]) -> None:
        """Format network test results for report"""
        status = data.get('overall_status', 'unknown')
        report.append(f"Status: {status.upper()}")
        
        port_tests = data.get('port_availability', {})
        accessible_ports = sum(1 for info in port_tests.values() if info.get('accessible'))
        report.append(f"Port Accessibility: {accessible_ports}/{len(port_tests)} ports accessible")
        
        conflicts = len(data.get('port_conflicts', []))
        report.append(f"Port Conflicts: {conflicts} detected")
    
    def _format_deployment_results(self, data: Dict, report: List[str]) -> None:
        """Format deployment test results for report"""
        status = data.get('overall_status', 'unknown')
        report.append(f"Status: {status.upper()}")
        
        deployment = data.get('deployment_status', {})
        if 'running_containers' in deployment:
            running = len(deployment['running_containers'])
            expected = deployment.get('expected_containers', 0)
            report.append(f"Containers: {running}/{expected} running")
        
        # Resource usage
        resources = data.get('resource_usage', {})
        if 'memory' in resources:
            mem_usage = resources['memory'].get('usage_percent', 0)
            report.append(f"Memory Usage: {mem_usage:.1f}%")
        
        if 'cpu' in resources:
            cpu_load = resources['cpu'].get('load_1min', 0)
            report.append(f"CPU Load (1m): {cpu_load}")
    
    def save_report(self, report_text: str) -> str:
        """Save the comprehensive report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON data
        json_file = os.path.join(self.report_dir, f'comprehensive_report_{timestamp}.json')
        with open(json_file, 'w') as f:
            json.dump(self.report_data, f, indent=2, default=str)
        
        # Save text report
        text_file = os.path.join(self.report_dir, f'comprehensive_report_{timestamp}.txt')
        with open(text_file, 'w') as f:
            f.write(report_text)
        
        # Save as latest
        latest_json = os.path.join(self.report_dir, 'latest_comprehensive_report.json')
        latest_text = os.path.join(self.report_dir, 'latest_comprehensive_report.txt')
        
        with open(latest_json, 'w') as f:
            json.dump(self.report_data, f, indent=2, default=str)
        
        with open(latest_text, 'w') as f:
            f.write(report_text)
        
        return text_file
    
    def run_comprehensive_analysis(self) -> str:
        """Run complete comprehensive analysis and generate report"""
        logger.info("Starting comprehensive analysis...")
        
        try:
            self.load_test_results()
            self.analyze_results()
            self.generate_recommendations()
            report_text = self.generate_report()
            report_file = self.save_report(report_text)
            
            logger.info(f"Comprehensive analysis complete. Report saved to: {report_file}")
            return report_text
            
        except Exception as e:
            logger.error(f"Comprehensive analysis failed: {str(e)}")
            raise

def main():
    """Main execution function"""
    generator = ComprehensiveReportGenerator()
    
    try:
        report = generator.run_comprehensive_analysis()
        logger.info(report)
        
        # Determine exit code based on assessment
        assessment = generator.report_data['overall_assessment']
        if assessment in ['excellent', 'good']:
            return 0
        elif assessment == 'fair':
            return 1
        else:
            return 2
            
    except Exception as e:
        logger.error(f"Report generation failed: {str(e)}")
        return 3

if __name__ == "__main__":
    exit(main())