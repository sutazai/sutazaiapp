#!/usr/bin/env python3
import os
import sys
import json
import subprocess
import platform
import psutil
import logging
from typing import Dict, List, Any

class SystemValidator:
    def __init__(self):
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s: %(message)s',
            handlers=[
                logging.FileHandler('/var/log/sutazai/system_validation.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Validation report
        self.validation_report = {
            'system_info': {},
            'resource_usage': {},
            'performance_metrics': {},
            'security_checks': {},
            'recommendations': []
        }
    
    def collect_system_info(self) -> Dict[str, Any]:
        """
        Collect comprehensive system information.
        
        Returns:
            Dict containing system details
        """
        system_info = {
            'os': {
                'name': platform.system(),
                'version': platform.version(),
                'release': platform.release(),
                'machine': platform.machine()
            },
            'python': {
                'version': platform.python_version(),
                'implementation': platform.python_implementation(),
                'compiler': platform.python_compiler()
            },
            'hardware': {
                'processor': platform.processor(),
                'architecture': platform.architecture()
            }
        }
        
        self.validation_report['system_info'] = system_info
        return system_info
    
    def check_resource_usage(self) -> Dict[str, float]:
        """
        Monitor system resource usage.
        
        Returns:
            Dict containing CPU, memory, and disk usage
        """
        resource_usage = {
            'cpu_usage': psutil.cpu_percent(interval=1),
            'memory_usage': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent
        }
        
        self.validation_report['resource_usage'] = resource_usage
        
        # Add recommendations based on resource usage
        if resource_usage['cpu_usage'] > 80:
            self.validation_report['recommendations'].append('High CPU usage: Consider optimizing processes')
        
        if resource_usage['memory_usage'] > 85:
            self.validation_report['recommendations'].append('High memory usage: Consider adding more RAM or optimizing memory-intensive applications')
        
        if resource_usage['disk_usage'] > 90:
            self.validation_report['recommendations'].append('Low disk space: Clean up unnecessary files or expand storage')
        
        return resource_usage
    
    def run_performance_tests(self) -> Dict[str, float]:
        """
        Run performance benchmarks.
        
        Returns:
            Dict containing performance metrics
        """
        performance_metrics = {}
        
        # Python startup time
        try:
            startup_time_output = subprocess.check_output([sys.executable, '-m', 'timeit', 'pass'], 
                                                          universal_newlines=True)
            # Extract the last line which contains the time
            time_line = startup_time_output.strip().split('\n')[-1]
            # Extract the numeric value
            time_value = float(time_line.split()[-2].replace('loop,', ''))
            performance_metrics['python_startup_time'] = time_value
        except Exception as e:
            self.logger.error(f"Performance test failed: {e}")
        
        # Disk I/O performance
        try:
            io_test_file = '/tmp/io_test.txt'
            with open(io_test_file, 'wb') as f:
                f.write(b'0' * 1024 * 1024 * 100)  # 100 MB file
            
            start_time = psutil.time.time()
            with open(io_test_file, 'rb') as f:
                f.read()
            
            io_time = psutil.time.time() - start_time
            performance_metrics['disk_read_speed'] = 100 / io_time  # MB/s
            
            os.remove(io_test_file)
        except Exception as e:
            self.logger.error(f"Disk I/O test failed: {e}")
        
        self.validation_report['performance_metrics'] = performance_metrics
        return performance_metrics
    
    def security_checks(self) -> Dict[str, bool]:
        """
        Perform basic security checks.
        
        Returns:
            Dict containing security check results
        """
        security_checks = {
            'firewall_enabled': self._check_firewall(),
            'antivirus_running': self._check_antivirus(),
            'system_updates_current': self._check_system_updates()
        }
        
        self.validation_report['security_checks'] = security_checks
        
        # Add security recommendations
        for check, status in security_checks.items():
            if not status:
                self.validation_report['recommendations'].append(f'Security concern: {check} is not configured optimally')
        
        return security_checks
    
    def _check_firewall(self) -> bool:
        """Check if firewall is enabled."""
        try:
            if platform.system() == 'Linux':
                result = subprocess.run(['sudo', 'ufw', 'status'], capture_output=True, text=True)
                return 'active' in result.stdout.lower()
            return False
        except Exception:
            return False
    
    def _check_antivirus(self) -> bool:
        """Check if antivirus is running."""
        # Placeholder for antivirus check
        return False
    
    def _check_system_updates(self) -> bool:
        """Check if system is up to date."""
        try:
            if platform.system() == 'Linux':
                result = subprocess.run(['apt', 'list', '--upgradable'], capture_output=True, text=True)
                return len(result.stdout.strip().split('\n')) <= 1
            return False
        except Exception:
            return False
    
    def generate_report(self) -> str:
        """
        Generate a comprehensive system validation report.
        
        Returns:
            str: Path to the generated report
        """
        report_path = '/var/log/sutazai/system_validation_report.json'
        
        try:
            with open(report_path, 'w') as f:
                json.dump(self.validation_report, f, indent=2)
            
            # Print summary
            print("\n🔍 System Validation Summary 🔍")
            print(f"OS: {self.validation_report['system_info']['os']['name']} {self.validation_report['system_info']['os']['version']}")
            print(f"Python: {self.validation_report['system_info']['python']['version']}")
            
            print("\n📊 Resource Usage:")
            for resource, usage in self.validation_report['resource_usage'].items():
                print(f"  {resource.replace('_', ' ').title()}: {usage}%")
            
            print("\n🚨 Recommendations:")
            for rec in self.validation_report['recommendations']:
                print(f"  - {rec}")
        
        except Exception as e:
            self.logger.error(f"Report generation failed: {e}")
        
        return report_path

def main():
    validator = SystemValidator()
    
    # Run comprehensive system validation
    validator.collect_system_info()
    validator.check_resource_usage()
    validator.run_performance_tests()
    validator.security_checks()
    
    # Generate report
    report_path = validator.generate_report()
    print(f"\nDetailed report saved to: {report_path}")

if __name__ == '__main__':
    main()