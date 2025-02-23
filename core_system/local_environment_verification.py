import importlib
import json
import logging
import os
import platform
import socket
import subprocess
import sys
import uuid

import pkg_resources
import psutil

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

class LocalEnvironmentVerifier:
    def __init__(self):
        self.environment_report = {
            'system_info': {},
            'python_info': {},
            'network_info': {},
            'dependencies': {},
            'potential_issues': []
        }

    def verify_system_info(self):
        """Collect comprehensive system information."""
        try:
            self.environment_report['system_info'] = {
                'os_name': platform.system(),
                'os_version': platform.version(),
                'os_release': platform.release(),
                'machine_type': platform.machine(),
                'processor': platform.processor(),
                'cpu_count': os.cpu_count(),
                'total_memory': f"{psutil.virtual_memory().total / (1024**3):.2f} GB",
                'hostname': socket.gethostname(),
                'unique_id': str(uuid.getnode())
            }
        except Exception as e:
            self.environment_report['potential_issues'].append(f"System info collection error: {str(e)}")

    def verify_python_environment(self):
        """Verify Python environment details."""
        try:
            self.environment_report['python_info'] = {
                'version': sys.version,
                'executable_path': sys.executable,
                'prefix': sys.prefix,
                'platform': sys.platform
            }
        except Exception as e:
            self.environment_report['potential_issues'].append(f"Python environment verification error: {str(e)}")

    def check_network_connectivity(self):
        """Check network connectivity and DNS resolution."""
        try:
            # Test internet connectivity
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            self.environment_report['network_info']['internet_connectivity'] = True
            
            # DNS resolution test
            socket.gethostbyname('google.com')
            self.environment_report['network_info']['dns_resolution'] = True
        except (socket.error, socket.timeout) as e:
            self.environment_report['network_info']['internet_connectivity'] = False
            self.environment_report['network_info']['dns_resolution'] = False
            self.environment_report['potential_issues'].append(f"Network connectivity issue: {str(e)}")

    def verify_dependencies(self):
        """Verify installed Python packages and their versions."""
        try:
            dependencies = {}
            for package in pkg_resources.working_set:
                dependencies[package.key] = package.version
            
            self.environment_report['dependencies'] = dependencies
            
            # Check for critical dependencies
            critical_dependencies = [
                'fastapi', 'uvicorn', 'sqlalchemy', 'alembic', 
                'pydantic', 'redis', 'psycopg2', 'pytest'
            ]
            
            missing_dependencies = [dep for dep in critical_dependencies if dep not in dependencies]
            if missing_dependencies:
                self.environment_report['potential_issues'].append(f"Missing critical dependencies: {missing_dependencies}")
        
        except Exception as e:
            self.environment_report['potential_issues'].append(f"Dependency verification error: {str(e)}")

    def check_environment_variables(self):
        """Check critical environment variables."""
        critical_env_vars = [
            'DATABASE_URL', 'REDIS_URL', 'SECRET_KEY', 
            'DEBUG', 'ENVIRONMENT'
        ]
        
        for var in critical_env_vars:
            value = os.environ.get(var)
            if not value:
                self.environment_report['potential_issues'].append(f"Missing environment variable: {var}")

    def generate_verification_report(self):
        """Generate a comprehensive environment verification report."""
        report_path = 'local_environment_verification_report.json'
        
        with open(report_path, 'w') as f:
            json.dump(self.environment_report, f, indent=2)
        
        logging.info(f"Environment verification report saved to {report_path}")
        
        # Log potential issues
        if self.environment_report['potential_issues']:
            logging.warning("Potential environment issues detected:")
            for issue in self.environment_report['potential_issues']:
                logging.warning(f"- {issue}")
        else:
            logging.info("No potential environment issues detected.")

def main():
    verifier = LocalEnvironmentVerifier()
    
    try:
        verifier.verify_system_info()
        verifier.verify_python_environment()
        verifier.check_network_connectivity()
        verifier.verify_dependencies()
        verifier.check_environment_variables()
        verifier.generate_verification_report()
    except Exception as e:
        logging.error(f"Local environment verification failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main() 