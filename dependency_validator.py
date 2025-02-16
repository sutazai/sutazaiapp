import sys
import pkg_resources
import subprocess
import logging
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

class DependencyValidator:
    def __init__(self, requirements_file='requirements.txt'):
        self.requirements_file = requirements_file
        self.issues = []

    def validate_dependencies(self) -> Dict[str, List[str]]:
        """
        Validate installed dependencies against requirements.txt
        """
        results = {
            'missing_dependencies': [],
            'version_mismatches': [],
            'outdated_packages': []
        }

        try:
            with open(self.requirements_file, 'r') as f:
                required_packages = [line.strip().split('==')[0] for line in f if line.strip() and not line.startswith('#')]

            for package in required_packages:
                try:
                    pkg_resources.get_distribution(package)
                except pkg_resources.DistributionNotFound:
                    results['missing_dependencies'].append(package)

            # Check for outdated packages
            try:
                outdated_output = subprocess.check_output([sys.executable, '-m', 'pip', 'list', '--outdated']).decode()
                for line in outdated_output.split('\n')[2:]:
                    if line.strip():
                        parts = line.split()
                        results['outdated_packages'].append(f"{parts[0]} (Current: {parts[1]}, Latest: {parts[2]})")
            except subprocess.CalledProcessError:
                logging.warning("Could not check for outdated packages")

        except Exception as e:
            logging.error(f"Dependency validation error: {e}")

        return results

    def generate_report(self):
        """Generate a comprehensive dependency report"""
        validation_results = self.validate_dependencies()
        
        print("\n🔍 Dependency Validation Report 🔍")
        
        if validation_results['missing_dependencies']:
            print("\n❌ Missing Dependencies:")
            for dep in validation_results['missing_dependencies']:
                print(f"  - {dep}")
        
        if validation_results['outdated_packages']:
            print("\n⚠️ Outdated Packages:")
            for pkg in validation_results['outdated_packages']:
                print(f"  - {pkg}")
        
        if not (validation_results['missing_dependencies'] or validation_results['outdated_packages']):
            print("\n✅ All dependencies are up to date!")

def main():
    validator = DependencyValidator()
    validator.generate_report()

if __name__ == '__main__':
    main() 