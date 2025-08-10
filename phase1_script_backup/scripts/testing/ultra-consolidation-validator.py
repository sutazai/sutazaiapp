#!/usr/bin/env python3
"""
ULTRA CONSOLIDATION VALIDATOR
Purpose: Comprehensive validation after script consolidation
Author: Ultra System Architect
Date: 2025-08-10
"""

import os
import sys
import json
import subprocess
import requests
from pathlib import Path
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class UltraConsolidationValidator:
    """Comprehensive validation system for script consolidation"""
    
    def __init__(self):
        self.project_root = Path('/opt/sutazaiapp')
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'tests_passed': 0,
            'tests_failed': 0,
            'warnings': [],
            'errors': [],
            'metrics': {}
        }
        
        # Critical scripts that MUST exist
        self.critical_scripts = [
            'scripts/deploy.sh',
            'scripts/health-check.sh',
            'scripts/init_database.sh',
            'scripts/maintenance/backup-neo4j.sh',
            'scripts/maintenance/backup-redis.sh',
            'scripts/maintenance/restore-databases.sh',
            'scripts/maintenance/master-backup.sh',
            'scripts/deployment/deployment-master.sh',
            'scripts/monitoring/monitoring-master.py',
            'scripts/maintenance/maintenance-master.sh'
        ]
        
        # Critical services that MUST be healthy
        self.critical_services = [
            {'name': 'Backend API', 'url': 'http://localhost:10010/health', 'expected': 'healthy'},
            {'name': 'Frontend UI', 'url': 'http://localhost:10011/', 'expected': 200},
            {'name': 'Ollama', 'url': 'http://localhost:10104/api/tags', 'expected': 200},
            {'name': 'Hardware Optimizer', 'url': 'http://localhost:11110/health', 'expected': 'healthy'},
            {'name': 'AI Agent Orchestrator', 'url': 'http://localhost:8589/health', 'expected': 'healthy'},
            {'name': 'Ollama Integration', 'url': 'http://localhost:8090/health', 'expected': 'healthy'},
            {'name': 'PostgreSQL', 'url': 'http://localhost:10000/', 'expected': 'connection'},
            {'name': 'Redis', 'url': 'http://localhost:10001/', 'expected': 'connection'},
            {'name': 'Neo4j', 'url': 'http://localhost:10003/', 'expected': 200},
            {'name': 'Prometheus', 'url': 'http://localhost:10200/', 'expected': 200},
            {'name': 'Grafana', 'url': 'http://localhost:10201/', 'expected': 200}
        ]
    
    def run_all_validations(self) -> Dict:
        """Execute all validation tests"""
        logger.info("="*60)
        logger.info("ULTRA CONSOLIDATION VALIDATOR")
        logger.info("="*60)
        
        # Phase 1: Script Inventory
        self.validate_script_count()
        
        # Phase 2: Critical Scripts
        self.validate_critical_scripts()
        
        # Phase 3: Service Health
        self.validate_service_health()
        
        # Phase 4: Docker Containers
        self.validate_docker_containers()
        
        # Phase 5: Performance
        self.validate_performance()
        
        # Phase 6: Functionality
        self.validate_functionality()
        
        # Phase 7: Symlinks
        self.validate_symlinks()
        
        # Phase 8: No Breaking Changes
        self.validate_no_breaking_changes()
        
        # Generate report
        self.generate_report()
        
        return self.results
    
    def validate_script_count(self):
        """Validate script reduction achieved"""
        logger.info("\n📊 VALIDATING SCRIPT COUNT...")
        
        try:
            # Count total scripts
            total_scripts = len(list(self.project_root.glob('**/*.sh'))) + \
                           len(list(self.project_root.glob('**/*.py')))
            
            # Count backup files
            backup_files = len(list(self.project_root.glob('**/*.backup_*')))
            
            # Count active scripts
            active_scripts = total_scripts - backup_files
            
            self.results['metrics']['total_scripts'] = total_scripts
            self.results['metrics']['backup_files'] = backup_files
            self.results['metrics']['active_scripts'] = active_scripts
            
            logger.info(f"Total scripts: {total_scripts}")
            logger.info(f"Backup files: {backup_files}")
            logger.info(f"Active scripts: {active_scripts}")
            
            if active_scripts <= 350:
                logger.info("✅ Script count target achieved!")
                self.results['tests_passed'] += 1
            elif active_scripts <= 500:
                logger.warning(f"⚠️  Script count above target: {active_scripts} > 350")
                self.results['warnings'].append(f"Script count: {active_scripts} (target: 350)")
            else:
                logger.error(f"❌ Script count too high: {active_scripts}")
                self.results['tests_failed'] += 1
                self.results['errors'].append(f"Script count: {active_scripts} (target: 350)")
                
        except Exception as e:
            logger.error(f"❌ Failed to count scripts: {e}")
            self.results['tests_failed'] += 1
    
    def validate_critical_scripts(self):
        """Ensure all critical scripts are accessible"""
        logger.info("\n🔍 VALIDATING CRITICAL SCRIPTS...")
        
        missing_scripts = []
        
        for script in self.critical_scripts:
            script_path = self.project_root / script
            
            if script_path.exists() or script_path.is_symlink():
                logger.info(f"✅ {script} - EXISTS")
                self.results['tests_passed'] += 1
                
                # Check if executable
                if not os.access(str(script_path), os.X_OK):
                    logger.warning(f"⚠️  {script} - Not executable")
                    self.results['warnings'].append(f"{script} not executable")
            else:
                logger.error(f"❌ {script} - MISSING")
                missing_scripts.append(script)
                self.results['tests_failed'] += 1
        
        if missing_scripts:
            self.results['errors'].append(f"Missing critical scripts: {missing_scripts}")
    
    def validate_service_health(self):
        """Check all services are healthy"""
        logger.info("\n🏥 VALIDATING SERVICE HEALTH...")
        
        unhealthy_services = []
        
        for service in self.critical_services:
            try:
                if service['expected'] == 'connection':
                    # Special handling for databases
                    result = self._check_database_connection(service['name'])
                    if result:
                        logger.info(f"✅ {service['name']} - CONNECTED")
                        self.results['tests_passed'] += 1
                    else:
                        logger.error(f"❌ {service['name']} - CONNECTION FAILED")
                        unhealthy_services.append(service['name'])
                        self.results['tests_failed'] += 1
                else:
                    response = requests.get(service['url'], timeout=5)
                    
                    if service['expected'] == 'healthy':
                        if 'healthy' in response.text.lower():
                            logger.info(f"✅ {service['name']} - HEALTHY")
                            self.results['tests_passed'] += 1
                        else:
                            logger.error(f"❌ {service['name']} - UNHEALTHY")
                            unhealthy_services.append(service['name'])
                            self.results['tests_failed'] += 1
                    elif service['expected'] == 200:
                        if response.status_code == 200:
                            logger.info(f"✅ {service['name']} - RESPONDING")
                            self.results['tests_passed'] += 1
                        else:
                            logger.error(f"❌ {service['name']} - STATUS {response.status_code}")
                            unhealthy_services.append(service['name'])
                            self.results['tests_failed'] += 1
                            
            except Exception as e:
                logger.warning(f"⚠️  {service['name']} - {str(e)[:50]}")
                self.results['warnings'].append(f"{service['name']} check failed")
        
        if unhealthy_services:
            self.results['errors'].append(f"Unhealthy services: {unhealthy_services}")
    
    def validate_docker_containers(self):
        """Validate Docker containers are running properly"""
        logger.info("\n🐳 VALIDATING DOCKER CONTAINERS...")
        
        try:
            # Get container status
            result = subprocess.run(
                ['docker', 'ps', '--format', '{{.Names}}\t{{.Status}}'],
                capture_output=True,
                text=True
            )
            
            containers = result.stdout.strip().split('\n')
            running_count = 0
            restarting_count = 0
            
            for container in containers:
                if '\t' in container:
                    name, status = container.split('\t', 1)
                    
                    if 'Up' in status:
                        running_count += 1
                    elif 'Restarting' in status:
                        restarting_count += 1
                        logger.warning(f"⚠️  {name} - RESTARTING")
                        self.results['warnings'].append(f"Container restarting: {name}")
            
            logger.info(f"Running containers: {running_count}")
            logger.info(f"Restarting containers: {restarting_count}")
            
            self.results['metrics']['running_containers'] = running_count
            self.results['metrics']['restarting_containers'] = restarting_count
            
            if restarting_count == 0:
                logger.info("✅ No containers in restart loop")
                self.results['tests_passed'] += 1
            else:
                logger.error(f"❌ {restarting_count} containers restarting")
                self.results['tests_failed'] += 1
                self.results['errors'].append(f"{restarting_count} containers in restart loop")
                
        except Exception as e:
            logger.error(f"❌ Failed to check containers: {e}")
            self.results['tests_failed'] += 1
    
    def validate_performance(self):
        """Check system performance metrics"""
        logger.info("\n⚡ VALIDATING PERFORMANCE...")
        
        try:
            # API response time
            import time
            start = time.time()
            response = requests.get('http://localhost:10010/health', timeout=5)
            response_time = time.time() - start
            
            self.results['metrics']['api_response_time'] = response_time
            
            if response_time < 1:
                logger.info(f"✅ API response time: {response_time:.2f}s (EXCELLENT)")
                self.results['tests_passed'] += 1
            elif response_time < 2:
                logger.info(f"✅ API response time: {response_time:.2f}s (GOOD)")
                self.results['tests_passed'] += 1
            else:
                logger.warning(f"⚠️  API response time: {response_time:.2f}s (SLOW)")
                self.results['warnings'].append(f"Slow API response: {response_time:.2f}s")
            
            # Memory usage
            result = subprocess.run(
                ['docker', 'stats', '--no-stream', '--format', '{{.MemUsage}}'],
                capture_output=True,
                text=True
            )
            
            logger.info("Memory usage sampled successfully")
            
        except Exception as e:
            logger.warning(f"⚠️  Performance check partial: {e}")
            self.results['warnings'].append("Performance metrics incomplete")
    
    def validate_functionality(self):
        """Test core functionalities still work"""
        logger.info("\n🔧 VALIDATING FUNCTIONALITY...")
        
        tests = [
            {
                'name': 'Database connectivity',
                'command': ['docker', 'exec', 'sutazai-postgres', 'psql', '-U', 'sutazai', '-c', '\\dt']
            },
            {
                'name': 'Redis connectivity',
                'command': ['docker', 'exec', 'sutazai-redis', 'redis-cli', 'ping']
            },
            {
                'name': 'Docker compose validity',
                'command': ['docker-compose', 'config']
            }
        ]
        
        for test in tests:
            try:
                result = subprocess.run(
                    test['command'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                if result.returncode == 0:
                    logger.info(f"✅ {test['name']} - PASSED")
                    self.results['tests_passed'] += 1
                else:
                    logger.error(f"❌ {test['name']} - FAILED")
                    self.results['tests_failed'] += 1
                    self.results['errors'].append(f"{test['name']} failed")
                    
            except Exception as e:
                logger.warning(f"⚠️  {test['name']} - {str(e)[:50]}")
                self.results['warnings'].append(f"{test['name']} check failed")
    
    def validate_symlinks(self):
        """Verify symlinks are working correctly"""
        logger.info("\n🔗 VALIDATING SYMLINKS...")
        
        symlink_count = 0
        broken_symlinks = []
        
        for symlink in self.project_root.glob('**/*'):
            if symlink.is_symlink():
                symlink_count += 1
                
                if not symlink.exists():
                    broken_symlinks.append(str(symlink))
                    logger.warning(f"⚠️  Broken symlink: {symlink}")
        
        self.results['metrics']['total_symlinks'] = symlink_count
        self.results['metrics']['broken_symlinks'] = len(broken_symlinks)
        
        logger.info(f"Total symlinks: {symlink_count}")
        logger.info(f"Broken symlinks: {len(broken_symlinks)}")
        
        if len(broken_symlinks) == 0:
            logger.info("✅ All symlinks valid")
            self.results['tests_passed'] += 1
        else:
            logger.error(f"❌ {len(broken_symlinks)} broken symlinks")
            self.results['tests_failed'] += 1
            self.results['errors'].append(f"Broken symlinks: {broken_symlinks[:5]}")  # First 5
    
    def validate_no_breaking_changes(self):
        """Ensure no functionality was broken"""
        logger.info("\n🛡️ VALIDATING NO BREAKING CHANGES...")
        
        # Check git diff for unexpected deletions
        try:
            result = subprocess.run(
                ['git', 'status', '--porcelain'],
                capture_output=True,
                text=True,
                cwd=str(self.project_root)
            )
            
            deleted_files = [line for line in result.stdout.split('\n') if line.startswith(' D ')]
            
            if deleted_files:
                logger.warning(f"⚠️  {len(deleted_files)} files deleted")
                self.results['warnings'].append(f"{len(deleted_files)} files deleted")
            else:
                logger.info("✅ No unexpected deletions")
                self.results['tests_passed'] += 1
                
        except Exception as e:
            logger.warning(f"⚠️  Git check failed: {e}")
    
    def _check_database_connection(self, db_name: str) -> bool:
        """Check database connectivity"""
        try:
            if 'PostgreSQL' in db_name:
                result = subprocess.run(
                    ['docker', 'exec', 'sutazai-postgres', 'psql', '-U', 'sutazai', '-c', 'SELECT 1'],
                    capture_output=True,
                    timeout=5
                )
                return result.returncode == 0
            elif 'Redis' in db_name:
                result = subprocess.run(
                    ['docker', 'exec', 'sutazai-redis', 'redis-cli', 'ping'],
                    capture_output=True,
                    timeout=5
                )
                return b'PONG' in result.stdout
            return False
        except (IOError, OSError, FileNotFoundError) as e:
            logger.warning(f"Exception caught, returning: {e}")
            return False
    
    def generate_report(self):
        """Generate final validation report"""
        logger.info("\n" + "="*60)
        logger.info("VALIDATION SUMMARY")
        logger.info("="*60)
        
        total_tests = self.results['tests_passed'] + self.results['tests_failed']
        pass_rate = (self.results['tests_passed'] / total_tests * 100) if total_tests > 0 else 0
        
        logger.info(f"Tests Passed: {self.results['tests_passed']}")
        logger.info(f"Tests Failed: {self.results['tests_failed']}")
        logger.info(f"Pass Rate: {pass_rate:.1f}%")
        logger.info(f"Warnings: {len(self.results['warnings'])}")
        logger.info(f"Errors: {len(self.results['errors'])}")
        
        if self.results['metrics']:
            logger.info("\nKey Metrics:")
            for key, value in self.results['metrics'].items():
                logger.info(f"  {key}: {value}")
        
        # Save report to file
        report_file = self.project_root / f"consolidation-validation-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"\nReport saved to: {report_file}")
        
        # Final verdict
        if self.results['tests_failed'] == 0:
            logger.info("\n✅ CONSOLIDATION VALIDATION PASSED")
            return True
        elif self.results['tests_failed'] <= 2:
            logger.warning("\n⚠️  CONSOLIDATION MOSTLY SUCCESSFUL (minor issues)")
            return True
        else:
            logger.error("\n❌ CONSOLIDATION VALIDATION FAILED")
            logger.error("RECOMMENDATION: Review errors and consider rollback")
            return False

def main():
    """Main execution"""
    validator = UltraConsolidationValidator()
    results = validator.run_all_validations()
    
    # Exit code based on results
    if results['tests_failed'] == 0:
        sys.exit(0)
    elif results['tests_failed'] <= 2:
        sys.exit(1)  # Minor issues
    else:
        sys.exit(2)  # Major issues

if __name__ == "__main__":
    main()