#!/usr/bin/env python3
"""
SutazAI Disaster Recovery Validation Script
Quick validation of disaster recovery preparedness and system resilience.
"""

import os
import sys
import json
import time
import subprocess
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import sqlite3
import psutil
import docker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DisasterRecoveryValidator:
    """Validates disaster recovery preparedness"""
    
    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'UNKNOWN',
            'checks': {},
            'recommendations': [],
            'critical_issues': [],
            'warnings': []
        }
        
        # Initialize Docker client
        try:
            self.docker_client = docker.from_env()
        except Exception as e:
            logger.warning(f"Docker client not available: {e}")
            self.docker_client = None

    def validate_all(self) -> Dict[str, Any]:
        """Run all disaster recovery validations"""
        logger.info("Starting disaster recovery validation")
        
        checks = [
            ('emergency_shutdown_coordinator', self.check_emergency_shutdown_coordinator),
            ('backup_coordinator', self.check_backup_coordinator),
            ('system_resources', self.check_system_resources),
            ('service_health', self.check_service_health),
            ('database_integrity', self.check_database_integrity),
            ('backup_integrity', self.check_backup_integrity),
            ('network_connectivity', self.check_network_connectivity),
            ('storage_health', self.check_storage_health),
            ('monitoring_systems', self.check_monitoring_systems),
            ('recovery_procedures', self.check_recovery_procedures)
        ]
        
        passed = 0
        failed = 0
        warnings = 0
        
        for check_name, check_func in checks:
            try:
                logger.info(f"Running check: {check_name}")
                result = check_func()
                self.results['checks'][check_name] = result
                
                if result['status'] == 'PASSED':
                    passed += 1
                elif result['status'] == 'FAILED':
                    failed += 1
                    self.results['critical_issues'].extend(result.get('issues', []))
                elif result['status'] == 'WARNING':
                    warnings += 1
                    self.results['warnings'].extend(result.get('issues', []))
                
            except Exception as e:
                logger.error(f"Check failed: {check_name} - {e}")
                self.results['checks'][check_name] = {
                    'status': 'FAILED',
                    'error': str(e),
                    'issues': [f"Check execution failed: {str(e)}"]
                }
                failed += 1
        
        # Determine overall status
        if failed == 0:
            self.results['overall_status'] = 'PASSED' if warnings == 0 else 'PASSED_WITH_WARNINGS'
        else:
            self.results['overall_status'] = 'FAILED'
        
        # Generate recommendations
        self._generate_recommendations()
        
        logger.info(f"Validation complete: {self.results['overall_status']} "
                   f"({passed} passed, {failed} failed, {warnings} warnings)")
        
        return self.results

    def check_emergency_shutdown_coordinator(self) -> Dict[str, Any]:
        """Check emergency shutdown coordinator functionality"""
        issues = []
        
        # Check if emergency shutdown coordinator exists
        coordinator_path = "/opt/sutazaiapp/disaster-recovery/emergency-shutdown-coordinator.py"
        if not os.path.exists(coordinator_path):
            return {
                'status': 'FAILED',
                'issues': ['Emergency shutdown coordinator script not found'],
                'details': {'path': coordinator_path}
            }
        
        # Check if coordinator can be imported/executed
        try:
            result = subprocess.run([
                'python3', coordinator_path, 'status'
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                issues.append('Emergency shutdown coordinator cannot execute status command')
        except subprocess.TimeoutExpired:
            issues.append('Emergency shutdown coordinator status check timeout')
        except Exception as e:
            issues.append(f'Emergency shutdown coordinator check failed: {str(e)}')
        
        # Check deadman switch file
        deadman_file = "/tmp/sutazai-deadman-switch"
        if os.path.exists(deadman_file):
            try:
                stat = os.stat(deadman_file)
                age = time.time() - stat.st_mtime
                if age > 300:  # 5 minutes
                    issues.append(f'Deadman switch file is stale (age: {age:.0f}s)')
            except Exception as e:
                issues.append(f'Cannot check deadman switch file: {str(e)}')
        
        # Check shutdown lock file (should not exist normally)
        lock_file = "/tmp/sutazai-shutdown.lock"
        if os.path.exists(lock_file):
            issues.append('Shutdown lock file exists - system may be in emergency shutdown state')
        
        status = 'FAILED' if issues else 'PASSED'
        return {
            'status': status,
            'issues': issues,
            'details': {
                'coordinator_exists': os.path.exists(coordinator_path),
                'deadman_switch_exists': os.path.exists(deadman_file),
                'shutdown_lock_exists': os.path.exists(lock_file)
            }
        }

    def check_backup_coordinator(self) -> Dict[str, Any]:
        """Check backup coordinator functionality"""
        issues = []
        
        # Check if backup coordinator exists
        coordinator_path = "/opt/sutazaiapp/disaster-recovery/backup-coordinator.py"
        if not os.path.exists(coordinator_path):
            return {
                'status': 'FAILED',
                'issues': ['Backup coordinator script not found'],
                'details': {'path': coordinator_path}
            }
        
        # Check backup directories
        backup_dirs = [
            "/opt/sutazaiapp/backups",
            "/mnt/offsite-backups"
        ]
        
        backup_status = {}
        for backup_dir in backup_dirs:
            if not os.path.exists(backup_dir):
                issues.append(f'Backup directory does not exist: {backup_dir}')
                backup_status[backup_dir] = 'missing'
            else:
                try:
                    # Check if writable
                    test_file = os.path.join(backup_dir, 'test_write')
                    with open(test_file, 'w') as f:
                        f.write('test')
                    os.remove(test_file)
                    backup_status[backup_dir] = 'accessible'
                except Exception as e:
                    issues.append(f'Backup directory not writable: {backup_dir} - {str(e)}')
                    backup_status[backup_dir] = 'not_writable'
        
        # Check recent backups
        backup_metadata_db = "/opt/sutazaiapp/data/backup-metadata.db"
        if os.path.exists(backup_metadata_db):
            try:
                conn = sqlite3.connect(backup_metadata_db)
                cursor = conn.execute('''
                    SELECT job_name, MAX(timestamp) as last_backup
                    FROM backup_metadata 
                    GROUP BY job_name
                ''')
                
                recent_backups = {}
                for row in cursor.fetchall():
                    job_name, last_backup = row
                    if last_backup:
                        backup_time = datetime.fromisoformat(last_backup)
                        age = datetime.now() - backup_time
                        recent_backups[job_name] = age.total_seconds()
                        
                        # Check if backup is too old
                        if age.total_seconds() > 86400:  # 24 hours
                            issues.append(f'Backup too old for job {job_name}: {age.days} days')
                
                conn.close()
                backup_status['recent_backups'] = recent_backups
                
            except Exception as e:
                issues.append(f'Cannot check backup metadata: {str(e)}')
        else:
            issues.append('Backup metadata database not found')
        
        status = 'FAILED' if issues else 'PASSED'
        return {
            'status': status,
            'issues': issues,
            'details': backup_status
        }

    def check_system_resources(self) -> Dict[str, Any]:
        """Check system resource availability"""
        issues = []
        warnings = []
        
        # Check disk usage
        disk_usage = psutil.disk_usage('/')
        disk_percent = (disk_usage.used / disk_usage.total) * 100
        
        if disk_percent > 95:
            issues.append(f'Critical disk usage: {disk_percent:.1f}%')
        elif disk_percent > 85:
            warnings.append(f'High disk usage: {disk_percent:.1f}%')
        
        # Check memory usage
        memory = psutil.virtual_memory()
        if memory.percent > 95:
            issues.append(f'Critical memory usage: {memory.percent:.1f}%')
        elif memory.percent > 85:
            warnings.append(f'High memory usage: {memory.percent:.1f}%')
        
        # Check CPU load
        cpu_count = psutil.cpu_count()
        load_avg = os.getloadavg()[0]  # 1-minute load average
        load_percent = (load_avg / cpu_count) * 100
        
        if load_percent > 200:  # 200% of CPU capacity
            issues.append(f'Critical CPU load: {load_percent:.1f}%')
        elif load_percent > 150:
            warnings.append(f'High CPU load: {load_percent:.1f}%')
        
        # Determine status
        if issues:
            status = 'FAILED'
        elif warnings:
            status = 'WARNING'
        else:
            status = 'PASSED'
        
        return {
            'status': status,
            'issues': issues + warnings,
            'details': {
                'disk_usage_percent': disk_percent,
                'memory_usage_percent': memory.percent,
                'cpu_load_percent': load_percent,
                'available_disk_gb': disk_usage.free / (1024**3),
                'available_memory_gb': memory.available / (1024**3)
            }
        }

    def check_service_health(self) -> Dict[str, Any]:
        """Check health of critical services"""
        issues = []
        
        # Services to check
        services = [
            ('frontend', 'http://localhost:8501/health'),
            ('backend', 'http://localhost:8000/health'),
            ('agent-orchestrator', 'http://localhost:8002/health'),
            ('monitoring', 'http://localhost:3000')  # Grafana
        ]
        
        service_status = {}
        
        for service_name, health_url in services:
            try:
                import urllib.request
                urllib.request.urlopen(health_url, timeout=5)
                service_status[service_name] = 'healthy'
            except Exception as e:
                issues.append(f'Service health check failed: {service_name} - {str(e)}')
                service_status[service_name] = 'unhealthy'
        
        # Check Docker containers if available
        if self.docker_client:
            try:
                containers = self.docker_client.containers.list()
                running_containers = len(containers)
                
                if running_containers == 0:
                    issues.append('No Docker containers running')
                
                service_status['docker_containers'] = running_containers
                
                # Check for unhealthy containers
                unhealthy_containers = []
                for container in containers:
                    if container.status != 'running':
                        unhealthy_containers.append(container.name)
                
                if unhealthy_containers:
                    issues.append(f'Unhealthy containers: {", ".join(unhealthy_containers)}')
                
            except Exception as e:
                issues.append(f'Docker container check failed: {str(e)}')
        
        status = 'FAILED' if issues else 'PASSED'
        return {
            'status': status,
            'issues': issues,
            'details': service_status
        }

    def check_database_integrity(self) -> Dict[str, Any]:
        """Check database integrity"""
        issues = []
        
        # Check critical databases
        databases = [
            "/opt/sutazaiapp/data/backup-metadata.db",
            "/tmp/sutazai-dr-tests/test.db"  # Test database
        ]
        
        db_status = {}
        
        for db_path in databases:
            db_name = os.path.basename(db_path)
            
            if not os.path.exists(db_path):
                # Only critical if it's a production database
                if 'backup-metadata' in db_name:
                    issues.append(f'Critical database missing: {db_name}')
                db_status[db_name] = 'missing'
                continue
            
            try:
                # Check database integrity
                conn = sqlite3.connect(db_path)
                cursor = conn.execute("PRAGMA integrity_check;")
                result = cursor.fetchone()
                
                if result[0] != 'ok':
                    issues.append(f'Database integrity check failed: {db_name} - {result[0]}')
                    db_status[db_name] = 'corrupted'
                else:
                    db_status[db_name] = 'healthy'
                
                conn.close()
                
            except Exception as e:
                issues.append(f'Database check failed: {db_name} - {str(e)}')
                db_status[db_name] = 'error'
        
        status = 'FAILED' if issues else 'PASSED'
        return {
            'status': status,
            'issues': issues,
            'details': db_status
        }

    def check_backup_integrity(self) -> Dict[str, Any]:
        """Check backup integrity"""
        issues = []
        
        # Run backup integrity test
        try:
            result = subprocess.run([
                'python3', '/opt/sutazaiapp/disaster-recovery/disaster-recovery-test-suite.py',
                'run', '--test', 'backup_integrity_validation'
            ], capture_output=True, text=True, timeout=120)
            
            if result.returncode != 0:
                issues.append('Backup integrity test failed')
            else:
                # Parse result to check success
                try:
                    test_result = json.loads(result.stdout)
                    if test_result.get('result') != 'passed':
                        issues.append('Backup integrity validation failed')
                except json.JSONDecodeError:
                    issues.append('Cannot parse backup integrity test result')
        
        except subprocess.TimeoutExpired:
            issues.append('Backup integrity test timeout')
        except Exception as e:
            issues.append(f'Backup integrity test error: {str(e)}')
        
        status = 'FAILED' if issues else 'PASSED'
        return {
            'status': status,
            'issues': issues,
            'details': {'test_executed': True}
        }

    def check_network_connectivity(self) -> Dict[str, Any]:
        """Check network connectivity"""
        issues = []
        
        # Check localhost connectivity
        try:
            import socket
            socket.gethostbyname('localhost')
        except Exception as e:
            issues.append(f'Localhost resolution failed: {str(e)}')
        
        # Check internal service connectivity
        internal_services = [
            ('localhost', 8501),  # Frontend
            ('localhost', 8000),  # Backend
            ('localhost', 8002),  # Orchestrator
        ]
        
        connectivity_status = {}
        
        for host, port in internal_services:
            try:
                import socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5)
                result = sock.connect_ex((host, port))
                sock.close()
                
                if result == 0:
                    connectivity_status[f'{host}:{port}'] = 'reachable'
                else:
                    connectivity_status[f'{host}:{port}'] = 'unreachable'
                    issues.append(f'Cannot connect to {host}:{port}')
                    
            except Exception as e:
                issues.append(f'Network connectivity test failed for {host}:{port}: {str(e)}')
                connectivity_status[f'{host}:{port}'] = 'error'
        
        status = 'FAILED' if issues else 'PASSED'
        return {
            'status': status,
            'issues': issues,
            'details': connectivity_status
        }

    def check_storage_health(self) -> Dict[str, Any]:
        """Check storage health"""
        issues = []
        warnings = []
        
        # Check critical directories
        critical_dirs = [
            "/opt/sutazaiapp",
            "/opt/sutazaiapp/data",
            "/opt/sutazaiapp/logs",
            "/opt/sutazaiapp/backups"
        ]
        
        dir_status = {}
        
        for dir_path in critical_dirs:
            if not os.path.exists(dir_path):
                issues.append(f'Critical directory missing: {dir_path}')
                dir_status[dir_path] = 'missing'
                continue
            
            # Check if writable
            try:
                test_file = os.path.join(dir_path, 'test_write')
                with open(test_file, 'w') as f:
                    f.write('test')
                os.remove(test_file)
                dir_status[dir_path] = 'writable'
            except Exception as e:
                issues.append(f'Directory not writable: {dir_path} - {str(e)}')
                dir_status[dir_path] = 'not_writable'
        
        # Check mount points
        try:
            with open('/proc/mounts', 'r') as f:
                mounts = f.read()
            
            required_mounts = ['/']
            mount_status = {}
            
            for mount in required_mounts:
                if mount in mounts:
                    mount_status[mount] = 'mounted'
                else:
                    issues.append(f'Required mount point not found: {mount}')
                    mount_status[mount] = 'not_mounted'
            
        except Exception as e:
            warnings.append(f'Cannot check mount points: {str(e)}')
        
        # Determine status
        if issues:
            status = 'FAILED'
        elif warnings:
            status = 'WARNING'
        else:
            status = 'PASSED'
        
        return {
            'status': status,
            'issues': issues + warnings,
            'details': {
                'directories': dir_status,
                'mounts': mount_status if 'mount_status' in locals() else {}
            }
        }

    def check_monitoring_systems(self) -> Dict[str, Any]:
        """Check monitoring systems"""
        issues = []
        warnings = []
        
        # Check log files
        log_files = [
            "/opt/sutazaiapp/logs/emergency-shutdown.log",
            "/opt/sutazaiapp/logs/backup-coordinator.log",
            "/opt/sutazaiapp/logs/disaster-recovery-tests.log"
        ]
        
        log_status = {}
        
        for log_file in log_files:
            if os.path.exists(log_file):
                try:
                    stat = os.stat(log_file)
                    age = time.time() - stat.st_mtime
                    
                    if age > 86400:  # 24 hours
                        warnings.append(f'Log file not updated recently: {os.path.basename(log_file)}')
                    
                    log_status[os.path.basename(log_file)] = {
                        'exists': True,
                        'age_hours': age / 3600,
                        'size_mb': stat.st_size / (1024 * 1024)
                    }
                except Exception as e:
                    warnings.append(f'Cannot check log file: {os.path.basename(log_file)} - {str(e)}')
            else:
                warnings.append(f'Log file missing: {os.path.basename(log_file)}')
                log_status[os.path.basename(log_file)] = {'exists': False}
        
        # Check if monitoring processes are running
        monitoring_processes = ['prometheus', 'grafana', 'alertmanager']
        process_status = {}
        
        for proc_name in monitoring_processes:
            found = False
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if proc_name in proc.info['name'].lower() or \
                       any(proc_name in cmd.lower() for cmd in proc.info['cmdline'] or []):
                        found = True
                        break
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            process_status[proc_name] = found
            if not found:
                warnings.append(f'Monitoring process not found: {proc_name}')
        
        # Determine status
        if issues:
            status = 'FAILED'
        elif warnings:
            status = 'WARNING'
        else:
            status = 'PASSED'
        
        return {
            'status': status,
            'issues': issues + warnings,
            'details': {
                'log_files': log_status,
                'processes': process_status
            }
        }

    def check_recovery_procedures(self) -> Dict[str, Any]:
        """Check recovery procedures and documentation"""
        issues = []
        
        # Check if recovery scripts exist
        recovery_scripts = [
            "/opt/sutazaiapp/disaster-recovery/emergency-shutdown-coordinator.py",
            "/opt/sutazaiapp/disaster-recovery/backup-coordinator.py",
            "/opt/sutazaiapp/disaster-recovery/disaster-recovery-test-suite.py"
        ]
        
        script_status = {}
        
        for script_path in recovery_scripts:
            script_name = os.path.basename(script_path)
            
            if os.path.exists(script_path):
                # Check if executable
                if os.access(script_path, os.X_OK):
                    script_status[script_name] = 'executable'
                else:
                    script_status[script_name] = 'not_executable'
                    issues.append(f'Recovery script not executable: {script_name}')
            else:
                script_status[script_name] = 'missing'
                issues.append(f'Recovery script missing: {script_name}')
        
        # Check documentation
        docs = [
            "/opt/sutazaiapp/disaster-recovery/disaster-recovery-runbook.md",
            "/opt/sutazaiapp/disaster-recovery/disaster-recovery-report.md"
        ]
        
        doc_status = {}
        
        for doc_path in docs:
            doc_name = os.path.basename(doc_path)
            
            if os.path.exists(doc_path):
                stat = os.stat(doc_path)
                age = time.time() - stat.st_mtime
                doc_status[doc_name] = {
                    'exists': True,
                    'age_days': age / 86400,
                    'size_kb': stat.st_size / 1024
                }
            else:
                issues.append(f'Recovery documentation missing: {doc_name}')
                doc_status[doc_name] = {'exists': False}
        
        status = 'FAILED' if issues else 'PASSED'
        return {
            'status': status,
            'issues': issues,
            'details': {
                'scripts': script_status,
                'documentation': doc_status
            }
        }

    def _generate_recommendations(self):
        """Generate recommendations based on check results"""
        recommendations = []
        
        # Analyze failed checks
        for check_name, result in self.results['checks'].items():
            if result['status'] == 'FAILED':
                if check_name == 'system_resources':
                    recommendations.append("Increase system resources or implement resource cleanup procedures")
                elif check_name == 'service_health':
                    recommendations.append("Investigate and resolve service health issues")
                elif check_name == 'database_integrity':
                    recommendations.append("Restore databases from backups or repair corruption")
                elif check_name == 'backup_integrity':
                    recommendations.append("Review and fix backup procedures")
                elif check_name == 'network_connectivity':
                    recommendations.append("Resolve network connectivity issues")
                elif check_name == 'storage_health':
                    recommendations.append("Fix storage issues and ensure directory accessibility")
        
        # Analyze warnings
        warning_count = len(self.results['warnings'])
        if warning_count > 0:
            recommendations.append(f"Address {warning_count} warning conditions to improve system resilience")
        
        # General recommendations
        if self.results['overall_status'] == 'PASSED':
            recommendations.append("System appears healthy - consider running full disaster recovery tests")
        else:
            recommendations.append("Address critical issues before disaster recovery procedures will be effective")
        
        self.results['recommendations'] = recommendations

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="SutazAI Disaster Recovery Validator")
    parser.add_argument("--output", help="Output file for results (JSON format)")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    validator = DisasterRecoveryValidator()
    
    try:
        results = validator.validate_all()
        
        # Output results
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to: {args.output}")
        else:
            logger.info(json.dumps(results, indent=2))
        
        # Exit with appropriate code
        if results['overall_status'] == 'FAILED':
            sys.exit(1)
        elif results['overall_status'] == 'PASSED_WITH_WARNINGS':
            sys.exit(2)
        else:
            sys.exit(0)
    
    except KeyboardInterrupt:
        logger.info("Validation interrupted")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()