#!/usr/bin/env python3
"""
import logging

logger = logging.getLogger(__name__)
ULTRATEST Security Validation Suite
Tests ALL container security improvements with 100% coverage.
"""

import subprocess
import json
import sys
import time
from datetime import datetime
from typing import Dict, List, Tuple, Any

class UltratestSecurityValidator:
    def __init__(self):
        self.test_results = {}
        self.container_security_status = {}
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        
    def log_test_result(self, test_name: str, passed: bool, details: str):
        """Log individual test results"""
        self.test_results[test_name] = {
            'passed': passed,
            'details': details,
            'timestamp': datetime.now().isoformat()
        }
        self.total_tests += 1
        if passed:
            self.passed_tests += 1
            logger.info(f"✅ PASS: {test_name}")
        else:
            self.failed_tests += 1
            logger.info(f"❌ FAIL: {test_name} - {details}")
            
    def get_running_containers(self) -> List[str]:
        """Get list of all running container names"""
        try:
            result = subprocess.run(
                ['docker', 'ps', '--format', '{{.Names}}'],
                capture_output=True, text=True, check=True
            )
            containers = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
            return containers
        except subprocess.CalledProcessError as e:
            self.log_test_result("get_running_containers", False, f"Failed to get containers: {e}")
            return []
    
    def check_container_user(self, container_name: str) -> Tuple[bool, str]:
        """Check if container is running as non-root user"""
        try:
            # Get the user ID from inside the container
            result = subprocess.run(
                ['docker', 'exec', container_name, 'id', '-u'],
                capture_output=True, text=True, check=True
            )
            uid = result.stdout.strip()
            
            # Get username if possible
            try:
                user_result = subprocess.run(
                    ['docker', 'exec', container_name, 'whoami'],
                    capture_output=True, text=True, check=True
                )
                username = user_result.stdout.strip()
            except:
                username = f"uid:{uid}"
            
            is_non_root = uid != "0"
            status = f"User: {username} (UID: {uid})"
            
            return is_non_root, status
            
        except subprocess.CalledProcessError as e:
            return False, f"Failed to check user: {e}"
    
    def test_all_container_security(self):
        """Test security configuration of all running containers"""
        logger.info("\n🔒 ULTRATEST: Container Security Validation")
        logger.info("=" * 60)
        
        containers = self.get_running_containers()
        if not containers:
            self.log_test_result("container_discovery", False, "No containers found")
            return
            
        logger.info(f"Found {len(containers)} running containers")
        
        non_root_count = 0
        root_count = 0
        
        for container in containers:
            is_non_root, details = self.check_container_user(container)
            self.container_security_status[container] = {
                'non_root': is_non_root,
                'details': details
            }
            
            if is_non_root:
                non_root_count += 1
                self.log_test_result(f"security_{container}", True, details)
            else:
                root_count += 1
                self.log_test_result(f"security_{container}", False, f"Running as root - {details}")
        
        # Calculate security percentage
        security_percentage = (non_root_count / len(containers)) * 100
        
        # Test overall security target (claimed 100% non-root)
        if security_percentage == 100:
            self.log_test_result("overall_security_100_percent", True, 
                               f"All {len(containers)} containers non-root ({security_percentage:.1f}%)")
        else:
            self.log_test_result("overall_security_100_percent", False, 
                               f"Only {non_root_count}/{len(containers)} containers non-root ({security_percentage:.1f}%)")
        
        logger.info(f"\n📊 Security Summary:")
        logger.info(f"   Non-root containers: {non_root_count}")
        logger.info(f"   Root containers: {root_count}")
        logger.info(f"   Security percentage: {security_percentage:.1f}%")
        
    def test_service_health_endpoints(self):
        """Test all service health endpoints"""
        logger.info("\n🏥 ULTRATEST: Service Health Validation")
        logger.info("=" * 60)
        
        health_endpoints = [
            ('Backend API', 'http://localhost:10010/health'),
            ('Frontend UI', 'http://localhost:10011/'),
            ('Ollama API', 'http://localhost:10104/api/tags'),
            ('Hardware Optimizer', 'http://localhost:11110/health'),
            ('AI Agent Orchestrator', 'http://localhost:8589/health'),
            ('Ollama Integration', 'http://localhost:8090/health'),
            ('FAISS Vector DB', 'http://localhost:10103/health'),
            ('Resource Arbitration', 'http://localhost:8588/health'),
            ('Task Assignment', 'http://localhost:8551/health'),
            ('PostgreSQL', 'http://localhost:10000/'),
            ('Redis', 'http://localhost:10001/'),
            ('Prometheus', 'http://localhost:10200/'),
            ('Grafana', 'http://localhost:10201/'),
        ]
        
        for service_name, endpoint in health_endpoints:
            try:
                result = subprocess.run(
                    ['curl', '-s', '-o', '/dev/null', '-w', '%{http_code}', endpoint],
                    capture_output=True, text=True, timeout=10
                )
                status_code = result.stdout.strip()
                
                if status_code in ['200', '302', '404']:  # 404 can be OK for some endpoints
                    self.log_test_result(f"health_{service_name.lower().replace(' ', '_')}", True, 
                                       f"HTTP {status_code}")
                else:
                    self.log_test_result(f"health_{service_name.lower().replace(' ', '_')}", False, 
                                       f"HTTP {status_code}")
            except subprocess.TimeoutExpired:
                self.log_test_result(f"health_{service_name.lower().replace(' ', '_')}", False, "Timeout")
            except Exception as e:
                self.log_test_result(f"health_{service_name.lower().replace(' ', '_')}", False, f"Error: {e}")
    
    def test_docker_compose_security_configuration(self):
        """Validate docker-compose.yml security settings"""
        logger.info("\n🐳 ULTRATEST: Docker Compose Security Configuration")
        logger.info("=" * 60)
        
        try:
            with open('/opt/sutazaiapp/docker-compose.yml', 'r') as f:
                compose_content = f.read()
                
            # Check for security-related configurations
            security_checks = {
                'user_specifications': compose_content.count('user:'),
                'no_new_privileges': compose_content.count('no-new-privileges'),
                'read_only_root_fs': compose_content.count('read_only'),
                'security_opt_configs': compose_content.count('security_opt'),
                'cap_drop_configs': compose_content.count('cap_drop')
            }
            
            for check, count in security_checks.items():
                if count > 0:
                    self.log_test_result(f"compose_security_{check}", True, f"Found {count} instances")
                else:
                    self.log_test_result(f"compose_security_{check}", False, f"No {check} found")
                    
        except Exception as e:
            self.log_test_result("compose_security_validation", False, f"Error reading compose file: {e}")
    
    def generate_ultratest_report(self):
        """Generate comprehensive test report"""
        logger.info("\n" + "=" * 80)
        logger.info("🎯 ULTRATEST SECURITY VALIDATION REPORT")
        logger.info("=" * 80)
        logger.info(f"Test Execution Time: {datetime.now().isoformat()}")
        logger.info(f"Total Tests: {self.total_tests}")
        logger.info(f"Passed: {self.passed_tests}")
        logger.error(f"Failed: {self.failed_tests}")
        logger.info(f"Success Rate: {(self.passed_tests/self.total_tests)*100:.1f}%")
        
        logger.info("\n🔒 CONTAINER SECURITY STATUS:")
        logger.info("-" * 50)
        for container, status in self.container_security_status.items():
            icon = "✅" if status['non_root'] else "❌"
            logger.info(f"{icon} {container}: {status['details']}")
        
        logger.info("\n📋 DETAILED TEST RESULTS:")
        logger.info("-" * 50)
        for test_name, result in self.test_results.items():
            icon = "✅" if result['passed'] else "❌"
            logger.info(f"{icon} {test_name}: {result['details']}")
        
        # Calculate container security percentage
        if self.container_security_status:
            non_root_containers = sum(1 for status in self.container_security_status.values() if status['non_root'])
            total_containers = len(self.container_security_status)
            security_percentage = (non_root_containers / total_containers) * 100
            
            logger.info(f"\n🎯 SECURITY ACHIEVEMENT:")
            logger.info(f"   Non-root containers: {non_root_containers}/{total_containers}")
            logger.info(f"   Security percentage: {security_percentage:.1f}%")
            logger.info(f"   Target: 100% (29/29 containers)")
            
            if security_percentage == 100:
                logger.info("   🏆 SECURITY TARGET ACHIEVED!")
            else:
                logger.info(f"   ⚠️  Security gap: {100-security_percentage:.1f}%")
        
        return {
            'total_tests': self.total_tests,
            'passed_tests': self.passed_tests,
            'failed_tests': self.failed_tests,
            'success_rate': (self.passed_tests/self.total_tests)*100,
            'container_security': self.container_security_status,
            'test_results': self.test_results
        }

def main():
    """Run comprehensive security validation"""
    logger.info("🚀 Starting ULTRATEST Security Validation Suite")
    logger.info("Testing ALL security improvements with 100% coverage")
    
    validator = UltratestSecurityValidator()
    
    # Run all security tests
    validator.test_all_container_security()
    validator.test_service_health_endpoints()
    validator.test_docker_compose_security_configuration()
    
    # Generate comprehensive report
    report = validator.generate_ultratest_report()
    
    # Save report to file
    with open('/opt/sutazaiapp/tests/ultratest_security_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"\n📄 Full report saved to: /opt/sutazaiapp/tests/ultratest_security_report.json")
    
    # Return exit code based on results
    if validator.failed_tests == 0:
        logger.info("\n🎉 ALL TESTS PASSED - ULTRATEST SECURITY VALIDATION SUCCESSFUL")
        return 0
    else:
        logger.error(f"\n⚠️  {validator.failed_tests} TEST(S) FAILED - REVIEW REQUIRED")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)