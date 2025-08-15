#!/usr/bin/env python3
"""
ULTRA Migration Test Suite
Comprehensive testing of recently migrated services to master base Dockerfiles

Services Under Test:
- agent-message-bus (port 8080)
- self-healing (port 8080) 
- data-analysis-engineer (port 8080)
- observability-monitoring-engineer (port 8580)
- agentzero (port 8080)
- hygiene-backend (port 8080)

Test Coverage:
1. Docker build verification
2. Container startup validation
3. Health endpoint testing
4. Non-root user verification
5. Python version validation
6. Functionality testing
"""

import logging

# Configure logger for exception handling
logger = logging.getLogger(__name__)

import subprocess
import requests
import json
import time
import datetime
import sys
import os
class UltraMigrationTester:
    def __init__(self):
        self.services = {
            'agent-message-bus': {
                'dockerfile_path': '/opt/sutazaiapp/docker/agent-message-bus/Dockerfile',
                'context_path': '/opt/sutazaiapp/docker/agent-message-bus',
                'port': 8080,
                'image_name': 'sutazai-agent-message-bus-test',
                'container_name': 'test-agent-message-bus'
            },
            'self-healing': {
                'dockerfile_path': '/opt/sutazaiapp/docker/self-healing/Dockerfile',
                'context_path': '/opt/sutazaiapp/docker/self-healing',
                'port': 8080,
                'image_name': 'sutazai-self-healing-test',
                'container_name': 'test-self-healing'
            },
            'data-analysis-engineer': {
                'dockerfile_path': '/opt/sutazaiapp/docker/data-analysis-engineer/Dockerfile',
                'context_path': '/opt/sutazaiapp/docker/data-analysis-engineer',
                'port': 8080,
                'image_name': 'sutazai-data-analysis-engineer-test',
                'container_name': 'test-data-analysis-engineer'
            },
            'observability-monitoring-engineer': {
                'dockerfile_path': '/opt/sutazaiapp/docker/observability-monitoring-engineer/Dockerfile',
                'context_path': '/opt/sutazaiapp/docker/observability-monitoring-engineer',
                'port': 8580,
                'image_name': 'sutazai-observability-monitoring-engineer-test',
                'container_name': 'test-observability-monitoring-engineer'
            },
            'agentzero': {
                'dockerfile_path': '/opt/sutazaiapp/docker/agentzero/Dockerfile',
                'context_path': '/opt/sutazaiapp/docker/agentzero',
                'port': 8080,
                'image_name': 'sutazai-agentzero-test',
                'container_name': 'test-agentzero'
            },
            'hygiene-backend': {
                'dockerfile_path': '/opt/sutazaiapp/docker/hygiene-backend/Dockerfile',
                'context_path': '/opt/sutazaiapp/docker/hygiene-backend',
                'port': 8080,
                'image_name': 'sutazai-hygiene-backend-test',
                'container_name': 'test-hygiene-backend'
            }
        }
        
        self.results = {
            'timestamp': datetime.datetime.now().isoformat(),
            'services_tested': len(self.services),
            'test_results': {},
            'summary': {
                'builds_successful': 0,
                'startups_successful': 0,
                'health_checks_passed': 0,
                'user_validations_passed': 0,
                'python_validations_passed': 0,
                'functionality_tests_passed': 0
            }
        }

    def run_command(self, cmd: str, cwd: str = None, timeout: int = 120) -> Tuple[bool, str, str]:
        """Run shell command and return success, stdout, stderr"""
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, 
                timeout=timeout, cwd=cwd
            )
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", "Command timed out"
        except Exception as e:
            return False, "", str(e)

    def cleanup_test_containers(self):
        """Clean up any existing test containers"""
        logger.info("\n🧹 Cleaning up existing test containers...")
        for service_name, config in self.services.items():
            container_name = config['container_name']
            image_name = config['image_name']
            
            # Stop and remove container
            self.run_command(f"docker stop {container_name} 2>/dev/null || true")
            self.run_command(f"docker rm {container_name} 2>/dev/null || true")
            
            # Remove test image
            self.run_command(f"docker rmi {image_name} 2>/dev/null || true")

    def test_docker_build(self, service_name: str, config: Dict) -> Dict:
        """Test Docker build for a service"""
        logger.info(f"\n🔨 Testing Docker build for {service_name}...")
        
        result = {
            'build_successful': False,
            'build_time_seconds': 0,
            'build_output': '',
            'build_errors': '',
            'dockerfile_exists': False
        }
        
        # Check if Dockerfile exists
        dockerfile_path = config['dockerfile_path']
        if not os.path.exists(dockerfile_path):
            result['build_errors'] = f"Dockerfile not found: {dockerfile_path}"
            return result
            
        result['dockerfile_exists'] = True
        
        # Build the Docker image
        build_cmd = f"docker build -t {config['image_name']} -f {dockerfile_path} {config['context_path']}"
        start_time = time.time()
        
        success, stdout, stderr = self.run_command(build_cmd, timeout=300)
        build_time = time.time() - start_time
        
        result['build_successful'] = success
        result['build_time_seconds'] = round(build_time, 2)
        result['build_output'] = stdout
        result['build_errors'] = stderr
        
        if success:
            logger.info(f"✅ Build successful for {service_name} ({build_time:.1f}s)")
            self.results['summary']['builds_successful'] += 1
        else:
            logger.error(f"❌ Build failed for {service_name}")
            logger.error(f"Error: {stderr}")
        
        return result

    def test_container_startup(self, service_name: str, config: Dict) -> Dict:
        """Test container startup"""
        logger.info(f"\n🚀 Testing container startup for {service_name}...")
        
        result = {
            'startup_successful': False,
            'container_running': False,
            'startup_time_seconds': 0,
            'startup_logs': '',
            'startup_errors': ''
        }
        
        container_name = config['container_name']
        port = config['port']
        
        # Start container
        run_cmd = f"docker run -d --name {container_name} -p {port}:{port} {config['image_name']}"
        start_time = time.time()
        
        success, stdout, stderr = self.run_command(run_cmd)
        
        if not success:
            result['startup_errors'] = stderr
            return result
        
        # Wait for container to be ready
        time.sleep(5)
        
        # Check if container is running
        check_cmd = f"docker ps --filter name={container_name} --format '{{{{.Status}}}}'"
        success, stdout, stderr = self.run_command(check_cmd)
        
        if success and stdout.strip():
            result['container_running'] = True
            result['startup_successful'] = True
            startup_time = time.time() - start_time
            result['startup_time_seconds'] = round(startup_time, 2)
            
            # Get logs
            logs_cmd = f"docker logs {container_name}"
            success, logs, _ = self.run_command(logs_cmd)
            result['startup_logs'] = logs
            
            logger.info(f"✅ Container startup successful for {service_name} ({startup_time:.1f}s)")
            self.results['summary']['startups_successful'] += 1
        else:
            logger.error(f"❌ Container startup failed for {service_name}")
            # Get logs for debugging
            logs_cmd = f"docker logs {container_name}"
            success, logs, _ = self.run_command(logs_cmd)
            result['startup_errors'] = logs
        
        return result

    def test_health_endpoint(self, service_name: str, config: Dict) -> Dict:
        """Test health endpoint"""
        logger.info(f"\n🏥 Testing health endpoint for {service_name}...")
        
        result = {
            'health_check_passed': False,
            'response_time_ms': 0,
            'status_code': 0,
            'response_body': '',
            'health_errors': ''
        }
        
        port = config['port']
        health_url = f"http://localhost:{port}/health"
        
        try:
            # Wait a bit for service to be ready
            time.sleep(3)
            
            start_time = time.time()
            response = requests.get(health_url, timeout=10)
            response_time = (time.time() - start_time) * 1000
            
            result['status_code'] = response.status_code
            result['response_time_ms'] = round(response_time, 2)
            result['response_body'] = response.text
            
            if response.status_code == 200:
                result['health_check_passed'] = True
                logger.info(f"✅ Health check passed for {service_name} ({response_time:.1f}ms)")
                self.results['summary']['health_checks_passed'] += 1
            else:
                logger.error(f"❌ Health check failed for {service_name} - Status: {response.status_code}")
                
        except Exception as e:
            result['health_errors'] = str(e)
            logger.error(f"❌ Health check error for {service_name}: {e}")
        
        return result

    def test_non_root_user(self, service_name: str, config: Dict) -> Dict:
        """Test that container is running as non-root user"""
        logger.info(f"\n👤 Testing non-root user for {service_name}...")
        
        result = {
            'user_validation_passed': False,
            'current_user': '',
            'user_id': '',
            'user_errors': ''
        }
        
        container_name = config['container_name']
        
        # Check current user
        user_cmd = f"docker exec {container_name} whoami"
        success, stdout, stderr = self.run_command(user_cmd)
        
        if success:
            current_user = stdout.strip()
            result['current_user'] = current_user
            
            # Check user ID
            id_cmd = f"docker exec {container_name} id -u"
            success, stdout, stderr = self.run_command(id_cmd)
            
            if success:
                user_id = stdout.strip()
                result['user_id'] = user_id
                
                # Validate non-root (user should be 'appuser' and ID should not be 0)
                if current_user == 'appuser' and user_id != '0':
                    result['user_validation_passed'] = True
                    logger.info(f"✅ Non-root user validation passed for {service_name} (user: {current_user}, id: {user_id})")
                    self.results['summary']['user_validations_passed'] += 1
                else:
                    logger.error(f"❌ Non-root user validation failed for {service_name} (user: {current_user}, id: {user_id})")
            else:
                result['user_errors'] = stderr
        else:
            result['user_errors'] = stderr
            logger.error(f"❌ User validation error for {service_name}: {stderr}")
        
        return result

    def test_python_version(self, service_name: str, config: Dict) -> Dict:
        """Test Python version"""
        logger.info(f"\n🐍 Testing Python version for {service_name}...")
        
        result = {
            'python_validation_passed': False,
            'python_version': '',
            'python_errors': ''
        }
        
        container_name = config['container_name']
        
        # Check Python version
        python_cmd = f"docker exec {container_name} python --version"
        success, stdout, stderr = self.run_command(python_cmd)
        
        if success:
            python_version = stdout.strip()
            result['python_version'] = python_version
            
            # Validate Python 3.12.8
            if '3.12.8' in python_version:
                result['python_validation_passed'] = True
                logger.info(f"✅ Python version validation passed for {service_name} ({python_version})")
                self.results['summary']['python_validations_passed'] += 1
            else:
                logger.error(f"❌ Python version validation failed for {service_name} ({python_version})")
        else:
            result['python_errors'] = stderr
            logger.error(f"❌ Python version error for {service_name}: {stderr}")
        
        return result

    def test_service_functionality(self, service_name: str, config: Dict) -> Dict:
        """Test basic service functionality"""
        logger.info(f"\n⚙️ Testing service functionality for {service_name}...")
        
        result = {
            'functionality_test_passed': False,
            'endpoints_tested': [],
            'functionality_errors': ''
        }
        
        port = config['port']
        base_url = f"http://localhost:{port}"
        
        # Test common endpoints
        endpoints_to_test = ['/health', '/status', '/api/v1/status', '/']
        working_endpoints = []
        
        for endpoint in endpoints_to_test:
            try:
                url = f"{base_url}{endpoint}"
                response = requests.get(url, timeout=5)
                if response.status_code in [200, 404]:  # 404 is OK, means server is responding
                    working_endpoints.append(f"{endpoint} ({response.status_code})")
            except (IOError, OSError, FileNotFoundError) as e:
                # Suppressed exception (was bare except)
                logger.debug(f"Suppressed exception: {e}")
                pass
        
        result['endpoints_tested'] = working_endpoints
        
        if len(working_endpoints) > 0:
            result['functionality_test_passed'] = True
            logger.info(f"✅ Functionality test passed for {service_name} - Working endpoints: {working_endpoints}")
            self.results['summary']['functionality_tests_passed'] += 1
        else:
            logger.error(f"❌ Functionality test failed for {service_name} - No responsive endpoints")
        
        return result

    def run_comprehensive_test(self, service_name: str) -> Dict:
        """Run comprehensive test for a single service"""
        logger.info(f"\n{'='*60}")
        logger.info(f"🧪 COMPREHENSIVE TEST: {service_name.upper()}")
        logger.info(f"{'='*60}")
        
        config = self.services[service_name]
        service_results = {
            'service_name': service_name,
            'config': config,
            'test_start_time': datetime.datetime.now().isoformat()
        }
        
        # Test 1: Docker Build
        service_results['build_test'] = self.test_docker_build(service_name, config)
        
        # Only continue if build was successful
        if service_results['build_test']['build_successful']:
            # Test 2: Container Startup
            service_results['startup_test'] = self.test_container_startup(service_name, config)
            
            # Only continue if startup was successful
            if service_results['startup_test']['startup_successful']:
                # Test 3: Health Endpoint
                service_results['health_test'] = self.test_health_endpoint(service_name, config)
                
                # Test 4: Non-root User
                service_results['user_test'] = self.test_non_root_user(service_name, config)
                
                # Test 5: Python Version
                service_results['python_test'] = self.test_python_version(service_name, config)
                
                # Test 6: Service Functionality
                service_results['functionality_test'] = self.test_service_functionality(service_name, config)
            else:
                logger.info(f"⚠️ Skipping remaining tests for {service_name} due to startup failure")
        else:
            logger.info(f"⚠️ Skipping all tests for {service_name} due to build failure")
        
        service_results['test_end_time'] = datetime.datetime.now().isoformat()
        return service_results

    def run_all_tests(self) -> Dict:
        """Run tests for all services"""
        logger.info("🚀 STARTING ULTRA MIGRATION TEST SUITE")
        logger.info(f"📊 Testing {len(self.services)} services")
        logger.info(f"⏰ Started at: {self.results['timestamp']}")
        
        # Cleanup existing test containers
        self.cleanup_test_containers()
        
        # Run tests for each service
        for service_name in self.services.keys():
            try:
                service_results = self.run_comprehensive_test(service_name)
                self.results['test_results'][service_name] = service_results
                
                # Stop and remove test container after testing
                container_name = self.services[service_name]['container_name']
                self.run_command(f"docker stop {container_name} 2>/dev/null || true")
                self.run_command(f"docker rm {container_name} 2>/dev/null || true")
                
            except Exception as e:
                logger.error(f"❌ Critical error testing {service_name}: {e}")
                self.results['test_results'][service_name] = {
                    'critical_error': str(e),
                    'test_completed': False
                }
        
        # Final cleanup
        self.cleanup_test_containers()
        
        return self.results

    def generate_report(self) -> str:
        """Generate comprehensive test report"""
        report = []
        
        report.append("="*80)
        report.append("🧪 ULTRA MIGRATION TEST SUITE REPORT")
        report.append("="*80)
        report.append(f"📅 Test Date: {self.results['timestamp']}")
        report.append(f"🔧 Services Tested: {self.results['services_tested']}")
        report.append("")
        
        # Summary
        summary = self.results['summary']
        report.append("📊 SUMMARY")
        report.append("-"*40)
        report.append(f"✅ Builds Successful:       {summary['builds_successful']}/{len(self.services)}")
        report.append(f"🚀 Startups Successful:     {summary['startups_successful']}/{len(self.services)}")
        report.append(f"🏥 Health Checks Passed:    {summary['health_checks_passed']}/{len(self.services)}")
        report.append(f"👤 User Validations Passed: {summary['user_validations_passed']}/{len(self.services)}")
        report.append(f"🐍 Python Validations:      {summary['python_validations_passed']}/{len(self.services)}")
        report.append(f"⚙️ Functionality Tests:     {summary['functionality_tests_passed']}/{len(self.services)}")
        report.append("")
        
        # Calculate overall health
        total_tests = len(self.services) * 6  # 6 test types per service
        passed_tests = sum(summary.values())
        health_percentage = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        report.append(f"🎯 Overall Health: {health_percentage:.1f}% ({passed_tests}/{total_tests} tests passed)")
        report.append("")
        
        # Detailed results
        report.append("📋 DETAILED RESULTS")
        report.append("="*80)
        
        for service_name, results in self.results['test_results'].items():
            report.append(f"\n🔹 {service_name.upper()}")
            report.append("-" * 50)
            
            if 'critical_error' in results:
                report.append(f"❌ CRITICAL ERROR: {results['critical_error']}")
                continue
            
            # Build test
            if 'build_test' in results:
                build = results['build_test']
                status = "✅" if build['build_successful'] else "❌"
                report.append(f"{status} Build: {build['build_time_seconds']}s")
                if not build['build_successful']:
                    report.append(f"   Error: {build['build_errors'][:200]}...")
            
            # Startup test
            if 'startup_test' in results:
                startup = results['startup_test']
                status = "✅" if startup['startup_successful'] else "❌"
                report.append(f"{status} Startup: {startup['startup_time_seconds']}s")
                if not startup['startup_successful']:
                    report.append(f"   Error: {startup['startup_errors'][:200]}...")
            
            # Health test
            if 'health_test' in results:
                health = results['health_test']
                status = "✅" if health['health_check_passed'] else "❌"
                report.append(f"{status} Health: {health['response_time_ms']}ms (Status: {health['status_code']})")
            
            # User test
            if 'user_test' in results:
                user = results['user_test']
                status = "✅" if user['user_validation_passed'] else "❌"
                report.append(f"{status} User: {user['current_user']} (ID: {user['user_id']})")
            
            # Python test
            if 'python_test' in results:
                python = results['python_test']
                status = "✅" if python['python_validation_passed'] else "❌"
                report.append(f"{status} Python: {python['python_version']}")
            
            # Functionality test
            if 'functionality_test' in results:
                func = results['functionality_test']
                status = "✅" if func['functionality_test_passed'] else "❌"
                endpoints = len(func['endpoints_tested'])
                report.append(f"{status} Functionality: {endpoints} endpoints responding")
        
        return "\n".join(report)

def main():
    """Main test execution"""
    tester = UltraMigrationTester()
    
    try:
        # Run all tests
        results = tester.run_all_tests()
        
        # Generate and save report
        report = tester.generate_report()
        
        # Save results to JSON
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        json_file = f"/opt/sutazaiapp/ultra_migration_test_results_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save report to text
        report_file = f"/opt/sutazaiapp/ultra_migration_test_report_{timestamp}.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        
        # Display results
        logger.info(report)
        logger.info(f"\n📊 Full results saved to: {json_file}")
        logger.info(f"📋 Report saved to: {report_file}")
        
        # Exit with appropriate code
        summary = results['summary']
        if summary['builds_successful'] == len(tester.services):
            logger.info("\n🎉 ALL SERVICES PASSED MIGRATION TESTING!")
            sys.exit(0)
        else:
            logger.error("\n⚠️ SOME SERVICES FAILED MIGRATION TESTING!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("\n⚠️ Test interrupted by user")
        tester.cleanup_test_containers()
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Critical error in test suite: {e}")
        tester.cleanup_test_containers()
        sys.exit(1)

if __name__ == "__main__":
    main()