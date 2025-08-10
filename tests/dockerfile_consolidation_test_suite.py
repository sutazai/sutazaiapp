#!/usr/bin/env python3
"""
SutazAI Dockerfile Consolidation Test Suite
Ultra QA Validator - Comprehensive Python Test Suite

This module provides detailed testing for Dockerfile consolidation,
complementing the shell validation script with deep Python-based testing.

Author: ULTRA QA VALIDATOR  
Date: August 10, 2025
Version: 1.0.0
"""

import os
import sys
import json
import time
import docker
import requests
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import tempfile
import yaml
import logging
from datetime import datetime
import concurrent.futures
import psutil

# Optional imports
try:
    import pytest
except ImportError:
    pytest = None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DockerfileValidationSuite:
    """Comprehensive Dockerfile consolidation validation test suite."""
    
    def __init__(self):
        """Initialize the validation suite."""
        self.project_root = Path(__file__).parent.parent
        self.docker_client = docker.from_env()
        self.test_results = {
            'timestamp': datetime.now().isoformat(),
            'tests_passed': 0,
            'tests_failed': 0,
            'tests_skipped': 0,
            'detailed_results': {}
        }
        
    def discover_dockerfiles(self) -> Dict[str, Path]:
        """Discover all Dockerfiles in the project."""
        dockerfiles = {}
        
        # Find all Dockerfiles
        for dockerfile in self.project_root.rglob("Dockerfile*"):
            # Skip backup and test files
            if any(skip in str(dockerfile) for skip in ['.backup', 'test-', 'backup']):
                continue
                
            service_name = dockerfile.parent.name
            if dockerfile.name != "Dockerfile":
                service_name = f"{service_name}_{dockerfile.stem}"
                
            dockerfiles[service_name] = dockerfile
            
        logger.info(f"Discovered {len(dockerfiles)} Dockerfiles")
        return dockerfiles
    
    def test_dockerfile_syntax(self, dockerfile_path: Path) -> Tuple[bool, str]:
        """Test Dockerfile syntax validity."""
        try:
            # Use docker build --dry-run equivalent check
            with tempfile.NamedTemporaryFile(mode='w', suffix='.Dockerfile', delete=False) as tmp:
                tmp.write(dockerfile_path.read_text())
                tmp.flush()
                
                # Basic syntax validation
                result = subprocess.run([
                    'docker', 'build', '--no-cache', '--dry-run', 
                    '-f', tmp.name, str(dockerfile_path.parent)
                ], capture_output=True, text=True, timeout=60)
                
                os.unlink(tmp.name)
                
                if result.returncode == 0:
                    return True, "Syntax valid"
                else:
                    return False, result.stderr
                    
        except Exception as e:
            return False, f"Syntax validation error: {str(e)}"
    
    def test_base_image_usage(self, dockerfile_path: Path) -> Tuple[bool, str]:
        """Test if Dockerfile uses consolidated base images."""
        try:
            content = dockerfile_path.read_text()
            
            # Check for base image usage
            base_patterns = [
                'python.*agent.*master',
                'nodejs.*agent.*master',
                'sutazai.*base'
            ]
            
            uses_base = any(
                pattern in content.lower() 
                for pattern in base_patterns
                for line in content.splitlines() 
                if line.strip().startswith('FROM') and pattern in line.lower()
            )
            
            if uses_base:
                return True, "Uses consolidated base image"
            else:
                # Check if it's a base image itself
                if 'base' in str(dockerfile_path).lower():
                    return True, "Is a base image template"
                else:
                    return False, "Does not use consolidated base image"
                    
        except Exception as e:
            return False, f"Base image check error: {str(e)}"
    
    def test_security_compliance(self, dockerfile_path: Path) -> Tuple[bool, List[str]]:
        """Test Dockerfile security compliance."""
        violations = []
        
        try:
            content = dockerfile_path.read_text()
            lines = content.splitlines()
            
            # Check for non-root user
            has_user_directive = any(line.strip().startswith('USER ') and 'root' not in line.lower() for line in lines)
            if not has_user_directive:
                violations.append("Missing non-root USER directive")
            
            # Check for hardcoded secrets
            secret_patterns = ['password', 'secret', 'key', 'token', 'api_key']
            for i, line in enumerate(lines, 1):
                if any(pattern in line.lower() and '=' in line for pattern in secret_patterns):
                    violations.append(f"Potential hardcoded secret on line {i}")
            
            # Check for insecure practices
            if any('--trusted-host' in line for line in lines):
                violations.append("Insecure pip install with --trusted-host")
            
            # Check for missing health check
            if not any(line.strip().startswith('HEALTHCHECK') for line in lines):
                violations.append("Missing HEALTHCHECK directive")
            
            # Check for proper package updates
            has_apt_update = any('apt-get update' in line for line in lines)
            has_apt_install = any('apt-get install' in line for line in lines)
            
            if has_apt_install and not has_apt_update:
                violations.append("apt-get install without update")
            
            return len(violations) == 0, violations
            
        except Exception as e:
            return False, [f"Security check error: {str(e)}"]
    
    def test_build_optimization(self, dockerfile_path: Path) -> Tuple[bool, Dict]:
        """Test Dockerfile build optimization."""
        metrics = {
            'layer_count': 0,
            'has_multistage': False,
            'has_cache_optimization': False,
            'estimated_size_mb': 0
        }
        
        try:
            content = dockerfile_path.read_text()
            lines = content.splitlines()
            
            # Count layers (RUN, COPY, ADD commands)
            layer_commands = ['RUN', 'COPY', 'ADD']
            metrics['layer_count'] = sum(
                1 for line in lines 
                if any(line.strip().startswith(cmd) for cmd in layer_commands)
            )
            
            # Check for multi-stage build
            from_count = sum(1 for line in lines if line.strip().startswith('FROM'))
            metrics['has_multistage'] = from_count > 1
            
            # Check for cache optimization
            cache_indicators = ['--no-cache-dir', 'rm -rf', 'apt-get clean']
            metrics['has_cache_optimization'] = any(
                indicator in content for indicator in cache_indicators
            )
            
            # Estimate size based on base image and layers
            base_sizes = {
                'python:3.12': 150,
                'node:': 200,
                'alpine': 50,
                'ubuntu': 100
            }
            
            estimated_size = 100  # Default
            for base, size in base_sizes.items():
                if base in content.lower():
                    estimated_size = size
                    break
            
            # Add estimated overhead per layer
            estimated_size += metrics['layer_count'] * 10
            metrics['estimated_size_mb'] = estimated_size
            
            # Determine if optimized
            is_optimized = (
                metrics['layer_count'] <= 15 and  # Reasonable layer count
                metrics['has_cache_optimization'] and
                metrics['estimated_size_mb'] <= 500  # Reasonable size
            )
            
            return is_optimized, metrics
            
        except Exception as e:
            return False, {'error': str(e)}
    
    def test_service_build(self, service_name: str, dockerfile_path: Path, timeout: int = 300) -> Tuple[bool, Dict]:
        """Test actual Docker build of a service."""
        build_info = {
            'build_time_seconds': 0,
            'image_size_mb': 0,
            'build_output': '',
            'success': False
        }
        
        try:
            start_time = time.time()
            
            # Build the image
            image_tag = f"sutazai-test-{service_name.lower()}:validation"
            
            # Build context is the directory containing the Dockerfile
            build_context = str(dockerfile_path.parent)
            
            image, build_logs = self.docker_client.images.build(
                path=build_context,
                dockerfile=str(dockerfile_path.name),
                tag=image_tag,
                timeout=timeout,
                rm=True,
                forcerm=True
            )
            
            build_info['build_time_seconds'] = time.time() - start_time
            build_info['success'] = True
            
            # Get image size
            image.reload()
            build_info['image_size_mb'] = round(image.attrs['Size'] / (1024 * 1024), 1)
            
            # Collect build output
            build_info['build_output'] = '\n'.join([log.get('stream', '') for log in build_logs if 'stream' in log])
            
            # Clean up the test image
            try:
                self.docker_client.images.remove(image.id, force=True)
            except (AssertionError, Exception) as e:
                # TODO: Review this exception handling
                logger.error(f"Unexpected exception: {e}", exc_info=True)
                pass  # Ignore cleanup errors
            
            return True, build_info
            
        except docker.errors.BuildError as e:
            build_info['build_output'] = str(e)
            build_info['build_time_seconds'] = time.time() - start_time
            return False, build_info
            
        except Exception as e:
            build_info['build_output'] = f"Build error: {str(e)}"
            return False, build_info
    
    def test_health_check_functionality(self, service_name: str) -> Tuple[bool, Dict]:
        """Test health check functionality for running services."""
        health_info = {
            'container_found': False,
            'health_status': 'unknown',
            'health_working': False
        }
        
        try:
            # Find running container
            container_name = f"sutazai-{service_name.lower()}"
            containers = self.docker_client.containers.list()
            
            container = None
            for c in containers:
                if container_name in c.name or service_name.lower() in c.name.lower():
                    container = c
                    break
            
            if not container:
                return False, health_info
            
            health_info['container_found'] = True
            container.reload()
            
            # Get health status
            health_status = container.attrs.get('State', {}).get('Health', {}).get('Status', 'none')
            health_info['health_status'] = health_status
            
            if health_status == 'healthy':
                health_info['health_working'] = True
                return True, health_info
            elif health_status == 'starting':
                # Wait a bit for health check to complete
                time.sleep(15)
                container.reload()
                new_status = container.attrs.get('State', {}).get('Health', {}).get('Status', 'none')
                health_info['health_status'] = new_status
                health_info['health_working'] = new_status == 'healthy'
                return new_status == 'healthy', health_info
            else:
                return False, health_info
                
        except Exception as e:
            health_info['error'] = str(e)
            return False, health_info
    
    def test_resource_usage(self, service_name: str) -> Tuple[bool, Dict]:
        """Test resource usage of running services."""
        resource_info = {
            'cpu_usage_percent': 0,
            'memory_usage_mb': 0,
            'memory_limit_mb': 0,
            'optimized': False
        }
        
        try:
            # Find running container
            container_name = f"sutazai-{service_name.lower()}"
            containers = self.docker_client.containers.list()
            
            container = None
            for c in containers:
                if container_name in c.name or service_name.lower() in c.name.lower():
                    container = c
                    break
            
            if not container:
                return False, resource_info
            
            # Get container stats
            stats = container.stats(stream=False)
            
            # Calculate CPU usage
            cpu_stats = stats['cpu_stats']
            precpu_stats = stats['precpu_stats']
            
            cpu_delta = cpu_stats['cpu_usage']['total_usage'] - precpu_stats['cpu_usage']['total_usage']
            system_delta = cpu_stats['system_cpu_usage'] - precpu_stats['system_cpu_usage']
            
            if system_delta > 0:
                cpu_usage = (cpu_delta / system_delta) * len(cpu_stats['cpu_usage']['percpu_usage']) * 100.0
                resource_info['cpu_usage_percent'] = round(cpu_usage, 2)
            
            # Calculate memory usage
            memory_usage = stats['memory_stats']['usage']
            memory_limit = stats['memory_stats']['limit']
            
            resource_info['memory_usage_mb'] = round(memory_usage / (1024 * 1024), 1)
            resource_info['memory_limit_mb'] = round(memory_limit / (1024 * 1024), 1)
            
            # Determine if usage is optimized
            cpu_optimized = resource_info['cpu_usage_percent'] < 80  # Less than 80% CPU
            memory_optimized = resource_info['memory_usage_mb'] < 1000  # Less than 1GB for most services
            
            resource_info['optimized'] = cpu_optimized and memory_optimized
            
            return resource_info['optimized'], resource_info
            
        except Exception as e:
            resource_info['error'] = str(e)
            return False, resource_info
    
    def run_comprehensive_validation(self) -> Dict:
        """Run comprehensive validation of all Dockerfiles."""
        logger.info("Starting comprehensive Dockerfile validation")
        
        # Discover all Dockerfiles
        dockerfiles = self.discover_dockerfiles()
        
        validation_results = {
            'summary': {
                'total_dockerfiles': len(dockerfiles),
                'syntax_valid': 0,
                'uses_base_images': 0,
                'security_compliant': 0,
                'build_optimized': 0,
                'builds_successful': 0,
                'health_checks_working': 0,
                'resource_optimized': 0
            },
            'detailed_results': {},
            'recommendations': []
        }
        
        # Test each Dockerfile
        for service_name, dockerfile_path in dockerfiles.items():
            logger.info(f"Validating service: {service_name}")
            
            service_results = {
                'dockerfile_path': str(dockerfile_path),
                'tests': {}
            }
            
            # Test 1: Syntax validation
            syntax_valid, syntax_message = self.test_dockerfile_syntax(dockerfile_path)
            service_results['tests']['syntax'] = {
                'passed': syntax_valid,
                'message': syntax_message
            }
            if syntax_valid:
                validation_results['summary']['syntax_valid'] += 1
            
            # Test 2: Base image usage
            uses_base, base_message = self.test_base_image_usage(dockerfile_path)
            service_results['tests']['base_image'] = {
                'passed': uses_base,
                'message': base_message
            }
            if uses_base:
                validation_results['summary']['uses_base_images'] += 1
            
            # Test 3: Security compliance
            security_compliant, security_violations = self.test_security_compliance(dockerfile_path)
            service_results['tests']['security'] = {
                'passed': security_compliant,
                'violations': security_violations
            }
            if security_compliant:
                validation_results['summary']['security_compliant'] += 1
            
            # Test 4: Build optimization
            build_optimized, build_metrics = self.test_build_optimization(dockerfile_path)
            service_results['tests']['optimization'] = {
                'passed': build_optimized,
                'metrics': build_metrics
            }
            if build_optimized:
                validation_results['summary']['build_optimized'] += 1
            
            # Test 5: Build success (only for key services)
            key_services = ['backend', 'frontend', 'hardware-resource-optimizer']
            if any(key in service_name.lower() for key in key_services):
                build_success, build_info = self.test_service_build(service_name, dockerfile_path)
                service_results['tests']['build'] = {
                    'passed': build_success,
                    'info': build_info
                }
                if build_success:
                    validation_results['summary']['builds_successful'] += 1
            
            # Test 6: Health check functionality (for running services)
            health_working, health_info = self.test_health_check_functionality(service_name)
            service_results['tests']['health_check'] = {
                'passed': health_working,
                'info': health_info
            }
            if health_working:
                validation_results['summary']['health_checks_working'] += 1
            
            # Test 7: Resource optimization (for running services)
            resource_optimized, resource_info = self.test_resource_usage(service_name)
            service_results['tests']['resources'] = {
                'passed': resource_optimized,
                'info': resource_info
            }
            if resource_optimized:
                validation_results['summary']['resource_optimized'] += 1
            
            validation_results['detailed_results'][service_name] = service_results
        
        # Generate recommendations
        self._generate_recommendations(validation_results)
        
        # Calculate overall score
        total_tests = len(dockerfiles) * 4  # Core tests: syntax, base_image, security, optimization
        passed_tests = (
            validation_results['summary']['syntax_valid'] +
            validation_results['summary']['uses_base_images'] +
            validation_results['summary']['security_compliant'] +
            validation_results['summary']['build_optimized']
        )
        
        validation_results['overall_score'] = round((passed_tests / total_tests) * 100, 1) if total_tests > 0 else 0
        validation_results['grade'] = self._calculate_grade(validation_results['overall_score'])
        
        logger.info(f"Validation completed. Overall score: {validation_results['overall_score']}/100")
        return validation_results
    
    def _generate_recommendations(self, results: Dict):
        """Generate actionable recommendations based on validation results."""
        recommendations = []
        summary = results['summary']
        total = summary['total_dockerfiles']
        
        if summary['syntax_valid'] < total:
            recommendations.append(f"{total - summary['syntax_valid']} Dockerfiles have syntax issues that need fixing")
        
        if summary['uses_base_images'] < total * 0.5:
            recommendations.append("More services should migrate to use consolidated base image templates")
        
        if summary['security_compliant'] < total:
            recommendations.append(f"{total - summary['security_compliant']} services have security violations that need addressing")
        
        if summary['build_optimized'] < total * 0.8:
            recommendations.append("Build optimization can be improved in multiple services")
        
        if summary['builds_successful'] < 3:  # Assuming 3 key services
            recommendations.append("Some key services are failing to build successfully")
        
        results['recommendations'] = recommendations
    
    def _calculate_grade(self, score: float) -> str:
        """Calculate letter grade based on score."""
        if score >= 90:
            return 'A'
        elif score >= 80:
            return 'B'
        elif score >= 70:
            return 'C'
        elif score >= 60:
            return 'D'
        else:
            return 'F'
    
    def save_results(self, results: Dict, output_file: str):
        """Save validation results to JSON file."""
        output_path = self.project_root / output_file
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to: {output_path}")

def main():
    """Main execution function."""
    validator = DockerfileValidationSuite()
    
    # Run comprehensive validation
    results = validator.run_comprehensive_validation()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"dockerfile_validation_results_{timestamp}.json"
    validator.save_results(results, results_file)
    
    # Print summary
    print("\n" + "="*60)
    print("  DOCKERFILE CONSOLIDATION VALIDATION RESULTS")
    print("="*60)
    print(f"Total Dockerfiles:      {results['summary']['total_dockerfiles']}")
    print(f"Syntax Valid:           {results['summary']['syntax_valid']}")
    print(f"Uses Base Images:       {results['summary']['uses_base_images']}")
    print(f"Security Compliant:     {results['summary']['security_compliant']}")
    print(f"Build Optimized:        {results['summary']['build_optimized']}")
    print(f"Builds Successful:      {results['summary']['builds_successful']}")
    print(f"Overall Score:          {results['overall_score']}/100")
    print(f"Grade:                  {results['grade']}")
    print("\nRecommendations:")
    for rec in results['recommendations']:
        print(f"  • {rec}")
    print("\n" + "="*60)
    
    # Return exit code based on score
    return 0 if results['overall_score'] >= 80 else 1

if __name__ == "__main__":
    sys.exit(main())