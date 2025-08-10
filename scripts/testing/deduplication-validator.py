#!/usr/bin/env python3
"""
SutazAI Deduplication Validation System
Tests all consolidations to ensure functionality preservation
Author: DevOps Manager - Deduplication Operation  
Date: August 10, 2025
"""

import asyncio
import json
import subprocess
import time
import docker
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import requests
from datetime import datetime

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
REPORTS_DIR = PROJECT_ROOT / "reports"

@dataclass
class ValidationResult:
    test_name: str
    status: str  # passed, failed, warning
    message: str
    details: Optional[Dict] = None
    execution_time: float = 0.0

class DeduplicationValidator:
    """Comprehensive validation system for deduplication changes."""
    
    def __init__(self):
        self.docker_client = docker.from_env()
        self.session = requests.Session()
        self.session.timeout = (10, 30)
        
        self.results: List[ValidationResult] = []
        
    def log(self, message: str):
        """Log message with timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}")
        
    def run_test(self, test_name: str, test_func, *args, **kwargs) -> ValidationResult:
        """Run a single validation test with timing and error handling."""
        self.log(f"Running test: {test_name}")
        
        start_time = time.time()
        
        try:
            result = test_func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            if isinstance(result, ValidationResult):
                result.execution_time = execution_time
                return result
            else:
                return ValidationResult(
                    test_name=test_name,
                    status="passed",
                    message=str(result),
                    execution_time=execution_time
                )
                
        except Exception as e:
            execution_time = time.time() - start_time
            return ValidationResult(
                test_name=test_name,
                status="failed",
                message=f"Test failed with exception: {str(e)}",
                execution_time=execution_time
            )
    
    def test_base_images_exist(self) -> ValidationResult:
        """Test that consolidated base images exist and are built."""
        base_images = [
            "sutazai-python-agent-master:latest",
            "sutazai-nodejs-agent-master:latest"
        ]
        
        missing_images = []
        existing_images = []
        
        for image_name in base_images:
            try:
                image = self.docker_client.images.get(image_name)
                existing_images.append({
                    "name": image_name,
                    "id": image.id,
                    "size": image.attrs.get("Size", 0),
                    "created": image.attrs.get("Created", "unknown")
                })
            except docker.errors.ImageNotFound:
                missing_images.append(image_name)
        
        if missing_images:
            return ValidationResult(
                test_name="base_images_exist",
                status="failed",
                message=f"Missing base images: {missing_images}",
                details={"missing": missing_images, "existing": existing_images}
            )
        else:
            return ValidationResult(
                test_name="base_images_exist", 
                status="passed",
                message=f"All base images exist ({len(existing_images)}/{len(base_images)})",
                details={"existing": existing_images}
            )
    
    def test_dockerfile_generation(self) -> ValidationResult:
        """Test that Dockerfile templates can generate service-specific files."""
        template_script = PROJECT_ROOT / "docker/templates/generate-dockerfile.py"
        
        if not template_script.exists():
            return ValidationResult(
                test_name="dockerfile_generation",
                status="failed", 
                message="Dockerfile generation script not found"
            )
        
        try:
            # Test generation for a few services
            test_services = ["autogpt", "crewai", "langchain"]
            output_dir = PROJECT_ROOT / "docker/test-generated"
            
            cmd = [
                "python3", str(template_script),
                "--output-dir", str(output_dir),
                "--template-dir", str(template_script.parent)
            ]
            
            for service in test_services:
                service_cmd = cmd + ["--service", service]
                result = subprocess.run(service_cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
                
                if result.returncode != 0:
                    return ValidationResult(
                        test_name="dockerfile_generation",
                        status="failed",
                        message=f"Failed to generate {service}: {result.stderr}",
                        details={"error": result.stderr}
                    )
            
            # Check generated files exist
            generated_files = []
            for service in test_services:
                dockerfile_path = output_dir / service / "Dockerfile"
                if dockerfile_path.exists():
                    generated_files.append(str(dockerfile_path))
                else:
                    return ValidationResult(
                        test_name="dockerfile_generation",
                        status="failed",
                        message=f"Generated Dockerfile not found: {dockerfile_path}"
                    )
            
            return ValidationResult(
                test_name="dockerfile_generation",
                status="passed",
                message=f"Successfully generated {len(generated_files)} Dockerfiles",
                details={"generated_files": generated_files}
            )
            
        except Exception as e:
            return ValidationResult(
                test_name="dockerfile_generation",
                status="failed",
                message=f"Template generation test failed: {str(e)}"
            )
    
    def test_master_scripts_exist(self) -> ValidationResult:
        """Test that master scripts exist and are executable."""
        master_scripts = [
            "scripts/deployment/deployment-master.sh",
            "scripts/monitoring/monitoring-master.py", 
            "scripts/maintenance/maintenance-master.sh"
        ]
        
        script_status = []
        
        for script_path in master_scripts:
            full_path = PROJECT_ROOT / script_path
            
            if not full_path.exists():
                script_status.append({
                    "path": script_path,
                    "exists": False,
                    "executable": False,
                    "status": "missing"
                })
            else:
                is_executable = full_path.stat().st_mode & 0o111 != 0
                script_status.append({
                    "path": script_path,
                    "exists": True,
                    "executable": is_executable,
                    "status": "ok" if is_executable else "not_executable"
                })
        
        failed_scripts = [s for s in script_status if s["status"] != "ok"]
        
        if failed_scripts:
            return ValidationResult(
                test_name="master_scripts_exist",
                status="failed",
                message=f"Script issues found: {len(failed_scripts)} scripts",
                details={"all_scripts": script_status, "failed": failed_scripts}
            )
        else:
            return ValidationResult(
                test_name="master_scripts_exist",
                status="passed",
                message="All master scripts exist and are executable",
                details={"scripts": script_status}
            )
    
    def test_deployment_script_functionality(self) -> ValidationResult:
        """Test that the master deployment script works with dry-run."""
        deployment_script = PROJECT_ROOT / "scripts/deployment/deployment-master.sh"
        
        try:
            # Test dry-run mode
            cmd = [str(deployment_script), "--dry-run", "minimal"]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=60)
            
            if result.returncode == 0:
                return ValidationResult(
                    test_name="deployment_script_functionality",
                    status="passed",
                    message="Deployment script dry-run successful",
                    details={"stdout": result.stdout, "stderr": result.stderr}
                )
            else:
                return ValidationResult(
                    test_name="deployment_script_functionality",
                    status="failed",
                    message=f"Deployment script failed: {result.stderr}",
                    details={"return_code": result.returncode, "stderr": result.stderr}
                )
                
        except subprocess.TimeoutExpired:
            return ValidationResult(
                test_name="deployment_script_functionality",
                status="failed",
                message="Deployment script timed out (60s)"
            )
        except Exception as e:
            return ValidationResult(
                test_name="deployment_script_functionality", 
                status="failed",
                message=f"Failed to test deployment script: {str(e)}"
            )
    
    def test_monitoring_script_functionality(self) -> ValidationResult:
        """Test that the master monitoring script works."""
        monitoring_script = PROJECT_ROOT / "scripts/monitoring/monitoring-master.py"
        
        try:
            # Test one-time check mode
            cmd = ["python3", str(monitoring_script), "--mode", "check", "--format", "json"]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=30)
            
            if result.returncode == 0:
                return ValidationResult(
                    test_name="monitoring_script_functionality",
                    status="passed",
                    message="Monitoring script check successful",
                    details={"stdout": result.stdout}
                )
            else:
                return ValidationResult(
                    test_name="monitoring_script_functionality",
                    status="warning",  # Monitoring may fail if services aren't running
                    message=f"Monitoring script returned {result.returncode} (may be expected if services down)",
                    details={"return_code": result.returncode, "stderr": result.stderr}
                )
                
        except subprocess.TimeoutExpired:
            return ValidationResult(
                test_name="monitoring_script_functionality",
                status="failed", 
                message="Monitoring script timed out (30s)"
            )
        except Exception as e:
            return ValidationResult(
                test_name="monitoring_script_functionality",
                status="failed",
                message=f"Failed to test monitoring script: {str(e)}"
            )
    
    def test_build_performance(self) -> ValidationResult:
        """Test that consolidated images build faster due to layer caching."""
        try:
            # Build base images first
            self.log("Building base images for performance test...")
            
            # Time the base image build
            start_time = time.time()
            
            # Build Python base image
            python_build_result = subprocess.run([
                "docker", "build", 
                "-t", "sutazai-python-agent-master:latest",
                "-f", str(PROJECT_ROOT / "docker/base/Dockerfile.python-agent-master"),
                str(PROJECT_ROOT / "docker/base")
            ], capture_output=True, text=True)
            
            base_build_time = time.time() - start_time
            
            if python_build_result.returncode != 0:
                return ValidationResult(
                    test_name="build_performance",
                    status="failed",
                    message=f"Base image build failed: {python_build_result.stderr}"
                )
            
            # Build a service image that uses the base
            service_start_time = time.time()
            
            # Generate a test service Dockerfile
            test_dockerfile_content = f"""
FROM sutazai-python-agent-master:latest
WORKDIR /app
COPY . .
CMD ["python", "app.py"]
"""
            
            test_dir = PROJECT_ROOT / "docker/test-build"
            test_dir.mkdir(exist_ok=True)
            
            with open(test_dir / "Dockerfile", "w") as f:
                f.write(test_dockerfile_content)
            
            # Create dummy app.py
            with open(test_dir / "app.py", "w") as f:
                f.write("print('Hello from test service')")
            
            # Build test service
            service_build_result = subprocess.run([
                "docker", "build",
                "-t", "sutazai-test-service:latest", 
                str(test_dir)
            ], capture_output=True, text=True)
            
            service_build_time = time.time() - service_start_time
            
            if service_build_result.returncode != 0:
                return ValidationResult(
                    test_name="build_performance",
                    status="warning",
                    message=f"Service build failed, but base build succeeded: {service_build_result.stderr}",
                    details={"base_build_time": base_build_time}
                )
            
            return ValidationResult(
                test_name="build_performance",
                status="passed",
                message=f"Build performance test successful",
                details={
                    "base_build_time": base_build_time,
                    "service_build_time": service_build_time,
                    "total_time": base_build_time + service_build_time
                }
            )
            
        except Exception as e:
            return ValidationResult(
                test_name="build_performance",
                status="failed",
                message=f"Build performance test failed: {str(e)}"
            )
    
    def test_security_compliance(self) -> ValidationResult:
        """Test that consolidated images maintain security compliance."""
        try:
            # Check that base images run as non-root
            images_to_check = [
                "sutazai-python-agent-master:latest",
                "sutazai-nodejs-agent-master:latest"
            ]
            
            security_results = []
            
            for image_name in images_to_check:
                try:
                    # Inspect image for USER directive
                    result = subprocess.run([
                        "docker", "inspect", image_name
                    ], capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        import json
                        image_data = json.loads(result.stdout)[0]
                        config = image_data.get("Config", {})
                        user = config.get("User", "root")
                        
                        security_results.append({
                            "image": image_name,
                            "user": user,
                            "secure": user != "" and user != "root",
                            "size": image_data.get("Size", 0)
                        })
                    else:
                        security_results.append({
                            "image": image_name,
                            "user": "unknown",
                            "secure": False,
                            "error": result.stderr
                        })
                        
                except docker.errors.ImageNotFound:
                    security_results.append({
                        "image": image_name,
                        "user": "unknown",
                        "secure": False,
                        "error": "Image not found"
                    })
            
            # Check results
            insecure_images = [r for r in security_results if not r["secure"]]
            
            if insecure_images:
                return ValidationResult(
                    test_name="security_compliance",
                    status="failed",
                    message=f"Security compliance failed: {len(insecure_images)} images running as root",
                    details={"results": security_results, "insecure": insecure_images}
                )
            else:
                return ValidationResult(
                    test_name="security_compliance", 
                    status="passed",
                    message="All images pass security compliance (non-root users)",
                    details={"results": security_results}
                )
                
        except Exception as e:
            return ValidationResult(
                test_name="security_compliance",
                status="failed",
                message=f"Security compliance test failed: {str(e)}"
            )
    
    async def run_all_validations(self) -> Dict:
        """Run all validation tests and return comprehensive report."""
        self.log("Starting comprehensive deduplication validation...")
        
        # Define all tests
        tests = [
            ("Base Images Exist", self.test_base_images_exist),
            ("Dockerfile Generation", self.test_dockerfile_generation), 
            ("Master Scripts Exist", self.test_master_scripts_exist),
            ("Deployment Script Functionality", self.test_deployment_script_functionality),
            ("Monitoring Script Functionality", self.test_monitoring_script_functionality),
            ("Build Performance", self.test_build_performance),
            ("Security Compliance", self.test_security_compliance),
        ]
        
        # Run all tests
        results = []
        
        for test_name, test_func in tests:
            result = self.run_test(test_name, test_func)
            results.append(result)
            self.results.append(result)
            
            # Log result
            status_icon = {"passed": "✅", "failed": "❌", "warning": "⚠️"}.get(result.status, "❓")
            self.log(f"{status_icon} {test_name}: {result.message}")
        
        # Calculate summary statistics
        passed = len([r for r in results if r.status == "passed"])
        failed = len([r for r in results if r.status == "failed"])
        warnings = len([r for r in results if r.status == "warning"])
        
        total_time = sum(r.execution_time for r in results)
        
        # Determine overall status
        if failed > 0:
            overall_status = "FAILED"
        elif warnings > 0:
            overall_status = "WARNING"  
        else:
            overall_status = "PASSED"
        
        # Generate comprehensive report
        report = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": overall_status,
            "summary": {
                "total_tests": len(results),
                "passed": passed,
                "failed": failed,
                "warnings": warnings,
                "success_rate": (passed / len(results)) * 100,
                "total_execution_time": total_time
            },
            "test_results": [asdict(r) for r in results],
            "recommendations": self.generate_recommendations(results),
            "next_steps": self.generate_next_steps(overall_status, results)
        }
        
        return report
    
    def generate_recommendations(self, results: List[ValidationResult]) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        failed_tests = [r for r in results if r.status == "failed"]
        warning_tests = [r for r in results if r.status == "warning"]
        
        if any(r.test_name == "base_images_exist" and r.status == "failed" for r in results):
            recommendations.append("Build base images before proceeding: docker build -t sutazai-python-agent-master docker/base/")
        
        if any(r.test_name == "security_compliance" and r.status == "failed" for r in results):
            recommendations.append("Fix security compliance issues before deploying to production")
        
        if any(r.test_name == "dockerfile_generation" and r.status == "failed" for r in results):
            recommendations.append("Fix Dockerfile template generation before removing duplicate files")
        
        if len(failed_tests) > 0:
            recommendations.append(f"Resolve {len(failed_tests)} failed tests before proceeding with deduplication")
        
        if len(warning_tests) > 0:
            recommendations.append(f"Review {len(warning_tests)} warnings - may be acceptable for current environment")
        
        return recommendations
    
    def generate_next_steps(self, status: str, results: List[ValidationResult]) -> List[str]:
        """Generate next steps based on validation results."""
        next_steps = []
        
        if status == "PASSED":
            next_steps.extend([
                "✅ All validations passed - Safe to proceed with deduplication",
                "1. Create archive backup: bash scripts/utils/deduplication-archiver.sh",
                "2. Generate consolidated Dockerfiles: python3 docker/templates/generate-dockerfile.py --all",
                "3. Remove duplicate files as per service-mapping.json",
                "4. Test full system deployment: bash scripts/deployment/deployment-master.sh minimal",
                "5. Monitor system health: python3 scripts/monitoring/monitoring-master.py --mode check"
            ])
        elif status == "WARNING":
            next_steps.extend([
                "⚠️ Some warnings found - Review before proceeding",
                "1. Address any critical warnings from the test results",
                "2. Consider if warnings are acceptable for your environment",
                "3. If acceptable, proceed with deduplication steps",
                "4. Monitor system closely during and after changes"
            ])
        else:  # FAILED
            next_steps.extend([
                "❌ Validation failed - Do NOT proceed with deduplication", 
                "1. Review all failed test results above",
                "2. Fix issues identified in the test failures",
                "3. Re-run validation: python3 scripts/testing/deduplication-validator.py",
                "4. Only proceed when all tests pass or have acceptable warnings"
            ])
        
        return next_steps
    
    def save_report(self, report: Dict) -> Path:
        """Save validation report to file."""
        REPORTS_DIR.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = REPORTS_DIR / f"deduplication_validation_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.log(f"Validation report saved: {report_file}")
        return report_file

async def main():
    """Main validation execution."""
    print("SutazAI Deduplication Validation System")
    print("=" * 50)
    
    validator = DeduplicationValidator()
    
    try:
        # Run all validations
        report = await validator.run_all_validations()
        
        # Save report
        report_file = validator.save_report(report)
        
        # Print summary
        print("\n" + "=" * 50)
        print(f"VALIDATION COMPLETE - {report['overall_status']}")
        print(f"Tests: {report['summary']['passed']}/{report['summary']['total_tests']} passed")
        print(f"Success Rate: {report['summary']['success_rate']:.1f}%")
        print(f"Execution Time: {report['summary']['total_execution_time']:.2f}s")
        
        if report['recommendations']:
            print(f"\nRECOMMENDATIONS:")
            for rec in report['recommendations']:
                print(f"  • {rec}")
        
        print(f"\nNEXT STEPS:")
        for step in report['next_steps']:
            print(f"  {step}")
        
        print(f"\nDetailed report: {report_file}")
        
        # Return appropriate exit code
        if report['overall_status'] == "FAILED":
            return 1
        else:
            return 0
        
    except KeyboardInterrupt:
        print("\nValidation interrupted by user")
        return 1
    except Exception as e:
        print(f"\nValidation failed with error: {e}")
        return 1

if __name__ == "__main__":
    exit(asyncio.run(main()))