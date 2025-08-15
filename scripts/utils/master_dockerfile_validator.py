#!/usr/bin/env python3
"""
SutazAI Master Dockerfile Consolidation Validator
Ultra QA Validator - Master Test Orchestration & Reporting

This is the master validator that orchestrates all Dockerfile consolidation
validation tests and produces a comprehensive final report.

Author: ULTRA QA VALIDATOR  
Date: August 10, 2025
Version: 1.0.0
"""

import os
import sys
import json
import asyncio
import subprocess
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import concurrent.futures

# Add the project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import validation modules
try:
    from tests.dockerfile_consolidation_test_suite import DockerfileValidationSuite
    from tests.dockerfile_performance_validator import DockerfilePerformanceValidator
    from tests.dockerfile_security_validator import DockerfileSecurityValidator
except ImportError as e:
    logger.error(f"Error importing validation modules: {e}")
    logger.info("Make sure all validation modules are present in the tests directory")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('dockerfile_validation_master.log')
    ]
)
logger = logging.getLogger(__name__)

class MasterDockerfileValidator:
    """Master orchestrator for all Dockerfile consolidation validation tests."""
    
    def __init__(self):
        """Initialize master validator."""
        self.project_root = project_root
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize sub-validators
        self.consolidation_validator = DockerfileValidationSuite()
        self.performance_validator = DockerfilePerformanceValidator()
        self.security_validator = DockerfileSecurityValidator()
        
        # Results storage
        self.validation_results = {
            'master_validation': {
                'timestamp': datetime.now().isoformat(),
                'validation_id': f'master_dockerfile_validation_{self.timestamp}',
                'version': '1.0.0'
            },
            'consolidation_results': {},
            'performance_results': {},
            'security_results': {},
            'shell_script_results': {},
            'final_assessment': {}
        }
    
    def run_shell_validation(self) -> Dict:
        """Run the shell-based validation script."""
        logger.info("Running shell-based validation script")
        
        shell_script = self.project_root / "scripts" / "validate-dockerfiles.sh"
        
        if not shell_script.exists():
            return {
                'executed': False,
                'error': 'Shell validation script not found',
                'exit_code': 1
            }
        
        try:
            # Make script executable
            os.chmod(shell_script, 0o755)
            
            # Run the shell script
            start_time = datetime.now()
            result = subprocess.run(
                [str(shell_script)],
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                timeout=1800  # 30 minute timeout
            )
            end_time = datetime.now()
            
            # Parse JSON report if it exists
            json_report = None
            report_files = list(self.project_root.glob("dockerfile_validation_report_*.json"))
            if report_files:
                latest_report = max(report_files, key=lambda p: p.stat().st_mtime)
                try:
                    with open(latest_report) as f:
                        json_report = json.load(f)
                except Exception as e:
                    logger.warning(f"Could not parse shell validation report: {e}")
            
            return {
                'executed': True,
                'exit_code': result.returncode,
                'duration_seconds': (end_time - start_time).total_seconds(),
                'stdout': result.stdout,
                'stderr': result.stderr,
                'json_report': json_report,
                'success': result.returncode == 0
            }
            
        except subprocess.TimeoutExpired:
            return {
                'executed': True,
                'error': 'Shell validation script timed out',
                'exit_code': 124,
                'timeout': True
            }
        except Exception as e:
            logger.error(f"Shell validation failed: {e}")
            return {
                'executed': False,
                'error': str(e),
                'exit_code': 1
            }
    
    def run_consolidation_validation(self) -> Dict:
        """Run comprehensive consolidation validation."""
        logger.info("Running consolidation validation tests")
        
        try:
            results = self.consolidation_validator.run_comprehensive_validation()
            
            # Save individual results
            results_file = f"consolidation_validation_{self.timestamp}.json"
            self.consolidation_validator.save_results(results, results_file)
            
            return {
                'executed': True,
                'success': results.get('overall_score', 0) >= 70,
                'results': results,
                'results_file': results_file
            }
            
        except Exception as e:
            logger.error(f"Consolidation validation failed: {e}")
            return {
                'executed': False,
                'error': str(e),
                'success': False
            }
    
    async def run_performance_validation(self) -> Dict:
        """Run performance validation tests."""
        logger.info("Running performance validation tests")
        
        try:
            # Define key services to test
            services_config = [
                {'name': 'backend', 'port': 10010, 'endpoint': '/health'},
                {'name': 'frontend', 'port': 10011, 'endpoint': '/'},
                {'name': 'ai-agent-orchestrator', 'port': 8589, 'endpoint': '/health'},
                {'name': 'ollama-integration', 'port': 8090, 'endpoint': '/health'},
                {'name': 'hardware-resource-optimizer', 'port': 11110, 'endpoint': '/health'}
            ]
            
            results = await self.performance_validator.validate_service_performance(services_config)
            
            # Save individual results
            results_file = f"performance_validation_{self.timestamp}.json"
            self.performance_validator.save_performance_results(results, results_file)
            
            return {
                'executed': True,
                'success': results.get('overall_performance_grade', 'F') in ['A', 'B', 'C'],
                'results': results,
                'results_file': results_file
            }
            
        except Exception as e:
            logger.error(f"Performance validation failed: {e}")
            return {
                'executed': False,
                'error': str(e),
                'success': False
            }
    
    def run_security_validation(self) -> Dict:
        """Run security validation tests."""
        logger.info("Running security validation tests")
        
        try:
            # Discover Dockerfiles
            dockerfiles = {}
            for dockerfile in self.project_root.rglob("Dockerfile*"):
                if any(skip in str(dockerfile) for skip in ['.backup', 'test-', 'backup']):
                    continue
                
                service_name = dockerfile.parent.name
                if dockerfile.name != "Dockerfile":
                    service_name = f"{service_name}_{dockerfile.stem}"
                
                dockerfiles[service_name] = dockerfile
            
            # Key containers to scan
            key_containers = [
                'backend', 'frontend', 'ai-agent-orchestrator', 
                'hardware-resource-optimizer', 'ollama-integration'
            ]
            
            results = self.security_validator.run_comprehensive_security_validation(
                dockerfiles, key_containers
            )
            
            # Save individual results
            results_file = f"security_validation_{self.timestamp}.json"
            self.security_validator.save_security_results(results, results_file)
            
            return {
                'executed': True,
                'success': results.get('summary', {}).get('overall_security_grade', 'F') in ['A', 'B'],
                'results': results,
                'results_file': results_file
            }
            
        except Exception as e:
            logger.error(f"Security validation failed: {e}")
            return {
                'executed': False,
                'error': str(e),
                'success': False
            }
    
    def calculate_final_assessment(self) -> Dict:
        """Calculate final assessment based on all validation results."""
        logger.info("Calculating final assessment")
        
        assessment = {
            'timestamp': datetime.now().isoformat(),
            'validation_scores': {},
            'overall_score': 0,
            'overall_grade': 'F',
            'passed_validations': 0,
            'total_validations': 0,
            'critical_issues': [],
            'recommendations': [],
            'deployment_readiness': 'NOT_READY'
        }
        
        # Shell validation score
        shell_results = self.validation_results.get('shell_script_results', {})
        if shell_results.get('executed') and shell_results.get('json_report'):
            shell_score = shell_results['json_report'].get('validation_scores', {}).get('overall_score', 0)
            assessment['validation_scores']['shell_validation'] = shell_score
            if shell_results.get('success'):
                assessment['passed_validations'] += 1
        assessment['total_validations'] += 1
        
        # Consolidation validation score
        consolidation_results = self.validation_results.get('consolidation_results', {})
        if consolidation_results.get('executed'):
            consolidation_score = consolidation_results.get('results', {}).get('overall_score', 0)
            assessment['validation_scores']['consolidation_validation'] = consolidation_score
            if consolidation_results.get('success'):
                assessment['passed_validations'] += 1
        assessment['total_validations'] += 1
        
        # Performance validation score
        performance_results = self.validation_results.get('performance_results', {})
        if performance_results.get('executed'):
            perf_grade = performance_results.get('results', {}).get('overall_performance_grade', 'F')
            perf_score = {'A': 95, 'B': 85, 'C': 75, 'D': 65, 'F': 0}.get(perf_grade, 0)
            assessment['validation_scores']['performance_validation'] = perf_score
            if performance_results.get('success'):
                assessment['passed_validations'] += 1
        assessment['total_validations'] += 1
        
        # Security validation score
        security_results = self.validation_results.get('security_results', {})
        if security_results.get('executed'):
            sec_grade = security_results.get('results', {}).get('summary', {}).get('overall_security_grade', 'F')
            sec_score = {'A': 95, 'B': 85, 'C': 75, 'D': 65, 'F': 0}.get(sec_grade, 0)
            assessment['validation_scores']['security_validation'] = sec_score
            if security_results.get('success'):
                assessment['passed_validations'] += 1
        assessment['total_validations'] += 1
        
        # Calculate weighted overall score
        scores = list(assessment['validation_scores'].values())
        if scores:
            # Weighted average: Security 30%, Performance 25%, Consolidation 25%, Shell 20%
            weights = [0.20, 0.25, 0.25, 0.30]  # shell, consolidation, performance, security
            weighted_score = sum(score * weight for score, weight in zip(scores, weights[:len(scores)]))
            assessment['overall_score'] = round(weighted_score, 1)
        
        # Calculate overall grade
        score = assessment['overall_score']
        if score >= 90:
            assessment['overall_grade'] = 'A'
        elif score >= 80:
            assessment['overall_grade'] = 'B'
        elif score >= 70:
            assessment['overall_grade'] = 'C'
        elif score >= 60:
            assessment['overall_grade'] = 'D'
        else:
            assessment['overall_grade'] = 'F'
        
        # Identify critical issues
        critical_issues = []
        
        # Security critical issues
        if security_results.get('results', {}).get('summary', {}).get('critical_vulnerabilities', 0) > 0:
            critical_issues.append("Critical security vulnerabilities found in containers")
        
        # Performance critical issues
        if performance_results.get('results', {}).get('performance_summary', {}).get('failing', 0) > 0:
            critical_issues.append("Performance tests failing for critical services")
        
        # Build critical issues
        if consolidation_results.get('results', {}).get('summary', {}).get('failed_builds', 0) > 0:
            critical_issues.append("Docker builds failing for some services")
        
        assessment['critical_issues'] = critical_issues
        
        # Generate recommendations
        recommendations = []
        
        if assessment['overall_score'] < 80:
            recommendations.append("Overall validation score below production threshold (80%)")
        
        if assessment['passed_validations'] < assessment['total_validations']:
            failed_count = assessment['total_validations'] - assessment['passed_validations']
            recommendations.append(f"{failed_count} validation test(s) failed - address before deployment")
        
        if critical_issues:
            recommendations.append("Address critical issues before proceeding with production deployment")
        
        if not critical_issues and assessment['overall_score'] >= 80:
            recommendations.append("System ready for production deployment")
        
        assessment['recommendations'] = recommendations
        
        # Determine deployment readiness
        if len(critical_issues) == 0 and assessment['overall_score'] >= 80:
            assessment['deployment_readiness'] = 'READY'
        elif len(critical_issues) == 0 and assessment['overall_score'] >= 70:
            assessment['deployment_readiness'] = 'READY_WITH_MONITORING'
        elif assessment['overall_score'] >= 60:
            assessment['deployment_readiness'] = 'NEEDS_IMPROVEMENT'
        else:
            assessment['deployment_readiness'] = 'NOT_READY'
        
        return assessment
    
    async def run_all_validations(self) -> Dict:
        """Run all validation tests in parallel where possible."""
        logger.info("Starting master Dockerfile consolidation validation")
        
        # Run shell validation first (baseline)
        self.validation_results['shell_script_results'] = self.run_shell_validation()
        
        # Run Python-based validations in parallel
        async def run_validations():
            # Create tasks for async validations
            tasks = [
                asyncio.create_task(self.run_performance_validation()),
            ]
            
            # Run sync validations in executor
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                consolidation_future = loop.run_in_executor(
                    executor, self.run_consolidation_validation
                )
                security_future = loop.run_in_executor(
                    executor, self.run_security_validation
                )
                
                # Wait for all validations to complete
                performance_result = await tasks[0]
                consolidation_result = await consolidation_future
                security_result = await security_future
            
            return {
                'performance': performance_result,
                'consolidation': consolidation_result,
                'security': security_result
            }
        
        # Execute all validations
        validation_results = await run_validations()
        
        # Store results
        self.validation_results['performance_results'] = validation_results['performance']
        self.validation_results['consolidation_results'] = validation_results['consolidation']
        self.validation_results['security_results'] = validation_results['security']
        
        # Calculate final assessment
        self.validation_results['final_assessment'] = self.calculate_final_assessment()
        
        return self.validation_results
    
    def save_master_report(self):
        """Save the master validation report."""
        report_file = self.project_root / f"master_dockerfile_validation_report_{self.timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(self.validation_results, f, indent=2)
        
        logger.info(f"Master validation report saved: {report_file}")
        return report_file
    
    def print_executive_summary(self):
        """Print executive summary of validation results."""
        assessment = self.validation_results.get('final_assessment', {})
        
        logger.info("\n" + "="*80)
        logger.info("  DOCKERFILE CONSOLIDATION VALIDATION - EXECUTIVE SUMMARY")
        logger.info("="*80)
        logger.info(f"Validation Timestamp:   {assessment.get('timestamp', 'Unknown')}")
        logger.info(f"Overall Score:          {assessment.get('overall_score', 0)}/100")
        logger.info(f"Overall Grade:          {assessment.get('overall_grade', 'F')}")
        logger.info(f"Passed Validations:     {assessment.get('passed_validations', 0)}/{assessment.get('total_validations', 0)}")
        logger.info(f"Deployment Readiness:   {assessment.get('deployment_readiness', 'NOT_READY')}")
        logger.info()
        
        # Validation scores breakdown
        scores = assessment.get('validation_scores', {})
        if scores:
            logger.info("VALIDATION SCORES:")
            for validation_type, score in scores.items():
                logger.info(f"  • {validation_type.replace('_', ' ').title()}: {score}/100")
            logger.info()
        
        # Critical issues
        critical_issues = assessment.get('critical_issues', [])
        if critical_issues:
            logger.error("🚨 CRITICAL ISSUES:")
            for issue in critical_issues:
                logger.info(f"  • {issue}")
            logger.info()
        
        # Recommendations
        recommendations = assessment.get('recommendations', [])
        if recommendations:
            logger.info("RECOMMENDATIONS:")
            for rec in recommendations:
                logger.info(f"  • {rec}")
            logger.info()
        
        # Final status
        deployment_status = assessment.get('deployment_readiness', 'NOT_READY')
        if deployment_status == 'READY':
            logger.info("✅ VALIDATION PASSED - System ready for production deployment")
        elif deployment_status == 'READY_WITH_MONITORING':
            logger.info("⚠️  VALIDATION CONDITIONALLY PASSED - Deploy with enhanced monitoring")
        elif deployment_status == 'NEEDS_IMPROVEMENT':
            logger.info("⚠️  VALIDATION NEEDS IMPROVEMENT - Address issues before deployment")
        else:
            logger.error("❌ VALIDATION FAILED - System not ready for production deployment")
        
        logger.info("="*80)

async def main():
    """Main execution function."""
    try:
        # Initialize master validator
        master_validator = MasterDockerfileValidator()
        
        # Run all validations
        results = await master_validator.run_all_validations()
        
        # Save master report
        report_file = master_validator.save_master_report()
        
        # Print executive summary
        master_validator.print_executive_summary()
        
        logger.info(f"\nDetailed master report: {report_file}")
        
        # Return appropriate exit code
        final_assessment = results.get('final_assessment', {})
        deployment_ready = final_assessment.get('deployment_readiness', 'NOT_READY')
        
        if deployment_ready in ['READY', 'READY_WITH_MONITORING']:
            return 0  # Success
        else:
            return 1  # Failure
            
    except KeyboardInterrupt:
        logger.info("Validation interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Master validation failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))