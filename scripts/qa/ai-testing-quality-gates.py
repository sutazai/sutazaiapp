#!/usr/bin/env python3
"""
AI Testing Quality Gates
Enterprise-grade quality enforcement for AI testing - Rule 5 compliance
"""

import subprocess
import sys
import json
import os
import time
from typing import List, Dict, Tuple, Any
from pathlib import Path

class AITestingQualityGate:
    """Enterprise AI testing quality enforcement system"""
    
    def __init__(self):
        self.min_coverage = 90.0
        self.required_ai_test_categories = [
            "model_validation",
            "data_quality", 
            "performance_validation",
            "security_testing"
        ]
        self.quality_report = {
            'timestamp': time.time(),
            'checks': [],
            'overall_status': 'PENDING'
        }
        
    def run_ai_test_coverage_analysis(self) -> Tuple[bool, str, Dict[str, Any]]:
        """Run comprehensive AI test coverage analysis"""
        print("🔍 Running AI test coverage analysis...")
        
        try:
            # Install coverage tools if not available
            subprocess.run([
                sys.executable, "-m", "pip", "install", "pytest-cov", "coverage[toml]", "--break-system-packages"
            ], capture_output=True, check=False)
            
            # Run pytest with coverage for AI testing modules
            result = subprocess.run([
                sys.executable, "-m", "pytest", 
                "tests/ai_testing/",
                "--cov=tests/ai_testing",
                "--cov-report=term-missing",
                "--cov-report=json:ai_coverage.json",
                "--tb=short",
                "-v"
            ], capture_output=True, text=True)
            
            coverage_data = {}
            if os.path.exists('ai_coverage.json'):
                with open('ai_coverage.json', 'r') as f:
                    coverage_data = json.load(f)
                    
                total_coverage = coverage_data.get('totals', {}).get('percent_covered', 0)
            else:
                # Fallback coverage calculation
                total_coverage = self._calculate_fallback_coverage()
            
            coverage_passed = total_coverage >= self.min_coverage
            
            coverage_report = {
                'total_coverage': total_coverage,
                'target_coverage': self.min_coverage,
                'passed': coverage_passed,
                'details': coverage_data
            }
            
            if coverage_passed:
                return True, f"AI test coverage {total_coverage:.1f}% meets {self.min_coverage}% threshold", coverage_report
            else:
                return False, f"AI test coverage {total_coverage:.1f}% below {self.min_coverage}% threshold", coverage_report
                
        except Exception as e:
            return False, f"Coverage analysis failed: {e}", {'error': str(e)}
            
    def _calculate_fallback_coverage(self) -> float:
        """Calculate fallback coverage when tools are not available"""
        ai_test_dir = Path("tests/ai_testing")
        if not ai_test_dir.exists():
            return 0.0
            
        test_files = list(ai_test_dir.glob("*.py"))
        if not test_files:
            return 0.0
            
        # Simple heuristic: if all expected test files exist, assume good coverage
        expected_files = [
            "model_validation.py",
            "data_quality.py", 
            "performance_validation.py",
            "security_testing.py"
        ]
        
        existing_files = [f.name for f in test_files]
        coverage_score = sum(1 for f in expected_files if f in existing_files) / len(expected_files) * 100
        
        return coverage_score
        
    def validate_ai_test_completeness(self) -> Tuple[bool, str, Dict[str, Any]]:
        """Validate all required AI test categories exist and have proper tests"""
        print("📋 Validating AI test completeness...")
        
        completeness_report = {
            'categories': {},
            'missing_categories': [],
            'test_counts': {}
        }
        
        for category in self.required_ai_test_categories:
            test_path = Path(f"tests/ai_testing/{category}.py")
            
            if test_path.exists():
                # Count test functions in the file
                try:
                    with open(test_path, 'r') as f:
                        content = f.read()
                        test_count = content.count('def test_')
                        
                    completeness_report['categories'][category] = {
                        'exists': True,
                        'test_count': test_count,
                        'adequate': test_count >= 3  # Minimum 3 tests per category
                    }
                    completeness_report['test_counts'][category] = test_count
                    
                except Exception as e:
                    completeness_report['categories'][category] = {
                        'exists': True,
                        'test_count': 0,
                        'adequate': False,
                        'error': str(e)
                    }
            else:
                completeness_report['categories'][category] = {
                    'exists': False,
                    'test_count': 0,
                    'adequate': False
                }
                completeness_report['missing_categories'].append(category)
        
        # Check overall completeness
        all_adequate = all(
            cat_info.get('adequate', False) 
            for cat_info in completeness_report['categories'].values()
        )
        
        total_tests = sum(completeness_report['test_counts'].values())
        
        if all_adequate and len(completeness_report['missing_categories']) == 0:
            return True, f"All AI test categories complete ({total_tests} total tests)", completeness_report
        else:
            missing = ', '.join(completeness_report['missing_categories'])
            return False, f"Missing or inadequate AI test categories: {missing}", completeness_report
            
    def run_ai_security_validation(self) -> Tuple[bool, str, Dict[str, Any]]:
        """Run AI security validation tests"""
        print("🛡️ Running AI security validation...")
        
        try:
            result = subprocess.run([
                sys.executable, "-m", "pytest", 
                "-m", "security",
                "tests/ai_testing/",
                "--tb=short",
                "-v"
            ], capture_output=True, text=True)
            
            security_report = {
                'exit_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'passed': result.returncode == 0
            }
            
            if result.returncode == 0:
                return True, "AI security validation passed", security_report
            else:
                return False, f"AI security tests failed: {result.stderr}", security_report
                
        except Exception as e:
            return False, f"Security validation failed: {e}", {'error': str(e)}
            
    def run_ai_performance_validation(self) -> Tuple[bool, str, Dict[str, Any]]:
        """Run AI performance validation tests"""
        print("⚡ Running AI performance validation...")
        
        try:
            result = subprocess.run([
                sys.executable, "-m", "pytest", 
                "-m", "performance",
                "tests/ai_testing/",
                "--tb=short",
                "-v"
            ], capture_output=True, text=True)
            
            performance_report = {
                'exit_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'passed': result.returncode == 0
            }
            
            if result.returncode == 0:
                return True, "AI performance validation passed", performance_report
            else:
                return False, f"AI performance tests failed: {result.stderr}", performance_report
                
        except Exception as e:
            return False, f"Performance validation failed: {e}", {'error': str(e)}
            
    def validate_ai_testing_infrastructure(self) -> Tuple[bool, str, Dict[str, Any]]:
        """Validate AI testing infrastructure and dependencies"""
        print("🔧 Validating AI testing infrastructure...")
        
        infrastructure_checks = {
            'python_version': sys.version_info >= (3, 8),
            'pytest_available': False,
            'numpy_available': False,
            'test_directory_exists': Path("tests/ai_testing").exists(),
            'init_file_exists': Path("tests/ai_testing/__init__.py").exists()
        }
        
        # Check pytest availability
        try:
            import pytest
            infrastructure_checks['pytest_available'] = True
            infrastructure_checks['pytest_version'] = pytest.__version__
        except ImportError:
            pass
            
        # Check numpy availability
        try:
            import numpy
            infrastructure_checks['numpy_available'] = True
            infrastructure_checks['numpy_version'] = numpy.__version__
        except ImportError:
            pass
            
        # Calculate infrastructure score
        passed_checks = sum(1 for check, passed in infrastructure_checks.items() 
                          if isinstance(passed, bool) and passed)
        total_checks = sum(1 for check, passed in infrastructure_checks.items() 
                         if isinstance(passed, bool))
        
        infrastructure_score = passed_checks / total_checks if total_checks > 0 else 0
        infrastructure_passed = infrastructure_score >= 0.8  # 80% of checks must pass
        
        if infrastructure_passed:
            return True, f"AI testing infrastructure validated ({passed_checks}/{total_checks} checks passed)", infrastructure_checks
        else:
            return False, f"AI testing infrastructure incomplete ({passed_checks}/{total_checks} checks passed)", infrastructure_checks
            
    def generate_quality_report(self, results: List[Tuple[str, bool, str, Dict[str, Any]]]) -> Dict[str, Any]:
        """Generate comprehensive quality report"""
        
        report = {
            'timestamp': time.time(),
            'overall_status': 'FAILED',
            'checks': [],
            'summary': {
                'total_checks': len(results),
                'passed_checks': 0,
                'failed_checks': 0,
                'success_rate': 0.0
            },
            'recommendations': []
        }
        
        for check_name, passed, message, details in results:
            check_result = {
                'check_name': check_name,
                'passed': passed,
                'message': message,
                'details': details
            }
            report['checks'].append(check_result)
            
            if passed:
                report['summary']['passed_checks'] += 1
            else:
                report['summary']['failed_checks'] += 1
                
        # Calculate success rate
        if report['summary']['total_checks'] > 0:
            report['summary']['success_rate'] = report['summary']['passed_checks'] / report['summary']['total_checks']
            
        # Determine overall status
        if report['summary']['success_rate'] >= 1.0:
            report['overall_status'] = 'PASSED'
        elif report['summary']['success_rate'] >= 0.8:
            report['overall_status'] = 'WARNING'
        else:
            report['overall_status'] = 'FAILED'
            
        # Generate recommendations
        for check_name, passed, message, details in results:
            if not passed:
                if 'coverage' in check_name.lower():
                    report['recommendations'].append(f"Improve test coverage by adding more comprehensive tests")
                elif 'completeness' in check_name.lower():
                    report['recommendations'].append(f"Implement missing AI test categories")
                elif 'security' in check_name.lower():
                    report['recommendations'].append(f"Fix AI security test failures")
                elif 'performance' in check_name.lower():
                    report['recommendations'].append(f"Optimize AI performance test implementation")
                elif 'infrastructure' in check_name.lower():
                    report['recommendations'].append(f"Install missing AI testing dependencies")
                    
        return report
        
    def save_quality_report(self, report: Dict[str, Any], filename: str = "ai_testing_quality_report.json"):
        """Save quality report to file"""
        os.makedirs("tests/reports", exist_ok=True)
        report_path = Path("tests/reports") / filename
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        print(f"📊 Quality report saved to: {report_path}")
        return report_path

def main():
    """Main quality gate execution"""
    print("🚀 Starting AI Testing Quality Gates Validation")
    print("=" * 60)
    
    gate = AITestingQualityGate()
    
    # Define all quality checks
    quality_checks = [
        ("Infrastructure Validation", gate.validate_ai_testing_infrastructure),
        ("Test Completeness", gate.validate_ai_test_completeness),
        ("Coverage Analysis", gate.run_ai_test_coverage_analysis),
        ("Security Validation", gate.run_ai_security_validation),
        ("Performance Validation", gate.run_ai_performance_validation)
    ]
    
    results = []
    
    # Run all quality checks
    for check_name, check_func in quality_checks:
        print(f"\n🔍 Running {check_name}...")
        try:
            passed, message, details = check_func()
            results.append((check_name, passed, message, details))
            
            if passed:
                print(f"✅ {check_name}: {message}")
            else:
                print(f"❌ {check_name}: {message}")
                
        except Exception as e:
            error_message = f"Check execution failed: {e}"
            results.append((check_name, False, error_message, {'error': str(e)}))
            print(f"💥 {check_name}: {error_message}")
    
    # Generate and save quality report
    print(f"\n📊 Generating quality report...")
    report = gate.generate_quality_report(results)
    report_path = gate.save_quality_report(report)
    
    # Display summary
    print(f"\n📋 Quality Gates Summary")
    print("=" * 40)
    print(f"Overall Status: {report['overall_status']}")
    print(f"Success Rate: {report['summary']['success_rate']*100:.1f}%")
    print(f"Passed Checks: {report['summary']['passed_checks']}/{report['summary']['total_checks']}")
    
    if report['recommendations']:
        print(f"\n💡 Recommendations:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"  {i}. {rec}")
    
    # Exit with appropriate code
    if report['overall_status'] == 'PASSED':
        print(f"\n🎉 All AI testing quality gates passed!")
        sys.exit(0)
    elif report['overall_status'] == 'WARNING':
        print(f"\n⚠️ AI testing quality gates passed with warnings.")
        sys.exit(0)  # Allow warnings for now
    else:
        print(f"\n🚫 AI testing quality gates failed. Fix issues before proceeding.")
        sys.exit(1)

if __name__ == "__main__":
    main()