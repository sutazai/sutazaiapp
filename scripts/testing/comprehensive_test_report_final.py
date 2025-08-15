#!/usr/bin/env python3
"""
SutazAI Final Comprehensive Test Report Generator
===============================================

This script generates a final comprehensive test report combining results from:
- AI-powered comprehensive testing
- Specialized testing
- Performance testing
- Security testing

Provides executive summary and detailed analysis with recommendations.
"""

import json
import asyncio
from datetime import datetime
from pathlib import Path
import logging
from typing import Dict, Any, List
import glob

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComprehensiveTestReportGenerator:
    """Generate final comprehensive test report"""
    
    def __init__(self):
        self.reports_dir = Path("/opt/sutazaiapp/data/workflow_reports")
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
    def load_latest_reports(self) -> Dict[str, Any]:
        """Load the latest test reports from all test suites"""
        reports = {}
        
        # Find latest reports of each type
        report_types = [
            ("comprehensive", "comprehensive_test_report_*.json"),
            ("specialized", "specialized_test_report_*.json"),
            ("performance", "performance_test_report_*.json"),
            ("security", "security_test_report_*.json")
        ]
        
        for report_type, pattern in report_types:
            files = glob.glob(str(self.reports_dir / pattern))
            if files:
                latest_file = max(files, key=lambda x: Path(x).stat().st_mtime)
                try:
                    with open(latest_file, 'r') as f:
                        reports[report_type] = json.load(f)
                    logger.info(f"Loaded {report_type} report: {latest_file}")
                except Exception as e:
                    logger.error(f"Failed to load {report_type} report: {e}")
                    reports[report_type] = {"error": str(e)}
            else:
                logger.warning(f"No {report_type} reports found")
                reports[report_type] = {"error": "No reports found"}
        
        return reports
    
    def generate_executive_summary(self, reports: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary from all test results"""
        
        summary = {
            "report_timestamp": datetime.now().isoformat(),
            "system_name": "SutazAI Task Automation System",
            "version": "1.0.0",
            "test_execution_summary": {}
        }
        
        # Extract key metrics from each report type
        comprehensive = reports.get("comprehensive", {})
        specialized = reports.get("specialized", {})
        performance = reports.get("performance", {})
        security = reports.get("security", {})
        
        # Comprehensive test results
        if "test_execution_summary" in comprehensive:
            comp_summary = comprehensive["test_execution_summary"]
            summary["test_execution_summary"]["comprehensive"] = {
                "total_tests": comp_summary.get("total_tests", 0),
                "passed_tests": comp_summary.get("passed_tests", 0),
                "success_rate": comp_summary.get("success_rate", 0),
                "status": comp_summary.get("status", "UNKNOWN"),
                "health_score": comprehensive.get("system_health_score", {}).get("score", 0)
            }
        
        # Specialized test results
        if "total_tests" in specialized:
            summary["test_execution_summary"]["specialized"] = {
                "total_tests": specialized.get("total_tests", 0),
                "passed_tests": specialized.get("passed_tests", 0),
                "success_rate": specialized.get("success_rate", 0),
                "status": specialized.get("status", "UNKNOWN")
            }
        
        # Performance test results
        if "performance_summary" in performance:
            perf_summary = performance["performance_summary"]
            summary["test_execution_summary"]["performance"] = {
                "performance_grade": perf_summary.get("performance_grade", "UNKNOWN"),
                "performance_score": perf_summary.get("overall_performance_score", 0),
                "avg_response_time": perf_summary.get("key_metrics", {}).get("avg_response_time", 0),
                "requests_per_second": perf_summary.get("key_metrics", {}).get("requests_per_second", 0)
            }
        
        # Security test results
        if "security_summary" in security:
            sec_summary = security["security_summary"]
            summary["test_execution_summary"]["security"] = {
                "security_grade": sec_summary.get("security_grade", "UNKNOWN"),
                "security_score": sec_summary.get("security_score", 0),
                "security_status": sec_summary.get("overall_status", "UNKNOWN"),
                "issues_count": len(sec_summary.get("security_issues", []))
            }
        
        return summary
    
    def calculate_overall_system_score(self, reports: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall system score based on all test results"""
        
        # Weight factors for different test categories
        weights = {
            "functionality": 0.35,  # Comprehensive + Specialized tests
            "performance": 0.25,
            "security": 0.25,
            "reliability": 0.15    # Error handling, edge cases
        }
        
        scores = {}
        
        # Functionality score (from comprehensive tests)
        comprehensive = reports.get("comprehensive", {})
        if "test_execution_summary" in comprehensive:
            functionality_score = comprehensive["test_execution_summary"].get("success_rate", 0)
            scores["functionality"] = functionality_score
        else:
            scores["functionality"] = 0
        
        # Performance score
        performance = reports.get("performance", {})
        if "performance_summary" in performance:
            perf_score = performance["performance_summary"].get("overall_performance_score", 0)
            scores["performance"] = perf_score
        else:
            scores["performance"] = 0
        
        # Security score
        security = reports.get("security", {})
        if "security_summary" in security:
            sec_score = security["security_summary"].get("security_score", 0)
            scores["security"] = sec_score
        else:
            scores["security"] = 0
        
        # Reliability score (from specialized tests and error handling)
        specialized = reports.get("specialized", {})
        if "success_rate" in specialized:
            reliability_score = specialized.get("success_rate", 0)
            scores["reliability"] = reliability_score
        else:
            scores["reliability"] = 0
        
        # Calculate weighted overall score
        overall_score = sum(scores[category] * weights[category] for category in weights.keys())
        
        # Determine overall grade
        if overall_score >= 90:
            overall_grade = "EXCELLENT"
            status = "PRODUCTION_READY"
        elif overall_score >= 80:
            overall_grade = "GOOD"
            status = "PRODUCTION_READY"
        elif overall_score >= 70:
            overall_grade = "FAIR"
            status = "NEEDS_IMPROVEMENT"
        elif overall_score >= 60:
            overall_grade = "POOR"
            status = "MAJOR_ISSUES"
        else:
            overall_grade = "CRITICAL"
            status = "NOT_PRODUCTION_READY"
        
        return {
            "overall_score": overall_score,
            "overall_grade": overall_grade,
            "system_status": status,
            "category_scores": scores,
            "weights": weights,
            "score_breakdown": {
                category: scores[category] * weights[category]
                for category in weights.keys()
            }
        }
    
    def generate_recommendations(self, reports: Dict[str, Any], overall_assessment: Dict[str, Any]) -> List[str]:
        """Generate comprehensive recommendations based on all test results"""
        
        recommendations = []
        
        # High priority recommendations based on overall score
        if overall_assessment["overall_score"] < 80:
            recommendations.append("CRITICAL: System requires significant improvements before production deployment")
        
        # Security recommendations (highest priority)
        security = reports.get("security", {})
        if "security_summary" in security:
            sec_issues = security["security_summary"].get("security_issues", [])
            if sec_issues:
                recommendations.append("SECURITY: Address all security vulnerabilities immediately")
                recommendations.extend([f"  - {issue}" for issue in sec_issues[:3]])  # Top 3 issues
        
        # Performance recommendations
        performance = reports.get("performance", {})
        if "performance_summary" in performance:
            perf_issues = performance["performance_summary"].get("performance_issues", [])
            if perf_issues:
                recommendations.append("PERFORMANCE: Optimize system performance")
                recommendations.extend([f"  - {issue}" for issue in perf_issues[:2]])  # Top 2 issues
        
        # Functional recommendations
        comprehensive = reports.get("comprehensive", {})
        if "recommendations" in comprehensive:
            func_recommendations = comprehensive.get("recommendations", [])
            if func_recommendations:
                recommendations.append("FUNCTIONALITY: Address functional issues")
                recommendations.extend([f"  - {rec}" for rec in func_recommendations[:2]])
        
        # Infrastructure recommendations
        if overall_assessment["category_scores"].get("reliability", 0) < 80:
            recommendations.append("RELIABILITY: Improve system reliability and error handling")
        
        # Deployment recommendations
        if overall_assessment["system_status"] == "PRODUCTION_READY":
            recommendations.append("✅ System is ready for production deployment")
            recommendations.append("  - Consider implementing monitoring and alerting")
            recommendations.append("  - Set up automated backups")
            recommendations.append("  - Configure load balancing for high availability")
        elif overall_assessment["system_status"] == "NEEDS_IMPROVEMENT":
            recommendations.append("⚠️ Address identified issues before production deployment")
        else:
            recommendations.append("❌ System is NOT ready for production deployment")
        
        return recommendations
    
    def generate_detailed_analysis(self, reports: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed analysis of all test results"""
        
        analysis = {
            "test_coverage": {},
            "failure_analysis": {},
            "performance_analysis": {},
            "security_analysis": {},
            "component_health": {}
        }
        
        # Test coverage analysis
        comprehensive = reports.get("comprehensive", {})
        if "test_categories" in comprehensive:
            categories = comprehensive["test_categories"]
            analysis["test_coverage"] = {
                "total_categories": len(categories),
                "passed_categories": sum(1 for cat in categories.values() if cat.get("status") == "PASSED"),
                "coverage_percentage": (sum(1 for cat in categories.values() if cat.get("status") == "PASSED") / len(categories)) * 100 if categories else 0,
                "category_details": categories
            }
        
        # Failure analysis
        failed_tests = []
        
        # Collect failed tests from comprehensive suite
        if "detailed_results" in comprehensive:
            for category, results in comprehensive["detailed_results"].items():
                if isinstance(results, list):
                    for result in results:
                        if hasattr(result, 'status') or (isinstance(result, str) and 'failed' in result.lower()):
                            # Parse the result string or object
                            if 'status=\'failed\'' in str(result) or 'status=\'error\'' in str(result):
                                failed_tests.append({
                                    "category": category,
                                    "test": str(result)[:100] + "..." if len(str(result)) > 100 else str(result)
                                })
        
        analysis["failure_analysis"] = {
            "total_failures": len(failed_tests),
            "failure_details": failed_tests[:10]  # Top 10 failures
        }
        
        # Performance analysis
        performance = reports.get("performance", {})
        if "basic_load_test" in performance:
            load_test = performance["basic_load_test"]
            analysis["performance_analysis"] = {
                "load_test_results": {
                    "concurrent_users": load_test.get("concurrent_users", 0),
                    "success_rate": load_test.get("success_rate", 0),
                    "avg_response_time": load_test.get("avg_response_time", 0),
                    "requests_per_second": load_test.get("requests_per_second", 0)
                },
                "performance_grade": performance.get("performance_summary", {}).get("performance_grade", "UNKNOWN")
            }
        
        # Security analysis
        security = reports.get("security", {})
        if "security_summary" in security:
            sec_summary = security["security_summary"]
            analysis["security_analysis"] = {
                "security_score": sec_summary.get("security_score", 0),
                "vulnerabilities": sec_summary.get("security_issues", []),
                "checks_passed": sec_summary.get("passed_checks", 0),
                "total_checks": sec_summary.get("total_checks", 0)
            }
        
        # Component health analysis
        if "detailed_results" in comprehensive:
            component_health = {}
            for component, results in comprehensive["detailed_results"].items():
                if isinstance(results, list):
                    total = len(results)
                    passed = sum(1 for r in results if 'status=\'passed\'' in str(r))
                    component_health[component] = {
                        "total_tests": total,
                        "passed_tests": passed,
                        "health_percentage": (passed / total * 100) if total > 0 else 0
                    }
            
            analysis["component_health"] = component_health
        
        return analysis
    
    async def generate_final_report(self) -> Dict[str, Any]:
        """Generate the final comprehensive test report"""
        logger.info("Generating final comprehensive test report...")
        
        # Load all test reports
        reports = self.load_latest_reports()
        
        # Generate executive summary
        executive_summary = self.generate_executive_summary(reports)
        
        # Calculate overall system score
        overall_assessment = self.calculate_overall_system_score(reports)
        
        # Generate recommendations
        recommendations = self.generate_recommendations(reports, overall_assessment)
        
        # Generate detailed analysis
        detailed_analysis = self.generate_detailed_analysis(reports)
        
        # Compile final report
        final_report = {
            "report_metadata": {
                "report_type": "Final Comprehensive Testing Report",
                "generated_timestamp": datetime.now().isoformat(),
                "system_under_test": "SutazAI Task Automation System",
                "version": "1.0.0",
                "test_execution_period": "Single comprehensive execution",
                "report_generator": "AI-Powered Testing QA Validator"
            },
            "executive_summary": executive_summary,
            "overall_assessment": overall_assessment,
            "recommendations": recommendations,
            "detailed_analysis": detailed_analysis,
            "source_reports": {
                "comprehensive_tests": bool(reports.get("comprehensive")),
                "specialized_tests": bool(reports.get("specialized")),
                "performance_tests": bool(reports.get("performance")),
                "security_tests": bool(reports.get("security"))
            },
            "raw_reports": reports  # Include raw data for reference
        }
        
        # Save final report
        await self._save_final_report(final_report)
        
        return final_report
    
    async def _save_final_report(self, report: Dict[str, Any]) -> None:
        """Save the final comprehensive report"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save JSON report
            json_file = self.reports_dir / f"FINAL_COMPREHENSIVE_TEST_REPORT_{timestamp}.json"
            with open(json_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            # Generate and save markdown report
            md_file = self.reports_dir / f"FINAL_COMPREHENSIVE_TEST_REPORT_{timestamp}.md"
            await self._generate_markdown_report(report, md_file)
            
            logger.info(f"Final comprehensive test report saved:")
            logger.info(f"  JSON: {json_file}")
            logger.info(f"  Markdown: {md_file}")
            
        except Exception as e:
            logger.error(f"Failed to save final report: {e}")
    
    async def _generate_markdown_report(self, report: Dict[str, Any], file_path: Path) -> None:
        """Generate markdown version of the final report"""
        
        overall = report["overall_assessment"]
        exec_summary = report["executive_summary"]
        
        markdown_content = f"""# SutazAI Task Automation System - Final Comprehensive Test Report

## Executive Summary

**System:** {exec_summary['system_name']} v{exec_summary['version']}  
**Report Generated:** {exec_summary['report_timestamp']}  
**Overall Grade:** {overall['overall_grade']}  
**System Status:** {overall['system_status']}  
**Overall Score:** {overall['overall_score']:.1f}/100

## Test Results Overview

| Test Category | Score | Grade | Status |
|---------------|-------|-------|---------|
| Functionality | {overall['category_scores'].get('functionality', 0):.1f}% | {self._get_grade(overall['category_scores'].get('functionality', 0))} | {'✅' if overall['category_scores'].get('functionality', 0) >= 80 else '❌'} |
| Performance | {overall['category_scores'].get('performance', 0):.1f}% | {self._get_grade(overall['category_scores'].get('performance', 0))} | {'✅' if overall['category_scores'].get('performance', 0) >= 80 else '❌'} |
| Security | {overall['category_scores'].get('security', 0):.1f}% | {self._get_grade(overall['category_scores'].get('security', 0))} | {'✅' if overall['category_scores'].get('security', 0) >= 80 else '❌'} |
| Reliability | {overall['category_scores'].get('reliability', 0):.1f}% | {self._get_grade(overall['category_scores'].get('reliability', 0))} | {'✅' if overall['category_scores'].get('reliability', 0) >= 80 else '❌'} |

## Key Findings

### Comprehensive Functional Testing
"""
        
        if "comprehensive" in exec_summary.get("test_execution_summary", {}):
            comp = exec_summary["test_execution_summary"]["comprehensive"]
            markdown_content += f"""
- **Total Tests:** {comp.get('total_tests', 0)}
- **Success Rate:** {comp.get('success_rate', 0):.1f}%
- **System Health Score:** {comp.get('health_score', 0):.1f}%
- **Status:** {comp.get('status', 'UNKNOWN')}
"""
        
        if "performance" in exec_summary.get("test_execution_summary", {}):
            perf = exec_summary["test_execution_summary"]["performance"]
            markdown_content += f"""
### Performance Testing
- **Performance Grade:** {perf.get('performance_grade', 'UNKNOWN')}
- **Average Response Time:** {perf.get('avg_response_time', 0):.3f} seconds
- **Requests per Second:** {perf.get('requests_per_second', 0):.1f}
"""
        
        if "security" in exec_summary.get("test_execution_summary", {}):
            sec = exec_summary["test_execution_summary"]["security"]
            markdown_content += f"""
### Security Testing
- **Security Grade:** {sec.get('security_grade', 'UNKNOWN')}
- **Security Score:** {sec.get('security_score', 0):.1f}%
- **Security Issues:** {sec.get('issues_count', 0)}
- **Status:** {sec.get('security_status', 'UNKNOWN')}
"""
        
        markdown_content += f"""
## Recommendations

"""
        for i, rec in enumerate(report["recommendations"][:10], 1):  # Top 10 recommendations
            markdown_content += f"{i}. {rec}\n"
        
        markdown_content += f"""
## Detailed Analysis

### Component Health Status
"""
        
        if "component_health" in report["detailed_analysis"]:
            for component, health in report["detailed_analysis"]["component_health"].items():
                status_icon = "✅" if health["health_percentage"] >= 80 else "⚠️" if health["health_percentage"] >= 60 else "❌"
                markdown_content += f"- {status_icon} **{component}:** {health['passed_tests']}/{health['total_tests']} tests passed ({health['health_percentage']:.1f}%)\n"
        
        markdown_content += f"""
## Conclusion

The SutazAI Task Automation System has undergone comprehensive testing across functionality, performance, security, and reliability dimensions. 

**Overall Assessment:** The system achieved a **{overall['overall_grade']}** grade with an overall score of **{overall['overall_score']:.1f}/100**.

**Deployment Recommendation:** {"✅ APPROVED for production deployment" if overall['system_status'] == 'PRODUCTION_READY' else "❌ NOT APPROVED for production deployment" if overall['system_status'] == 'NOT_PRODUCTION_READY' else "⚠️ Conditional approval - address identified issues"}

---

*Report generated by SutazAI AI-Powered Testing QA Validator*  
*Timestamp: {datetime.now().isoformat()}*
"""
        
        with open(file_path, 'w') as f:
            f.write(markdown_content)
    
    def _get_grade(self, score: float) -> str:
        """Convert numeric score to letter grade"""
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"

async def main():
    """Generate and display final comprehensive test report"""
    generator = ComprehensiveTestReportGenerator()
    final_report = await generator.generate_final_report()
    
    logger.info("\n" + "="*100)
    logger.info("SUTAZAI TASK AUTOMATION SYSTEM - FINAL COMPREHENSIVE TEST REPORT")
    logger.info("="*100)
    
    overall = final_report["overall_assessment"]
    logger.info(f"🎯 OVERALL GRADE: {overall['overall_grade']}")
    logger.info(f"📊 OVERALL SCORE: {overall['overall_score']:.1f}/100")
    logger.info(f"🚀 SYSTEM STATUS: {overall['system_status']}")
    
    logger.info(f"\n📋 CATEGORY BREAKDOWN:")
    for category, score in overall["category_scores"].items():
        status_icon = "✅" if score >= 80 else "⚠️" if score >= 60 else "❌"
        logger.info(f"  {status_icon} {category.title()}: {score:.1f}%")
    
    logger.info(f"\n🔍 TOP RECOMMENDATIONS:")
    for i, rec in enumerate(final_report["recommendations"][:5], 1):
        logger.info(f"  {i}. {rec}")
    
    logger.info(f"\n📈 SYSTEM READINESS:")
    if overall['system_status'] == 'PRODUCTION_READY':
        logger.info("  ✅ System is READY for production deployment")
    elif overall['system_status'] == 'NEEDS_IMPROVEMENT':
        logger.info("  ⚠️ System needs improvement before production deployment")
    else:
        logger.info("  ❌ System is NOT ready for production deployment")
    
    logger.info("="*100)
    logger.info(f"📄 Full reports saved in: {generator.reports_dir}")
    logger.info("="*100)
    
    return final_report

if __name__ == "__main__":
    asyncio.run(main())