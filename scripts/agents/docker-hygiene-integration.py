#!/usr/bin/env python3
"""
Purpose: Integration layer between Docker Structure Validator and Hygiene Enforcement
Usage: Called by hygiene-agent-orchestrator.py for Docker-specific validation
Requirements: Python 3.8+

This module provides Docker-specific hygiene checks for the main hygiene system.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add project root to path
sys.path.insert(0, '/opt/sutazaiapp')

from scripts.agents.docker_structure_validator import DockerStructureValidator

logger = logging.getLogger(__name__)

class DockerHygieneAgent:
    """Docker-specific hygiene enforcement agent"""
    
    def __init__(self, project_root: str = "/opt/sutazaiapp"):
        self.project_root = Path(project_root)
        self.validator = DockerStructureValidator(project_root)
        
    def check_hygiene(self) -> Dict[str, Any]:
        """Run Docker hygiene checks"""
        logger.info("🐳 Running Docker hygiene checks...")
        
        # Run validation without auto-fix first
        validation_results = self.validator.run_validation(auto_fix=False)
        
        # Extract hygiene violations
        violations = self._extract_violations(validation_results)
        
        return {
            "agent": "docker",
            "status": "pass" if not violations else "fail",
            "violations": violations,
            "metrics": self._calculate_metrics(validation_results),
            "auto_fix_available": self._check_auto_fix_available(violations)
        }
    
    def _extract_violations(self, validation_results: Dict) -> List[Dict]:
        """Extract hygiene violations from validation results"""
        violations = []
        
        # Structure violations
        structure = validation_results.get("structure_compliance", {})
        for issue in structure.get("issues", []):
            violations.append({
                "type": "structure",
                "severity": issue.get("severity", "medium"),
                "location": issue.get("path", "unknown"),
                "message": issue.get("fix", "Structure issue"),
                "rule": "Rule 11: Docker Structure"
            })
        
        # Dockerfile violations
        for path, analysis in validation_results.get("dockerfile_analysis", {}).items():
            # Security violations
            for issue in analysis.get("security_problems", []):
                violations.append({
                    "type": "security",
                    "severity": "critical",
                    "location": path,
                    "line": issue.get("line", 0),
                    "message": f"Security issue: {issue['type']}",
                    "rule": "Rule 11: Docker Security"
                })
            
            # Standard violations
            for issue in analysis.get("issues", []):
                violations.append({
                    "type": "dockerfile",
                    "severity": issue.get("severity", "medium"),
                    "location": path,
                    "line": issue.get("line", 0),
                    "message": issue.get("message", "Dockerfile issue"),
                    "rule": "Rule 11: Dockerfile Standards"
                })
        
        # Compose violations
        compose = validation_results.get("compose_validation", {})
        for issue in compose.get("issues", []):
            violations.append({
                "type": "compose",
                "severity": issue.get("severity", "medium"),
                "location": issue.get("file", "docker-compose.yml"),
                "service": issue.get("service", ""),
                "message": issue.get("message", "Compose issue"),
                "rule": "Rule 11: Docker Compose"
            })
        
        return violations
    
    def _calculate_metrics(self, validation_results: Dict) -> Dict[str, Any]:
        """Calculate Docker hygiene metrics"""
        report = validation_results.get("compliance_report", {})
        
        return {
            "overall_score": report.get("overall_score", 0),
            "structure_score": report.get("scores", {}).get("structure", 0),
            "dockerfile_score": report.get("scores", {}).get("dockerfiles", 0),
            "compose_score": report.get("scores", {}).get("compose", 0),
            "total_dockerfiles": report.get("summary", {}).get("total_dockerfiles", 0),
            "critical_issues": len(report.get("critical_issues", []))
        }
    
    def _check_auto_fix_available(self, violations: List[Dict]) -> bool:
        """Check if auto-fix is available for violations"""
        auto_fixable_types = {
            "structure", "no_apt_cleanup", "no_workdir", 
            "missing_dockerignore", "no_user_context"
        }
        
        for violation in violations:
            if violation.get("type") in auto_fixable_types:
                return True
                
        return False
    
    def apply_auto_fixes(self) -> Dict[str, Any]:
        """Apply automatic fixes for Docker hygiene violations"""
        logger.info("🔧 Applying Docker hygiene auto-fixes...")
        
        # Run validation with auto-fix enabled
        validation_results = self.validator.run_validation(auto_fix=True)
        
        return {
            "agent": "docker",
            "fixes_applied": validation_results.get("auto_fixes", []),
            "status": "success" if validation_results.get("auto_fixes") else "no_fixes_needed"
        }
    
    def get_recommendations(self) -> List[Dict[str, str]]:
        """Get Docker-specific hygiene recommendations"""
        validation_results = self.validator.run_validation(auto_fix=False)
        report = validation_results.get("compliance_report", {})
        
        recommendations = []
        
        # Add report recommendations
        for rec in report.get("recommendations", []):
            recommendations.append({
                "agent": "docker",
                "category": rec.get("category", "general"),
                "priority": rec.get("priority", "medium"),
                "action": rec.get("action", ""),
                "impact": "high" if rec.get("priority") == "critical" else "medium"
            })
        
        # Add specific recommendations based on violations
        violations = self._extract_violations(validation_results)
        
        if any(v["type"] == "security" for v in violations):
            recommendations.append({
                "agent": "docker",
                "category": "security",
                "priority": "critical",
                "action": "Address security vulnerabilities in Dockerfiles immediately",
                "impact": "critical"
            })
        
        if not (self.project_root / "docker" / ".dockerignore").exists():
            recommendations.append({
                "agent": "docker",
                "category": "optimization",
                "priority": "high",
                "action": "Create .dockerignore to reduce build context size",
                "impact": "high"
            })
        
        return recommendations


# Integration function for hygiene orchestrator
def run_docker_hygiene_check(project_root: str = "/opt/sutazaiapp") -> Dict[str, Any]:
    """Entry point for hygiene orchestrator"""
    agent = DockerHygieneAgent(project_root)
    return agent.check_hygiene()


def apply_docker_hygiene_fixes(project_root: str = "/opt/sutazaiapp") -> Dict[str, Any]:
    """Apply Docker hygiene fixes"""
    agent = DockerHygieneAgent(project_root)
    return agent.apply_auto_fixes()


if __name__ == "__main__":
    # Test the integration
    results = run_docker_hygiene_check()
    print(json.dumps(results, indent=2))