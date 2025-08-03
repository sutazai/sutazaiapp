#!/usr/bin/env python3
"""
Purpose: Intelligent rule control manager for codebase hygiene enforcement
Usage: python rule-control-manager.py [--port PORT]
Requirements: fastapi, uvicorn, pydantic, aiofiles
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set
from enum import Enum
import asyncio
import logging

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path("/opt/sutazaiapp")
CONFIG_DIR = PROJECT_ROOT / "config"
RULES_CONFIG_FILE = CONFIG_DIR / "hygiene-rules.json"
PROFILES_CONFIG_FILE = CONFIG_DIR / "rule-profiles.json"
RULE_STATE_FILE = CONFIG_DIR / "rule-states.json"

class RuleSeverity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class RuleCategory(str, Enum):
    STRUCTURE = "structure"
    CODE_QUALITY = "code_quality"
    DOCUMENTATION = "documentation"
    SECURITY = "security"
    PERFORMANCE = "performance"
    TESTING = "testing"
    DEPLOYMENT = "deployment"

class RuleProfile(str, Enum):
    STRICT = "strict"
    MODERATE = "moderate"
    MINIMAL = "minimal"
    CUSTOM = "custom"

class Rule(BaseModel):
    id: str
    name: str
    description: str
    category: RuleCategory
    severity: RuleSeverity
    enabled: bool = True
    dependencies: List[str] = Field(default_factory=list)
    impact: str
    recommendation: str
    auto_fix_available: bool = False
    metadata: Dict = Field(default_factory=dict)

class RuleToggleRequest(BaseModel):
    rule_id: str
    enabled: bool
    force: bool = False

class BulkRuleToggleRequest(BaseModel):
    rule_ids: List[str]
    enabled: bool
    force: bool = False

class ProfileUpdateRequest(BaseModel):
    profile: RuleProfile
    custom_rules: Optional[Dict[str, bool]] = None

class RuleImpactAnalysis(BaseModel):
    affected_rules: List[str]
    warnings: List[str]
    recommendations: List[str]
    severity_impact: Dict[str, int]

class RuleControlManager:
    """Manages rule states, profiles, and dependencies"""
    
    def __init__(self):
        self.rules: Dict[str, Rule] = {}
        self.profiles: Dict[str, Dict[str, bool]] = {}
        self.current_profile: RuleProfile = RuleProfile.MODERATE
        self.rule_states: Dict[str, bool] = {}
        self._load_configurations()
        
    def _load_configurations(self):
        """Load rule definitions and profiles"""
        # Load default rules
        self._load_default_rules()
        
        # Load profiles
        if PROFILES_CONFIG_FILE.exists():
            with open(PROFILES_CONFIG_FILE) as f:
                profile_data = json.load(f)
                self.profiles = profile_data.get("profiles", {})
                self.current_profile = RuleProfile(profile_data.get("current", "moderate"))
        else:
            self._create_default_profiles()
            
        # Load saved rule states
        if RULE_STATE_FILE.exists():
            with open(RULE_STATE_FILE) as f:
                saved_states = json.load(f)
                self.rule_states = saved_states.get("states", {})
                # Apply saved states to rules
                for rule_id, enabled in self.rule_states.items():
                    if rule_id in self.rules:
                        self.rules[rule_id].enabled = enabled
    
    def _load_default_rules(self):
        """Load all hygiene rules from CLAUDE.md instructions"""
        default_rules = [
            Rule(
                id="no_fantasy_elements",
                name="No Fantasy Elements",
                description="Only real, production-ready implementations allowed",
                category=RuleCategory.CODE_QUALITY,
                severity=RuleSeverity.CRITICAL,
                impact="Prevents speculative or placeholder code",
                recommendation="Use concrete implementations with real libraries",
                auto_fix_available=True
            ),
            Rule(
                id="no_breaking_changes",
                name="Do Not Break Existing Functionality",
                description="Every change must respect what already works",
                category=RuleCategory.CODE_QUALITY,
                severity=RuleSeverity.CRITICAL,
                dependencies=["functionality_first_cleanup"],
                impact="Ensures system stability and prevents regressions",
                recommendation="Test thoroughly before making changes"
            ),
            Rule(
                id="analyze_everything",
                name="Analyze Everything—Every Time",
                description="Conduct thorough review before proceeding",
                category=RuleCategory.CODE_QUALITY,
                severity=RuleSeverity.HIGH,
                impact="Ensures comprehensive understanding of codebase",
                recommendation="Use automated analysis tools"
            ),
            Rule(
                id="reuse_before_creating",
                name="Reuse Before Creating",
                description="Always check for and reuse existing scripts",
                category=RuleCategory.STRUCTURE,
                severity=RuleSeverity.MEDIUM,
                dependencies=["eliminate_script_chaos"],
                impact="Reduces code duplication",
                recommendation="Search codebase before creating new files"
            ),
            Rule(
                id="professional_project",
                name="Professional Project Standards",
                description="Treat as professional project, not playground",
                category=RuleCategory.CODE_QUALITY,
                severity=RuleSeverity.HIGH,
                impact="Maintains professional code standards",
                recommendation="Follow best practices consistently"
            ),
            Rule(
                id="centralized_documentation",
                name="Clear, Centralized Documentation",
                description="Documentation must be organized and consistent",
                category=RuleCategory.DOCUMENTATION,
                severity=RuleSeverity.HIGH,
                impact="Improves project maintainability",
                recommendation="Use /docs folder structure",
                auto_fix_available=True
            ),
            Rule(
                id="eliminate_script_chaos",
                name="Eliminate Script Chaos",
                description="Scripts must be centralized and documented",
                category=RuleCategory.STRUCTURE,
                severity=RuleSeverity.HIGH,
                impact="Reduces script sprawl and confusion",
                recommendation="Use /scripts folder with clear structure",
                auto_fix_available=True
            ),
            Rule(
                id="python_script_sanity",
                name="Python Script Sanity",
                description="Python scripts must be purposeful and organized",
                category=RuleCategory.CODE_QUALITY,
                severity=RuleSeverity.MEDIUM,
                impact="Ensures Python code quality",
                recommendation="Include docstrings and proper structure",
                auto_fix_available=True
            ),
            Rule(
                id="no_version_duplication",
                name="No Backend/Frontend Duplication",
                description="Single source of truth for all code",
                category=RuleCategory.STRUCTURE,
                severity=RuleSeverity.HIGH,
                impact="Prevents version conflicts",
                recommendation="Use git branches, not folder copies"
            ),
            Rule(
                id="functionality_first_cleanup",
                name="Functionality-First Cleanup",
                description="Never delete blindly - verify first",
                category=RuleCategory.CODE_QUALITY,
                severity=RuleSeverity.CRITICAL,
                impact="Prevents accidental functionality loss",
                recommendation="Test before removing any code"
            ),
            Rule(
                id="clean_docker_structure",
                name="Clean Docker Structure",
                description="Docker assets must be modular and predictable",
                category=RuleCategory.STRUCTURE,
                severity=RuleSeverity.MEDIUM,
                impact="Improves container maintainability",
                recommendation="Use /docker folder structure"
            ),
            Rule(
                id="single_deployment_script",
                name="Single Deployment Script",
                description="One canonical deployment script for all environments",
                category=RuleCategory.DEPLOYMENT,
                severity=RuleSeverity.HIGH,
                impact="Ensures consistent deployments",
                recommendation="Use deploy.sh as single source"
            ),
            Rule(
                id="no_garbage_no_rot",
                name="No Garbage, No Rot",
                description="Zero tolerance for junk or abandoned code",
                category=RuleCategory.CODE_QUALITY,
                severity=RuleSeverity.HIGH,
                impact="Keeps codebase clean",
                recommendation="Regular cleanup audits",
                auto_fix_available=True
            ),
            Rule(
                id="correct_ai_agent",
                name="Use Correct AI Agent",
                description="Route tasks to specialized AI agents",
                category=RuleCategory.PERFORMANCE,
                severity=RuleSeverity.MEDIUM,
                impact="Optimizes AI usage",
                recommendation="Match agent to task type"
            ),
            Rule(
                id="clean_documentation",
                name="Clean Documentation Standards",
                description="Documentation as critical as code",
                category=RuleCategory.DOCUMENTATION,
                severity=RuleSeverity.MEDIUM,
                dependencies=["centralized_documentation"],
                impact="Ensures documentation quality",
                recommendation="Update docs with code changes",
                auto_fix_available=True
            ),
            Rule(
                id="local_llm_ollama",
                name="Local LLM via Ollama",
                description="All local LLMs must use Ollama framework",
                category=RuleCategory.SECURITY,
                severity=RuleSeverity.LOW,
                impact="Standardizes LLM usage",
                recommendation="Default to TinyLlama"
            )
        ]
        
        for rule in default_rules:
            self.rules[rule.id] = rule
    
    def _create_default_profiles(self):
        """Create default rule profiles"""
        self.profiles = {
            "strict": {rule_id: True for rule_id in self.rules},
            "moderate": {
                rule_id: rule.severity in [RuleSeverity.CRITICAL, RuleSeverity.HIGH]
                for rule_id, rule in self.rules.items()
            },
            "minimal": {
                rule_id: rule.severity == RuleSeverity.CRITICAL
                for rule_id, rule in self.rules.items()
            }
        }
        self._save_profiles()
    
    def _save_profiles(self):
        """Save profiles to disk"""
        CONFIG_DIR.mkdir(exist_ok=True)
        with open(PROFILES_CONFIG_FILE, 'w') as f:
            json.dump({
                "profiles": self.profiles,
                "current": self.current_profile.value
            }, f, indent=2)
    
    def _save_rule_states(self):
        """Save current rule states to disk"""
        states = {rule_id: rule.enabled for rule_id, rule in self.rules.items()}
        with open(RULE_STATE_FILE, 'w') as f:
            json.dump({"states": states, "timestamp": datetime.now().isoformat()}, f, indent=2)
    
    def get_all_rules(self) -> List[Rule]:
        """Get all rules with current states"""
        return list(self.rules.values())
    
    def toggle_rule(self, rule_id: str, enabled: bool, force: bool = False) -> Rule:
        """Toggle a specific rule"""
        if rule_id not in self.rules:
            raise ValueError(f"Rule {rule_id} not found")
        
        rule = self.rules[rule_id]
        
        # Check dependencies if disabling
        if not enabled and not force:
            dependent_rules = self._get_dependent_rules(rule_id)
            if dependent_rules:
                enabled_deps = [r for r in dependent_rules if self.rules[r].enabled]
                if enabled_deps:
                    raise ValueError(
                        f"Cannot disable {rule_id}: required by {', '.join(enabled_deps)}"
                    )
        
        rule.enabled = enabled
        self._save_rule_states()
        
        # Update current profile to custom if manually toggling
        if self.current_profile != RuleProfile.CUSTOM:
            self.current_profile = RuleProfile.CUSTOM
            self._save_profiles()
        
        return rule
    
    def _get_dependent_rules(self, rule_id: str) -> List[str]:
        """Get rules that depend on the given rule"""
        dependents = []
        for rid, rule in self.rules.items():
            if rule_id in rule.dependencies:
                dependents.append(rid)
        return dependents
    
    def toggle_bulk(self, rule_ids: List[str], enabled: bool, force: bool = False) -> List[Rule]:
        """Toggle multiple rules at once"""
        updated_rules = []
        
        # If disabling, check all dependencies first
        if not enabled and not force:
            for rule_id in rule_ids:
                dependent_rules = self._get_dependent_rules(rule_id)
                enabled_deps = [
                    r for r in dependent_rules 
                    if self.rules[r].enabled and r not in rule_ids
                ]
                if enabled_deps:
                    raise ValueError(
                        f"Cannot disable {rule_id}: required by {', '.join(enabled_deps)}"
                    )
        
        # Toggle all rules
        for rule_id in rule_ids:
            if rule_id in self.rules:
                rule = self.toggle_rule(rule_id, enabled, force=True)
                updated_rules.append(rule)
        
        return updated_rules
    
    def apply_profile(self, profile: RuleProfile, custom_rules: Optional[Dict[str, bool]] = None):
        """Apply a rule profile"""
        if profile == RuleProfile.CUSTOM and custom_rules:
            # Apply custom rules
            for rule_id, enabled in custom_rules.items():
                if rule_id in self.rules:
                    self.rules[rule_id].enabled = enabled
            self.profiles["custom"] = custom_rules
        else:
            # Apply predefined profile
            if profile.value in self.profiles:
                for rule_id, enabled in self.profiles[profile.value].items():
                    if rule_id in self.rules:
                        self.rules[rule_id].enabled = enabled
        
        self.current_profile = profile
        self._save_profiles()
        self._save_rule_states()
    
    def analyze_impact(self, rule_ids: List[str], action: str = "disable") -> RuleImpactAnalysis:
        """Analyze the impact of enabling/disabling rules"""
        warnings = []
        recommendations = []
        affected_rules = []
        severity_impact = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        
        for rule_id in rule_ids:
            if rule_id not in self.rules:
                continue
                
            rule = self.rules[rule_id]
            severity_impact[rule.severity.value] += 1
            
            if action == "disable":
                # Check dependent rules
                dependents = self._get_dependent_rules(rule_id)
                if dependents:
                    affected_rules.extend(dependents)
                    warnings.append(
                        f"Disabling '{rule.name}' affects: {', '.join(dependents)}"
                    )
                
                # Add severity-based warnings
                if rule.severity == RuleSeverity.CRITICAL:
                    warnings.append(
                        f"'{rule.name}' is CRITICAL - disabling may cause serious issues"
                    )
                elif rule.severity == RuleSeverity.HIGH:
                    warnings.append(
                        f"'{rule.name}' is HIGH priority - consider the impact"
                    )
                
                # Add recommendations
                if rule.auto_fix_available:
                    recommendations.append(
                        f"Consider using auto-fix for '{rule.name}' instead of disabling"
                    )
            
            elif action == "enable":
                # Check dependencies that need to be enabled
                for dep_id in rule.dependencies:
                    if dep_id in self.rules and not self.rules[dep_id].enabled:
                        affected_rules.append(dep_id)
                        recommendations.append(
                            f"Enable '{self.rules[dep_id].name}' for '{rule.name}' to work properly"
                        )
        
        return RuleImpactAnalysis(
            affected_rules=list(set(affected_rules)),
            warnings=warnings,
            recommendations=recommendations,
            severity_impact=severity_impact
        )
    
    def get_recommendations(self, project_type: Optional[str] = None) -> Dict[str, List[str]]:
        """Get rule recommendations based on project type"""
        recommendations = {
            "enable": [],
            "disable": [],
            "configure": []
        }
        
        if project_type == "microservices":
            recommendations["enable"].extend([
                "clean_docker_structure",
                "single_deployment_script",
                "no_version_duplication"
            ])
            recommendations["configure"].append("Consider stricter API versioning rules")
            
        elif project_type == "monolith":
            recommendations["enable"].extend([
                "eliminate_script_chaos",
                "python_script_sanity",
                "centralized_documentation"
            ])
            recommendations["configure"].append("Focus on module separation rules")
            
        elif project_type == "ai_ml":
            recommendations["enable"].extend([
                "correct_ai_agent",
                "local_llm_ollama",
                "python_script_sanity"
            ])
            recommendations["disable"].append("clean_docker_structure")
            recommendations["configure"].append("Add data versioning rules")
        
        # General recommendations based on current state
        critical_disabled = [
            rule.id for rule in self.rules.values() 
            if rule.severity == RuleSeverity.CRITICAL and not rule.enabled
        ]
        if critical_disabled:
            recommendations["enable"].extend(critical_disabled)
        
        return recommendations

# Create FastAPI app
app = FastAPI(title="Rule Control API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize manager
manager = RuleControlManager()

@app.get("/api/rules")
async def get_all_rules():
    """Get all rules with current states"""
    rules = manager.get_all_rules()
    return {
        "rules": [rule.dict() for rule in rules],
        "current_profile": manager.current_profile.value,
        "total": len(rules),
        "enabled": sum(1 for r in rules if r.enabled),
        "disabled": sum(1 for r in rules if not r.enabled)
    }

@app.put("/api/rules/{rule_id}/toggle")
async def toggle_rule(rule_id: str, request: RuleToggleRequest):
    """Toggle a specific rule"""
    try:
        rule = manager.toggle_rule(rule_id, request.enabled, request.force)
        return {"success": True, "rule": rule.dict()}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.put("/api/rules/bulk")
async def toggle_bulk_rules(request: BulkRuleToggleRequest):
    """Enable/disable multiple rules"""
    try:
        updated_rules = manager.toggle_bulk(
            request.rule_ids, 
            request.enabled, 
            request.force
        )
        return {
            "success": True,
            "updated": len(updated_rules),
            "rules": [rule.dict() for rule in updated_rules]
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/profiles")
async def get_profiles():
    """Get available rule profiles"""
    return {
        "profiles": list(manager.profiles.keys()),
        "current": manager.current_profile.value,
        "definitions": manager.profiles
    }

@app.put("/api/profiles")
async def update_profile(request: ProfileUpdateRequest):
    """Apply a rule profile"""
    try:
        manager.apply_profile(request.profile, request.custom_rules)
        return {
            "success": True,
            "profile": request.profile.value,
            "rules_updated": len(manager.rules)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/rules/analyze")
async def analyze_impact(
    rule_ids: List[str] = Body(...),
    action: str = Body("disable")
):
    """Analyze impact of rule changes"""
    analysis = manager.analyze_impact(rule_ids, action)
    return analysis.dict()

@app.get("/api/rules/recommendations")
async def get_recommendations(project_type: Optional[str] = None):
    """Get rule recommendations"""
    return manager.get_recommendations(project_type)

@app.post("/api/rules/enable-all")
async def enable_all_rules():
    """Enable all rules"""
    rule_ids = list(manager.rules.keys())
    updated = manager.toggle_bulk(rule_ids, True, force=True)
    return {
        "success": True,
        "enabled": len(updated)
    }

@app.post("/api/rules/disable-all")
async def disable_all_rules():
    """Disable all rules (requires force)"""
    rule_ids = list(manager.rules.keys())
    try:
        updated = manager.toggle_bulk(rule_ids, False, force=True)
        return {
            "success": True,
            "disabled": len(updated),
            "warning": "All rules disabled - codebase hygiene enforcement is OFF"
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Rule Control Manager API")
    parser.add_argument("--port", type=int, default=8100, help="Port to run on")
    args = parser.parse_args()
    
    uvicorn.run(app, host="0.0.0.0", port=args.port)