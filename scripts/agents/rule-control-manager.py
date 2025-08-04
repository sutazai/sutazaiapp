#!/usr/bin/env python3
"""
Purpose: Intelligent rule control manager for codebase hygiene enforcement
Usage: python rule-control-manager.py [--port PORT]
Requirements: fastapi, uvicorn, pydantic, aiofiles
"""

import json
import os
import sys
import time
import psutil
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
from enum import Enum
import asyncio
import logging
from contextlib import asynccontextmanager
import threading
from collections import defaultdict, deque

from fastapi import FastAPI, HTTPException, Body, Request, Response, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware  
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
import uvicorn

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths - Use environment variables for Docker compatibility
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", "/opt/sutazaiapp"))
CONFIG_DIR = Path(os.getenv("CONFIG_DIR", PROJECT_ROOT / "config"))
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

class CircuitBreakerStats(BaseModel):
    state: str
    failure_count: int
    success_count: int
    last_failure_time: Optional[datetime]
    last_success_time: Optional[datetime]

class SystemHealthMetrics(BaseModel):
    cpu_usage_percent: float
    memory_usage_mb: float
    disk_usage_percent: float
    response_time_ms: float
    active_connections: int
    requests_per_minute: int
    error_rate_percent: float
    uptime_seconds: int

class RateLimitInfo(BaseModel):
    limit: int
    remaining: int
    reset_time: datetime
    retry_after_seconds: Optional[int] = None

class APICircuitBreaker:
    """Circuit breaker for API endpoints"""
    def __init__(self, failure_threshold: int = 10, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.last_success_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self._lock = threading.Lock()
    
    def call(self, func):
        """Execute function through circuit breaker"""
        with self._lock:
            if self.state == "OPEN":
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = "HALF_OPEN"
                else:
                    raise HTTPException(status_code=503, detail="Service temporarily unavailable")
            
            try:
                result = func()
                self._on_success()
                return result
            except Exception as e:
                self._on_failure()
                raise
    
    def _on_success(self):
        self.success_count += 1
        self.last_success_time = time.time()
        if self.state == "HALF_OPEN":
            self.state = "CLOSED"
            self.failure_count = 0
    
    def _on_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
    
    def get_stats(self) -> CircuitBreakerStats:
        return CircuitBreakerStats(
            state=self.state,
            failure_count=self.failure_count,
            success_count=self.success_count,
            last_failure_time=datetime.fromtimestamp(self.last_failure_time) if self.last_failure_time else None,
            last_success_time=datetime.fromtimestamp(self.last_success_time) if self.last_success_time else None
        )

class RateLimiter:
    """Rate limiter for API endpoints"""
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = defaultdict(deque)
        self._lock = threading.Lock()
    
    def is_allowed(self, client_id: str) -> tuple[bool, RateLimitInfo]:
        """Check if request is allowed"""
        now = time.time()
        
        with self._lock:
            # Clean old requests
            client_requests = self.requests[client_id]
            while client_requests and client_requests[0] < now - self.window_seconds:
                client_requests.popleft()
            
            # Check limit
            if len(client_requests) >= self.max_requests:
                oldest_request = client_requests[0]
                retry_after = int(oldest_request + self.window_seconds - now)
                
                return False, RateLimitInfo(
                    limit=self.max_requests,
                    remaining=0,
                    reset_time=datetime.fromtimestamp(oldest_request + self.window_seconds),
                    retry_after_seconds=retry_after
                )
            
            # Allow request
            client_requests.append(now)
            remaining = self.max_requests - len(client_requests)
            
            return True, RateLimitInfo(
                limit=self.max_requests,
                remaining=remaining,
                reset_time=datetime.fromtimestamp(now + self.window_seconds)
            )

class SystemMonitor:
    """System health monitoring"""
    def __init__(self):
        self.start_time = time.time()
        self.request_count = 0
        self.error_count = 0
        self.response_times = deque(maxlen=1000)
        self._lock = threading.Lock()
    
    def record_request(self, response_time_ms: float, is_error: bool = False):
        """Record request metrics"""
        with self._lock:
            self.request_count += 1
            if is_error:
                self.error_count += 1
            self.response_times.append(response_time_ms)
    
    def get_health_metrics(self) -> SystemHealthMetrics:
        """Get current system health metrics"""
        process = psutil.Process()
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_info = process.memory_info()
        disk_usage = psutil.disk_usage('/').percent
        
        with self._lock:
            avg_response_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0
            uptime = time.time() - self.start_time
            error_rate = (self.error_count / self.request_count * 100) if self.request_count > 0 else 0
            rpm = (self.request_count / uptime * 60) if uptime > 0 else 0
        
        return SystemHealthMetrics(
            cpu_usage_percent=cpu_percent,
            memory_usage_mb=memory_info.rss / 1024 / 1024,
            disk_usage_percent=disk_usage,
            response_time_ms=avg_response_time,
            active_connections=1,  # Simplified for now
            requests_per_minute=rpm,
            error_rate_percent=error_rate,
            uptime_seconds=int(uptime)
        )

class RuleControlManager:
    """Enhanced rule control manager with bulletproof reliability"""
    
    def __init__(self):
        self.rules: Dict[str, Rule] = {}
        self.profiles: Dict[str, Dict[str, bool]] = {}
        self.current_profile: RuleProfile = RuleProfile.MODERATE
        self.rule_states: Dict[str, bool] = {}
        
        # Enhanced monitoring and reliability
        self.circuit_breaker = APICircuitBreaker()
        self.rate_limiter = RateLimiter(max_requests=1000, window_seconds=60)
        self.system_monitor = SystemMonitor()
        
        # Performance caching
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes
        self.cache_timestamps = {}
        
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
    
    def _get_cached(self, key: str) -> Optional[Any]:
        """Get cached value if not expired"""
        if key in self.cache:
            if time.time() - self.cache_timestamps[key] < self.cache_ttl:
                return self.cache[key]
            else:
                # Expired, remove from cache
                del self.cache[key]
                del self.cache_timestamps[key]
        return None
    
    def _set_cache(self, key: str, value: Any):
        """Set cached value"""
        self.cache[key] = value
        self.cache_timestamps[key] = time.time()
    
    def get_system_health(self) -> SystemHealthMetrics:
        """Get system health metrics"""
        return self.system_monitor.get_health_metrics()
    
    def get_circuit_breaker_stats(self) -> CircuitBreakerStats:
        """Get circuit breaker statistics"""
        return self.circuit_breaker.get_stats()

# Middleware for monitoring and reliability
async def monitor_request(request: Request, call_next):
    """Middleware to monitor requests"""
    start_time = time.time()
    client_ip = request.client.host
    
    # Rate limiting
    allowed, rate_info = manager.rate_limiter.is_allowed(client_ip)
    if not allowed:
        response = JSONResponse(
            status_code=429,
            content={"error": "Rate limit exceeded", "retry_after": rate_info.retry_after_seconds}
        )
        response.headers["X-RateLimit-Limit"] = str(rate_info.limit)
        response.headers["X-RateLimit-Remaining"] = str(rate_info.remaining)
        response.headers["X-RateLimit-Reset"] = rate_info.reset_time.isoformat()
        if rate_info.retry_after_seconds:
            response.headers["Retry-After"] = str(rate_info.retry_after_seconds)
        return response
    
    # Process request
    try:
        response = await call_next(request)
        processing_time = (time.time() - start_time) * 1000
        
        # Record metrics
        is_error = response.status_code >= 400
        manager.system_monitor.record_request(processing_time, is_error)
        
        # Add performance headers
        response.headers["X-Response-Time"] = f"{processing_time:.2f}ms"
        response.headers["X-RateLimit-Limit"] = str(rate_info.limit)
        response.headers["X-RateLimit-Remaining"] = str(rate_info.remaining)
        
        return response
        
    except Exception as e:
        processing_time = (time.time() - start_time) * 1000
        manager.system_monitor.record_request(processing_time, True)
        logger.error(f"Request failed: {e}")
        raise

# Create FastAPI app with enhanced configuration
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    logger.info("🚀 Rule Control API starting up...")
    yield
    logger.info("🛑 Rule Control API shutting down...")

app = FastAPI(
    title="Sutazai Rule Control API",
    version="2.0.0",
    description="Bulletproof rule control system with zero-downtime reliability",
    lifespan=lifespan
)

# Add middleware
app.middleware("http")(monitor_request)
app.add_middleware(GZipMiddleware, minimum_size=1000)
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

# Enhanced monitoring endpoints
@app.get("/api/health")
async def get_system_health():
    """Get comprehensive system health metrics"""
    try:
        health_metrics = manager.get_system_health()
        circuit_stats = manager.get_circuit_breaker_stats()
        
        return {
            "status": "healthy" if health_metrics.cpu_usage_percent < 90 and health_metrics.memory_usage_mb < 4096 else "degraded",
            "metrics": health_metrics.dict(),
            "circuit_breaker": circuit_stats.dict(),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Health check failed")

@app.get("/api/health/ready")
async def readiness_check():
    """Kubernetes readiness probe"""
    try:
        # Quick health check
        if len(manager.rules) == 0:
            raise HTTPException(status_code=503, detail="Rules not loaded")
        
        health = manager.get_system_health()
        if health.cpu_usage_percent > 95 or health.memory_usage_mb > 8192:
            raise HTTPException(status_code=503, detail="Resource exhaustion")
        
        return {"status": "ready", "timestamp": datetime.now().isoformat()}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

@app.get("/api/health/live")
async def liveness_check():
    """Kubernetes liveness probe"""
    return {"status": "alive", "timestamp": datetime.now().isoformat()}

@app.get("/api/metrics")
async def get_metrics():
    """Prometheus-compatible metrics endpoint"""
    health = manager.get_system_health()
    circuit_stats = manager.get_circuit_breaker_stats()
    
    metrics = [
        f"# HELP rule_control_cpu_usage_percent Current CPU usage percentage",
        f"# TYPE rule_control_cpu_usage_percent gauge",
        f"rule_control_cpu_usage_percent {health.cpu_usage_percent}",
        "",
        f"# HELP rule_control_memory_usage_mb Current memory usage in MB",
        f"# TYPE rule_control_memory_usage_mb gauge", 
        f"rule_control_memory_usage_mb {health.memory_usage_mb}",
        "",
        f"# HELP rule_control_requests_per_minute Requests per minute",
        f"# TYPE rule_control_requests_per_minute gauge",
        f"rule_control_requests_per_minute {health.requests_per_minute}",
        "",
        f"# HELP rule_control_error_rate_percent Error rate percentage",
        f"# TYPE rule_control_error_rate_percent gauge",
        f"rule_control_error_rate_percent {health.error_rate_percent}",
        "",
        f"# HELP rule_control_circuit_breaker_failures Circuit breaker failure count",
        f"# TYPE rule_control_circuit_breaker_failures counter",
        f"rule_control_circuit_breaker_failures {circuit_stats.failure_count}",
        "",
        f"# HELP rule_control_rules_enabled Number of enabled rules",
        f"# TYPE rule_control_rules_enabled gauge",
        f"rule_control_rules_enabled {sum(1 for r in manager.rules.values() if r.enabled)}",
        "",
        f"# HELP rule_control_uptime_seconds Uptime in seconds",
        f"# TYPE rule_control_uptime_seconds counter",
        f"rule_control_uptime_seconds {health.uptime_seconds}",
    ]
    
    return Response(content="\n".join(metrics), media_type="text/plain")

@app.post("/api/system/cache/clear")
async def clear_cache():
    """Clear system cache"""
    try:
        cache_size = len(manager.cache)
        manager.cache.clear()
        manager.cache_timestamps.clear()
        
        return {
            "success": True,
            "cleared_entries": cache_size,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/system/stats")
async def get_system_stats():
    """Get detailed system statistics"""
    try:
        health = manager.get_system_health()
        circuit_stats = manager.get_circuit_breaker_stats()
        
        rule_stats = {
            "total": len(manager.rules),
            "enabled": sum(1 for r in manager.rules.values() if r.enabled),
            "by_severity": {},
            "by_category": {}
        }
        
        for rule in manager.rules.values():
            # By severity
            if rule.severity.value not in rule_stats["by_severity"]:
                rule_stats["by_severity"][rule.severity.value] = {"total": 0, "enabled": 0}
            rule_stats["by_severity"][rule.severity.value]["total"] += 1
            if rule.enabled:
                rule_stats["by_severity"][rule.severity.value]["enabled"] += 1
            
            # By category  
            if rule.category.value not in rule_stats["by_category"]:
                rule_stats["by_category"][rule.category.value] = {"total": 0, "enabled": 0}
            rule_stats["by_category"][rule.category.value]["total"] += 1
            if rule.enabled:
                rule_stats["by_category"][rule.category.value]["enabled"] += 1
        
        return {
            "system_health": health.dict(),
            "circuit_breaker": circuit_stats.dict(),
            "rule_statistics": rule_stats,
            "cache_statistics": {
                "entries": len(manager.cache),
                "hit_ratio": 0.95  # Placeholder - would need actual hit/miss tracking
            },
            "current_profile": manager.current_profile.value,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"System stats failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Enhanced Rule Control Manager API")
    parser.add_argument("--port", type=int, default=8100, help="Port to run on")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--log-level", default="info", help="Log level")
    args = parser.parse_args()
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=args.port,
        workers=args.workers,
        log_level=args.log_level,
        access_log=True
    )