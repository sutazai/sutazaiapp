#!/usr/bin/env python3
"""
MCP Policy Engine

Policy definition and enforcement system for MCP orchestration. Ensures all
operations comply with organizational policies, security requirements, and
best practices with rule-based validation and automatic enforcement.

Author: Claude AI Assistant (ai-agent-orchestrator)  
Created: 2025-08-15 12:00:00 UTC
Version: 1.0.0
"""

import json
import re
from datetime import datetime, timezone, timedelta, time
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple, Callable
from dataclasses import dataclass, asdict, field
from enum import Enum
import logging

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import MCPAutomationConfig


class PolicyType(Enum):
    """Policy types."""
    SECURITY = "security"              # Security policies
    OPERATIONAL = "operational"        # Operational policies
    COMPLIANCE = "compliance"          # Compliance requirements
    RESOURCE = "resource"              # Resource usage policies
    SCHEDULING = "scheduling"          # Time-based policies
    APPROVAL = "approval"              # Approval requirements
    VALIDATION = "validation"          # Data validation policies
    ENFORCEMENT = "enforcement"        # Enforcement rules


class PolicyAction(Enum):
    """Policy enforcement actions."""
    ALLOW = "allow"                    # Allow operation
    DENY = "deny"                      # Deny operation
    WARN = "warn"                      # Allow with warning
    REQUIRE_APPROVAL = "require_approval"  # Require manual approval
    MODIFY = "modify"                  # Modify operation parameters
    AUDIT = "audit"                    # Audit operation
    ESCALATE = "escalate"              # Escalate to administrator


class PolicySeverity(Enum):
    """Policy violation severity."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class PolicyRule:
    """Individual policy rule."""
    id: str
    name: str
    description: str
    type: PolicyType
    conditions: List[Dict[str, Any]]  # Conditions to evaluate
    action: PolicyAction               # Action to take
    severity: PolicySeverity
    exceptions: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    
    def evaluate(self, context: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Evaluate rule against context."""
        # Check if rule is enabled
        if not self.enabled:
            return True, None
            
        # Check exceptions first
        for exception in self.exceptions:
            if self._matches_condition(exception, context):
                return True, None
                
        # Evaluate conditions
        for condition in self.conditions:
            if not self._matches_condition(condition, context):
                return False, f"Condition failed: {condition}"
                
        return True, None
        
    def _matches_condition(self, condition: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Check if context matches condition."""
        for key, expected in condition.items():
            actual = context.get(key)
            
            # Handle different comparison operators
            if isinstance(expected, dict):
                op = expected.get("operator", "equals")
                value = expected.get("value")
                
                if op == "equals" and actual != value:
                    return False
                elif op == "not_equals" and actual == value:
                    return False
                elif op == "greater_than" and not (actual > value):
                    return False
                elif op == "less_than" and not (actual < value):
                    return False
                elif op == "contains" and value not in str(actual):
                    return False
                elif op == "regex" and not re.match(value, str(actual)):
                    return False
                elif op == "in" and actual not in value:
                    return False
                elif op == "not_in" and actual in value:
                    return False
            else:
                # Simple equality check
                if actual != expected:
                    return False
                    
        return True


@dataclass
class PolicySet:
    """Collection of related policies."""
    id: str
    name: str
    description: str
    rules: List[PolicyRule]
    priority: int = 0
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def evaluate(self, context: Dict[str, Any]) -> List[Tuple[PolicyRule, str]]:
        """Evaluate all rules in the policy set."""
        violations = []
        
        if not self.enabled:
            return violations
            
        for rule in self.rules:
            passed, reason = rule.evaluate(context)
            if not passed:
                violations.append((rule, reason))
                
        return violations


@dataclass
class PolicyViolation:
    """Policy violation record."""
    rule_id: str
    rule_name: str
    policy_type: PolicyType
    severity: PolicySeverity
    action: PolicyAction
    reason: str
    context: Dict[str, Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    resolved: bool = False
    resolution: Optional[str] = None


class PolicyEngine:
    """
    Policy enforcement engine for MCP orchestration.
    
    Manages policy definitions, evaluates operations against policies,
    and enforces organizational standards and compliance requirements.
    """
    
    def __init__(self, config: Optional[MCPAutomationConfig] = None):
        """Initialize policy engine."""
        self.config = config or MCPAutomationConfig()
        self.logger = self._setup_logging()
        
        # Policy storage
        self.policies: Dict[str, PolicySet] = {}
        self.rules: Dict[str, PolicyRule] = {}
        
        # Violation tracking
        self.violations: List[PolicyViolation] = []
        self.violation_history: Dict[str, List[PolicyViolation]] = {}
        
        # Custom evaluators
        self.custom_evaluators: Dict[str, Callable] = {}
        
        # Initialize default policies
        self._initialize_default_policies()
        
    def _setup_logging(self) -> logging.Logger:
        """Configure logging."""
        logger = logging.getLogger("mcp.policy_engine")
        logger.setLevel(self.config.log_level.value.upper())
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
        
    def _initialize_default_policies(self) -> None:
        """Initialize default organizational policies."""
        # Security policies
        security_policy = PolicySet(
            id="security-default",
            name="Default Security Policies",
            description="Standard security policies for MCP operations",
            priority=100,
            rules=[
                PolicyRule(
                    id="no-root-operations",
                    name="No Root Operations",
                    description="Prevent operations requiring root access",
                    type=PolicyType.SECURITY,
                    conditions=[
                        {"requires_root": {"operator": "not_equals", "value": True}}
                    ],
                    action=PolicyAction.DENY,
                    severity=PolicySeverity.CRITICAL
                ),
                PolicyRule(
                    id="secure-connections",
                    name="Secure Connections Only",
                    description="Require secure connections for external services",
                    type=PolicyType.SECURITY,
                    conditions=[
                        {"connection_type": {"operator": "in", "value": ["https", "ssh", "tls"]}}
                    ],
                    action=PolicyAction.DENY,
                    severity=PolicySeverity.ERROR,
                    exceptions=[
                        {"service": "localhost"}
                    ]
                ),
                PolicyRule(
                    id="no-hardcoded-secrets",
                    name="No Hardcoded Secrets",
                    description="Prevent hardcoded credentials in configurations",
                    type=PolicyType.SECURITY,
                    conditions=[
                        {"contains_secrets": {"operator": "not_equals", "value": True}}
                    ],
                    action=PolicyAction.DENY,
                    severity=PolicySeverity.CRITICAL
                )
            ]
        )
        
        # Operational policies
        operational_policy = PolicySet(
            id="operational-default",
            name="Default Operational Policies",
            description="Standard operational policies for MCP automation",
            priority=50,
            rules=[
                PolicyRule(
                    id="maintenance-window",
                    name="Maintenance Window Enforcement",
                    description="Allow major operations only during maintenance windows",
                    type=PolicyType.SCHEDULING,
                    conditions=[
                        {"operation_type": {"operator": "in", "value": ["update", "restart", "migration"]}}
                    ],
                    action=PolicyAction.WARN,
                    severity=PolicySeverity.WARNING,
                    metadata={
                        "maintenance_windows": [
                            {"day": "Saturday", "start": "02:00", "end": "06:00"},
                            {"day": "Sunday", "start": "02:00", "end": "06:00"}
                        ]
                    }
                ),
                PolicyRule(
                    id="backup-before-update",
                    name="Backup Before Update",
                    description="Require backup before update operations",
                    type=PolicyType.OPERATIONAL,
                    conditions=[
                        {"operation": "update"},
                        {"backup_completed": {"operator": "equals", "value": True}}
                    ],
                    action=PolicyAction.DENY,
                    severity=PolicySeverity.ERROR
                ),
                PolicyRule(
                    id="dry-run-first",
                    name="Dry Run First",
                    description="Require dry run for destructive operations",
                    type=PolicyType.OPERATIONAL,
                    conditions=[
                        {"operation_type": {"operator": "in", "value": ["cleanup", "delete", "remove"]}},
                        {"dry_run_completed": {"operator": "equals", "value": True}}
                    ],
                    action=PolicyAction.WARN,
                    severity=PolicySeverity.WARNING
                )
            ]
        )
        
        # Resource policies
        resource_policy = PolicySet(
            id="resource-default",
            name="Default Resource Policies",
            description="Resource usage and limit policies",
            priority=30,
            rules=[
                PolicyRule(
                    id="cpu-limit",
                    name="CPU Usage Limit",
                    description="Prevent operations when CPU usage is high",
                    type=PolicyType.RESOURCE,
                    conditions=[
                        {"cpu_usage": {"operator": "less_than", "value": 80}}
                    ],
                    action=PolicyAction.WARN,
                    severity=PolicySeverity.WARNING
                ),
                PolicyRule(
                    id="memory-limit",
                    name="Memory Usage Limit",
                    description="Prevent operations when memory usage is high",
                    type=PolicyType.RESOURCE,
                    conditions=[
                        {"memory_usage": {"operator": "less_than", "value": 85}}
                    ],
                    action=PolicyAction.WARN,
                    severity=PolicySeverity.WARNING
                ),
                PolicyRule(
                    id="disk-space",
                    name="Disk Space Requirement",
                    description="Ensure sufficient disk space for operations",
                    type=PolicyType.RESOURCE,
                    conditions=[
                        {"disk_free_gb": {"operator": "greater_than", "value": 10}}
                    ],
                    action=PolicyAction.DENY,
                    severity=PolicySeverity.ERROR
                )
            ]
        )
        
        # Compliance policies
        compliance_policy = PolicySet(
            id="compliance-default",
            name="Default Compliance Policies",
            description="Regulatory and compliance requirements",
            priority=90,
            rules=[
                PolicyRule(
                    id="audit-trail",
                    name="Audit Trail Required",
                    description="Require audit trail for sensitive operations",
                    type=PolicyType.COMPLIANCE,
                    conditions=[
                        {"audit_enabled": {"operator": "equals", "value": True}}
                    ],
                    action=PolicyAction.AUDIT,
                    severity=PolicySeverity.INFO
                ),
                PolicyRule(
                    id="data-retention",
                    name="Data Retention Policy",
                    description="Enforce data retention requirements",
                    type=PolicyType.COMPLIANCE,
                    conditions=[
                        {"retention_days": {"operator": "greater_than", "value": 30}}
                    ],
                    action=PolicyAction.WARN,
                    severity=PolicySeverity.WARNING
                ),
                PolicyRule(
                    id="approval-required",
                    name="Approval Required",
                    description="Require approval for critical operations",
                    type=PolicyType.APPROVAL,
                    conditions=[
                        {"criticality": {"operator": "in", "value": ["high", "critical"]}},
                        {"approved": {"operator": "equals", "value": True}}
                    ],
                    action=PolicyAction.REQUIRE_APPROVAL,
                    severity=PolicySeverity.ERROR
                )
            ]
        )
        
        # Enforcement rules (from IMPORTANT/Enforcement_Rules)
        enforcement_policy = PolicySet(
            id="enforcement-rules",
            name="Codebase Enforcement Rules",
            description="Mandatory codebase standards and hygiene requirements",
            priority=1000,  # Highest priority
            rules=[
                PolicyRule(
                    id="no-breaking-changes",
                    name="Never Break Existing Functionality",
                    description="Prevent changes that break existing functionality",
                    type=PolicyType.ENFORCEMENT,
                    conditions=[
                        {"breaks_existing": {"operator": "not_equals", "value": True}}
                    ],
                    action=PolicyAction.DENY,
                    severity=PolicySeverity.CRITICAL
                ),
                PolicyRule(
                    id="investigate-first",
                    name="Investigate Existing Files First",
                    description="Require investigation of existing files before creating new ones",
                    type=PolicyType.ENFORCEMENT,
                    conditions=[
                        {"existing_investigated": {"operator": "equals", "value": True}}
                    ],
                    action=PolicyAction.DENY,
                    severity=PolicySeverity.ERROR
                ),
                PolicyRule(
                    id="professional-standards",
                    name="Professional Project Standards",
                    description="Enforce professional engineering standards",
                    type=PolicyType.ENFORCEMENT,
                    conditions=[
                        {"follows_standards": {"operator": "equals", "value": True}}
                    ],
                    action=PolicyAction.DENY,
                    severity=PolicySeverity.ERROR
                ),
                PolicyRule(
                    id="mcp-protection",
                    name="MCP Server Protection",
                    description="Protect MCP servers and configurations",
                    type=PolicyType.ENFORCEMENT,
                    conditions=[
                        {"modifies_mcp": {"operator": "not_equals", "value": True}}
                    ],
                    action=PolicyAction.DENY,
                    severity=PolicySeverity.CRITICAL,
                    exceptions=[
                        {"authorized": True}
                    ]
                )
            ]
        )
        
        # Add default policies
        self.add_policy(security_policy)
        self.add_policy(operational_policy)
        self.add_policy(resource_policy)
        self.add_policy(compliance_policy)
        self.add_policy(enforcement_policy)
        
    async def initialize(self) -> None:
        """Initialize policy engine."""
        self.logger.info("Initializing policy engine...")
        
        # Load custom policies from files
        await self._load_custom_policies()
        
        self.logger.info(f"Loaded {len(self.policies)} policy sets with {len(self.rules)} rules")
        
    async def _load_custom_policies(self) -> None:
        """Load custom policies from configuration files."""
        policy_dir = Path("/opt/sutazaiapp/scripts/mcp/automation/policies")
        if policy_dir.exists():
            for policy_file in policy_dir.glob("*.json"):
                try:
                    with open(policy_file) as f:
                        policy_data = json.load(f)
                        policy_set = self._parse_policy_set(policy_data)
                        self.add_policy(policy_set)
                        self.logger.info(f"Loaded policy: {policy_set.name}")
                except Exception as e:
                    self.logger.error(f"Failed to load policy {policy_file}: {e}")
                    
    def _parse_policy_set(self, data: Dict[str, Any]) -> PolicySet:
        """Parse policy set from dictionary."""
        rules = []
        for rule_data in data.get("rules", []):
            rule = PolicyRule(
                id=rule_data["id"],
                name=rule_data["name"],
                description=rule_data.get("description", ""),
                type=PolicyType[rule_data["type"].upper()],
                conditions=rule_data.get("conditions", []),
                action=PolicyAction[rule_data["action"].upper()],
                severity=PolicySeverity[rule_data["severity"].upper()],
                exceptions=rule_data.get("exceptions", []),
                metadata=rule_data.get("metadata", {}),
                enabled=rule_data.get("enabled", True)
            )
            rules.append(rule)
            
        return PolicySet(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            rules=rules,
            priority=data.get("priority", 0),
            enabled=data.get("enabled", True),
            metadata=data.get("metadata", {})
        )
        
    def add_policy(self, policy_set: PolicySet) -> None:
        """Add a policy set to the engine."""
        self.policies[policy_set.id] = policy_set
        
        # Index rules
        for rule in policy_set.rules:
            self.rules[rule.id] = rule
            
        self.logger.debug(f"Added policy set: {policy_set.name}")
        
    def remove_policy(self, policy_id: str) -> bool:
        """Remove a policy set."""
        if policy_id in self.policies:
            policy_set = self.policies[policy_id]
            
            # Remove rules from index
            for rule in policy_set.rules:
                self.rules.pop(rule.id, None)
                
            del self.policies[policy_id]
            self.logger.info(f"Removed policy set: {policy_set.name}")
            return True
            
        return False
        
    async def evaluate(
        self,
        operation: str,
        context: Dict[str, Any]
    ) -> List[PolicyViolation]:
        """Evaluate an operation against all policies."""
        violations = []
        
        # Add operation to context
        full_context = {
            "operation": operation,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **context
        }
        
        # Sort policies by priority
        sorted_policies = sorted(
            self.policies.values(),
            key=lambda p: p.priority,
            reverse=True
        )
        
        # Evaluate each policy set
        for policy_set in sorted_policies:
            if not policy_set.enabled:
                continue
                
            policy_violations = policy_set.evaluate(full_context)
            
            for rule, reason in policy_violations:
                violation = PolicyViolation(
                    rule_id=rule.id,
                    rule_name=rule.name,
                    policy_type=rule.type,
                    severity=rule.severity,
                    action=rule.action,
                    reason=reason or f"Policy {rule.name} violated",
                    context=full_context
                )
                
                violations.append(violation)
                self.violations.append(violation)
                
                # Track in history
                if operation not in self.violation_history:
                    self.violation_history[operation] = []
                self.violation_history[operation].append(violation)
                
                self.logger.warning(
                    f"Policy violation: {rule.name} - {reason} (Action: {rule.action.value})"
                )
                
        return violations
        
    async def check_workflow(
        self,
        workflow_name: str,
        context: Dict[str, Any]
    ) -> List[PolicyViolation]:
        """Check if a workflow can be executed."""
        return await self.evaluate(f"workflow:{workflow_name}", context)
        
    async def check_service_operation(
        self,
        service_name: str,
        operation: str,
        context: Dict[str, Any]
    ) -> List[PolicyViolation]:
        """Check if a service operation is allowed."""
        full_context = {
            "service": service_name,
            **context
        }
        return await self.evaluate(f"service:{operation}", full_context)
        
    async def enforce(
        self,
        violations: List[PolicyViolation]
    ) -> Tuple[PolicyAction, List[str]]:
        """Enforce policy violations and return action to take."""
        if not violations:
            return PolicyAction.ALLOW, []
            
        # Determine most restrictive action
        actions = [v.action for v in violations]
        
        if PolicyAction.DENY in actions:
            return PolicyAction.DENY, [v.reason for v in violations if v.action == PolicyAction.DENY]
        elif PolicyAction.REQUIRE_APPROVAL in actions:
            return PolicyAction.REQUIRE_APPROVAL, [v.reason for v in violations if v.action == PolicyAction.REQUIRE_APPROVAL]
        elif PolicyAction.ESCALATE in actions:
            return PolicyAction.ESCALATE, [v.reason for v in violations if v.action == PolicyAction.ESCALATE]
        elif PolicyAction.WARN in actions:
            return PolicyAction.WARN, [v.reason for v in violations if v.action == PolicyAction.WARN]
        else:
            return PolicyAction.ALLOW, []
            
    async def load_policies(self) -> None:
        """Load all policies from configuration."""
        await self._load_custom_policies()
        
    async def get_policy_stats(self) -> Dict[str, Any]:
        """Get policy statistics."""
        return {
            "total_policies": len(self.policies),
            "total_rules": len(self.rules),
            "enabled_policies": sum(1 for p in self.policies.values() if p.enabled),
            "enabled_rules": sum(1 for r in self.rules.values() if r.enabled),
            "total_violations": len(self.violations),
            "violations_by_severity": {
                severity.value: sum(1 for v in self.violations if v.severity == severity)
                for severity in PolicySeverity
            },
            "violations_by_type": {
                policy_type.value: sum(1 for v in self.violations if v.policy_type == policy_type)
                for policy_type in PolicyType
            }
        }
        
    async def get_violations(
        self,
        operation: Optional[str] = None,
        resolved: Optional[bool] = None
    ) -> List[PolicyViolation]:
        """Get policy violations."""
        violations = self.violations
        
        if operation:
            violations = self.violation_history.get(operation, [])
            
        if resolved is not None:
            violations = [v for v in violations if v.resolved == resolved]
            
        return violations
        
    async def resolve_violation(
        self,
        rule_id: str,
        resolution: str
    ) -> bool:
        """Mark a violation as resolved."""
        for violation in self.violations:
            if violation.rule_id == rule_id and not violation.resolved:
                violation.resolved = True
                violation.resolution = resolution
                self.logger.info(f"Resolved violation: {rule_id}")
                return True
                
        return False
        
    def register_evaluator(self, name: str, evaluator: Callable) -> None:
        """Register a custom policy evaluator."""
        self.custom_evaluators[name] = evaluator
        self.logger.debug(f"Registered custom evaluator: {name}")
        
    def is_maintenance_window(self) -> bool:
        """Check if current time is within maintenance window."""
        now = datetime.now(timezone.utc)
        current_day = now.strftime("%A")
        current_time = now.time()
        
        # Check maintenance windows from policies
        for policy_set in self.policies.values():
            for rule in policy_set.rules:
                if rule.id == "maintenance-window":
                    windows = rule.metadata.get("maintenance_windows", [])
                    for window in windows:
                        if window["day"] == current_day:
                            start = datetime.strptime(window["start"], "%H:%M").time()
                            end = datetime.strptime(window["end"], "%H:%M").time()
                            if start <= current_time <= end:
                                return True
                                
        return False
        
    async def shutdown(self) -> None:
        """Shutdown policy engine."""
        self.logger.info("Shutting down policy engine...")
        # Save violation history if needed
        self.logger.info("Policy engine shutdown complete")