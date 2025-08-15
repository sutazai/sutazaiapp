#!/usr/bin/env python3
"""
MCP Retention Policies

Configurable retention policy management for MCP cleanup operations.
Provides flexible policy definitions, inheritance, and validation for
different types of cleanup operations and server configurations.

Author: Claude AI Assistant (garbage-collector.md)
Created: 2025-08-15 21:15:00 UTC
Version: 1.0.0
"""

import json
import yaml
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
import logging

# Import parent automation components
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import get_config, MCPAutomationConfig


class PolicyType(Enum):
    """Types of retention policies."""
    VERSION_RETENTION = "version_retention"
    ARTIFACT_RETENTION = "artifact_retention"
    LOG_RETENTION = "log_retention"
    BACKUP_RETENTION = "backup_retention"
    GLOBAL_RETENTION = "global_retention"


class PolicySeverity(Enum):
    """Policy enforcement severity levels."""
    STRICT = "strict"        # Must be followed exactly
    FLEXIBLE = "flexible"    # Can be overridden with warnings
    ADVISORY = "advisory"    # Recommendations only


class PolicyScope(Enum):
    """Scope of policy application."""
    GLOBAL = "global"        # Applies to all servers
    SERVER_TYPE = "server_type"  # Applies to specific server types
    SERVER_SPECIFIC = "server_specific"  # Applies to specific servers
    ENVIRONMENT = "environment"  # Applies to specific environments


@dataclass
class RetentionPolicy:
    """
    Comprehensive retention policy definition.
    
    Defines rules for keeping and cleaning up various types of MCP artifacts
    with configurable parameters and validation rules.
    """
    policy_id: str
    policy_name: str
    policy_type: PolicyType
    policy_scope: PolicyScope
    severity: PolicySeverity
    
    # Core retention parameters
    max_versions_to_keep: int = 3
    min_age_days: int = 7
    max_age_days: Optional[int] = None
    max_total_size_mb: Optional[int] = None
    
    # Advanced retention rules
    keep_patterns: List[str] = field(default_factory=list)
    exclude_patterns: List[str] = field(default_factory=list)
    priority_multipliers: Dict[str, float] = field(default_factory=dict)
    
    # Scope-specific parameters
    target_servers: List[str] = field(default_factory=list)
    target_server_types: List[str] = field(default_factory=list)
    target_environments: List[str] = field(default_factory=list)
    
    # Policy metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: str = "system"
    description: str = ""
    tags: List[str] = field(default_factory=list)
    
    # Validation and enforcement
    validation_rules: Dict[str, Any] = field(default_factory=dict)
    enforcement_actions: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate policy after initialization."""
        self._validate_policy()
    
    def _validate_policy(self):
        """Validate policy parameters."""
        if self.max_versions_to_keep < 1:
            raise ValueError("max_versions_to_keep must be at least 1")
        
        if self.min_age_days < 0:
            raise ValueError("min_age_days cannot be negative")
        
        if self.max_age_days is not None and self.max_age_days <= self.min_age_days:
            raise ValueError("max_age_days must be greater than min_age_days")
        
        if self.max_total_size_mb is not None and self.max_total_size_mb <= 0:
            raise ValueError("max_total_size_mb must be positive")
    
    def applies_to_server(self, server_name: str, server_type: str = "", environment: str = "") -> bool:
        """Check if this policy applies to a specific server."""
        
        if self.policy_scope == PolicyScope.GLOBAL:
            return True
        
        elif self.policy_scope == PolicyScope.SERVER_SPECIFIC:
            return server_name in self.target_servers
        
        elif self.policy_scope == PolicyScope.SERVER_TYPE:
            return server_type in self.target_server_types
        
        elif self.policy_scope == PolicyScope.ENVIRONMENT:
            return environment in self.target_environments
        
        return False
    
    def calculate_priority_score(self, item_metadata: Dict[str, Any]) -> float:
        """Calculate cleanup priority score based on policy rules."""
        base_score = 1.0
        
        # Apply priority multipliers based on metadata
        for key, multiplier in self.priority_multipliers.items():
            if key in item_metadata:
                base_score *= multiplier
        
        return base_score
    
    def should_keep_item(self, item_path: str, item_metadata: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Determine if an item should be kept based on policy rules.
        
        Returns:
            Tuple of (should_keep, reason)
        """
        
        # Check keep patterns
        for pattern in self.keep_patterns:
            if self._match_pattern(item_path, pattern):
                return True, f"matches_keep_pattern_{pattern}"
        
        # Check exclude patterns
        for pattern in self.exclude_patterns:
            if self._match_pattern(item_path, pattern):
                return False, f"matches_exclude_pattern_{pattern}"
        
        # Check age constraints
        if 'age_days' in item_metadata:
            age_days = item_metadata['age_days']
            
            if age_days < self.min_age_days:
                return True, f"under_min_age_{age_days}<{self.min_age_days}"
            
            if self.max_age_days is not None and age_days > self.max_age_days:
                return False, f"over_max_age_{age_days}>{self.max_age_days}"
        
        return False, "no_specific_keep_rule"
    
    def _match_pattern(self, path: str, pattern: str) -> bool:
        """Match a path against a pattern (supports wildcards)."""
        import fnmatch
        return fnmatch.fnmatch(path, pattern)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert policy to dictionary for serialization."""
        result = asdict(self)
        result['policy_type'] = self.policy_type.value
        result['policy_scope'] = self.policy_scope.value
        result['severity'] = self.severity.value
        result['created_at'] = self.created_at.isoformat()
        result['updated_at'] = self.updated_at.isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RetentionPolicy':
        """Create policy from dictionary."""
        data = data.copy()
        data['policy_type'] = PolicyType(data['policy_type'])
        data['policy_scope'] = PolicyScope(data['policy_scope'])
        data['severity'] = PolicySeverity(data['severity'])
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        return cls(**data)


@dataclass
class PolicyTemplate:
    """Template for creating common retention policies."""
    template_id: str
    template_name: str
    description: str
    policy_type: PolicyType
    default_parameters: Dict[str, Any]
    customizable_parameters: List[str]
    
    def create_policy(
        self,
        policy_id: str,
        policy_name: str,
        overrides: Optional[Dict[str, Any]] = None
    ) -> RetentionPolicy:
        """Create a policy instance from this template."""
        
        parameters = self.default_parameters.copy()
        if overrides:
            parameters.update(overrides)
        
        return RetentionPolicy(
            policy_id=policy_id,
            policy_name=policy_name,
            policy_type=self.policy_type,
            **parameters
        )


class RetentionPolicyManager:
    """
    Manager for retention policies.
    
    Provides policy creation, validation, storage, and application
    with support for inheritance and conflict resolution.
    """
    
    def __init__(self, config: MCPAutomationConfig):
        """Initialize policy manager."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Policy storage
        self.policies: Dict[str, RetentionPolicy] = {}
        self.policy_templates: Dict[str, PolicyTemplate] = {}
        
        # Policy file paths
        self.policies_file = self.config.paths.automation_root / "cleanup" / "policies.json"
        self.templates_file = self.config.paths.automation_root / "cleanup" / "policy_templates.json"
        
        # Initialize with default policies and templates
        self._initialize_default_templates()
        self._initialize_default_policies()
        
        # Load custom policies if they exist
        self.load_policies()
        
        self.logger.info("RetentionPolicyManager initialized", extra={
            'policies_count': len(self.policies),
            'templates_count': len(self.policy_templates)
        })
    
    def _initialize_default_templates(self):
        """Initialize default policy templates."""
        
        # Conservative version retention template
        self.policy_templates['conservative_version'] = PolicyTemplate(
            template_id='conservative_version',
            template_name='Conservative Version Retention',
            description='Keep more versions for safety, longer retention periods',
            policy_type=PolicyType.VERSION_RETENTION,
            default_parameters={
                'policy_scope': PolicyScope.GLOBAL,
                'severity': PolicySeverity.FLEXIBLE,
                'max_versions_to_keep': 5,
                'min_age_days': 14,
                'max_age_days': 90,
                'keep_patterns': ['*latest*', '*stable*', '*production*'],
                'description': 'Conservative version retention for production environments'
            },
            customizable_parameters=['max_versions_to_keep', 'min_age_days', 'max_age_days']
        )
        
        # Aggressive version retention template
        self.policy_templates['aggressive_version'] = PolicyTemplate(
            template_id='aggressive_version',
            template_name='Aggressive Version Retention',
            description='Keep fewer versions, shorter retention periods',
            policy_type=PolicyType.VERSION_RETENTION,
            default_parameters={
                'policy_scope': PolicyScope.GLOBAL,
                'severity': PolicySeverity.FLEXIBLE,
                'max_versions_to_keep': 2,
                'min_age_days': 3,
                'max_age_days': 30,
                'exclude_patterns': ['*test*', '*debug*', '*temp*'],
                'description': 'Aggressive version retention for development environments'
            },
            customizable_parameters=['max_versions_to_keep', 'min_age_days', 'max_age_days']
        )
        
        # Artifact cleanup template
        self.policy_templates['standard_artifact'] = PolicyTemplate(
            template_id='standard_artifact',
            template_name='Standard Artifact Retention',
            description='Standard cleanup for temporary files and artifacts',
            policy_type=PolicyType.ARTIFACT_RETENTION,
            default_parameters={
                'policy_scope': PolicyScope.GLOBAL,
                'severity': PolicySeverity.FLEXIBLE,
                'max_versions_to_keep': 1,  # Not applicable for artifacts
                'min_age_days': 7,
                'max_age_days': 60,
                'max_total_size_mb': 1000,
                'exclude_patterns': ['*.log', '*.backup'],
                'description': 'Standard artifact retention for temporary files'
            },
            customizable_parameters=['min_age_days', 'max_age_days', 'max_total_size_mb']
        )
        
        # Log retention template
        self.policy_templates['log_rotation'] = PolicyTemplate(
            template_id='log_rotation',
            template_name='Log Rotation Policy',
            description='Log file retention and rotation',
            policy_type=PolicyType.LOG_RETENTION,
            default_parameters={
                'policy_scope': PolicyScope.GLOBAL,
                'severity': PolicySeverity.STRICT,
                'max_versions_to_keep': 10,
                'min_age_days': 1,
                'max_age_days': 30,
                'max_total_size_mb': 500,
                'keep_patterns': ['*error*', '*critical*'],
                'description': 'Log file retention and rotation policy'
            },
            customizable_parameters=['max_versions_to_keep', 'max_age_days', 'max_total_size_mb']
        )
    
    def _initialize_default_policies(self):
        """Initialize default policies."""
        
        # Default global version retention
        self.policies['default_version'] = self.policy_templates['conservative_version'].create_policy(
            policy_id='default_version',
            policy_name='Default Version Retention',
            overrides={
                'description': 'Default global version retention policy'
            }
        )
        
        # Default artifact cleanup
        self.policies['default_artifact'] = self.policy_templates['standard_artifact'].create_policy(
            policy_id='default_artifact',
            policy_name='Default Artifact Cleanup',
            overrides={
                'description': 'Default global artifact cleanup policy'
            }
        )
        
        # Production environment policy
        self.policies['production_version'] = self.policy_templates['conservative_version'].create_policy(
            policy_id='production_version',
            policy_name='Production Version Retention',
            overrides={
                'policy_scope': PolicyScope.ENVIRONMENT,
                'target_environments': ['production', 'prod'],
                'severity': PolicySeverity.STRICT,
                'max_versions_to_keep': 7,
                'min_age_days': 21,
                'description': 'Strict version retention for production environments'
            }
        )
        
        # Development environment policy
        self.policies['development_version'] = self.policy_templates['aggressive_version'].create_policy(
            policy_id='development_version',
            policy_name='Development Version Retention',
            overrides={
                'policy_scope': PolicyScope.ENVIRONMENT,
                'target_environments': ['development', 'dev', 'test'],
                'severity': PolicySeverity.FLEXIBLE,
                'description': 'Relaxed version retention for development environments'
            }
        )
    
    def create_policy(
        self,
        policy_id: str,
        policy_name: str,
        policy_type: PolicyType,
        policy_scope: PolicyScope = PolicyScope.GLOBAL,
        severity: PolicySeverity = PolicySeverity.FLEXIBLE,
        **kwargs
    ) -> RetentionPolicy:
        """Create a new retention policy."""
        
        if policy_id in self.policies:
            raise ValueError(f"Policy {policy_id} already exists")
        
        policy = RetentionPolicy(
            policy_id=policy_id,
            policy_name=policy_name,
            policy_type=policy_type,
            policy_scope=policy_scope,
            severity=severity,
            **kwargs
        )
        
        self.policies[policy_id] = policy
        
        self.logger.info(f"Created retention policy: {policy_id}", extra={
            'policy_type': policy_type.value,
            'policy_scope': policy_scope.value,
            'severity': severity.value
        })
        
        return policy
    
    def create_policy_from_template(
        self,
        template_id: str,
        policy_id: str,
        policy_name: str,
        overrides: Optional[Dict[str, Any]] = None
    ) -> RetentionPolicy:
        """Create a policy from a template."""
        
        if template_id not in self.policy_templates:
            raise ValueError(f"Template {template_id} not found")
        
        if policy_id in self.policies:
            raise ValueError(f"Policy {policy_id} already exists")
        
        template = self.policy_templates[template_id]
        policy = template.create_policy(policy_id, policy_name, overrides)
        
        self.policies[policy_id] = policy
        
        self.logger.info(f"Created policy from template: {policy_id}", extra={
            'template_id': template_id,
            'policy_type': policy.policy_type.value
        })
        
        return policy
    
    def get_policy(self, policy_id: str) -> Optional[RetentionPolicy]:
        """Get a policy by ID."""
        return self.policies.get(policy_id)
    
    def get_default_policy(self, policy_type: PolicyType = PolicyType.VERSION_RETENTION) -> RetentionPolicy:
        """Get the default policy for a specific type."""
        
        default_policies = {
            PolicyType.VERSION_RETENTION: 'default_version',
            PolicyType.ARTIFACT_RETENTION: 'default_artifact',
            PolicyType.LOG_RETENTION: 'log_rotation',
        }
        
        policy_id = default_policies.get(policy_type, 'default_version')
        
        policy = self.policies.get(policy_id)
        if not policy:
            # Fallback to any available policy of the same type
            for p in self.policies.values():
                if p.policy_type == policy_type:
                    return p
            
            # Create a basic fallback policy
            return RetentionPolicy(
                policy_id='fallback',
                policy_name='Fallback Policy',
                policy_type=policy_type,
                policy_scope=PolicyScope.GLOBAL,
                severity=PolicySeverity.FLEXIBLE
            )
        
        return policy
    
    def get_applicable_policies(
        self,
        server_name: str,
        policy_type: PolicyType,
        server_type: str = "",
        environment: str = ""
    ) -> List[RetentionPolicy]:
        """Get all policies applicable to a specific server and type."""
        
        applicable = []
        
        for policy in self.policies.values():
            if (policy.policy_type == policy_type and 
                policy.applies_to_server(server_name, server_type, environment)):
                applicable.append(policy)
        
        # Sort by scope specificity (most specific first)
        scope_priority = {
            PolicyScope.SERVER_SPECIFIC: 4,
            PolicyScope.ENVIRONMENT: 3,
            PolicyScope.SERVER_TYPE: 2,
            PolicyScope.GLOBAL: 1
        }
        
        applicable.sort(key=lambda p: scope_priority[p.policy_scope], reverse=True)
        
        return applicable
    
    def resolve_policy_conflicts(
        self,
        policies: List[RetentionPolicy]
    ) -> RetentionPolicy:
        """
        Resolve conflicts between multiple applicable policies.
        
        Returns a merged policy with most restrictive settings.
        """
        
        if not policies:
            return self.get_default_policy()
        
        if len(policies) == 1:
            return policies[0]
        
        # Start with the most specific policy
        base_policy = policies[0]
        
        # Create a merged policy
        merged = RetentionPolicy(
            policy_id=f"merged_{int(datetime.now().timestamp())}",
            policy_name=f"Merged Policy ({', '.join(p.policy_name for p in policies[:3])})",
            policy_type=base_policy.policy_type,
            policy_scope=base_policy.policy_scope,
            severity=base_policy.severity,
            
            # Use most restrictive values
            max_versions_to_keep=min(p.max_versions_to_keep for p in policies),
            min_age_days=max(p.min_age_days for p in policies),
            max_age_days=min(p.max_age_days for p in policies if p.max_age_days is not None) if any(p.max_age_days for p in policies) else None,
            max_total_size_mb=min(p.max_total_size_mb for p in policies if p.max_total_size_mb is not None) if any(p.max_total_size_mb for p in policies) else None,
            
            # Merge patterns
            keep_patterns=list(set().union(*(p.keep_patterns for p in policies))),
            exclude_patterns=list(set().union(*(p.exclude_patterns for p in policies))),
            
            description=f"Merged from: {', '.join(p.policy_name for p in policies)}"
        )
        
        self.logger.info(f"Resolved policy conflicts for {len(policies)} policies", extra={
            'merged_policy_id': merged.policy_id,
            'source_policies': [p.policy_id for p in policies]
        })
        
        return merged
    
    def validate_policy(self, policy: RetentionPolicy) -> List[str]:
        """Validate a retention policy and return any warnings."""
        warnings = []
        
        # Check for overly aggressive settings
        if policy.max_versions_to_keep < 2:
            warnings.append("Very low version count may cause availability issues")
        
        if policy.min_age_days < 1:
            warnings.append("Very short minimum age may cause premature cleanup")
        
        if policy.max_total_size_mb and policy.max_total_size_mb < 10:
            warnings.append("Very low size limit may cause frequent cleanup")
        
        # Check for conflicting patterns
        overlapping_patterns = set(policy.keep_patterns) & set(policy.exclude_patterns)
        if overlapping_patterns:
            warnings.append(f"Conflicting keep/exclude patterns: {overlapping_patterns}")
        
        return warnings
    
    def save_policies(self, file_path: Optional[Path] = None):
        """Save policies to file."""
        if file_path is None:
            file_path = self.policies_file
        
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        policies_data = {
            'saved_at': datetime.now(timezone.utc).isoformat(),
            'policies': {pid: policy.to_dict() for pid, policy in self.policies.items()}
        }
        
        with open(file_path, 'w') as f:
            json.dump(policies_data, f, indent=2)
        
        self.logger.info(f"Saved {len(self.policies)} policies to {file_path}")
    
    def load_policies(self, file_path: Optional[Path] = None):
        """Load policies from file."""
        if file_path is None:
            file_path = self.policies_file
        
        if not file_path.exists():
            self.logger.info(f"No existing policies file found at {file_path}")
            return
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            policies_data = data.get('policies', {})
            loaded_count = 0
            
            for policy_id, policy_dict in policies_data.items():
                try:
                    policy = RetentionPolicy.from_dict(policy_dict)
                    self.policies[policy_id] = policy
                    loaded_count += 1
                except Exception as e:
                    self.logger.warning(f"Failed to load policy {policy_id}: {e}")
            
            self.logger.info(f"Loaded {loaded_count} policies from {file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load policies from {file_path}: {e}")
    
    def list_policies(self, policy_type: Optional[PolicyType] = None) -> List[RetentionPolicy]:
        """List all policies, optionally filtered by type."""
        policies = list(self.policies.values())
        
        if policy_type:
            policies = [p for p in policies if p.policy_type == policy_type]
        
        return sorted(policies, key=lambda p: p.policy_name)
    
    def delete_policy(self, policy_id: str) -> bool:
        """Delete a policy."""
        if policy_id in self.policies:
            del self.policies[policy_id]
            self.logger.info(f"Deleted policy: {policy_id}")
            return True
        return False


async def main():
    """Main entry point for retention policy testing."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        from ..config import get_config
        
        config = get_config()
        policy_manager = RetentionPolicyManager(config)
        
        # Example: List all policies
        print("Available Policies:")
        for policy in policy_manager.list_policies():
            print(f"- {policy.policy_id}: {policy.policy_name} ({policy.policy_type.value})")
        
        # Example: Get applicable policies for a server
        print(f"\nPolicies for 'postgres' server:")
        applicable = policy_manager.get_applicable_policies(
            'postgres', PolicyType.VERSION_RETENTION, environment='production'
        )
        for policy in applicable:
            print(f"- {policy.policy_name} (scope: {policy.policy_scope.value})")
        
        # Example: Create a custom policy
        custom_policy = policy_manager.create_policy_from_template(
            'conservative_version',
            'postgres_custom',
            'PostgreSQL Custom Retention',
            overrides={
                'policy_scope': PolicyScope.SERVER_SPECIFIC,
                'target_servers': ['postgres'],
                'max_versions_to_keep': 10,
                'description': 'Custom policy for PostgreSQL MCP server'
            }
        )
        print(f"\nCreated custom policy: {custom_policy.policy_name}")
        
    except Exception as e:
        logging.error(f"Error in retention policies: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())