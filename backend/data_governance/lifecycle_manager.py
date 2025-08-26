"""
Data Lifecycle Manager
=====================

Manages the complete lifecycle of data from creation to deletion,
implementing automated policies for retention, archival, and deletion.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
import json
from pathlib import Path

from .data_classifier import DataClassification, DataType, RegulationScope


class LifecycleStage(Enum):
    """Stages in the data lifecycle"""
    ACTIVE = "active"              # Currently in use
    INACTIVE = "inactive"          # No longer actively used
    ARCHIVED = "archived"          # Moved to archive storage
    QUARANTINED = "quarantined"    # Flagged for review
    SCHEDULED_DELETION = "scheduled_deletion"  # Marked for deletion
    DELETED = "deleted"            # Permanently removed


class RetentionAction(Enum):
    """Actions that can be taken during retention processing"""
    KEEP_ACTIVE = "keep_active"
    ARCHIVE = "archive"
    DELETE = "delete"
    QUARANTINE = "quarantine"
    EXTEND_RETENTION = "extend_retention"


@dataclass
class LifecyclePolicy:
    """Policy defining data lifecycle rules"""
    name: str
    description: str
    
    # Classification criteria
    classification: Optional[DataClassification] = None
    data_types: Optional[List[DataType]] = None
    source_patterns: Optional[List[str]] = None
    
    # Retention settings
    min_retention_days: int = 0
    max_retention_days: Optional[int] = None
    archive_after_days: Optional[int] = None
    delete_after_days: Optional[int] = None
    
    # Storage settings
    hot_storage_days: int = 30
    warm_storage_days: int = 365
    cold_storage_days: Optional[int] = None
    
    # Security requirements
    encryption_required: bool = False
    compression_enabled: bool = True
    backup_required: bool = False
    
    # Compliance settings
    immutable_period_days: Optional[int] = None
    legal_hold_capable: bool = False
    audit_required: bool = False
    
    # Automated actions
    auto_archive: bool = True
    auto_delete: bool = False  # Requires explicit approval for safety
    
    # Custom conditions and actions
    custom_conditions: Dict[str, Any] = field(default_factory=dict)
    custom_actions: List[str] = field(default_factory=list)
    
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    is_active: bool = True


@dataclass
class DataAsset:
    """Represents a data asset in the lifecycle management system"""
    id: str
    name: str
    source: str
    data_type: DataType
    classification: DataClassification
    
    # Lifecycle tracking
    created_at: datetime
    last_accessed: Optional[datetime] = None
    last_modified: Optional[datetime] = None
    current_stage: LifecycleStage = LifecycleStage.ACTIVE
    
    # Storage information
    storage_location: str = ""
    size_bytes: int = 0
    checksum: Optional[str] = None
    
    # Retention information
    retention_policy: Optional[str] = None
    retention_until: Optional[datetime] = None
    archive_after: Optional[datetime] = None
    delete_after: Optional[datetime] = None
    
    # Compliance tracking
    legal_holds: List[str] = field(default_factory=list)
    audit_trail: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)


@dataclass
class LifecycleAction:
    """Represents an action to be taken on a data asset"""
    asset_id: str
    action: RetentionAction
    scheduled_at: datetime
    reason: str
    policy_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Execution tracking
    executed_at: Optional[datetime] = None
    execution_status: str = "pending"  # pending, executing, completed, failed
    execution_details: Dict[str, Any] = field(default_factory=dict)


class DataLifecycleManager:
    """
    Manages the complete lifecycle of data assets including
    retention policies, archival, and deletion.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger("lifecycle_manager")
        
        # Storage for policies and assets
        self.policies: Dict[str, LifecyclePolicy] = {}
        self.assets: Dict[str, DataAsset] = {}
        self.pending_actions: Dict[str, LifecycleAction] = {}
        
        # Storage configuration
        self.storage_config = self.config.get('storage', {})
        self.hot_storage_path = Path(self.storage_config.get('hot_path', '/data/hot'))
        self.warm_storage_path = Path(self.storage_config.get('warm_path', '/data/warm'))
        self.cold_storage_path = Path(self.storage_config.get('cold_path', '/data/cold'))
        self.archive_storage_path = Path(self.storage_config.get('archive_path', '/data/archive'))
        
        # Execution settings
        self.batch_size = self.config.get('batch_size', 100)
        self.execution_interval = self.config.get('execution_interval_minutes', 60)
        
        # Safety settings
        self.require_approval_for_deletion = self.config.get('require_approval_for_deletion', True)
        self.max_deletions_per_batch = self.config.get('max_deletions_per_batch', 10)
        
        # Initialize default policies
        self._initialize_default_policies()
    
    def _initialize_default_policies(self):
        """Initialize default lifecycle policies"""
        
        # Public data policy
        self.add_policy(LifecyclePolicy(
            name="public_data_policy",
            description="Standard policy for public data",
            classification=DataClassification.PUBLIC,
            max_retention_days=90,
            delete_after_days=90,
            hot_storage_days=7,
            auto_delete=True,
            compression_enabled=True
        ))
        
        # Internal data policy
        self.add_policy(LifecyclePolicy(
            name="internal_data_policy",
            description="Standard policy for internal data",
            classification=DataClassification.INTERNAL,
            min_retention_days=90,
            max_retention_days=365,
            archive_after_days=30,
            delete_after_days=365,
            hot_storage_days=30,
            warm_storage_days=335,
            backup_required=True,
            auto_archive=True
        ))
        
        # Confidential data policy
        self.add_policy(LifecyclePolicy(
            name="confidential_data_policy",
            description="Policy for confidential data requiring encryption",
            classification=DataClassification.CONFIDENTIAL,
            min_retention_days=365,
            max_retention_days=1095,
            archive_after_days=180,
            hot_storage_days=30,
            warm_storage_days=150,
            cold_storage_days=915,
            encryption_required=True,
            backup_required=True,
            audit_required=True,
            legal_hold_capable=True,
            auto_archive=True
        ))
        
        # Restricted data policy
        self.add_policy(LifecyclePolicy(
            name="restricted_data_policy", 
            description="Policy for restricted data with strict compliance",
            classification=DataClassification.RESTRICTED,
            min_retention_days=1095,
            max_retention_days=2555,
            archive_after_days=365,
            hot_storage_days=30,
            warm_storage_days=335,
            cold_storage_days=2190,
            encryption_required=True,
            backup_required=True,
            audit_required=True,
            immutable_period_days=1095,
            legal_hold_capable=True,
            auto_archive=True,
            auto_delete=False  # Always require manual approval
        ))
        
        # AI model data policy
        self.add_policy(LifecyclePolicy(
            name="ai_model_data_policy",
            description="Policy for AI model training and inference data",
            data_types=[DataType.AI_MODEL],
            min_retention_days=180,
            max_retention_days=1095,
            archive_after_days=90,
            hot_storage_days=30,
            warm_storage_days=60,
            cold_storage_days=1005,
            compression_enabled=True,
            backup_required=True,
            auto_archive=True
        ))
        
        # System logs policy
        self.add_policy(LifecyclePolicy(
            name="system_logs_policy",
            description="Policy for system and audit logs",
            data_types=[DataType.SYSTEM_LOG],
            min_retention_days=30,
            max_retention_days=365,
            archive_after_days=7,
            delete_after_days=365,
            hot_storage_days=7,
            warm_storage_days=23,
            cold_storage_days=335,
            compression_enabled=True,
            auto_archive=True,
            auto_delete=True
        ))
    
    def add_policy(self, policy: LifecyclePolicy):
        """Add a lifecycle policy"""
        self.policies[policy.name] = policy
        self.logger.info(f"Added lifecycle policy: {policy.name}")
    
    def register_data_asset(self, asset: DataAsset) -> bool:
        """Register a data asset for lifecycle management"""
        try:
            # Apply appropriate policy
            policy = self.find_matching_policy(asset)
            if policy:
                asset.retention_policy = policy.name
                
                # Calculate retention dates
                if policy.archive_after_days:
                    asset.archive_after = asset.created_at + timedelta(days=policy.archive_after_days)
                
                if policy.delete_after_days:
                    asset.delete_after = asset.created_at + timedelta(days=policy.delete_after_days)
                elif policy.max_retention_days:
                    asset.delete_after = asset.created_at + timedelta(days=policy.max_retention_days)
            
            # Store asset
            self.assets[asset.id] = asset
            
            # Add audit entry
            self._add_audit_entry(asset, "registered", {"policy": policy.name if policy else None})
            
            self.logger.info(f"Registered data asset: {asset.id} with policy: {asset.retention_policy}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register data asset {asset.id}: {e}")
            return False
    
    def find_matching_policy(self, asset: DataAsset) -> Optional[LifecyclePolicy]:
        """Find the best matching policy for a data asset"""
        
        # Look for exact classification match first
        for policy in self.policies.values():
            if not policy.is_active:
                continue
                
            if policy.classification == asset.classification:
                if not policy.data_types or asset.data_type in policy.data_types:
                    return policy
        
        # Look for data type match
        for policy in self.policies.values():
            if not policy.is_active:
                continue
                
            if policy.data_types and asset.data_type in policy.data_types:
                return policy
        
        # Look for source pattern match
        for policy in self.policies.values():
            if not policy.is_active:
                continue
                
            if policy.source_patterns:
                for pattern in policy.source_patterns:
                    if pattern.lower() in asset.source.lower():
                        return policy
        
        # Default to a basic policy based on classification
        if asset.classification == DataClassification.PUBLIC:
            return self.policies.get("public_data_policy")
        elif asset.classification == DataClassification.INTERNAL:
            return self.policies.get("internal_data_policy")
        elif asset.classification == DataClassification.CONFIDENTIAL:
            return self.policies.get("confidential_data_policy")
        elif asset.classification == DataClassification.RESTRICTED:
            return self.policies.get("restricted_data_policy")
        
        return None
    
    async def evaluate_lifecycle_actions(self) -> List[LifecycleAction]:
        """Evaluate all assets and determine required lifecycle actions"""
        actions = []
        current_time = datetime.utcnow()
        
        for asset in self.assets.values():
            # Skip if asset has legal holds
            if asset.legal_holds:
                continue
            
            policy = self.policies.get(asset.retention_policy)
            if not policy:
                continue
            
            # Check for archival
            if (asset.current_stage == LifecycleStage.ACTIVE and 
                asset.archive_after and 
                current_time >= asset.archive_after):
                
                actions.append(LifecycleAction(
                    asset_id=asset.id,
                    action=RetentionAction.ARCHIVE,
                    scheduled_at=current_time,
                    reason=f"Asset reached archive date: {asset.archive_after}",
                    policy_name=policy.name
                ))
            
            # Check for deletion
            if (asset.delete_after and 
                current_time >= asset.delete_after and
                asset.current_stage not in [LifecycleStage.SCHEDULED_DELETION, LifecycleStage.DELETED]):
                
                actions.append(LifecycleAction(
                    asset_id=asset.id,
                    action=RetentionAction.DELETE,
                    scheduled_at=current_time,
                    reason=f"Asset reached deletion date: {asset.delete_after}",
                    policy_name=policy.name
                ))
            
            # Check for storage tier movement
            days_since_creation = (current_time - asset.created_at).days
            days_since_access = (current_time - (asset.last_accessed or asset.created_at)).days
            
            # Move to warm storage
            if (days_since_access >= policy.hot_storage_days and 
                'hot' in asset.storage_location):
                
                actions.append(LifecycleAction(
                    asset_id=asset.id,
                    action=RetentionAction.KEEP_ACTIVE,
                    scheduled_at=current_time,
                    reason="Move to warm storage tier",
                    metadata={"target_tier": "warm"}
                ))
            
            # Move to cold storage
            if (policy.cold_storage_days and 
                days_since_access >= policy.warm_storage_days and 
                'warm' in asset.storage_location):
                
                actions.append(LifecycleAction(
                    asset_id=asset.id,
                    action=RetentionAction.KEEP_ACTIVE,
                    scheduled_at=current_time,
                    reason="Move to cold storage tier",
                    metadata={"target_tier": "cold"}
                ))
        
        return actions
    
    async def execute_lifecycle_action(self, action: LifecycleAction) -> bool:
        """Execute a single lifecycle action"""
        try:
            action.execution_status = "executing"
            action.executed_at = datetime.utcnow()
            
            asset = self.assets.get(action.asset_id)
            if not asset:
                action.execution_status = "failed"
                action.execution_details["error"] = "Asset not found"
                return False
            
            if action.action == RetentionAction.ARCHIVE:
                success = await self._archive_asset(asset)
            elif action.action == RetentionAction.DELETE:
                success = await self._delete_asset(asset)
            elif action.action == RetentionAction.QUARANTINE:
                success = await self._quarantine_asset(asset)
            elif action.action == RetentionAction.KEEP_ACTIVE:
                success = await self._move_storage_tier(asset, action.metadata.get("target_tier"))
            else:
                success = False
                action.execution_details["error"] = f"Unknown action: {action.action}"
            
            action.execution_status = "completed" if success else "failed"
            
            # Add audit entry
            self._add_audit_entry(asset, action.action.value, {
                "success": success,
                "reason": action.reason,
                "execution_details": action.execution_details
            })
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to execute lifecycle action {action.action} for asset {action.asset_id}: {e}")
            action.execution_status = "failed"
            action.execution_details["error"] = str(e)
            return False
    
    async def _archive_asset(self, asset: DataAsset) -> bool:
        """Archive a data asset"""
        try:
            # Move to archive storage
            archive_path = self.archive_storage_path / asset.id
            
            # Update asset metadata
            asset.current_stage = LifecycleStage.ARCHIVED
            asset.storage_location = str(archive_path)
            
            self.logger.info(f"Archived asset {asset.id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to archive asset {asset.id}: {e}")
            return False
    
    async def _delete_asset(self, asset: DataAsset) -> bool:
        """Delete a data asset (with safety checks)"""
        try:
            # Safety check: require approval for sensitive data
            if (self.require_approval_for_deletion and 
                asset.classification in [DataClassification.CONFIDENTIAL, DataClassification.RESTRICTED]):
                
                asset.current_stage = LifecycleStage.SCHEDULED_DELETION
                self.logger.info(f"Asset {asset.id} scheduled for deletion (approval required)")
                return True
            
            # Perform actual deletion
            asset.current_stage = LifecycleStage.DELETED
            asset.storage_location = ""
            
            self.logger.info(f"Deleted asset {asset.id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete asset {asset.id}: {e}")
            return False
    
    async def _quarantine_asset(self, asset: DataAsset) -> bool:
        """Quarantine a data asset for review"""
        try:
            asset.current_stage = LifecycleStage.QUARANTINED
            self.logger.info(f"Quarantined asset {asset.id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to quarantine asset {asset.id}: {e}")
            return False
    
    async def _move_storage_tier(self, asset: DataAsset, target_tier: str) -> bool:
        """Move asset to different storage tier"""
        try:
            if target_tier == "warm":
                new_location = str(self.warm_storage_path / asset.id)
            elif target_tier == "cold":
                new_location = str(self.cold_storage_path / asset.id)
            else:
                return False
            
            asset.storage_location = new_location
            self.logger.info(f"Moved asset {asset.id} to {target_tier} storage")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to move asset {asset.id} to {target_tier} storage: {e}")
            return False
    
    def _add_audit_entry(self, asset: DataAsset, action: str, details: Dict[str, Any]):
        """Add an audit entry to the asset"""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "action": action,
            "details": details
        }
        asset.audit_trail.append(entry)
    
    def add_legal_hold(self, asset_id: str, hold_id: str, reason: str) -> bool:
        """Add a legal hold to prevent deletion"""
        asset = self.assets.get(asset_id)
        if asset:
            asset.legal_holds.append(hold_id)
            self._add_audit_entry(asset, "legal_hold_added", {"hold_id": hold_id, "reason": reason})
            self.logger.info(f"Added legal hold {hold_id} to asset {asset_id}")
            return True
        return False
    
    def remove_legal_hold(self, asset_id: str, hold_id: str) -> bool:
        """Remove a legal hold"""
        asset = self.assets.get(asset_id)
        if asset and hold_id in asset.legal_holds:
            asset.legal_holds.remove(hold_id)
            self._add_audit_entry(asset, "legal_hold_removed", {"hold_id": hold_id})
            self.logger.info(f"Removed legal hold {hold_id} from asset {asset_id}")
            return True
        return False
    
    def get_lifecycle_statistics(self) -> Dict[str, Any]:
        """Get comprehensive lifecycle statistics"""
        stats = {
            "total_assets": len(self.assets),
            "assets_by_stage": {},
            "assets_by_classification": {},
            "assets_by_policy": {},
            "pending_actions": len(self.pending_actions),
            "legal_holds_active": 0,
            "storage_usage": {},
            "upcoming_actions": []
        }
        
        # Count by stage
        for asset in self.assets.values():
            stage = asset.current_stage.value
            stats["assets_by_stage"][stage] = stats["assets_by_stage"].get(stage, 0) + 1
            
            # Count by classification
            classification = asset.classification.value
            stats["assets_by_classification"][classification] = stats["assets_by_classification"].get(classification, 0) + 1
            
            # Count by policy
            policy = asset.retention_policy or "no_policy"
            stats["assets_by_policy"][policy] = stats["assets_by_policy"].get(policy, 0) + 1
            
            # Count legal holds
            if asset.legal_holds:
                stats["legal_holds_active"] += len(asset.legal_holds)
        
        return stats