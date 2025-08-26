"""
Data Governance Framework
========================

Central orchestrator for data governance, integrating classification,
lifecycle management, compliance, and audit capabilities.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
import json
from pathlib import Path

from .data_classifier import DataClassifier, DataClassification, ClassificationResult
from .lifecycle_manager import DataLifecycleManager, DataAsset, LifecycleAction
from .audit_logger import DataAuditLogger
from .compliance_manager import ComplianceManager
from .lineage_tracker import DataLineageTracker
from .quality_monitor import DataQualityMonitor


@dataclass
class GovernancePolicy:
    """High-level governance policy that coordinates multiple components"""
    name: str
    description: str
    
    # Classification requirements
    auto_classify: bool = True
    classification_confidence_threshold: float = 0.7
    require_manual_review: bool = False
    
    # Lifecycle management
    enforce_retention: bool = True
    auto_archive: bool = True
    auto_delete: bool = False
    
    # Quality requirements
    quality_score_threshold: float = 0.8
    require_quality_validation: bool = True
    
    # Compliance requirements
    gdpr_compliance: bool = False
    ccpa_compliance: bool = False
    hipaa_compliance: bool = False
    sox_compliance: bool = False
    
    # Audit requirements
    audit_all_access: bool = False
    audit_sensitive_data: bool = True
    retain_audit_logs_days: int = 2555  # 7 years
    
    # Data lineage tracking
    track_lineage: bool = True
    lineage_depth: int = 5
    
    # Notification settings
    notify_on_violations: bool = True
    notify_on_policy_changes: bool = True
    
    created_at: datetime = field(default_factory=datetime.utcnow)
    is_active: bool = True


@dataclass
class GovernanceViolation:
    """Represents a governance policy violation"""
    id: str
    policy_name: str
    violation_type: str
    severity: str  # low, medium, high, critical
    description: str
    asset_id: Optional[str] = None
    detected_at: datetime = field(default_factory=datetime.utcnow)
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class DataGovernanceFramework:
    """
    Central data governance framework that orchestrates all data
    governance activities including classification, lifecycle management,
    compliance monitoring, and audit logging.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger("data_governance")
        
        # Initialize components
        self.classifier = DataClassifier()
        self.lifecycle_manager = DataLifecycleManager(self.config.get('lifecycle', {}))
        self.audit_logger = DataAuditLogger(self.config.get('audit', {}))
        self.compliance_manager = ComplianceManager(self.config.get('compliance', {}))
        self.lineage_tracker = DataLineageTracker(self.config.get('lineage', {}))
        self.quality_monitor = DataQualityMonitor(self.config.get('quality', {}))
        
        # Governance state
        self.policies: Dict[str, GovernancePolicy] = {}
        self.violations: Dict[str, GovernanceViolation] = {}
        
        # Processing queues
        self.classification_queue: asyncio.Queue = asyncio.Queue()
        self.lifecycle_queue: asyncio.Queue = asyncio.Queue()
        self.compliance_queue: asyncio.Queue = asyncio.Queue()
        
        # Processing settings
        self.batch_size = self.config.get('batch_size', 100)
        self.processing_interval = self.config.get('processing_interval_minutes', 15)
        
        # Initialize default governance policies
        self._initialize_default_policies()
        
        # Background tasks
        self._background_tasks: List[asyncio.Task] = []
    
    def _initialize_default_policies(self):
        """Initialize default governance policies"""
        
        # Standard enterprise policy
        self.add_policy(GovernancePolicy(
            name="enterprise_standard",
            description="Standard enterprise data governance policy",
            auto_classify=True,
            enforce_retention=True,
            auto_archive=True,
            quality_score_threshold=0.8,
            require_quality_validation=True,
            audit_sensitive_data=True,
            track_lineage=True,
            notify_on_violations=True
        ))
        
        # GDPR compliance policy
        self.add_policy(GovernancePolicy(
            name="gdpr_compliance",
            description="GDPR compliance policy for EU data subjects",
            auto_classify=True,
            classification_confidence_threshold=0.9,
            require_manual_review=True,
            enforce_retention=True,
            auto_archive=True,
            auto_delete=False,  # Require explicit approval
            quality_score_threshold=0.9,
            require_quality_validation=True,
            gdpr_compliance=True,
            audit_all_access=True,
            retain_audit_logs_days=2555,
            track_lineage=True,
            lineage_depth=10,
            notify_on_violations=True
        ))
        
        # Healthcare data policy
        self.add_policy(GovernancePolicy(
            name="healthcare_hipaa",
            description="HIPAA compliant policy for healthcare data",
            auto_classify=True,
            classification_confidence_threshold=0.95,
            require_manual_review=True,
            enforce_retention=True,
            auto_archive=True,
            auto_delete=False,
            quality_score_threshold=0.95,
            require_quality_validation=True,
            hipaa_compliance=True,
            audit_all_access=True,
            retain_audit_logs_days=2555,
            track_lineage=True,
            lineage_depth=15,
            notify_on_violations=True
        ))
        
        # Financial data policy
        self.add_policy(GovernancePolicy(
            name="financial_sox",
            description="SOX compliant policy for financial data",
            auto_classify=True,
            classification_confidence_threshold=0.9,
            require_manual_review=True,
            enforce_retention=True,
            auto_archive=True,
            auto_delete=False,
            quality_score_threshold=0.9,
            require_quality_validation=True,
            sox_compliance=True,
            audit_all_access=True,
            retain_audit_logs_days=2555,
            track_lineage=True,
            lineage_depth=10,
            notify_on_violations=True
        ))
    
    def add_policy(self, policy: GovernancePolicy):
        """Add a governance policy"""
        self.policies[policy.name] = policy
        self.logger.info(f"Added governance policy: {policy.name}")
    
    async def initialize(self) -> bool:
        """Initialize the governance framework"""
        try:
            self.logger.info("Initializing data governance framework")
            
            # Initialize all components
            components = [
                ('audit_logger', self.audit_logger),
                ('compliance_manager', self.compliance_manager), 
                ('lineage_tracker', self.lineage_tracker),
                ('quality_monitor', self.quality_monitor)
            ]
            
            for name, component in components:
                if hasattr(component, 'initialize'):
                    success = await component.initialize()
                    if not success:
                        self.logger.error(f"Failed to initialize {name}")
                        return False
            
            # Start background processing tasks
            await self._start_background_tasks()
            
            self.logger.info("Data governance framework initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize governance framework: {e}")
            return False
    
    async def process_new_data(self, data_id: str, content: str, 
                             metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process new data through the complete governance pipeline
        
        Args:
            data_id: Unique identifier for the data
            content: Data content to process
            metadata: Optional metadata about the data
            
        Returns:
            Dictionary with processing results
        """
        results = {
            'data_id': data_id,
            'processing_timestamp': datetime.utcnow().isoformat(),
            'classification': None,
            'lifecycle_policy_applied': None,
            'quality_score': None,
            'compliance_status': {},
            'violations': [],
            'lineage_recorded': False
        }
        
        try:
            # Step 1: Classify the data
            classification_result = self.classifier.classify_data(content, metadata)
            results['classification'] = {
                'level': classification_result.classification.value,
                'data_types': [dt.value for dt in classification_result.data_types],
                'regulations': [reg.value for reg in classification_result.regulations],
                'confidence': classification_result.confidence,
                'reasoning': classification_result.reasoning
            }
            
            # Step 2: Create data asset for lifecycle management
            asset = DataAsset(
                id=data_id,
                name=metadata.get('name', data_id),
                source=metadata.get('source', 'unknown'),
                data_type=list(classification_result.data_types)[0] if classification_result.data_types else None,
                classification=classification_result.classification,
                created_at=datetime.utcnow(),
                storage_location=metadata.get('storage_location', ''),
                size_bytes=len(content.encode('utf-8')),
                metadata=metadata or {}
            )
            
            # Register asset with lifecycle manager
            lifecycle_success = self.lifecycle_manager.register_data_asset(asset)
            if lifecycle_success:
                results['lifecycle_policy_applied'] = asset.retention_policy
            
            # Step 3: Assess data quality
            quality_score = await self.quality_monitor.assess_quality(data_id, content, metadata)
            results['quality_score'] = quality_score
            
            # Step 4: Check compliance requirements
            compliance_status = await self.compliance_manager.assess_compliance(
                data_id, classification_result, content, metadata
            )
            results['compliance_status'] = compliance_status
            
            # Step 5: Record data lineage
            if metadata and 'source_systems' in metadata:
                lineage_success = await self.lineage_tracker.record_data_flow(
                    data_id, metadata['source_systems'], metadata.get('transformations', [])
                )
                results['lineage_recorded'] = lineage_success
            
            # Step 6: Check for policy violations
            violations = await self._check_policy_violations(data_id, classification_result, quality_score, compliance_status)
            results['violations'] = [v.__dict__ for v in violations]
            
            # Step 7: Log governance activities
            await self.audit_logger.log_data_processing(data_id, results)
            
            self.logger.info(f"Processed data {data_id} through governance pipeline")
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to process data {data_id}: {e}")
            results['error'] = str(e)
            return results
    
    async def _check_policy_violations(self, data_id: str, classification: ClassificationResult,
                                     quality_score: float, compliance_status: Dict[str, Any]) -> List[GovernanceViolation]:
        """Check for governance policy violations"""
        violations = []
        
        # Get applicable policy (for now, use enterprise standard)
        policy = self.policies.get("enterprise_standard")
        if not policy:
            return violations
        
        # Check classification confidence
        if classification.confidence < policy.classification_confidence_threshold:
            violations.append(GovernanceViolation(
                id=f"{data_id}_low_classification_confidence",
                policy_name=policy.name,
                violation_type="classification_confidence",
                severity="medium",
                description=f"Classification confidence {classification.confidence:.2f} below threshold {policy.classification_confidence_threshold}",
                asset_id=data_id,
                metadata={"confidence": classification.confidence, "threshold": policy.classification_confidence_threshold}
            ))
        
        # Check data quality
        if quality_score < policy.quality_score_threshold:
            violations.append(GovernanceViolation(
                id=f"{data_id}_low_quality_score",
                policy_name=policy.name,
                violation_type="data_quality",
                severity="high" if quality_score < 0.5 else "medium",
                description=f"Data quality score {quality_score:.2f} below threshold {policy.quality_score_threshold}",
                asset_id=data_id,
                metadata={"quality_score": quality_score, "threshold": policy.quality_score_threshold}
            ))
        
        # Check compliance failures
        for regulation, status in compliance_status.items():
            if isinstance(status, dict) and not status.get('compliant', True):
                violations.append(GovernanceViolation(
                    id=f"{data_id}_compliance_{regulation}",
                    policy_name=policy.name,
                    violation_type="compliance_failure",
                    severity="critical",
                    description=f"Failed {regulation.upper()} compliance check",
                    asset_id=data_id,
                    metadata={"regulation": regulation, "details": status}
                ))
        
        # Store violations
        for violation in violations:
            self.violations[violation.id] = violation
        
        return violations
    
    async def run_periodic_governance_checks(self):
        """Run periodic governance checks and maintenance"""
        try:
            self.logger.info("Running periodic governance checks")
            
            # Evaluate lifecycle actions
            pending_actions = await self.lifecycle_manager.evaluate_lifecycle_actions()
            
            # Execute approved actions
            for action in pending_actions[:self.batch_size]:
                if action.action.value in ['archive', 'keep_active']:  # Safe actions
                    await self.lifecycle_manager.execute_lifecycle_action(action)
                else:
                    # Queue for manual approval
                    await self.lifecycle_queue.put(action)
            
            # Run compliance audits
            await self.compliance_manager.run_periodic_audits()
            
            # Check data quality across assets
            await self.quality_monitor.run_quality_checks()
            
            # Clean up old violations
            await self._cleanup_resolved_violations()
            
            self.logger.info("Completed periodic governance checks")
            
        except Exception as e:
            self.logger.error(f"Error in periodic governance checks: {e}")
    
    async def _cleanup_resolved_violations(self):
        """Clean up resolved violations older than retention period"""
        retention_cutoff = datetime.utcnow() - timedelta(days=90)  # Keep for 90 days
        
        violations_to_remove = []
        for violation_id, violation in self.violations.items():
            if (violation.resolved_at and 
                violation.resolved_at < retention_cutoff):
                violations_to_remove.append(violation_id)
        
        for violation_id in violations_to_remove:
            del self.violations[violation_id]
        
        if violations_to_remove:
            self.logger.info(f"Cleaned up {len(violations_to_remove)} resolved violations")
    
    async def _start_background_tasks(self):
        """Start background processing tasks"""
        
        # Periodic governance checks
        async def periodic_checks():
            while True:
                try:
                    await self.run_periodic_governance_checks()
                    await asyncio.sleep(self.processing_interval * 60)  # Convert to seconds
                except Exception as e:
                    self.logger.error(f"Error in periodic checks task: {e}")
                    await asyncio.sleep(60)  # Wait 1 minute before retry
        
        # Queue processors
        async def process_lifecycle_queue():
            while True:
                try:
                    action = await self.lifecycle_queue.get()
                    # Process lifecycle action (could involve approval workflow)
                    self.logger.info(f"Processing lifecycle action: {action.action} for {action.asset_id}")
                    self.lifecycle_queue.task_done()
                except Exception as e:
                    self.logger.error(f"Error processing lifecycle queue: {e}")
        
        # Start tasks
        self._background_tasks = [
            asyncio.create_task(periodic_checks()),
            asyncio.create_task(process_lifecycle_queue())
        ]
        
        self.logger.info("Started background governance tasks")
    
    async def shutdown(self):
        """Shutdown the governance framework"""
        try:
            # Cancel background tasks
            for task in self._background_tasks:
                task.cancel()
            
            # Wait for tasks to complete
            if self._background_tasks:
                await asyncio.gather(*self._background_tasks, return_exceptions=True)
            
            # Shutdown components
            components = [self.audit_logger, self.compliance_manager, 
                         self.lineage_tracker, self.quality_monitor]
            
            for component in components:
                if hasattr(component, 'shutdown'):
                    await component.shutdown()
            
            self.logger.info("Data governance framework shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during governance framework shutdown: {e}")
    
    def get_governance_dashboard_data(self) -> Dict[str, Any]:
        """Get data for governance dashboard"""
        
        # Get statistics from all components
        lifecycle_stats = self.lifecycle_manager.get_lifecycle_statistics()
        
        # Count violations by severity
        violation_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        active_violations = [v for v in self.violations.values() if not v.resolved_at]
        
        for violation in active_violations:
            violation_counts[violation.severity] = violation_counts.get(violation.severity, 0) + 1
        
        # Recent activity
        recent_violations = sorted(
            [v for v in self.violations.values() if 
             (datetime.utcnow() - v.detected_at).days <= 7],
            key=lambda x: x.detected_at,
            reverse=True
        )[:10]
        
        dashboard_data = {
            "overview": {
                "total_policies": len(self.policies),
                "active_policies": len([p for p in self.policies.values() if p.is_active]),
                "total_violations": len(self.violations),
                "active_violations": len(active_violations),
                "violation_counts": violation_counts
            },
            "lifecycle": lifecycle_stats,
            "recent_violations": [
                {
                    "id": v.id,
                    "type": v.violation_type,
                    "severity": v.severity,
                    "description": v.description,
                    "detected_at": v.detected_at.isoformat()
                }
                for v in recent_violations
            ],
            "policies": [
                {
                    "name": p.name,
                    "description": p.description,
                    "is_active": p.is_active,
                    "created_at": p.created_at.isoformat()
                }
                for p in self.policies.values()
            ]
        }
        
        return dashboard_data
    
        """Mark a violation as resolved"""
        violation = self.violations.get(violation_id)
        if violation:
            violation.resolved_at = datetime.utcnow()
            self.logger.info(f"Resolved violation {violation_id}")
            return True
