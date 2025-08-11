"""
Compliance Manager
=================

Manages regulatory compliance including GDPR, CCPA, HIPAA, and SOX.
Provides automated compliance checks, reporting, and violation detection.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
import json
import re

from .data_classifier import DataClassification, DataType, RegulationScope, ClassificationResult


class ComplianceStatus(Enum):
    """Compliance status levels"""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    REQUIRES_REVIEW = "requires_review"
    UNKNOWN = "unknown"


class ViolationType(Enum):
    """Types of compliance violations"""
    DATA_RETENTION = "data_retention"
    ACCESS_CONTROL = "access_control"
    CONSENT_MANAGEMENT = "consent_management"
    DATA_MINIMIZATION = "data_minimization"
    ENCRYPTION_REQUIRED = "encryption_required"
    AUDIT_TRAIL = "audit_trail"
    RIGHT_TO_DELETION = "right_to_deletion"
    DATA_PORTABILITY = "data_portability"
    BREACH_NOTIFICATION = "breach_notification"
    PRIVACY_IMPACT = "privacy_impact"


@dataclass
class ComplianceRule:
    """Defines a compliance rule for a specific regulation"""
    id: str
    regulation: RegulationScope
    name: str
    description: str
    
    # Applicability criteria
    data_types: Optional[Set[DataType]] = None
    classifications: Optional[Set[DataClassification]] = None
    jurisdictions: Optional[Set[str]] = None
    
    # Rule logic
    conditions: Dict[str, Any] = field(default_factory=dict)
    requirements: Dict[str, Any] = field(default_factory=dict)
    
    # Violation settings
    violation_type: ViolationType = ViolationType.DATA_RETENTION
    severity: str = "medium"  # low, medium, high, critical
    
    # Metadata
    legal_reference: Optional[str] = None
    implementation_guide: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    is_active: bool = True


@dataclass
class ComplianceViolation:
    """Represents a compliance violation"""
    id: str
    rule_id: str
    regulation: RegulationScope
    violation_type: ViolationType
    severity: str
    
    # Context
    data_id: Optional[str] = None
    description: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    
    # Resolution tracking
    detected_at: datetime = field(default_factory=datetime.utcnow)
    resolved_at: Optional[datetime] = None
    resolution_method: Optional[str] = None
    resolution_notes: Optional[str] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComplianceReport:
    """Comprehensive compliance report"""
    report_id: str
    regulation: RegulationScope
    generated_at: datetime = field(default_factory=datetime.utcnow)
    
    # Coverage summary
    total_data_assets: int = 0
    compliant_assets: int = 0
    non_compliant_assets: int = 0
    requires_review_assets: int = 0
    
    # Violations
    total_violations: int = 0
    critical_violations: int = 0
    high_violations: int = 0
    medium_violations: int = 0
    low_violations: int = 0
    resolved_violations: int = 0
    
    # Detailed findings
    violations: List[ComplianceViolation] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    # Risk assessment
    overall_risk_level: str = "medium"
    risk_factors: List[str] = field(default_factory=list)
    
    # Metadata
    report_period_start: Optional[datetime] = None
    report_period_end: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ComplianceManager:
    """
    Manages regulatory compliance across multiple frameworks
    including GDPR, CCPA, HIPAA, and SOX.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger("compliance_manager")
        
        # Compliance rules by regulation
        self.rules: Dict[RegulationScope, List[ComplianceRule]] = {}
        self.violations: Dict[str, ComplianceViolation] = {}
        
        # Configuration
        self.enabled_regulations = set(self.config.get('enabled_regulations', [
            'gdpr', 'ccpa', 'hipaa', 'sox'
        ]))
        
        # Processing settings
        self.batch_size = self.config.get('batch_size', 100)
        self.audit_interval_hours = self.config.get('audit_interval_hours', 24)
        
        # Initialize compliance rules
        self._initialize_compliance_rules()
    
    async def initialize(self) -> bool:
        """Initialize the compliance manager"""
        try:
            self.logger.info("Initializing compliance manager")
            
            # Validate configuration
            for regulation in self.enabled_regulations:
                if regulation.upper() not in [r.value.upper() for r in RegulationScope]:
                    self.logger.warning(f"Unknown regulation: {regulation}")
            
            self.logger.info(f"Compliance manager initialized for regulations: {', '.join(self.enabled_regulations)}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize compliance manager: {e}")
            return False
    
    def _initialize_compliance_rules(self):
        """Initialize compliance rules for supported regulations"""
        
        # GDPR Rules
        if 'gdpr' in self.enabled_regulations:
            self.rules[RegulationScope.GDPR] = self._create_gdpr_rules()
        
        # CCPA Rules
        if 'ccpa' in self.enabled_regulations:
            self.rules[RegulationScope.CCPA] = self._create_ccpa_rules()
        
        # HIPAA Rules
        if 'hipaa' in self.enabled_regulations:
            self.rules[RegulationScope.HIPAA] = self._create_hipaa_rules()
        
        # SOX Rules
        if 'sox' in self.enabled_regulations:
            self.rules[RegulationScope.SOX] = self._create_sox_rules()
    
    def _create_gdpr_rules(self) -> List[ComplianceRule]:
        """Create GDPR compliance rules"""
        rules = []
        
        # Data retention limits
        rules.append(ComplianceRule(
            id="gdpr_retention_limit",
            regulation=RegulationScope.GDPR,
            name="Data Retention Limit",
            description="Personal data must not be kept longer than necessary",
            data_types={DataType.PII},
            requirements={
                "max_retention_days": 1095,  # 3 years default
                "require_justification": True,
                "automatic_deletion": True
            },
            violation_type=ViolationType.DATA_RETENTION,
            severity="high",
            legal_reference="GDPR Article 5(1)(e)"
        ))
        
        # Consent management
        rules.append(ComplianceRule(
            id="gdpr_consent_required",
            regulation=RegulationScope.GDPR,
            name="Lawful Basis for Processing",
            description="Must have lawful basis for processing personal data",
            data_types={DataType.PII},
            requirements={
                "consent_required": True,
                "consent_recorded": True,
                "withdrawal_mechanism": True
            },
            violation_type=ViolationType.CONSENT_MANAGEMENT,
            severity="critical",
            legal_reference="GDPR Article 6"
        ))
        
        # Right to deletion
        rules.append(ComplianceRule(
            id="gdpr_right_to_deletion",
            regulation=RegulationScope.GDPR,
            name="Right to Erasure",
            description="Must support right to deletion/erasure",
            data_types={DataType.PII},
            requirements={
                "deletion_capability": True,
                "response_time_days": 30,
                "cascade_deletion": True
            },
            violation_type=ViolationType.RIGHT_TO_DELETION,
            severity="high",
            legal_reference="GDPR Article 17"
        ))
        
        # Data minimization
        rules.append(ComplianceRule(
            id="gdpr_data_minimization",
            regulation=RegulationScope.GDPR,
            name="Data Minimization",
            description="Only collect data that is necessary for the purpose",
            data_types={DataType.PII},
            requirements={
                "purpose_limitation": True,
                "necessity_justification": True,
                "regular_review": True
            },
            violation_type=ViolationType.DATA_MINIMIZATION,
            severity="medium",
            legal_reference="GDPR Article 5(1)(c)"
        ))
        
        # Encryption requirements
        rules.append(ComplianceRule(
            id="gdpr_encryption",
            regulation=RegulationScope.GDPR,
            name="Security Measures",
            description="Appropriate technical measures to protect personal data",
            data_types={DataType.PII},
            classifications={DataClassification.CONFIDENTIAL, DataClassification.RESTRICTED},
            requirements={
                "encryption_at_rest": True,
                "encryption_in_transit": True,
                "access_controls": True
            },
            violation_type=ViolationType.ENCRYPTION_REQUIRED,
            severity="high",
            legal_reference="GDPR Article 32"
        ))
        
        return rules
    
    def _create_ccpa_rules(self) -> List[ComplianceRule]:
        """Create CCPA compliance rules"""
        rules = []
        
        # Right to know
        rules.append(ComplianceRule(
            id="ccpa_right_to_know",
            regulation=RegulationScope.CCPA,
            name="Right to Know",
            description="Consumers have right to know what personal information is collected",
            data_types={DataType.PII},
            requirements={
                "disclosure_capability": True,
                "data_inventory": True,
                "response_time_days": 45
            },
            violation_type=ViolationType.DATA_PORTABILITY,
            severity="medium",
            legal_reference="CCPA Section 1798.100"
        ))
        
        # Right to delete
        rules.append(ComplianceRule(
            id="ccpa_right_to_delete",
            regulation=RegulationScope.CCPA,
            name="Right to Delete",
            description="Consumers have right to delete personal information",
            data_types={DataType.PII},
            requirements={
                "deletion_capability": True,
                "response_time_days": 45,
                "verification_process": True
            },
            violation_type=ViolationType.RIGHT_TO_DELETION,
            severity="high",
            legal_reference="CCPA Section 1798.105"
        ))
        
        return rules
    
    def _create_hipaa_rules(self) -> List[ComplianceRule]:
        """Create HIPAA compliance rules"""
        rules = []
        
        # PHI encryption
        rules.append(ComplianceRule(
            id="hipaa_phi_encryption",
            regulation=RegulationScope.HIPAA,
            name="PHI Encryption Required",
            description="Protected Health Information must be encrypted",
            data_types={DataType.PHI},
            requirements={
                "encryption_at_rest": True,
                "encryption_in_transit": True,
                "key_management": True
            },
            violation_type=ViolationType.ENCRYPTION_REQUIRED,
            severity="critical",
            legal_reference="HIPAA Security Rule 164.312(a)(2)(iv)"
        ))
        
        # Access controls
        rules.append(ComplianceRule(
            id="hipaa_access_control",
            regulation=RegulationScope.HIPAA,
            name="Access Control",
            description="PHI access must be controlled and logged",
            data_types={DataType.PHI},
            requirements={
                "role_based_access": True,
                "access_logging": True,
                "regular_review": True
            },
            violation_type=ViolationType.ACCESS_CONTROL,
            severity="high",
            legal_reference="HIPAA Security Rule 164.312(a)"
        ))
        
        # Retention requirements
        rules.append(ComplianceRule(
            id="hipaa_retention",
            regulation=RegulationScope.HIPAA,
            name="Record Retention",
            description="PHI records must be retained for required period",
            data_types={DataType.PHI},
            requirements={
                "min_retention_days": 2190,  # 6 years
                "secure_storage": True,
                "disposal_procedures": True
            },
            violation_type=ViolationType.DATA_RETENTION,
            severity="medium",
            legal_reference="HIPAA Privacy Rule 164.530(j)"
        ))
        
        return rules
    
    def _create_sox_rules(self) -> List[ComplianceRule]:
        """Create SOX compliance rules"""
        rules = []
        
        # Financial data retention
        rules.append(ComplianceRule(
            id="sox_financial_retention",
            regulation=RegulationScope.SOX,
            name="Financial Record Retention",
            description="Financial records must be retained for 7 years",
            data_types={DataType.FINANCIAL},
            requirements={
                "min_retention_days": 2555,  # 7 years
                "immutable_storage": True,
                "audit_trail": True
            },
            violation_type=ViolationType.DATA_RETENTION,
            severity="critical",
            legal_reference="SOX Section 802"
        ))
        
        # Audit trail requirements
        rules.append(ComplianceRule(
            id="sox_audit_trail",
            regulation=RegulationScope.SOX,
            name="Audit Trail Requirements",
            description="All financial data changes must be audited",
            data_types={DataType.FINANCIAL},
            requirements={
                "comprehensive_logging": True,
                "immutable_logs": True,
                "retention_7_years": True
            },
            violation_type=ViolationType.AUDIT_TRAIL,
            severity="high",
            legal_reference="SOX Section 404"
        ))
        
        return rules
    
    async def assess_compliance(self, data_id: str, classification: ClassificationResult,
                              content: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Assess compliance for a specific data asset
        
        Args:
            data_id: Unique identifier for the data
            classification: Data classification result
            content: Data content (may be sample/hash for large data)
            metadata: Additional metadata about the data
            
        Returns:
            Dictionary with compliance assessment results
        """
        assessment_results = {}
        
        try:
            # Check each applicable regulation
            for regulation in classification.regulations:
                if regulation in self.rules:
                    regulation_results = await self._assess_regulation_compliance(
                        data_id, regulation, classification, content, metadata
                    )
                    assessment_results[regulation.value] = regulation_results
            
            # If no specific regulations apply, check general rules
            if not assessment_results:
                # Apply general data protection rules based on classification
                if classification.classification in [DataClassification.CONFIDENTIAL, DataClassification.RESTRICTED]:
                    general_results = await self._assess_general_compliance(
                        data_id, classification, content, metadata
                    )
                    assessment_results['general'] = general_results
            
            self.logger.debug(f"Completed compliance assessment for data {data_id}")
            return assessment_results
            
        except Exception as e:
            self.logger.error(f"Failed to assess compliance for data {data_id}: {e}")
            return {"error": str(e)}
    
    async def _assess_regulation_compliance(self, data_id: str, regulation: RegulationScope,
                                          classification: ClassificationResult,
                                          content: str, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess compliance for a specific regulation"""
        
        results = {
            "regulation": regulation.value,
            "overall_compliant": True,
            "violations": [],
            "warnings": [],
            "recommendations": []
        }
        
        rules = self.rules.get(regulation, [])
        
        for rule in rules:
            if not rule.is_active:
                continue
            
            # Check if rule applies to this data
            if not self._rule_applies_to_data(rule, classification, metadata):
                continue
            
            # Evaluate rule compliance
            compliance_check = await self._evaluate_rule_compliance(
                data_id, rule, classification, content, metadata
            )
            
            if not compliance_check["compliant"]:
                results["overall_compliant"] = False
                
                # Create violation record
                violation = ComplianceViolation(
                    id=f"{data_id}_{rule.id}_{int(datetime.utcnow().timestamp())}",
                    rule_id=rule.id,
                    regulation=regulation,
                    violation_type=rule.violation_type,
                    severity=rule.severity,
                    data_id=data_id,
                    description=compliance_check["description"],
                    details=compliance_check["details"],
                    metadata={"rule_name": rule.name, "legal_reference": rule.legal_reference}
                )
                
                # Store violation
                self.violations[violation.id] = violation
                results["violations"].append(violation.__dict__)
            
            # Add warnings and recommendations
            if compliance_check.get("warnings"):
                results["warnings"].extend(compliance_check["warnings"])
            
            if compliance_check.get("recommendations"):
                results["recommendations"].extend(compliance_check["recommendations"])
        
        return results
    
    def _rule_applies_to_data(self, rule: ComplianceRule, classification: ClassificationResult,
                            metadata: Optional[Dict[str, Any]]) -> bool:
        """Check if a compliance rule applies to the given data"""
        
        # Check data types
        if rule.data_types:
            if not classification.data_types.intersection(rule.data_types):
                return False
        
        # Check classifications
        if rule.classifications:
            if classification.classification not in rule.classifications:
                return False
        
        # Check jurisdictions (if metadata contains jurisdiction info)
        if rule.jurisdictions and metadata:
            data_jurisdiction = metadata.get('jurisdiction', '').lower()
            if data_jurisdiction and data_jurisdiction not in [j.lower() for j in rule.jurisdictions]:
                return False
        
        return True
    
    async def _evaluate_rule_compliance(self, data_id: str, rule: ComplianceRule,
                                      classification: ClassificationResult, content: str,
                                      metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate compliance for a specific rule"""
        
        result = {
            "compliant": True,
            "description": "",
            "details": {},
            "warnings": [],
            "recommendations": []
        }
        
        try:
            # Evaluate based on rule type
            if rule.violation_type == ViolationType.DATA_RETENTION:
                result = await self._check_retention_compliance(data_id, rule, metadata)
            
            elif rule.violation_type == ViolationType.ENCRYPTION_REQUIRED:
                result = await self._check_encryption_compliance(data_id, rule, metadata)
            
            elif rule.violation_type == ViolationType.ACCESS_CONTROL:
                result = await self._check_access_control_compliance(data_id, rule, metadata)
            
            elif rule.violation_type == ViolationType.CONSENT_MANAGEMENT:
                result = await self._check_consent_compliance(data_id, rule, metadata)
            
            elif rule.violation_type == ViolationType.DATA_MINIMIZATION:
                result = await self._check_minimization_compliance(data_id, rule, content, metadata)
            
            elif rule.violation_type == ViolationType.AUDIT_TRAIL:
                result = await self._check_audit_trail_compliance(data_id, rule, metadata)
            
            else:
                # Generic compliance check
                result = await self._check_generic_compliance(data_id, rule, metadata)
            
        except Exception as e:
            self.logger.error(f"Error evaluating rule {rule.id} for data {data_id}: {e}")
            result = {
                "compliant": False,
                "description": f"Error evaluating compliance: {str(e)}",
                "details": {"error": str(e)},
                "warnings": [],
                "recommendations": []
            }
        
        return result
    
    async def _check_retention_compliance(self, data_id: str, rule: ComplianceRule, 
                                        metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Check data retention compliance"""
        
        result = {
            "compliant": True,
            "description": "Data retention compliance verified",
            "details": {},
            "warnings": [],
            "recommendations": []
        }
        
        if not metadata:
            return {
                "compliant": False,
                "description": "Cannot verify retention compliance without metadata",
                "details": {},
                "warnings": [],
                "recommendations": ["Add metadata with creation date and retention policy"]
            }
        
        created_at = metadata.get('created_at')
        if not created_at:
            result["compliant"] = False
            result["description"] = "Missing creation date for retention assessment"
            result["recommendations"].append("Add creation timestamp to data metadata")
            return result
        
        # Parse creation date
        if isinstance(created_at, str):
            try:
                created_at = datetime.fromisoformat(created_at)
            except ValueError:
                result["compliant"] = False
                result["description"] = "Invalid creation date format"
                return result
        
        # Check retention requirements
        requirements = rule.requirements
        current_age_days = (datetime.utcnow() - created_at).days
        
        # Check minimum retention
        min_retention = requirements.get('min_retention_days', 0)
        if current_age_days < min_retention:
            result["warnings"].append(f"Data is {current_age_days} days old, minimum retention is {min_retention} days")
        
        # Check maximum retention
        max_retention = requirements.get('max_retention_days')
        if max_retention and current_age_days > max_retention:
            result["compliant"] = False
            result["description"] = f"Data exceeds maximum retention period ({max_retention} days)"
            result["details"]["current_age_days"] = current_age_days
            result["details"]["max_retention_days"] = max_retention
            result["recommendations"].append("Schedule data for deletion or archival")
        
        return result
    
    async def _check_encryption_compliance(self, data_id: str, rule: ComplianceRule,
                                         metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Check encryption compliance"""
        
        result = {
            "compliant": True,
            "description": "Encryption compliance verified",
            "details": {},
            "warnings": [],
            "recommendations": []
        }
        
        requirements = rule.requirements
        
        # Check encryption at rest
        if requirements.get('encryption_at_rest'):
            encrypted_at_rest = metadata.get('encrypted_at_rest', False) if metadata else False
            if not encrypted_at_rest:
                result["compliant"] = False
                result["description"] = "Data not encrypted at rest"
                result["recommendations"].append("Enable encryption at rest for sensitive data")
        
        # Check encryption in transit
        if requirements.get('encryption_in_transit'):
            encrypted_in_transit = metadata.get('encrypted_in_transit', False) if metadata else False
            if not encrypted_in_transit:
                result["compliant"] = False
                result["description"] = "Data not encrypted in transit"
                result["recommendations"].append("Ensure all data transfers use encryption")
        
        return result
    
    async def _check_access_control_compliance(self, data_id: str, rule: ComplianceRule,
                                             metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Check access control compliance"""
        
        result = {
            "compliant": True,
            "description": "Access control compliance verified",
            "details": {},
            "warnings": [],
            "recommendations": []
        }
        
        requirements = rule.requirements
        
        # Check role-based access
        if requirements.get('role_based_access'):
            has_rbac = metadata.get('access_control_enabled', False) if metadata else False
            if not has_rbac:
                result["compliant"] = False
                result["description"] = "Role-based access control not implemented"
                result["recommendations"].append("Implement role-based access controls")
        
        # Check access logging
        if requirements.get('access_logging'):
            has_logging = metadata.get('access_logging_enabled', False) if metadata else False
            if not has_logging:
                result["compliant"] = False
                result["description"] = "Access logging not enabled"
                result["recommendations"].append("Enable comprehensive access logging")
        
        return result
    
    async def _check_consent_compliance(self, data_id: str, rule: ComplianceRule,
                                      metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Check consent management compliance"""
        
        result = {
            "compliant": True,
            "description": "Consent compliance verified",
            "details": {},
            "warnings": [],
            "recommendations": []
        }
        
        requirements = rule.requirements
        
        # Check if consent is required and present
        if requirements.get('consent_required', True):
            consent_status = metadata.get('consent_status') if metadata else None
            if not consent_status or consent_status != 'granted':
                result["compliant"] = False
                result["description"] = "Valid consent not found"
                result["recommendations"].append("Obtain and record valid consent")
        
        return result
    
    async def _check_minimization_compliance(self, data_id: str, rule: ComplianceRule,
                                           content: str, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Check data minimization compliance"""
        
        result = {
            "compliant": True,
            "description": "Data minimization compliance verified",
            "details": {},
            "warnings": [],
            "recommendations": []
        }
        
        # This is a simplified check - in practice would be more sophisticated
        requirements = rule.requirements
        
        if requirements.get('purpose_limitation'):
            purpose = metadata.get('purpose') if metadata else None
            if not purpose:
                result["warnings"].append("Data purpose not documented")
                result["recommendations"].append("Document the specific purpose for data collection")
        
        return result
    
    async def _check_audit_trail_compliance(self, data_id: str, rule: ComplianceRule,
                                          metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Check audit trail compliance"""
        
        result = {
            "compliant": True,
            "description": "Audit trail compliance verified",
            "details": {},
            "warnings": [],
            "recommendations": []
        }
        
        requirements = rule.requirements
        
        if requirements.get('comprehensive_logging'):
            has_audit_trail = metadata.get('audit_trail_enabled', False) if metadata else False
            if not has_audit_trail:
                result["compliant"] = False
                result["description"] = "Comprehensive audit trail not enabled"
                result["recommendations"].append("Enable comprehensive audit logging")
        
        return result
    
    async def _check_generic_compliance(self, data_id: str, rule: ComplianceRule,
                                      metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Generic compliance check for rules not specifically handled"""
        
        return {
            "compliant": True,
            "description": f"Generic compliance check for rule {rule.name}",
            "details": {"rule_id": rule.id},
            "warnings": ["Generic compliance check - manual review recommended"],
            "recommendations": ["Review compliance requirements manually"]
        }
    
    async def _assess_general_compliance(self, data_id: str, classification: ClassificationResult,
                                       content: str, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess general data protection compliance"""
        
        results = {
            "regulation": "general",
            "overall_compliant": True,
            "violations": [],
            "warnings": [],
            "recommendations": []
        }
        
        # Basic security requirements for sensitive data
        if classification.classification in [DataClassification.CONFIDENTIAL, DataClassification.RESTRICTED]:
            if metadata:
                if not metadata.get('encrypted_at_rest', False):
                    results["warnings"].append("Sensitive data should be encrypted at rest")
                    results["recommendations"].append("Enable encryption for sensitive data")
                
                if not metadata.get('access_control_enabled', False):
                    results["warnings"].append("Access controls should be implemented for sensitive data")
                    results["recommendations"].append("Implement role-based access controls")
        
        return results
    
    async def run_periodic_audits(self):
        """Run periodic compliance audits"""
        try:
            self.logger.info("Running periodic compliance audits")
            
            # This would typically iterate through all data assets
            # For now, we'll just log the activity
            
            audit_results = {
                "audit_timestamp": datetime.utcnow().isoformat(),
                "regulations_checked": list(self.rules.keys()),
                "total_violations": len(self.violations),
                "active_violations": len([v for v in self.violations.values() if not v.resolved_at])
            }
            
            self.logger.info(f"Periodic audit completed: {audit_results}")
            
        except Exception as e:
            self.logger.error(f"Error in periodic compliance audit: {e}")
    
    def generate_compliance_report(self, regulation: RegulationScope,
                                 start_date: Optional[datetime] = None,
                                 end_date: Optional[datetime] = None) -> ComplianceReport:
        """Generate a comprehensive compliance report"""
        
        if not end_date:
            end_date = datetime.utcnow()
        if not start_date:
            start_date = end_date - timedelta(days=30)
        
        report = ComplianceReport(
            report_id=f"compliance_{regulation.value}_{int(datetime.utcnow().timestamp())}",
            regulation=regulation,
            report_period_start=start_date,
            report_period_end=end_date
        )
        
        # Filter violations for this regulation and time period
        relevant_violations = [
            v for v in self.violations.values()
            if v.regulation == regulation and
            start_date <= v.detected_at <= end_date
        ]
        
        report.violations = relevant_violations
        report.total_violations = len(relevant_violations)
        
        # Count by severity
        for violation in relevant_violations:
            if violation.severity == "critical":
                report.critical_violations += 1
            elif violation.severity == "high":
                report.high_violations += 1
            elif violation.severity == "medium":
                report.medium_violations += 1
            elif violation.severity == "low":
                report.low_violations += 1
            
            if violation.resolved_at:
                report.resolved_violations += 1
        
        # Determine overall risk level
        if report.critical_violations > 0:
            report.overall_risk_level = "critical"
        elif report.high_violations > 5:
            report.overall_risk_level = "high" 
        elif report.medium_violations > 10:
            report.overall_risk_level = "medium"
        else:
            report.overall_risk_level = "low"
        
        # Add recommendations
        if report.critical_violations > 0:
            report.recommendations.append("Address critical violations immediately")
        if report.high_violations > 0:
            report.recommendations.append("Prioritize resolution of high-severity violations")
        
        return report
    
    def resolve_violation(self, violation_id: str, resolution_method: str, 
                         resolution_notes: str) -> bool:
        """Mark a compliance violation as resolved"""
        
        violation = self.violations.get(violation_id)
        if violation:
            violation.resolved_at = datetime.utcnow()
            violation.resolution_method = resolution_method
            violation.resolution_notes = resolution_notes
            
            self.logger.info(f"Resolved compliance violation {violation_id}: {resolution_method}")
            return True
        
        return False
    
    def get_compliance_statistics(self) -> Dict[str, Any]:
        """Get compliance statistics across all regulations"""
        
        stats = {
            "total_violations": len(self.violations),
            "active_violations": 0,
            "resolved_violations": 0,
            "violations_by_regulation": {},
            "violations_by_severity": {},
            "violations_by_type": {},
            "recent_violations": [],
            "enabled_regulations": list(self.enabled_regulations)
        }
        
        # Calculate statistics
        recent_cutoff = datetime.utcnow() - timedelta(days=7)
        
        for violation in self.violations.values():
            # Count active vs resolved
            if violation.resolved_at:
                stats["resolved_violations"] += 1
            else:
                stats["active_violations"] += 1
            
            # Count by regulation
            reg = violation.regulation.value
            stats["violations_by_regulation"][reg] = stats["violations_by_regulation"].get(reg, 0) + 1
            
            # Count by severity
            severity = violation.severity
            stats["violations_by_severity"][severity] = stats["violations_by_severity"].get(severity, 0) + 1
            
            # Count by type
            v_type = violation.violation_type.value
            stats["violations_by_type"][v_type] = stats["violations_by_type"].get(v_type, 0) + 1
            
            # Recent violations
            if violation.detected_at >= recent_cutoff:
                stats["recent_violations"].append({
                    "id": violation.id,
                    "regulation": violation.regulation.value,
                    "type": violation.violation_type.value,
                    "severity": violation.severity,
                    "description": violation.description,
                    "detected_at": violation.detected_at.isoformat()
                })
        
        # Sort recent violations by date
        stats["recent_violations"].sort(key=lambda x: x["detected_at"], reverse=True)
        stats["recent_violations"] = stats["recent_violations"][:20]  # Top 20
        
        return stats
    
    async def shutdown(self):
        """Shutdown the compliance manager"""
        try:
            self.logger.info("Shutting down compliance manager")
            # Any cleanup tasks would go here
            self.logger.info("Compliance manager shutdown complete")
        except Exception as e:
            self.logger.error(f"Error during compliance manager shutdown: {e}")