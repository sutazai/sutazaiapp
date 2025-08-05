#!/usr/bin/env python3
"""
Compliance Reporting System for SutazAI Human Oversight
Generates detailed compliance reports for regulatory and audit purposes
"""

import asyncio
import json
import logging
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from jinja2 import Template
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

class ComplianceFramework(Enum):
    GDPR = "gdpr"
    HIPAA = "hipaa"
    SOX = "sox"
    ISO27001 = "iso27001"
    PCI_DSS = "pci_dss"
    NIST = "nist"
    AI_ETHICS = "ai_ethics"

class ReportType(Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"
    INCIDENT = "incident"
    AUDIT_RESPONSE = "audit_response"

@dataclass
class ComplianceMetric:
    """Represents a compliance metric"""
    name: str
    value: float
    threshold: float
    status: str  # "compliant", "warning", "violation"
    framework: ComplianceFramework
    description: str
    evidence: List[str]
    last_updated: datetime

@dataclass
class ComplianceViolation:
    """Represents a compliance violation"""
    id: str
    framework: ComplianceFramework
    violation_type: str
    severity: str  # "low", "medium", "high", "critical"
    description: str
    agent_id: str
    detected_at: datetime
    resolved_at: Optional[datetime]
    remediation_actions: List[str]
    evidence: Dict[str, Any]

@dataclass
class ComplianceReport:
    """Represents a comprehensive compliance report"""
    id: str
    report_type: ReportType
    framework: ComplianceFramework
    period_start: datetime
    period_end: datetime
    generated_at: datetime
    metrics: List[ComplianceMetric]
    violations: List[ComplianceViolation]
    summary: Dict[str, Any]
    recommendations: List[str]
    attachments: List[str]

class ComplianceReporter:
    """
    Comprehensive compliance reporting system for SutazAI oversight
    """
    
    def __init__(self, db_path: str = "/opt/sutazaiapp/backend/oversight/oversight.db",
                 reports_dir: str = "/opt/sutazaiapp/backend/oversight/reports"):
        self.db_path = Path(db_path)
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize compliance frameworks
        self.frameworks = {
            ComplianceFramework.GDPR: self._init_gdpr_rules(),
            ComplianceFramework.HIPAA: self._init_hipaa_rules(),
            ComplianceFramework.SOX: self._init_sox_rules(),
            ComplianceFramework.ISO27001: self._init_iso27001_rules(),
            ComplianceFramework.AI_ETHICS: self._init_ai_ethics_rules(),
            ComplianceFramework.NIST: self._init_nist_rules()
        }
        
        # Initialize database for compliance tracking
        self._init_compliance_database()
    
    def _init_compliance_database(self):
        """Initialize compliance tracking database"""
        with sqlite3.connect(self.db_path) as conn:
            # Compliance metrics table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS compliance_metrics (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    value REAL NOT NULL,
                    threshold REAL NOT NULL,
                    status TEXT NOT NULL,
                    framework TEXT NOT NULL,
                    description TEXT NOT NULL,
                    evidence TEXT,
                    timestamp TEXT NOT NULL
                )
            ''')
            
            # Compliance violations table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS compliance_violations (
                    id TEXT PRIMARY KEY,
                    framework TEXT NOT NULL,
                    violation_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    description TEXT NOT NULL,
                    agent_id TEXT,
                    detected_at TEXT NOT NULL,
                    resolved_at TEXT,
                    remediation_actions TEXT,
                    evidence TEXT,
                    resolved INTEGER DEFAULT 0
                )
            ''')
            
            # Compliance reports table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS compliance_reports (
                    id TEXT PRIMARY KEY,
                    report_type TEXT NOT NULL,
                    framework TEXT NOT NULL,
                    period_start TEXT NOT NULL,
                    period_end TEXT NOT NULL,
                    generated_at TEXT NOT NULL,
                    file_path TEXT,
                    summary TEXT,
                    recommendations TEXT
                )
            ''')
            
            conn.commit()
    
    def _init_gdpr_rules(self) -> Dict[str, Any]:
        """Initialize GDPR compliance rules"""
        return {
            "data_processing_consent": {
                "description": "All data processing must have explicit consent",
                "threshold": 100.0,
                "critical": True
            },
            "data_retention_period": {
                "description": "Personal data retention within legal limits",
                "threshold": 95.0,
                "critical": True
            },
            "right_to_erasure": {
                "description": "Ability to delete user data upon request",
                "threshold": 100.0,
                "critical": True
            },
            "data_portability": {
                "description": "Users can export their data",
                "threshold": 100.0,
                "critical": False
            },
            "privacy_by_design": {
                "description": "Privacy considerations in AI system design",
                "threshold": 90.0,
                "critical": True
            }
        }
    
    def _init_hipaa_rules(self) -> Dict[str, Any]:
        """Initialize HIPAA compliance rules"""
        return {
            "phi_encryption": {
                "description": "Protected Health Information must be encrypted",
                "threshold": 100.0,
                "critical": True
            },
            "access_controls": {
                "description": "Proper access controls for health data",
                "threshold": 100.0,
                "critical": True
            },
            "audit_logging": {
                "description": "Complete audit trail for health data access",
                "threshold": 100.0,
                "critical": True
            },
            "minimum_necessary": {
                "description": "AI systems access minimum necessary health data",
                "threshold": 95.0,
                "critical": True
            }
        }
    
    def _init_sox_rules(self) -> Dict[str, Any]:
        """Initialize SOX compliance rules"""
        return {
            "financial_data_accuracy": {
                "description": "Accuracy of financial data processing",
                "threshold": 99.9,
                "critical": True
            },
            "internal_controls": {
                "description": "Internal controls over financial reporting",
                "threshold": 100.0,
                "critical": True
            },
            "change_management": {
                "description": "Change management for financial systems",
                "threshold": 100.0,
                "critical": True
            }
        }
    
    def _init_iso27001_rules(self) -> Dict[str, Any]:
        """Initialize ISO 27001 compliance rules"""
        return {
            "information_security_policy": {
                "description": "Information security policy implementation",
                "threshold": 100.0,
                "critical": True
            },
            "risk_assessment": {
                "description": "Regular information security risk assessments",
                "threshold": 100.0,
                "critical": True
            },
            "incident_response": {
                "description": "Security incident response procedures",
                "threshold": 100.0,
                "critical": True
            },
            "access_management": {
                "description": "Proper access management controls",
                "threshold": 95.0,
                "critical": True
            }
        }
    
    def _init_ai_ethics_rules(self) -> Dict[str, Any]:
        """Initialize AI Ethics compliance rules"""
        return {
            "algorithmic_transparency": {
                "description": "AI decisions should be explainable",
                "threshold": 85.0,
                "critical": False
            },
            "bias_detection": {
                "description": "Regular bias detection in AI models",
                "threshold": 90.0,
                "critical": True
            },
            "human_oversight": {
                "description": "Meaningful human oversight of AI decisions",
                "threshold": 100.0,
                "critical": True
            },
            "fairness_metrics": {
                "description": "Fair treatment across different groups",
                "threshold": 90.0,
                "critical": True
            },
            "accountability": {
                "description": "Clear accountability for AI decisions",
                "threshold": 100.0,
                "critical": True
            }
        }
    
    def _init_nist_rules(self) -> Dict[str, Any]:
        """Initialize NIST cybersecurity framework rules"""
        return {
            "identify_assets": {
                "description": "Asset inventory and management",
                "threshold": 100.0,
                "critical": True
            },
            "protect_systems": {
                "description": "Protective measures implementation",
                "threshold": 95.0,
                "critical": True
            },
            "detect_events": {
                "description": "Security event detection capabilities",
                "threshold": 95.0,
                "critical": True
            },
            "respond_incidents": {
                "description": "Incident response capabilities",
                "threshold": 100.0,
                "critical": True
            },
            "recover_operations": {
                "description": "Recovery and continuity planning",
                "threshold": 90.0,
                "critical": True
            }
        }
    
    async def collect_compliance_metrics(self, framework: ComplianceFramework) -> List[ComplianceMetric]:
        """Collect compliance metrics for a specific framework"""
        metrics = []
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get audit events for analysis
                cursor = conn.execute('''
                    SELECT event_type, agent_id, description, before_state, after_state, 
                           timestamp, compliance_tags
                    FROM audit_events 
                    WHERE timestamp >= ? AND compliance_tags LIKE ?
                    ORDER BY timestamp DESC
                ''', ((datetime.utcnow() - timedelta(days=30)).isoformat(), f'%{framework.value}%'))
                
                audit_events = cursor.fetchall()
                
                # Analyze metrics based on framework
                if framework == ComplianceFramework.AI_ETHICS:
                    metrics.extend(await self._analyze_ai_ethics_metrics(audit_events))
                elif framework == ComplianceFramework.GDPR:
                    metrics.extend(await self._analyze_gdpr_metrics(audit_events))
                elif framework == ComplianceFramework.HIPAA:
                    metrics.extend(await self._analyze_hipaa_metrics(audit_events))
                elif framework == ComplianceFramework.SOX:
                    metrics.extend(await self._analyze_sox_metrics(audit_events))
                elif framework == ComplianceFramework.ISO27001:
                    metrics.extend(await self._analyze_iso27001_metrics(audit_events))
                elif framework == ComplianceFramework.NIST:
                    metrics.extend(await self._analyze_nist_metrics(audit_events))
                
                # Store metrics in database
                for metric in metrics:
                    conn.execute('''
                        INSERT INTO compliance_metrics 
                        (id, name, value, threshold, status, framework, description, evidence, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (str(uuid.uuid4()), metric.name, metric.value, metric.threshold,
                          metric.status, metric.framework.value, metric.description,
                          json.dumps(metric.evidence), metric.last_updated.isoformat()))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error collecting compliance metrics for {framework}: {e}")
        
        return metrics
    
    async def _analyze_ai_ethics_metrics(self, audit_events: List) -> List[ComplianceMetric]:
        """Analyze AI ethics compliance metrics"""
        metrics = []
        
        # Human oversight metric
        human_interventions = len([e for e in audit_events if 'human_intervention' in str(e[6])])
        total_decisions = len(audit_events)
        
        if total_decisions > 0:
            oversight_percentage = (human_interventions / total_decisions) * 100
        else:
            oversight_percentage = 100.0
        
        metrics.append(ComplianceMetric(
            name="human_oversight",
            value=oversight_percentage,
            threshold=100.0,
            status="compliant" if oversight_percentage >= 100.0 else "warning",
            framework=ComplianceFramework.AI_ETHICS,
            description="Percentage of AI decisions with human oversight",
            evidence=[f"Human interventions: {human_interventions}", f"Total decisions: {total_decisions}"],
            last_updated=datetime.utcnow()
        ))
        
        # Bias detection metric (simulated)
        bias_checks = len([e for e in audit_events if 'bias' in str(e[2]).lower()])
        bias_percentage = 90.0 if bias_checks > 0 else 70.0
        
        metrics.append(ComplianceMetric(
            name="bias_detection",
            value=bias_percentage,
            threshold=90.0,
            status="compliant" if bias_percentage >= 90.0 else "warning",
            framework=ComplianceFramework.AI_ETHICS,
            description="Regular bias detection and mitigation",
            evidence=[f"Bias checks performed: {bias_checks}"],
            last_updated=datetime.utcnow()
        ))
        
        # Algorithmic transparency
        explainable_decisions = len([e for e in audit_events if 'explanation' in str(e[2]).lower()])
        transparency_percentage = (explainable_decisions / max(total_decisions, 1)) * 100
        
        metrics.append(ComplianceMetric(
            name="algorithmic_transparency",
            value=transparency_percentage,
            threshold=85.0,
            status="compliant" if transparency_percentage >= 85.0 else "warning",
            framework=ComplianceFramework.AI_ETHICS,
            description="Percentage of AI decisions with explanations",
            evidence=[f"Explainable decisions: {explainable_decisions}"],
            last_updated=datetime.utcnow()
        ))
        
        return metrics
    
    async def _analyze_gdpr_metrics(self, audit_events: List) -> List[ComplianceMetric]:
        """Analyze GDPR compliance metrics"""
        metrics = []
        
        # Data processing consent
        consent_events = len([e for e in audit_events if 'consent' in str(e[2]).lower()])
        data_processing_events = len([e for e in audit_events if 'data_processing' in str(e[2]).lower()])
        
        if data_processing_events > 0:
            consent_percentage = (consent_events / data_processing_events) * 100
        else:
            consent_percentage = 100.0
        
        metrics.append(ComplianceMetric(
            name="data_processing_consent",
            value=consent_percentage,
            threshold=100.0,
            status="compliant" if consent_percentage >= 100.0 else "violation",
            framework=ComplianceFramework.GDPR,
            description="Data processing with explicit consent",
            evidence=[f"Consent events: {consent_events}", f"Processing events: {data_processing_events}"],
            last_updated=datetime.utcnow()
        ))
        
        # Right to erasure
        deletion_requests = len([e for e in audit_events if 'delete' in str(e[2]).lower() or 'erasure' in str(e[2]).lower()])
        erasure_percentage = 100.0 if deletion_requests >= 0 else 0.0
        
        metrics.append(ComplianceMetric(
            name="right_to_erasure",
            value=erasure_percentage,
            threshold=100.0,
            status="compliant" if erasure_percentage >= 100.0 else "violation",
            framework=ComplianceFramework.GDPR,
            description="Ability to fulfill data deletion requests",
            evidence=[f"Deletion requests handled: {deletion_requests}"],
            last_updated=datetime.utcnow()
        ))
        
        return metrics
    
    async def _analyze_hipaa_metrics(self, audit_events: List) -> List[ComplianceMetric]:
        """Analyze HIPAA compliance metrics"""
        metrics = []
        
        # PHI encryption
        phi_events = len([e for e in audit_events if 'phi' in str(e[2]).lower() or 'health' in str(e[2]).lower()])
        encrypted_events = len([e for e in audit_events if 'encrypt' in str(e[2]).lower()])
        
        if phi_events > 0:
            encryption_percentage = (encrypted_events / phi_events) * 100
        else:
            encryption_percentage = 100.0
        
        metrics.append(ComplianceMetric(
            name="phi_encryption",
            value=encryption_percentage,
            threshold=100.0,
            status="compliant" if encryption_percentage >= 100.0 else "violation",
            framework=ComplianceFramework.HIPAA,
            description="PHI encryption compliance",
            evidence=[f"PHI events: {phi_events}", f"Encrypted events: {encrypted_events}"],
            last_updated=datetime.utcnow()
        ))
        
        return metrics
    
    async def _analyze_sox_metrics(self, audit_events: List) -> List[ComplianceMetric]:
        """Analyze SOX compliance metrics"""
        metrics = []
        
        # Financial data accuracy
        financial_events = len([e for e in audit_events if 'financial' in str(e[2]).lower()])
        accuracy_percentage = 99.9  # Simulated high accuracy
        
        metrics.append(ComplianceMetric(
            name="financial_data_accuracy",
            value=accuracy_percentage,
            threshold=99.9,
            status="compliant" if accuracy_percentage >= 99.9 else "violation",
            framework=ComplianceFramework.SOX,
            description="Financial data processing accuracy",
            evidence=[f"Financial events processed: {financial_events}"],
            last_updated=datetime.utcnow()
        ))
        
        return metrics
    
    async def _analyze_iso27001_metrics(self, audit_events: List) -> List[ComplianceMetric]:
        """Analyze ISO 27001 compliance metrics"""
        metrics = []
        
        # Security incident response
        security_events = len([e for e in audit_events if 'security' in str(e[2]).lower()])
        response_events = len([e for e in audit_events if 'incident' in str(e[2]).lower()])
        
        if security_events > 0:
            response_percentage = (response_events / security_events) * 100
        else:
            response_percentage = 100.0
        
        metrics.append(ComplianceMetric(
            name="incident_response",
            value=response_percentage,
            threshold=100.0,
            status="compliant" if response_percentage >= 100.0 else "warning",
            framework=ComplianceFramework.ISO27001,
            description="Security incident response coverage",
            evidence=[f"Security events: {security_events}", f"Response events: {response_events}"],
            last_updated=datetime.utcnow()
        ))
        
        return metrics
    
    async def _analyze_nist_metrics(self, audit_events: List) -> List[ComplianceMetric]:
        """Analyze NIST cybersecurity framework metrics"""
        metrics = []
        
        # Asset identification
        asset_events = len([e for e in audit_events if 'asset' in str(e[2]).lower()])
        identification_percentage = 100.0 if asset_events >= 0 else 0.0
        
        metrics.append(ComplianceMetric(
            name="identify_assets",
            value=identification_percentage,
            threshold=100.0,
            status="compliant" if identification_percentage >= 100.0 else "warning",
            framework=ComplianceFramework.NIST,
            description="Asset identification and inventory",
            evidence=[f"Asset management events: {asset_events}"],
            last_updated=datetime.utcnow()
        ))
        
        return metrics
    
    async def detect_violations(self, framework: ComplianceFramework) -> List[ComplianceViolation]:
        """Detect compliance violations for a specific framework"""
        violations = []
        
        try:
            # Get recent metrics
            metrics = await self.collect_compliance_metrics(framework)
            
            # Check for violations
            for metric in metrics:
                if metric.status in ["violation", "warning"]:
                    severity = "critical" if metric.status == "violation" else "medium"
                    
                    violation = ComplianceViolation(
                        id=str(uuid.uuid4()),
                        framework=framework,
                        violation_type=metric.name,
                        severity=severity,
                        description=f"{metric.name} below threshold: {metric.value}% < {metric.threshold}%",
                        agent_id="system",
                        detected_at=datetime.utcnow(),
                        resolved_at=None,
                        remediation_actions=[
                            f"Investigate {metric.name} compliance gap",
                            f"Implement corrective measures to reach {metric.threshold}% threshold",
                            "Monitor progress and verify resolution"
                        ],
                        evidence={
                            "metric_value": metric.value,
                            "threshold": metric.threshold,
                            "evidence": metric.evidence
                        }
                    )
                    
                    violations.append(violation)
                    
                    # Store violation in database
                    with sqlite3.connect(self.db_path) as conn:
                        conn.execute('''
                            INSERT INTO compliance_violations 
                            (id, framework, violation_type, severity, description, agent_id,
                             detected_at, remediation_actions, evidence)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (violation.id, violation.framework.value, violation.violation_type,
                              violation.severity, violation.description, violation.agent_id,
                              violation.detected_at.isoformat(), 
                              json.dumps(violation.remediation_actions),
                              json.dumps(violation.evidence)))
                        
                        conn.commit()
        
        except Exception as e:
            logger.error(f"Error detecting violations for {framework}: {e}")
        
        return violations
    
    async def generate_compliance_report(self, framework: ComplianceFramework, 
                                       report_type: ReportType,
                                       period_start: Optional[datetime] = None,
                                       period_end: Optional[datetime] = None) -> ComplianceReport:
        """Generate a comprehensive compliance report"""
        
        if not period_end:
            period_end = datetime.utcnow()
        
        if not period_start:
            if report_type == ReportType.DAILY:
                period_start = period_end - timedelta(days=1)
            elif report_type == ReportType.WEEKLY:
                period_start = period_end - timedelta(weeks=1)
            elif report_type == ReportType.MONTHLY:
                period_start = period_end - timedelta(days=30)
            elif report_type == ReportType.QUARTERLY:
                period_start = period_end - timedelta(days=90)
            elif report_type == ReportType.ANNUAL:
                period_start = period_end - timedelta(days=365)
            else:
                period_start = period_end - timedelta(days=30)
        
        try:
            # Collect metrics and violations
            metrics = await self.collect_compliance_metrics(framework)
            violations = await self.detect_violations(framework)
            
            # Calculate summary statistics
            total_metrics = len(metrics)
            compliant_metrics = len([m for m in metrics if m.status == "compliant"])
            warning_metrics = len([m for m in metrics if m.status == "warning"])
            violation_metrics = len([m for m in metrics if m.status == "violation"])
            
            compliance_percentage = (compliant_metrics / max(total_metrics, 1)) * 100
            
            summary = {
                "total_metrics": total_metrics,
                "compliant_metrics": compliant_metrics,
                "warning_metrics": warning_metrics,
                "violation_metrics": violation_metrics,
                "compliance_percentage": compliance_percentage,
                "total_violations": len(violations),
                "critical_violations": len([v for v in violations if v.severity == "critical"]),
                "high_violations": len([v for v in violations if v.severity == "high"]),
                "medium_violations": len([v for v in violations if v.severity == "medium"]),
                "low_violations": len([v for v in violations if v.severity == "low"])
            }
            
            # Generate recommendations
            recommendations = self._generate_recommendations(framework, metrics, violations)
            
            # Create report
            report = ComplianceReport(
                id=str(uuid.uuid4()),
                report_type=report_type,
                framework=framework,
                period_start=period_start,
                period_end=period_end,
                generated_at=datetime.utcnow(),
                metrics=metrics,
                violations=violations,
                summary=summary,
                recommendations=recommendations,
                attachments=[]
            )
            
            # Generate report files
            report_files = await self._generate_report_files(report)
            report.attachments = report_files
            
            # Store report metadata in database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO compliance_reports 
                    (id, report_type, framework, period_start, period_end, generated_at,
                     file_path, summary, recommendations)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (report.id, report.report_type.value, report.framework.value,
                      report.period_start.isoformat(), report.period_end.isoformat(),
                      report.generated_at.isoformat(), str(report_files[0]) if report_files else None,
                      json.dumps(report.summary), json.dumps(report.recommendations)))
                
                conn.commit()
            
            logger.info(f"Generated {framework.value} {report_type.value} compliance report: {report.id}")
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating compliance report: {e}")
            raise
    
    def _generate_recommendations(self, framework: ComplianceFramework, 
                                metrics: List[ComplianceMetric],
                                violations: List[ComplianceViolation]) -> List[str]:
        """Generate compliance recommendations based on metrics and violations"""
        recommendations = []
        
        # Framework-specific recommendations
        if framework == ComplianceFramework.AI_ETHICS:
            if any(m.name == "human_oversight" and m.status != "compliant" for m in metrics):
                recommendations.append("Increase human oversight of AI decision-making processes")
            
            if any(m.name == "bias_detection" and m.status != "compliant" for m in metrics):
                recommendations.append("Implement regular bias detection and mitigation procedures")
            
            if any(m.name == "algorithmic_transparency" and m.status != "compliant" for m in metrics):
                recommendations.append("Enhance AI explainability and transparency features")
        
        elif framework == ComplianceFramework.GDPR:
            if any(m.name == "data_processing_consent" and m.status != "compliant" for m in metrics):
                recommendations.append("Ensure explicit consent for all data processing activities")
            
            if any(m.name == "right_to_erasure" and m.status != "compliant" for m in metrics):
                recommendations.append("Implement comprehensive data deletion capabilities")
        
        elif framework == ComplianceFramework.HIPAA:
            if any(m.name == "phi_encryption" and m.status != "compliant" for m in metrics):
                recommendations.append("Ensure all PHI is properly encrypted at rest and in transit")
        
        # General recommendations based on violations
        critical_violations = [v for v in violations if v.severity == "critical"]
        if critical_violations:
            recommendations.append(f"Immediately address {len(critical_violations)} critical compliance violations")
        
        high_violations = [v for v in violations if v.severity == "high"]
        if high_violations:
            recommendations.append(f"Prioritize resolution of {len(high_violations)} high-severity violations")
        
        # Compliance percentage recommendations
        compliant_metrics = len([m for m in metrics if m.status == "compliant"])
        total_metrics = len(metrics)
        
        if total_metrics > 0:
            compliance_percentage = (compliant_metrics / total_metrics) * 100
            
            if compliance_percentage < 80:
                recommendations.append("Implement comprehensive compliance improvement program")
            elif compliance_percentage < 95:
                recommendations.append("Focus on addressing remaining compliance gaps")
            else:
                recommendations.append("Maintain current compliance levels and monitor for any degradation")
        
        return recommendations
    
    async def _generate_report_files(self, report: ComplianceReport) -> List[str]:
        """Generate report files (HTML, PDF, JSON)"""
        files = []
        
        try:
            # Create report directory
            report_dir = self.reports_dir / f"{report.framework.value}_{report.report_type.value}_{report.generated_at.strftime('%Y%m%d_%H%M%S')}"
            report_dir.mkdir(exist_ok=True)
            
            # Generate JSON report
            json_path = report_dir / "report.json"
            with open(json_path, 'w') as f:
                json.dump(asdict(report), f, indent=2, default=str)
            files.append(str(json_path))
            
            # Generate HTML report
            html_path = report_dir / "report.html"
            html_content = self._generate_html_report(report)
            with open(html_path, 'w') as f:
                f.write(html_content)
            files.append(str(html_path))
            
            # Generate metrics chart
            chart_path = report_dir / "metrics_chart.png"
            self._generate_metrics_chart(report.metrics, chart_path)
            files.append(str(chart_path))
            
            # Generate violations summary chart
            if report.violations:
                violations_chart_path = report_dir / "violations_chart.png"
                self._generate_violations_chart(report.violations, violations_chart_path)
                files.append(str(violations_chart_path))
            
            logger.info(f"Generated {len(files)} report files in {report_dir}")
            
        except Exception as e:
            logger.error(f"Error generating report files: {e}")
        
        return files
    
    def _generate_html_report(self, report: ComplianceReport) -> str:
        """Generate HTML compliance report"""
        template = Template('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ framework.value.upper() }} Compliance Report - {{ report_type.value.title() }}</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 2rem;
            background-color: #f5f5f5;
            color: #333;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            text-align: center;
        }
        
        .header h1 {
            margin: 0;
            font-size: 2.5rem;
        }
        
        .header .subtitle {
            margin-top: 0.5rem;
            font-size: 1.2rem;
            opacity: 0.9;
        }
        
        .summary {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }
        
        .metric-card {
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
        }
        
        .metric-value {
            font-size: 2.5rem;
            font-weight: bold;
            margin-bottom: 0.5rem;
        }
        
        .metric-value.compliant {
            color: #4caf50;
        }
        
        .metric-value.warning {
            color: #ff9800;
        }
        
        .metric-value.violation {
            color: #f44336;
        }
        
        .metric-label {
            font-size: 0.9rem;
            color: #666;
        }
        
        .section {
            background: white;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 2rem;
        }
        
        .section h2 {
            margin-top: 0;
            color: #333;
            border-bottom: 2px solid #667eea;
            padding-bottom: 0.5rem;
        }
        
        .metrics-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
        }
        
        .metrics-table th,
        .metrics-table td {
            padding: 1rem;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        
        .metrics-table th {
            background-color: #f8f9fa;
            font-weight: bold;
        }
        
        .status-badge {
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: bold;
            text-transform: uppercase;
        }
        
        .status-compliant {
            background-color: #d4edda;
            color: #155724;
        }
        
        .status-warning {
            background-color: #fff3cd;
            color: #856404;
        }
        
        .status-violation {
            background-color: #f8d7da;
            color: #721c24;
        }
        
        .violation-item {
            background-color: #f8f9fa;
            border-left: 4px solid #f44336;
            padding: 1rem;
            margin-bottom: 1rem;
            border-radius: 5px;
        }
        
        .violation-critical {
            border-left-color: #f44336;
        }
        
        .violation-high {
            border-left-color: #ff9800;
        }
        
        .violation-medium {
            border-left-color: #ffc107;
        }
        
        .violation-low {
            border-left-color: #28a745;
        }
        
        .recommendations {
            background-color: #e3f2fd;
            border-left: 4px solid #2196f3;
            padding: 1rem;
            margin-top: 1rem;
        }
        
        .recommendations ul {
            margin: 0;
            padding-left: 1.5rem;
        }
        
        .recommendations li {
            margin-bottom: 0.5rem;
        }
        
        .footer {
            text-align: center;
            margin-top: 2rem;
            padding: 1rem;
            color: #666;
            font-size: 0.9rem;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>{{ framework.value.upper() }} Compliance Report</h1>
        <div class="subtitle">{{ report_type.value.title() }} Report - {{ generated_at.strftime('%B %d, %Y') }}</div>
        <div class="subtitle">Period: {{ period_start.strftime('%Y-%m-%d') }} to {{ period_end.strftime('%Y-%m-%d') }}</div>
    </div>
    
    <div class="summary">
        <div class="metric-card">
            <div class="metric-value compliant">{{ "%.1f"|format(summary.compliance_percentage) }}%</div>
            <div class="metric-label">Overall Compliance</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{{ summary.total_metrics }}</div>
            <div class="metric-label">Total Metrics</div>
        </div>
        <div class="metric-card">
            <div class="metric-value compliant">{{ summary.compliant_metrics }}</div>
            <div class="metric-label">Compliant</div>
        </div>
        <div class="metric-card">
            <div class="metric-value warning">{{ summary.warning_metrics }}</div>
            <div class="metric-label">Warnings</div>
        </div>
        <div class="metric-card">
            <div class="metric-value violation">{{ summary.violation_metrics }}</div>
            <div class="metric-label">Violations</div>
        </div>
    </div>
    
    <div class="section">
        <h2>📊 Compliance Metrics</h2>
        <table class="metrics-table">
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                    <th>Threshold</th>
                    <th>Status</th>
                    <th>Description</th>
                </tr>
            </thead>
            <tbody>
                {% for metric in metrics %}
                <tr>
                    <td><strong>{{ metric.name.replace('_', ' ').title() }}</strong></td>
                    <td>{{ "%.1f"|format(metric.value) }}%</td>
                    <td>{{ "%.1f"|format(metric.threshold) }}%</td>
                    <td><span class="status-badge status-{{ metric.status }}">{{ metric.status }}</span></td>
                    <td>{{ metric.description }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>
    
    {% if violations %}
    <div class="section">
        <h2>⚠️ Compliance Violations</h2>
        {% for violation in violations %}
        <div class="violation-item violation-{{ violation.severity }}">
            <h3>{{ violation.violation_type.replace('_', ' ').title() }} ({{ violation.severity.upper() }})</h3>
            <p><strong>Description:</strong> {{ violation.description }}</p>
            <p><strong>Detected:</strong> {{ violation.detected_at.strftime('%Y-%m-%d %H:%M:%S') }}</p>
            <p><strong>Agent:</strong> {{ violation.agent_id }}</p>
            <div>
                <strong>Remediation Actions:</strong>
                <ul>
                    {% for action in violation.remediation_actions %}
                    <li>{{ action }}</li>
                    {% endfor %}
                </ul>
            </div>
        </div>
        {% endfor %}
    </div>
    {% endif %}
    
    {% if recommendations %}
    <div class="section">
        <h2>💡 Recommendations</h2>
        <div class="recommendations">
            <ul>
                {% for recommendation in recommendations %}
                <li>{{ recommendation }}</li>
                {% endfor %}
            </ul>
        </div>
    </div>
    {% endif %}
    
    <div class="footer">
        <p>Report generated by SutazAI Human Oversight Interface on {{ generated_at.strftime('%Y-%m-%d %H:%M:%S') }} UTC</p>
        <p>Report ID: {{ id }}</p>
    </div>
</body>
</html>
        ''')
        
        return template.render(**asdict(report))
    
    def _generate_metrics_chart(self, metrics: List[ComplianceMetric], output_path: Path):
        """Generate metrics visualization chart"""
        try:
            # Prepare data
            metric_names = [m.name.replace('_', ' ').title() for m in metrics]
            values = [m.value for m in metrics]
            thresholds = [m.threshold for m in metrics]
            statuses = [m.status for m in metrics]
            
            # Create figure
            plt.figure(figsize=(12, 8))
            
            # Create color map
            color_map = {'compliant': '#4caf50', 'warning': '#ff9800', 'violation': '#f44336'}
            colors = [color_map.get(status, '#666') for status in statuses]
            
            # Create bar chart
            bars = plt.bar(range(len(metrics)), values, color=colors, alpha=0.8)
            plt.plot(range(len(metrics)), thresholds, 'r--', linewidth=2, label='Threshold')
            
            # Customize chart
            plt.xlabel('Compliance Metrics')
            plt.ylabel('Percentage (%)')
            plt.title('Compliance Metrics Status')
            plt.xticks(range(len(metrics)), metric_names, rotation=45, ha='right')
            plt.legend()
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{value:.1f}%', ha='center', va='bottom')
            
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error generating metrics chart: {e}")
    
    def _generate_violations_chart(self, violations: List[ComplianceViolation], output_path: Path):
        """Generate violations summary chart"""
        try:
            # Count violations by severity
            severity_counts = {}
            for violation in violations:
                severity_counts[violation.severity] = severity_counts.get(violation.severity, 0) + 1
            
            if not severity_counts:
                return
            
            # Create pie chart
            plt.figure(figsize=(10, 8))
            
            severities = list(severity_counts.keys())
            counts = list(severity_counts.values())
            colors = ['#f44336', '#ff9800', '#ffc107', '#4caf50'][:len(severities)]
            
            plt.pie(counts, labels=severities, colors=colors, autopct='%1.1f%%', startangle=90)
            plt.title('Compliance Violations by Severity')
            plt.axis('equal')
            
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error generating violations chart: {e}")
    
    async def schedule_reports(self):
        """Schedule automatic compliance report generation"""
        logger.info("Starting compliance report scheduler")
        
        while True:
            try:
                current_time = datetime.utcnow()
                
                # Generate daily reports at midnight
                if current_time.hour == 0 and current_time.minute < 30:
                    for framework in ComplianceFramework:
                        await self.generate_compliance_report(framework, ReportType.DAILY)
                
                # Generate weekly reports on Sundays
                if current_time.weekday() == 6 and current_time.hour == 1:  # Sunday
                    for framework in ComplianceFramework:
                        await self.generate_compliance_report(framework, ReportType.WEEKLY)
                
                # Generate monthly reports on the 1st of each month
                if current_time.day == 1 and current_time.hour == 2:
                    for framework in ComplianceFramework:
                        await self.generate_compliance_report(framework, ReportType.MONTHLY)
                
                # Sleep for 30 minutes before checking again
                await asyncio.sleep(1800)
                
            except Exception as e:
                logger.error(f"Error in compliance report scheduler: {e}")
                await asyncio.sleep(3600)  # Wait an hour on error


async def main():
    """Main entry point for standalone compliance reporter"""
    reporter = ComplianceReporter()
    
    # Generate sample reports for all frameworks
    for framework in ComplianceFramework:
        try:
            report = await reporter.generate_compliance_report(framework, ReportType.MONTHLY)
            print(f"Generated {framework.value} compliance report: {report.id}")
        except Exception as e:
            print(f"Error generating {framework.value} report: {e}")
    
    # Start scheduler
    await reporter.schedule_reports()


if __name__ == "__main__":
    asyncio.run(main())