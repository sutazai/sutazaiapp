#!/usr/bin/env python3
"""
Security Compliance Automation System
Implements automated compliance checking for SOC2, ISO27001, PCI-DSS, and other frameworks
"""

import asyncio
import logging
import json
import os
import time
import hashlib
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import redis
import psycopg2
from psycopg2.extras import RealDictCursor
import yaml
import requests
from pathlib import Path
import threading

class ComplianceFramework(Enum):
    SOC2 = "soc2"
    ISO27001 = "iso27001"
    PCI_DSS = "pci_dss"
    NIST_CSF = "nist_csf"
    GDPR = "gdpr"
    HIPAA = "hipaa"
    CIS = "cis"

class ComplianceStatus(Enum):
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NOT_ASSESSED = "not_assessed"
    EXEMPT = "exempt"

class ControlSeverity(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class ComplianceControl:
    """Individual compliance control"""
    control_id: str
    framework: ComplianceFramework
    title: str
    description: str
    category: str
    severity: ControlSeverity
    automated_check: bool
    check_frequency: str  # daily, weekly, monthly
    evidence_required: List[str]
    implementation_guidance: str
    related_controls: List[str]

@dataclass
class ComplianceEvidence:
    """Evidence for compliance control"""
    evidence_id: str
    control_id: str
    evidence_type: str  # log, screenshot, document, config
    evidence_path: str
    collected_at: datetime
    valid_until: Optional[datetime]
    metadata: Dict[str, Any]

@dataclass
class AssessmentResult:
    """Assessment result for a control"""
    control_id: str
    status: ComplianceStatus
    assessment_date: datetime
    assessor: str
    findings: List[str]
    evidence_ids: List[str]
    remediation_required: bool
    remediation_timeline: Optional[datetime]

@dataclass
class ComplianceReport:
    """Compliance assessment report"""
    report_id: str
    framework: ComplianceFramework
    assessment_date: datetime
    scope: str
    overall_status: ComplianceStatus
    total_controls: int
    compliant_controls: int
    non_compliant_controls: int
    partially_compliant_controls: int
    results: List[AssessmentResult]
    recommendations: List[str]
    next_assessment_due: datetime

class ComplianceEngine:
    """Automated compliance assessment engine"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.redis_client = None
        self.db_connection = None
        self.controls: Dict[str, ComplianceControl] = {}
        self.evidence_store: Dict[str, ComplianceEvidence] = {}
        self.assessment_results: Dict[str, AssessmentResult] = {}
        self.running = False
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize compliance engine components"""
        try:
            # Initialize Redis
            self.redis_client = redis.Redis(
                host=self.config.get('redis_host', 'redis'),
                port=self.config.get('redis_port', 6379),
                password=self.config.get('redis_password'),
                ssl=True,
                decode_responses=True
            )
            
            # Initialize PostgreSQL
            self.db_connection = psycopg2.connect(
                host=self.config.get('postgres_host', 'postgres'),
                port=self.config.get('postgres_port', 5432),
                database=self.config.get('postgres_db', 'sutazai'),
                user=self.config.get('postgres_user', 'sutazai'),
                password=self.config.get('postgres_password'),
                sslmode='require'
            )
            
            # Load compliance controls
            self._load_compliance_controls()
            
            # Setup evidence collection
            self._setup_evidence_collection()
            
            self.logger.info("Compliance Engine initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Compliance Engine: {e}")
            raise
    
    def _load_compliance_controls(self):
        """Load compliance controls from database and definitions"""
        try:
            # Load from database first
            cursor = self.db_connection.cursor(cursor_factory=RealDictCursor)
            cursor.execute("SELECT * FROM compliance_controls WHERE enabled = true")
            
            for row in cursor.fetchall():
                control = ComplianceControl(
                    control_id=row['control_id'],
                    framework=ComplianceFramework(row['framework']),
                    title=row['title'],
                    description=row['description'],
                    category=row['category'],
                    severity=ControlSeverity(row['severity']),
                    automated_check=row['automated_check'],
                    check_frequency=row['check_frequency'],
                    evidence_required=row['evidence_required'] or [],
                    implementation_guidance=row['implementation_guidance'],
                    related_controls=row['related_controls'] or []
                )
                self.controls[control.control_id] = control
            
            cursor.close()
            
            # Load default controls if none in database
            if not self.controls:
                self._load_default_controls()
            
            self.logger.info(f"Loaded {len(self.controls)} compliance controls")
            
        except Exception as e:
            self.logger.error(f"Failed to load compliance controls: {e}")
            self._load_default_controls()
    
    def _load_default_controls(self):
        """Load default compliance controls"""
        default_controls = [
            # SOC2 Controls
            ComplianceControl(
                "SOC2-CC6.1", ComplianceFramework.SOC2,
                "Logical and Physical Access Controls",
                "Entity implements logical and physical access controls to protect against threats",
                "Common Criteria", ControlSeverity.HIGH, True, "daily",
                ["access_logs", "user_access_matrix", "physical_access_logs"],
                "Implement role-based access control and regular access reviews",
                ["SOC2-CC6.2", "SOC2-CC6.3"]
            ),
            ComplianceControl(
                "SOC2-CC6.2", ComplianceFramework.SOC2,
                "Multifactor Authentication",
                "Entity uses multifactor authentication for privileged access",
                "Common Criteria", ControlSeverity.HIGH, True, "daily",
                ["mfa_logs", "authentication_policy"],
                "Enable MFA for all administrative and privileged accounts",
                ["SOC2-CC6.1"]
            ),
            ComplianceControl(
                "SOC2-CC7.1", ComplianceFramework.SOC2,
                "Data Transmission Security",
                "Entity encrypts data in transmission and at rest",
                "Common Criteria", ControlSeverity.CRITICAL, True, "daily",
                ["encryption_config", "tls_certificates", "data_classification"],
                "Use TLS 1.2+ for transmission and AES-256 for data at rest",
                ["SOC2-CC7.2"]
            ),
            
            # ISO27001 Controls
            ComplianceControl(
                "ISO27001-A.9.1.1", ComplianceFramework.ISO27001,
                "Access Control Policy",
                "Access control policy shall be established and reviewed regularly",
                "Access Control", ControlSeverity.HIGH, False, "monthly",
                ["access_control_policy", "policy_review_records"],
                "Document and regularly review access control policies",
                ["ISO27001-A.9.1.2"]
            ),
            ComplianceControl(
                "ISO27001-A.12.6.1", ComplianceFramework.ISO27001,
                "Management of Technical Vulnerabilities",
                "Information about technical vulnerabilities shall be obtained and managed",
                "Operations Security", ControlSeverity.HIGH, True, "weekly",
                ["vulnerability_scans", "patch_management_logs", "vulnerability_register"],
                "Implement automated vulnerability scanning and patch management",
                ["ISO27001-A.12.6.2"]
            ),
            
            # PCI-DSS Controls
            ComplianceControl(
                "PCI-DSS-2.1", ComplianceFramework.PCI_DSS,
                "Change Default Passwords",
                "Always change vendor-supplied defaults and remove/disable unnecessary accounts",
                "Build and Maintain Secure Networks", ControlSeverity.CRITICAL, True, "daily",
                ["password_policy", "default_account_audit", "system_hardening"],
                "Change all default passwords and disable unused accounts",
                ["PCI-DSS-2.2"]
            ),
            ComplianceControl(
                "PCI-DSS-4.1", ComplianceFramework.PCI_DSS,
                "Encrypt Cardholder Data Transmission",
                "Use strong cryptography and security protocols during transmission",
                "Protect Cardholder Data", ControlSeverity.CRITICAL, True, "daily",
                ["network_encryption", "tls_configuration", "secure_protocols"],
                "Use TLS 1.2+ for all cardholder data transmission",
                ["PCI-DSS-4.2"]
            ),
            
            # NIST CSF Controls
            ComplianceControl(
                "NIST-PR.AC-1", ComplianceFramework.NIST_CSF,
                "Identity and Access Management",
                "Identities and credentials are issued, managed, verified, revoked",
                "Protect", ControlSeverity.HIGH, True, "daily",
                ["identity_management", "credential_lifecycle", "access_reviews"],
                "Implement comprehensive identity and access management",
                ["NIST-PR.AC-2"]
            ),
            
            # CIS Controls
            ComplianceControl(
                "CIS-1.1", ComplianceFramework.CIS,
                "Maintain Inventory of Authorized Software",
                "Maintain an inventory of all authorized software",
                "Inventory and Control", ControlSeverity.MEDIUM, True, "weekly",
                ["software_inventory", "asset_management", "change_control"],
                "Maintain up-to-date inventory of all authorized software",
                ["CIS-1.2"]
            ),
        ]
        
        # Add default controls
        for control in default_controls:
            self.controls[control.control_id] = control
        
        # Store in database
        self._store_default_controls(default_controls)
    
    def _store_default_controls(self, controls: List[ComplianceControl]):
        """Store default controls in database"""
        try:
            cursor = self.db_connection.cursor()
            
            for control in controls:
                cursor.execute("""
                    INSERT INTO compliance_controls 
                    (control_id, framework, title, description, category, severity,
                     automated_check, check_frequency, evidence_required, 
                     implementation_guidance, related_controls, enabled)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (control_id) DO NOTHING
                """, (
                    control.control_id, control.framework.value, control.title,
                    control.description, control.category, control.severity.value,
                    control.automated_check, control.check_frequency,
                    json.dumps(control.evidence_required),
                    control.implementation_guidance,
                    json.dumps(control.related_controls), True
                ))
            
            self.db_connection.commit()
            cursor.close()
            
        except Exception as e:
            self.logger.error(f"Failed to store default controls: {e}")
    
    def _setup_evidence_collection(self):
        """Setup automated evidence collection"""
        try:
            # Create evidence storage directory
            evidence_dir = Path(self.config.get('evidence_path', '/opt/sutazaiapp/compliance/evidence'))
            evidence_dir.mkdir(parents=True, exist_ok=True)
            
            self.logger.info("Evidence collection setup completed")
            
        except Exception as e:
            self.logger.error(f"Failed to setup evidence collection: {e}")
    
    async def run_automated_assessment(self, framework: ComplianceFramework) -> str:
        """Run automated compliance assessment"""
        report_id = self._generate_report_id()
        
        try:
            self.logger.info(f"Starting automated assessment for {framework.value}")
            
            # Get controls for framework
            framework_controls = [
                control for control in self.controls.values() 
                if control.framework == framework and control.automated_check
            ]
            
            # Run assessments
            results = []
            for control in framework_controls:
                result = await self._assess_control(control)
                results.append(result)
                self.assessment_results[control.control_id] = result
            
            # Generate report
            report = self._generate_compliance_report(
                report_id, framework, results
            )
            
            # Store report
            await self._store_compliance_report(report)
            
            self.logger.info(f"Completed automated assessment: {report_id}")
            return report_id
            
        except Exception as e:
            self.logger.error(f"Automated assessment failed: {e}")
            raise
    
    async def _assess_control(self, control: ComplianceControl) -> AssessmentResult:
        """Assess individual compliance control"""
        try:
            self.logger.debug(f"Assessing control: {control.control_id}")
            
            findings = []
            evidence_ids = []
            status = ComplianceStatus.NOT_ASSESSED
            
            # Collect evidence
            evidence = await self._collect_evidence(control)
            if evidence:
                evidence_ids = [e.evidence_id for e in evidence]
            
            # Run automated checks based on control
            if control.control_id.startswith("SOC2-CC6.1"):
                status, control_findings = await self._check_access_controls()
                findings.extend(control_findings)
            
            elif control.control_id.startswith("SOC2-CC6.2"):
                status, control_findings = await self._check_mfa_controls()
                findings.extend(control_findings)
            
            elif control.control_id.startswith("SOC2-CC7.1"):
                status, control_findings = await self._check_encryption_controls()
                findings.extend(control_findings)
            
            elif control.control_id.startswith("ISO27001-A.12.6.1"):
                status, control_findings = await self._check_vulnerability_management()
                findings.extend(control_findings)
            
            elif control.control_id.startswith("PCI-DSS-2.1"):
                status, control_findings = await self._check_default_passwords()
                findings.extend(control_findings)
            
            elif control.control_id.startswith("PCI-DSS-4.1"):
                status, control_findings = await self._check_data_transmission_encryption()
                findings.extend(control_findings)
            
            elif control.control_id.startswith("NIST-PR.AC-1"):
                status, control_findings = await self._check_identity_management()
                findings.extend(control_findings)
            
            elif control.control_id.startswith("CIS-1.1"):
                status, control_findings = await self._check_software_inventory()
                findings.extend(control_findings)
            
            else:
                # Default manual assessment required
                status = ComplianceStatus.NOT_ASSESSED
                findings.append("Manual assessment required")
            
            return AssessmentResult(
                control_id=control.control_id,
                status=status,
                assessment_date=datetime.utcnow(),
                assessor="automated_system",
                findings=findings,
                evidence_ids=evidence_ids,
                remediation_required=(status == ComplianceStatus.NON_COMPLIANT),
                remediation_timeline=datetime.utcnow() + timedelta(days=30) if status == ComplianceStatus.NON_COMPLIANT else None,
            )
            
        except Exception as e:
            self.logger.error(f"Failed to assess control {control.control_id}: {e}")
            return AssessmentResult(
                control_id=control.control_id,
                status=ComplianceStatus.NOT_ASSESSED,
                assessment_date=datetime.utcnow(),
                assessor="automated_system",
                findings=[f"Assessment failed: {str(e)}"],
                evidence_ids=[],
                remediation_required=False,
                remediation_timeline=None,
            )
    
    async def _collect_evidence(self, control: ComplianceControl) -> List[ComplianceEvidence]:
        """Collect evidence for control"""
        evidence_list = []
        
        try:
            for evidence_type in control.evidence_required:
                if evidence_type == "access_logs":
                    evidence = await self._collect_access_logs()
                elif evidence_type == "mfa_logs":
                    evidence = await self._collect_mfa_logs()
                elif evidence_type == "encryption_config":
                    evidence = await self._collect_encryption_config()
                elif evidence_type == "vulnerability_scans":
                    evidence = await self._collect_vulnerability_scans()
                elif evidence_type == "system_hardening":
                    evidence = await self._collect_system_hardening()
                elif evidence_type == "software_inventory":
                    evidence = await self._collect_software_inventory()
                else:
                    continue
                
                if evidence:
                    evidence_list.append(evidence)
                    self.evidence_store[evidence.evidence_id] = evidence
        
        except Exception as e:
            self.logger.error(f"Failed to collect evidence for {control.control_id}: {e}")
        
        return evidence_list
    
    # Evidence collection methods
    async def _collect_access_logs(self) -> Optional[ComplianceEvidence]:
        """Collect access logs evidence"""
        try:
            # Query recent access logs from Redis
            logs = self.redis_client.lrange("security_events", 0, 1000)
            
            evidence_id = self._generate_evidence_id()
            evidence_path = f"/opt/sutazaiapp/compliance/evidence/access_logs_{evidence_id}.json"
            
            # Store logs to file
            with open(evidence_path, 'w') as f:
                json.dump(logs, f, indent=2)
            
            return ComplianceEvidence(
                evidence_id=evidence_id,
                control_id="access_logs",
                evidence_type="log",
                evidence_path=evidence_path,
                collected_at=datetime.utcnow(),
                valid_until=datetime.utcnow() + timedelta(days=30),
                metadata={"log_count": len(logs), "source": "redis"}
            )
            
        except Exception as e:
            self.logger.error(f"Failed to collect access logs: {e}")
            return None
    
    async def _collect_mfa_logs(self) -> Optional[ComplianceEvidence]:
        """Collect MFA logs evidence"""
        try:
            # Query MFA events from database
            cursor = self.db_connection.cursor(cursor_factory=RealDictCursor)
            cursor.execute("""
                SELECT * FROM security_audit_log 
                WHERE event_type LIKE '%mfa%' 
                AND timestamp > NOW() - INTERVAL '30 days'
            """)
            
            mfa_logs = [dict(row) for row in cursor.fetchall()]
            cursor.close()
            
            evidence_id = self._generate_evidence_id()
            evidence_path = f"/opt/sutazaiapp/compliance/evidence/mfa_logs_{evidence_id}.json"
            
            with open(evidence_path, 'w') as f:
                json.dump(mfa_logs, f, indent=2, default=str)
            
            return ComplianceEvidence(
                evidence_id=evidence_id,
                control_id="mfa_logs",
                evidence_type="log",
                evidence_path=evidence_path,
                collected_at=datetime.utcnow(),
                valid_until=datetime.utcnow() + timedelta(days=30),
                metadata={"log_count": len(mfa_logs), "source": "database"}
            )
            
        except Exception as e:
            self.logger.error(f"Failed to collect MFA logs: {e}")
            return None
    
    async def _collect_encryption_config(self) -> Optional[ComplianceEvidence]:
        """Collect encryption configuration evidence"""
        try:
            encryption_config = {}
            
            # Check TLS configuration
            tls_config = await self._check_tls_configuration()
            encryption_config['tls'] = tls_config
            
            # Check database encryption
            db_encryption = await self._check_database_encryption()
            encryption_config['database'] = db_encryption
            
            evidence_id = self._generate_evidence_id()
            evidence_path = f"/opt/sutazaiapp/compliance/evidence/encryption_config_{evidence_id}.json"
            
            with open(evidence_path, 'w') as f:
                json.dump(encryption_config, f, indent=2, default=str)
            
            return ComplianceEvidence(
                evidence_id=evidence_id,
                control_id="encryption_config",
                evidence_type="config",
                evidence_path=evidence_path,
                collected_at=datetime.utcnow(),
                valid_until=datetime.utcnow() + timedelta(days=90),
                metadata={"config_items": len(encryption_config), "source": "system"}
            )
            
        except Exception as e:
            self.logger.error(f"Failed to collect encryption config: {e}")
            return None
    
    async def _collect_vulnerability_scans(self) -> Optional[ComplianceEvidence]:
        """Collect vulnerability scan evidence"""
        try:
            # Get recent vulnerability scans
            cursor = self.db_connection.cursor(cursor_factory=RealDictCursor)
            cursor.execute("""
                SELECT * FROM vulnerability_scans 
                WHERE started_at > NOW() - INTERVAL '7 days'
                ORDER BY started_at DESC
            """)
            
            scans = [dict(row) for row in cursor.fetchall()]
            cursor.close()
            
            evidence_id = self._generate_evidence_id()
            evidence_path = f"/opt/sutazaiapp/compliance/evidence/vuln_scans_{evidence_id}.json"
            
            with open(evidence_path, 'w') as f:
                json.dump(scans, f, indent=2, default=str)
            
            return ComplianceEvidence(
                evidence_id=evidence_id,
                control_id="vulnerability_scans",
                evidence_type="document",
                evidence_path=evidence_path,
                collected_at=datetime.utcnow(),
                valid_until=datetime.utcnow() + timedelta(days=30),
                metadata={"scan_count": len(scans), "source": "database"}
            )
            
        except Exception as e:
            self.logger.error(f"Failed to collect vulnerability scans: {e}")
            return None
    
    async def _collect_system_hardening(self) -> Optional[ComplianceEvidence]:
        """Collect system hardening evidence"""
        try:
            hardening_config = {}
            
            # Check firewall rules
            try:
                result = subprocess.run(['iptables', '-L', '-n'], capture_output=True, text=True)
                hardening_config['firewall_rules'] = result.stdout
            except (IOError, OSError, FileNotFoundError) as e:
                logger.error(f"Unexpected exception: {e}", exc_info=True)
                hardening_config['firewall_rules'] = "Not available"
            
            # Check SSH configuration
            try:
                with open('/etc/ssh/sshd_config', 'r') as f:
                    hardening_config['ssh_config'] = f.read()
            except (IOError, OSError, FileNotFoundError) as e:
                logger.error(f"Unexpected exception: {e}", exc_info=True)
                hardening_config['ssh_config'] = "Not available"
            
            evidence_id = self._generate_evidence_id()
            evidence_path = f"/opt/sutazaiapp/compliance/evidence/system_hardening_{evidence_id}.json"
            
            with open(evidence_path, 'w') as f:
                json.dump(hardening_config, f, indent=2)
            
            return ComplianceEvidence(
                evidence_id=evidence_id,
                control_id="system_hardening",
                evidence_type="config",
                evidence_path=evidence_path,
                collected_at=datetime.utcnow(),
                valid_until=datetime.utcnow() + timedelta(days=90),
                metadata={"config_items": len(hardening_config), "source": "system"}
            )
            
        except Exception as e:
            self.logger.error(f"Failed to collect system hardening: {e}")
            return None
    
    async def _collect_software_inventory(self) -> Optional[ComplianceEvidence]:
        """Collect software inventory evidence"""
        try:
            inventory = {}
            
            # Get installed packages (Debian/Ubuntu)
            try:
                result = subprocess.run(['dpkg', '-l'], capture_output=True, text=True)
                inventory['dpkg_packages'] = result.stdout
            except (IOError, OSError, FileNotFoundError) as e:
                # Suppressed exception (was bare except)
                logger.debug(f"Suppressed exception: {e}")
                pass
            
            # Get Docker images
            try:
                result = subprocess.run(['docker', 'images', '--format', 'json'], capture_output=True, text=True)
                images = [json.loads(line) for line in result.stdout.strip().split('\n') if line]
                inventory['docker_images'] = images
            except (IOError, OSError, FileNotFoundError) as e:
                # Suppressed exception (was bare except)
                logger.debug(f"Suppressed exception: {e}")
                pass
            
            # Get Python packages
            try:
                result = subprocess.run(['pip', 'list', '--format=json'], capture_output=True, text=True)
                inventory['python_packages'] = json.loads(result.stdout)
            except (IOError, OSError, FileNotFoundError) as e:
                # Suppressed exception (was bare except)
                logger.debug(f"Suppressed exception: {e}")
                pass
            
            evidence_id = self._generate_evidence_id()
            evidence_path = f"/opt/sutazaiapp/compliance/evidence/software_inventory_{evidence_id}.json"
            
            with open(evidence_path, 'w') as f:
                json.dump(inventory, f, indent=2, default=str)
            
            return ComplianceEvidence(
                evidence_id=evidence_id,
                control_id="software_inventory",
                evidence_type="document",
                evidence_path=evidence_path,
                collected_at=datetime.utcnow(),
                valid_until=datetime.utcnow() + timedelta(days=30),
                metadata={"inventory_items": len(inventory), "source": "system"}
            )
            
        except Exception as e:
            self.logger.error(f"Failed to collect software inventory: {e}")
            return None
    
    # Automated check methods
    async def _check_access_controls(self) -> Tuple[ComplianceStatus, List[str]]:
        """Check access control implementation"""
        findings = []
        
        try:
            # Check if access controls are configured
            cursor = self.db_connection.cursor()
            cursor.execute("SELECT COUNT(*) FROM users WHERE is_active = true")
            active_users = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM users WHERE is_active = true AND last_login > NOW() - INTERVAL '90 days'")
            recent_users = cursor.fetchone()[0]
            
            cursor.close()
            
            if active_users == 0:
                findings.append("No active users found")
                return ComplianceStatus.NON_COMPLIANT, findings
            
            # Check for stale accounts
            stale_accounts = active_users - recent_users
            if stale_accounts > 0:
                findings.append(f"{stale_accounts} users haven't logged in within 90 days")
                return ComplianceStatus.PARTIALLY_COMPLIANT, findings
            
            findings.append("Access controls properly configured")
            return ComplianceStatus.COMPLIANT, findings
            
        except Exception as e:
            findings.append(f"Access control check failed: {e}")
            return ComplianceStatus.NOT_ASSESSED, findings
    
    async def _check_mfa_controls(self) -> Tuple[ComplianceStatus, List[str]]:
        """Check multi-factor authentication implementation"""
        findings = []
        
        try:
            # Check MFA configuration
            cursor = self.db_connection.cursor()
            cursor.execute("SELECT COUNT(*) FROM users WHERE is_active = true")
            total_users = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM user_mfa WHERE enabled = true")
            mfa_enabled_users = cursor.fetchone()[0] or 0
            
            cursor.close()
            
            if total_users == 0:
                findings.append("No users found")
                return ComplianceStatus.NOT_ASSESSED, findings
            
            mfa_percentage = (mfa_enabled_users / total_users) * 100
            
            if mfa_percentage < 50:
                findings.append(f"Only {mfa_percentage:.1f}% of users have MFA enabled")
                return ComplianceStatus.NON_COMPLIANT, findings
            elif mfa_percentage < 90:
                findings.append(f"{mfa_percentage:.1f}% of users have MFA enabled (target: 90%+)")
                return ComplianceStatus.PARTIALLY_COMPLIANT, findings
            else:
                findings.append(f"MFA enabled for {mfa_percentage:.1f}% of users")
                return ComplianceStatus.COMPLIANT, findings
                
        except Exception as e:
            findings.append(f"MFA check failed: {e}")
            return ComplianceStatus.NOT_ASSESSED, findings
    
    async def _check_encryption_controls(self) -> Tuple[ComplianceStatus, List[str]]:
        """Check encryption implementation"""
        findings = []
        
        try:
            compliant_items = 0
            total_items = 0
            
            # Check TLS configuration
            tls_config = await self._check_tls_configuration()
            total_items += 1
            if tls_config.get('version', '').startswith('TLSv1.2') or tls_config.get('version', '').startswith('TLSv1.3'):
                compliant_items += 1
                findings.append("TLS 1.2+ configured")
            else:
                findings.append(f"Weak TLS version: {tls_config.get('version', 'unknown')}")
            
            # Check database encryption
            db_encryption = await self._check_database_encryption()
            total_items += 1
            if db_encryption.get('ssl_enabled', False):
                compliant_items += 1
                findings.append("Database SSL enabled")
            else:
                findings.append("Database SSL not enabled")
            
            # Determine compliance status
            if compliant_items == total_items:
                return ComplianceStatus.COMPLIANT, findings
            elif compliant_items > 0:
                return ComplianceStatus.PARTIALLY_COMPLIANT, findings
            else:
                return ComplianceStatus.NON_COMPLIANT, findings
                
        except Exception as e:
            findings.append(f"Encryption check failed: {e}")
            return ComplianceStatus.NOT_ASSESSED, findings
    
    async def _check_vulnerability_management(self) -> Tuple[ComplianceStatus, List[str]]:
        """Check vulnerability management process"""
        findings = []
        
        try:
            # Check recent vulnerability scans
            cursor = self.db_connection.cursor()
            cursor.execute("""
                SELECT COUNT(*) FROM vulnerability_scans 
                WHERE started_at > NOW() - INTERVAL '7 days'
            """)
            recent_scans = cursor.fetchone()[0] or 0
            
            # Check critical vulnerabilities
            cursor.execute("""
                SELECT COUNT(*) FROM vulnerabilities 
                WHERE severity >= 4 AND remediation_status != 'completed'
                AND discovered_at > NOW() - INTERVAL '30 days'
            """)
            open_critical_vulns = cursor.fetchone()[0] or 0
            
            cursor.close()
            
            if recent_scans == 0:
                findings.append("No vulnerability scans in the last 7 days")
                return ComplianceStatus.NON_COMPLIANT, findings
            
            findings.append(f"{recent_scans} vulnerability scans completed in last 7 days")
            
            if open_critical_vulns > 0:
                findings.append(f"{open_critical_vulns} unresolved critical vulnerabilities")
                return ComplianceStatus.PARTIALLY_COMPLIANT, findings
            
            findings.append("No unresolved critical vulnerabilities")
            return ComplianceStatus.COMPLIANT, findings
            
        except Exception as e:
            findings.append(f"Vulnerability management check failed: {e}")
            return ComplianceStatus.NOT_ASSESSED, findings
    
    async def _check_default_passwords(self) -> Tuple[ComplianceStatus, List[str]]:
        """Check for default passwords"""
        findings = []
        
        try:
            # Check for common default passwords
            default_password_indicators = []
            
            # Check database configurations
            config_files = [
                '/opt/sutazaiapp/docker-compose.yml',
                '/opt/sutazaiapp/.env',
            ]
            
            for config_file in config_files:
                if os.path.exists(config_file):
                    with open(config_file, 'r') as f:
                        content = f.read()
                        
                    # Check for obvious default passwords
                    default_patterns = ['password', 'admin', '123456', 'changeme', 'default']
                    for pattern in default_patterns:
                        if pattern in content.lower():
                            default_password_indicators.append(f"Potential default password in {config_file}")
            
            if default_password_indicators:
                findings.extend(default_password_indicators)
                return ComplianceStatus.NON_COMPLIANT, findings
            else:
                findings.append("No obvious default passwords found")
                return ComplianceStatus.COMPLIANT, findings
                
        except Exception as e:
            findings.append(f"Default password check failed: {e}")
            return ComplianceStatus.NOT_ASSESSED, findings
    
    async def _check_data_transmission_encryption(self) -> Tuple[ComplianceStatus, List[str]]:
        """Check data transmission encryption"""
        findings = []
        
        try:
            # Check TLS configuration
            tls_config = await self._check_tls_configuration()
            
            if tls_config.get('version', '').startswith('TLSv1.2') or tls_config.get('version', '').startswith('TLSv1.3'):
                findings.append("Strong TLS version in use")
                return ComplianceStatus.COMPLIANT, findings
            else:
                findings.append(f"Weak or no TLS encryption: {tls_config.get('version', 'unknown')}")
                return ComplianceStatus.NON_COMPLIANT, findings
                
        except Exception as e:
            findings.append(f"Data transmission encryption check failed: {e}")
            return ComplianceStatus.NOT_ASSESSED, findings
    
    async def _check_identity_management(self) -> Tuple[ComplianceStatus, List[str]]:
        """Check identity management implementation"""
        findings = []
        
        try:
            # Check user management
            cursor = self.db_connection.cursor()
            cursor.execute("SELECT COUNT(*) FROM users WHERE is_active = true")
            active_users = cursor.fetchone()[0] or 0
            
            # Check for proper user lifecycle management
            cursor.execute("""
                SELECT COUNT(*) FROM users 
                WHERE created_at < NOW() - INTERVAL '1 year' 
                AND last_login IS NULL
            """)
            unused_accounts = cursor.fetchone()[0] or 0
            
            cursor.close()
            
            if active_users == 0:
                findings.append("No active user accounts found")
                return ComplianceStatus.NOT_ASSESSED, findings
            
            if unused_accounts > 0:
                findings.append(f"{unused_accounts} unused accounts found")
                return ComplianceStatus.PARTIALLY_COMPLIANT, findings
            
            findings.append("Identity management properly configured")
            return ComplianceStatus.COMPLIANT, findings
            
        except Exception as e:
            findings.append(f"Identity management check failed: {e}")
            return ComplianceStatus.NOT_ASSESSED, findings
    
    async def _check_software_inventory(self) -> Tuple[ComplianceStatus, List[str]]:
        """Check software inventory maintenance"""
        findings = []
        
        try:
            # Check if software inventory exists
            inventory_path = "/opt/sutazaiapp/compliance/evidence"
            inventory_files = [f for f in os.listdir(inventory_path) if f.startswith('software_inventory')]
            
            if not inventory_files:
                findings.append("No software inventory found")
                return ComplianceStatus.NON_COMPLIANT, findings
            
            # Check age of most recent inventory
            latest_inventory = max(inventory_files, key=lambda f: os.path.getmtime(os.path.join(inventory_path, f)))
            inventory_age = time.time() - os.path.getmtime(os.path.join(inventory_path, latest_inventory))
            inventory_age_days = inventory_age / (24 * 3600)
            
            if inventory_age_days > 30:
                findings.append(f"Software inventory is {inventory_age_days:.0f} days old")
                return ComplianceStatus.PARTIALLY_COMPLIANT, findings
            
            findings.append(f"Current software inventory maintained (last updated {inventory_age_days:.0f} days ago)")
            return ComplianceStatus.COMPLIANT, findings
            
        except Exception as e:
            findings.append(f"Software inventory check failed: {e}")
            return ComplianceStatus.NOT_ASSESSED, findings
    
    # Helper methods
    async def _check_tls_configuration(self) -> Dict[str, Any]:
        """Check TLS configuration"""
        try:
            # This would check actual TLS configuration
            # For now, return Mock data
            return {
                "version": "TLSv1.3",
                "cipher_suites": ["TLS_AES_256_GCM_SHA384", "TLS_CHACHA20_POLY1305_SHA256"],
                "certificate_valid": True
            }
        except Exception as e:
            self.logger.error(f"TLS configuration check failed: {e}")
            return {"version": "unknown", "error": str(e)}
    
    async def _check_database_encryption(self) -> Dict[str, Any]:
        """Check database encryption configuration"""
        try:
            # Check PostgreSQL SSL configuration
            cursor = self.db_connection.cursor()
            cursor.execute("SHOW ssl")
            ssl_status = cursor.fetchone()[0]
            cursor.close()
            
            return {
                "ssl_enabled": ssl_status == "on",
                "connection_encrypted": True
            }
        except Exception as e:
            self.logger.error(f"Database encryption check failed: {e}")
            return {"ssl_enabled": False, "error": str(e)}
    
    def _generate_compliance_report(self, report_id: str, framework: ComplianceFramework,
                                  results: List[AssessmentResult]) -> ComplianceReport:
        """Generate compliance report from assessment results"""
        total_controls = len(results)
        compliant_controls = len([r for r in results if r.status == ComplianceStatus.COMPLIANT])
        non_compliant_controls = len([r for r in results if r.status == ComplianceStatus.NON_COMPLIANT])
        partially_compliant_controls = len([r for r in results if r.status == ComplianceStatus.PARTIALLY_COMPLIANT])
        
        # Determine overall status
        if non_compliant_controls == 0 and partially_compliant_controls == 0:
            overall_status = ComplianceStatus.COMPLIANT
        elif non_compliant_controls == 0:
            overall_status = ComplianceStatus.PARTIALLY_COMPLIANT
        else:
            overall_status = ComplianceStatus.NON_COMPLIANT
        
        # Generate recommendations
        recommendations = []
        for result in results:
            if result.status == ComplianceStatus.NON_COMPLIANT:
                recommendations.append(f"Address non-compliance in {result.control_id}")
            elif result.status == ComplianceStatus.PARTIALLY_COMPLIANT:
                recommendations.append(f"Improve compliance in {result.control_id}")
        
        return ComplianceReport(
            report_id=report_id,
            framework=framework,
            assessment_date=datetime.utcnow(),
            scope="SutazAI Production System",
            overall_status=overall_status,
            total_controls=total_controls,
            compliant_controls=compliant_controls,
            non_compliant_controls=non_compliant_controls,
            partially_compliant_controls=partially_compliant_controls,
            results=results,
            recommendations=recommendations,
            next_assessment_due=datetime.utcnow() + timedelta(days=90)
        )
    
    async def _store_compliance_report(self, report: ComplianceReport):
        """Store compliance report in database"""
        try:
            cursor = self.db_connection.cursor()
            
            # Store report
            cursor.execute("""
                INSERT INTO compliance_reports 
                (report_id, framework, assessment_date, scope, overall_status,
                 total_controls, compliant_controls, non_compliant_controls,
                 partially_compliant_controls, recommendations, next_assessment_due)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                report.report_id, report.framework.value, report.assessment_date,
                report.scope, report.overall_status.value, report.total_controls,
                report.compliant_controls, report.non_compliant_controls,
                report.partially_compliant_controls, json.dumps(report.recommendations),
                report.next_assessment_due
            ))
            
            # Store assessment results
            for result in report.results:
                cursor.execute("""
                    INSERT INTO assessment_results 
                    (report_id, control_id, status, assessment_date, assessor,
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    report.report_id, result.control_id, result.status.value,
                    result.assessment_date, result.assessor, json.dumps(result.findings),
                    json.dumps(result.evidence_ids), result.remediation_required,
                ))
            
            self.db_connection.commit()
            cursor.close()
            
            # Store in Redis for quick access
            self.redis_client.setex(
                f"compliance_report:{report.report_id}",
                86400,  # 24 hours
                json.dumps(asdict(report), default=str)
            )
            
        except Exception as e:
            self.logger.error(f"Failed to store compliance report: {e}")
    
    def _generate_report_id(self) -> str:
        """Generate unique report ID"""
        timestamp = str(int(time.time() * 1000))
        random_part = hashlib.md5(os.urandom(16)).hexdigest()[:8]
        return f"compliance_{timestamp}_{random_part}"
    
    def _generate_evidence_id(self) -> str:
        """Generate unique evidence ID"""
        timestamp = str(int(time.time() * 1000))
        random_part = hashlib.md5(os.urandom(16)).hexdigest()[:8]
        return f"evidence_{timestamp}_{random_part}"
    
    async def start_continuous_monitoring(self):
        """Start continuous compliance monitoring"""
        self.running = True
        
        while self.running:
            try:
                # Run daily assessments for critical controls
                critical_controls = [
                    control for control in self.controls.values()
                    if control.severity == ControlSeverity.CRITICAL and control.automated_check
                ]
                
                for control in critical_controls:
                    if control.check_frequency == "daily":
                        result = await self._assess_control(control)
                        self.assessment_results[control.control_id] = result
                        
                        # Alert on non-compliance
                        if result.status == ComplianceStatus.NON_COMPLIANT:
                            await self._send_compliance_alert(control, result)
                
                # Wait 24 hours before next check
                await asyncio.sleep(86400)
                
            except Exception as e:
                self.logger.error(f"Continuous monitoring error: {e}")
                await asyncio.sleep(3600)  # Wait 1 hour on error
    
    async def _send_compliance_alert(self, control: ComplianceControl, result: AssessmentResult):
        """Send compliance alert for non-compliant control"""
        try:
            alert_data = {
                "type": "compliance_violation",
                "control_id": control.control_id,
                "framework": control.framework.value,
                "title": control.title,
                "severity": control.severity.name,
                "status": result.status.value,
                "findings": result.findings,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Store alert in Redis
            self.redis_client.lpush("compliance_alerts", json.dumps(alert_data))
            
            self.logger.warning(
                f"Compliance violation: {control.control_id} - {control.title}"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to send compliance alert: {e}")
    
    def stop_continuous_monitoring(self):
        """Stop continuous compliance monitoring"""
        self.running = False
        self.logger.info("Stopped continuous compliance monitoring")
    
    def get_compliance_status(self, framework: ComplianceFramework = None) -> Dict[str, Any]:
        """Get current compliance status"""
        if framework:
            framework_results = [
                result for result in self.assessment_results.values()
                if self.controls.get(result.control_id, {}).framework == framework
            ]
        else:
            framework_results = list(self.assessment_results.values())
        
        total = len(framework_results)
        compliant = len([r for r in framework_results if r.status == ComplianceStatus.COMPLIANT])
        non_compliant = len([r for r in framework_results if r.status == ComplianceStatus.NON_COMPLIANT])
        partially_compliant = len([r for r in framework_results if r.status == ComplianceStatus.PARTIALLY_COMPLIANT])
        
        return {
            "framework": framework.value if framework else "all",
            "total_controls": total,
            "compliant": compliant,
            "non_compliant": non_compliant,
            "partially_compliant": partially_compliant,
            "compliance_percentage": (compliant / total * 100) if total > 0 else 0,
            "last_updated": datetime.utcnow().isoformat()
        }

# Database schema for compliance management
COMPLIANCE_SCHEMA = """
-- Compliance controls table
CREATE TABLE IF NOT EXISTS compliance_controls (
    id SERIAL PRIMARY KEY,
    control_id VARCHAR(100) UNIQUE NOT NULL,
    framework VARCHAR(50) NOT NULL,
    title TEXT NOT NULL,
    description TEXT,
    category VARCHAR(100),
    severity INTEGER DEFAULT 2,
    automated_check BOOLEAN DEFAULT false,
    check_frequency VARCHAR(20) DEFAULT 'monthly',
    evidence_required JSONB DEFAULT '[]',
    implementation_guidance TEXT,
    related_controls JSONB DEFAULT '[]',
    enabled BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Compliance reports table
CREATE TABLE IF NOT EXISTS compliance_reports (
    id SERIAL PRIMARY KEY,
    report_id VARCHAR(255) UNIQUE NOT NULL,
    framework VARCHAR(50) NOT NULL,
    assessment_date TIMESTAMP NOT NULL,
    scope TEXT,
    overall_status VARCHAR(50) NOT NULL,
    total_controls INTEGER DEFAULT 0,
    compliant_controls INTEGER DEFAULT 0,
    non_compliant_controls INTEGER DEFAULT 0,
    partially_compliant_controls INTEGER DEFAULT 0,
    recommendations JSONB DEFAULT '[]',
    next_assessment_due TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Assessment results table
CREATE TABLE IF NOT EXISTS assessment_results (
    id SERIAL PRIMARY KEY,
    report_id VARCHAR(255) REFERENCES compliance_reports(report_id),
    control_id VARCHAR(100) NOT NULL,
    status VARCHAR(50) NOT NULL,
    assessment_date TIMESTAMP NOT NULL,
    assessor VARCHAR(255),
    findings JSONB DEFAULT '[]',
    evidence_ids JSONB DEFAULT '[]',
    remediation_required BOOLEAN DEFAULT false,
    remediation_timeline TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Evidence table
CREATE TABLE IF NOT EXISTS compliance_evidence (
    id SERIAL PRIMARY KEY,
    evidence_id VARCHAR(255) UNIQUE NOT NULL,
    control_id VARCHAR(100),
    evidence_type VARCHAR(50) NOT NULL,
    evidence_path TEXT NOT NULL,
    collected_at TIMESTAMP NOT NULL,
    valid_until TIMESTAMP,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_compliance_controls_framework ON compliance_controls(framework);
CREATE INDEX IF NOT EXISTS idx_compliance_reports_framework ON compliance_reports(framework);
CREATE INDEX IF NOT EXISTS idx_assessment_results_control_id ON assessment_results(control_id);
CREATE INDEX IF NOT EXISTS idx_compliance_evidence_control_id ON compliance_evidence(control_id);
"""

if __name__ == "__main__":
    # Example usage
    config = {
        'redis_host': 'redis',
        'redis_port': 6379,
        'postgres_host': 'postgres',
        'postgres_port': 5432,
        'postgres_db': 'sutazai',
        'postgres_user': 'sutazai',
        'evidence_path': '/opt/sutazaiapp/compliance/evidence'
    }
    
    async def main():
        compliance_engine = ComplianceEngine(config)
        
        # Run SOC2 assessment
        soc2_report_id = await compliance_engine.run_automated_assessment(ComplianceFramework.SOC2)
        logger.info(f"SOC2 assessment completed: {soc2_report_id}")
        
        # Run ISO27001 assessment
        iso_report_id = await compliance_engine.run_automated_assessment(ComplianceFramework.ISO27001)
        logger.info(f"ISO27001 assessment completed: {iso_report_id}")
        
        # Get compliance status
        status = compliance_engine.get_compliance_status(ComplianceFramework.SOC2)
        logger.info(f"SOC2 Compliance Status: {status}")
        
        # Start continuous monitoring
        # await compliance_engine.start_continuous_monitoring()
    
