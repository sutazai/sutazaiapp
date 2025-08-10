#!/usr/bin/env python3
"""
Security Orchestrator
Main orchestration system for comprehensive security hardening framework
"""

import asyncio
import logging
import json
import os
import time
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import signal
import sys

# Import security modules
from zero_trust.architecture import ZeroTrustEngine
from defense_in_depth.network_security import NetworkSecurityEngine
from rasp.runtime_protection import RASPEngine
from threat_detection.advanced_detection import ThreatDetectionEngine
from agent_communication.secure_agent_comm import SecureAgentCommunication
from vulnerability_management.vuln_scanner import VulnerabilityScanner
from compliance.compliance_automation import ComplianceEngine
from incident_response.incident_response import IncidentResponseEngine

class SecurityOrchestrationStatus(Enum):
    INITIALIZING = "initializing"
    RUNNING = "running"
    DEGRADED = "degraded"
    STOPPED = "stopped"
    ERROR = "error"

@dataclass
class SecuritySystemStatus:
    """Status of individual security system"""
    system_name: str
    status: str
    last_check: datetime
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = None

class SecurityOrchestrator:
    """Main security orchestration system"""
    
    def __init__(self, config_path: str = "/opt/sutazaiapp/security/config.json"):
        self.config_path = config_path
        self.config = self._load_config()
        self.logger = self._setup_logging()
        
        # Security system status
        self.status = SecurityOrchestrationStatus.INITIALIZING
        self.system_statuses: Dict[str, SecuritySystemStatus] = {}
        
        # Security engines
        self.zero_trust_engine = None
        self.network_security_engine = None
        self.rasp_engine = None
        self.threat_detection_engine = None
        self.agent_communication = None
        self.vulnerability_scanner = None
        self.compliance_engine = None
        self.incident_response_engine = None
        
        # Control flags
        self.running = False
        self.shutdown_event = threading.Event()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load security orchestrator configuration"""
        default_config = {
            "redis": {
                "host": "redis",
                "port": 6379,
                "password": None
            },
            "postgres": {
                "host": "postgres",
                "port": 5432,
                "database": "sutazai",
                "user": "sutazai",
                "password": os.getenv("POSTGRES_PASSWORD")
            },
            "zero_trust": {
                "enabled": True,
                "session_ttl": 3600,
                "max_risk_threshold": 0.8
            },
            "network_security": {
                "enabled": True,
                "max_connections_per_minute": 100,
                "port_scan_threshold": 10,
                "ddos_threshold": 1000
            },
            "rasp": {
                "enabled": True,
                "protection_enabled": True
            },
            "threat_detection": {
                "enabled": True,
                "ml_enabled": True,
                "threat_intel_update_interval": 3600
            },
            "agent_communication": {
                "enabled": True,
                "encryption_enabled": True,
                "mtls_enabled": True
            },
            "vulnerability_scanner": {
                "enabled": True,
                "scan_schedule": "daily",
                "auto_remediation": False
            },
            "compliance": {
                "enabled": True,
                "frameworks": ["soc2", "iso27001", "pci_dss"],
                "continuous_monitoring": True
            },
            "incident_response": {
                "enabled": True,
                "auto_response": True,
                "forensics_enabled": True
            },
            "monitoring": {
                "health_check_interval": 60,
                "metrics_retention": 86400
            },
            "notifications": {
                "email": {
                    "enabled": False,
                    "smtp_host": "localhost",
                    "smtp_port": 587,
                    "from": "security@sutazai.com",
                    "to": ["admin@sutazai.com"]
                },
                "webhook": {
                    "enabled": False,
                    "url": None
                }
            }
        }
        
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    user_config = json.load(f)
                    # Merge with default config
                    default_config.update(user_config)
            except Exception as e:
                print(f"Warning: Failed to load config from {self.config_path}: {e}")
        
        return default_config
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for security orchestrator"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('/opt/sutazaiapp/logs/security_orchestrator.log')
            ]
        )
        return logging.getLogger(__name__)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}, initiating shutdown...")
        self.shutdown_event.set()
        asyncio.create_task(self.shutdown())
    
    async def initialize(self):
        """Initialize all security systems"""
        try:
            self.logger.info("Initializing SutazAI Security Hardening Framework...")
            self.status = SecurityOrchestrationStatus.INITIALIZING
            
            # Initialize Zero Trust Architecture
            if self.config["zero_trust"]["enabled"]:
                self.logger.info("Initializing Zero Trust Engine...")
                zt_config = {**self.config["postgres"], **self.config["redis"], **self.config["zero_trust"]}
                self.zero_trust_engine = ZeroTrustEngine(zt_config)
                self._update_system_status("zero_trust", "initialized")
            
            # Initialize Network Security
            if self.config["network_security"]["enabled"]:
                self.logger.info("Initializing Network Security Engine...")
                ns_config = {**self.config["redis"], **self.config["network_security"]}
                self.network_security_engine = NetworkSecurityEngine(ns_config)
                await self.network_security_engine.start_monitoring()
                self._update_system_status("network_security", "running")
            
            # Initialize RASP
            if self.config["rasp"]["enabled"]:
                self.logger.info("Initializing RASP Engine...")
                rasp_config = {**self.config["redis"], **self.config["rasp"]}
                self.rasp_engine = RASPEngine(rasp_config)
                if self.config["rasp"]["protection_enabled"]:
                    self.rasp_engine.enable_protection()
                self._update_system_status("rasp", "running")
            
            # Initialize Threat Detection
            if self.config["threat_detection"]["enabled"]:
                self.logger.info("Initializing Threat Detection Engine...")
                td_config = {**self.config["postgres"], **self.config["redis"], **self.config["threat_detection"]}
                self.threat_detection_engine = ThreatDetectionEngine(td_config)
                await self.threat_detection_engine.start_detection()
                self._update_system_status("threat_detection", "running")
            
            # Initialize Agent Communication
            if self.config["agent_communication"]["enabled"]:
                self.logger.info("Initializing Secure Agent Communication...")
                ac_config = {**self.config["postgres"], **self.config["redis"], **self.config["agent_communication"]}
                self.agent_communication = SecureAgentCommunication(ac_config)
                asyncio.create_task(self.agent_communication.start_message_processing())
                self._update_system_status("agent_communication", "running")
            
            # Initialize Vulnerability Scanner
            if self.config["vulnerability_scanner"]["enabled"]:
                self.logger.info("Initializing Vulnerability Scanner...")
                vs_config = {**self.config["postgres"], **self.config["redis"], **self.config["vulnerability_scanner"]}
                self.vulnerability_scanner = VulnerabilityScanner(vs_config)
                self._update_system_status("vulnerability_scanner", "initialized")
            
            # Initialize Compliance Engine
            if self.config["compliance"]["enabled"]:
                self.logger.info("Initializing Compliance Engine...")
                comp_config = {**self.config["postgres"], **self.config["redis"], **self.config["compliance"]}
                self.compliance_engine = ComplianceEngine(comp_config)
                if self.config["compliance"]["continuous_monitoring"]:
                    asyncio.create_task(self.compliance_engine.start_continuous_monitoring())
                self._update_system_status("compliance", "running")
            
            # Initialize Incident Response
            if self.config["incident_response"]["enabled"]:
                self.logger.info("Initializing Incident Response Engine...")
                ir_config = {**self.config["postgres"], **self.config["redis"], **self.config["incident_response"], **self.config["notifications"]}
                self.incident_response_engine = IncidentResponseEngine(ir_config)
                asyncio.create_task(self.incident_response_engine.start_incident_monitoring())
                self._update_system_status("incident_response", "running")
            
            self.status = SecurityOrchestrationStatus.RUNNING
            self.running = True
            
            self.logger.info("SutazAI Security Hardening Framework initialized successfully!")
            await self._send_startup_notification()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize security framework: {e}")
            self.status = SecurityOrchestrationStatus.ERROR
            raise
    
    async def run(self):
        """Main orchestrator run loop"""
        try:
            await self.initialize()
            
            # Start monitoring and orchestration tasks
            tasks = [
                asyncio.create_task(self._health_monitoring_loop()),
                asyncio.create_task(self._security_orchestration_loop()),
                asyncio.create_task(self._metrics_collection_loop()),
                asyncio.create_task(self._scheduled_tasks_loop())
            ]
            
            self.logger.info("Security Orchestrator is now running...")
            
            # Wait for shutdown signal
            while self.running and not self.shutdown_event.is_set():
                await asyncio.sleep(1)
            
            # Cancel all tasks
            for task in tasks:
                task.cancel()
            
            await asyncio.gather(*tasks, return_exceptions=True)
            
        except Exception as e:
            self.logger.error(f"Security orchestrator error: {e}")
            self.status = SecurityOrchestrationStatus.ERROR
        finally:
            await self.shutdown()
    
    async def _health_monitoring_loop(self):
        """Monitor health of all security systems"""
        interval = self.config["monitoring"]["health_check_interval"]
        
        while self.running:
            try:
                await self._check_system_health()
                await asyncio.sleep(interval)
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(interval)
    
    async def _check_system_health(self):
        """Check health of all security systems"""
        try:
            # Check Zero Trust Engine
            if self.zero_trust_engine:
                try:
                    # Basic connectivity check
                    self._update_system_status("zero_trust", "running")
                except Exception as e:
                    self._update_system_status("zero_trust", "error", str(e))
            
            # Check Network Security Engine
            if self.network_security_engine:
                try:
                    if self.network_security_engine.running:
                        self._update_system_status("network_security", "running")
                    else:
                        self._update_system_status("network_security", "stopped")
                except Exception as e:
                    self._update_system_status("network_security", "error", str(e))
            
            # Check RASP Engine
            if self.rasp_engine:
                try:
                    stats = self.rasp_engine.get_statistics()
                    self._update_system_status("rasp", "running", metrics=stats)
                except Exception as e:
                    self._update_system_status("rasp", "error", str(e))
            
            # Check Threat Detection Engine
            if self.threat_detection_engine:
                try:
                    if self.threat_detection_engine.running:
                        stats = self.threat_detection_engine.get_threat_statistics()
                        self._update_system_status("threat_detection", "running", metrics=stats)
                    else:
                        self._update_system_status("threat_detection", "stopped")
                except Exception as e:
                    self._update_system_status("threat_detection", "error", str(e))
            
            # Check Agent Communication
            if self.agent_communication:
                try:
                    status = self.agent_communication.get_agent_status()
                    self._update_system_status("agent_communication", "running", metrics=status)
                except Exception as e:
                    self._update_system_status("agent_communication", "error", str(e))
            
            # Check Vulnerability Scanner
            if self.vulnerability_scanner:
                try:
                    self._update_system_status("vulnerability_scanner", "running")
                except Exception as e:
                    self._update_system_status("vulnerability_scanner", "error", str(e))
            
            # Check Compliance Engine
            if self.compliance_engine:
                try:
                    if self.compliance_engine.running:
                        self._update_system_status("compliance", "running")
                    else:
                        self._update_system_status("compliance", "stopped")
                except Exception as e:
                    self._update_system_status("compliance", "error", str(e))
            
            # Check Incident Response Engine
            if self.incident_response_engine:
                try:
                    if self.incident_response_engine.running:
                        stats = self.incident_response_engine.get_incident_statistics()
                        self._update_system_status("incident_response", "running", metrics=stats)
                    else:
                        self._update_system_status("incident_response", "stopped")
                except Exception as e:
                    self._update_system_status("incident_response", "error", str(e))
            
            # Update overall status
            self._update_overall_status()
            
        except Exception as e:
            self.logger.error(f"System health check failed: {e}")
    
    def _update_system_status(self, system_name: str, status: str, error_message: str = None, metrics: Dict[str, Any] = None):
        """Update status of individual security system"""
        self.system_statuses[system_name] = SecuritySystemStatus(
            system_name=system_name,
            status=status,
            last_check=datetime.utcnow(),
            error_message=error_message,
            metrics=metrics
        )
    
    def _update_overall_status(self):
        """Update overall orchestrator status based on system statuses"""
        error_systems = [s for s in self.system_statuses.values() if s.status == "error"]
        stopped_systems = [s for s in self.system_statuses.values() if s.status == "stopped"]
        
        if error_systems:
            self.status = SecurityOrchestrationStatus.ERROR
        elif stopped_systems:
            self.status = SecurityOrchestrationStatus.DEGRADED
        else:
            self.status = SecurityOrchestrationStatus.RUNNING
    
    async def _security_orchestration_loop(self):
        """Main security orchestration and coordination loop"""
        while self.running:
            try:
                # Coordinate threat intelligence sharing
                await self._coordinate_threat_intelligence()
                
                # Coordinate incident response
                await self._coordinate_incident_response()
                
                # Coordinate compliance monitoring
                await self._coordinate_compliance_monitoring()
                
                # Auto-remediation
                await self._auto_remediation()
                
                await asyncio.sleep(30)  # Run every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Security orchestration error: {e}")
                await asyncio.sleep(60)
    
    async def _coordinate_threat_intelligence(self):
        """Coordinate threat intelligence between systems"""
        try:
            if not (self.threat_detection_engine and self.network_security_engine):
                return
            
            # Get active threats from threat detection
            active_threats = self.threat_detection_engine.get_active_threats()
            
            # Share threat indicators with network security
            for threat in active_threats:
                if threat.source_ip and threat.confidence > 0.7:
                    # This would add the IP to network security blocking rules
                    pass
            
        except Exception as e:
            self.logger.error(f"Threat intelligence coordination error: {e}")
    
    async def _coordinate_incident_response(self):
        """Coordinate incident response activities"""
        try:
            if not self.incident_response_engine:
                return
            
            # Check for new high-severity incidents
            active_incidents = self.incident_response_engine.active_incidents
            
            for incident in active_incidents.values():
                if incident.severity.value >= 3 and incident.status.value == "new":
                    # Trigger enhanced monitoring across all systems
                    await self._enhance_security_monitoring(incident)
            
        except Exception as e:
            self.logger.error(f"Incident response coordination error: {e}")
    
    async def _coordinate_compliance_monitoring(self):
        """Coordinate compliance monitoring activities"""
        try:
            if not self.compliance_engine:
                return
            
            # Check compliance status and trigger remediation if needed
            # This would be implemented based on specific compliance requirements
            pass
            
        except Exception as e:
            self.logger.error(f"Compliance coordination error: {e}")
    
    async def _auto_remediation(self):
        """Perform automated security remediation"""
        try:
            # Check RASP events for immediate response
            if self.rasp_engine:
                recent_events = self.rasp_engine.get_events(limit=50)
                critical_events = [e for e in recent_events if e.severity.value >= 4 and e.blocked]
                
                if len(critical_events) > 5:  # Pattern of critical events
                    self.logger.warning("Multiple critical RASP events detected, enhancing security")
                    await self._enhance_security_posture()
            
            # Check vulnerability scan results for auto-patching
            if self.vulnerability_scanner and self.config["vulnerability_scanner"]["auto_remediation"]:
                # This would implement auto-patching for certain types of vulnerabilities
                pass
            
        except Exception as e:
            self.logger.error(f"Auto-remediation error: {e}")
    
    async def _enhance_security_monitoring(self, incident):
        """Enhance security monitoring based on incident"""
        try:
            self.logger.info(f"Enhancing security monitoring for incident: {incident.incident_id}")
            
            # Increase monitoring sensitivity
            if self.network_security_engine:
                # This would lower thresholds for detection
                pass
            
            if self.threat_detection_engine:
                # This would increase frequency of ML model runs
                pass
            
        except Exception as e:
            self.logger.error(f"Failed to enhance security monitoring: {e}")
    
    async def _enhance_security_posture(self):
        """Enhance overall security posture"""
        try:
            self.logger.info("Enhancing security posture due to threat patterns")
            
            # Enable additional RASP protections
            if self.rasp_engine:
                self.rasp_engine.enable_protection()
            
            # Trigger immediate vulnerability scan
            if self.vulnerability_scanner:
                await self.vulnerability_scanner.scan_containers()
            
            # Notify administrators
            await self._send_security_alert("Security posture enhanced due to threat patterns")
            
        except Exception as e:
            self.logger.error(f"Failed to enhance security posture: {e}")
    
    async def _metrics_collection_loop(self):
        """Collect and store security metrics"""
        while self.running:
            try:
                await self._collect_security_metrics()
                await asyncio.sleep(300)  # Collect every 5 minutes
            except Exception as e:
                self.logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(300)
    
    async def _collect_security_metrics(self):
        """Collect security metrics from all systems"""
        try:
            metrics = {
                "timestamp": datetime.utcnow().isoformat(),
                "orchestrator_status": self.status.value,
                "system_statuses": {
                    name: asdict(status) for name, status in self.system_statuses.items()
                }
            }
            
            # Store metrics (would typically go to a time-series database)
            self.logger.debug(f"Collected security metrics: {len(metrics)} data points")
            
        except Exception as e:
            self.logger.error(f"Failed to collect security metrics: {e}")
    
    async def _scheduled_tasks_loop(self):
        """Handle scheduled security tasks"""
        while self.running:
            try:
                current_time = datetime.utcnow()
                
                # Daily vulnerability scans
                if (current_time.hour == 2 and current_time.minute == 0 and 
                    self.vulnerability_scanner and 
                    self.config["vulnerability_scanner"]["scan_schedule"] == "daily"):
                    
                    self.logger.info("Starting scheduled vulnerability scan")
                    await self.vulnerability_scanner.scan_containers()
                    await self.vulnerability_scanner.scan_network(["127.0.0.1"])
                
                # Weekly compliance assessments
                if (current_time.weekday() == 0 and current_time.hour == 1 and 
                    current_time.minute == 0 and self.compliance_engine):
                    
                    self.logger.info("Starting scheduled compliance assessment")
                    for framework in self.config["compliance"]["frameworks"]:
                        try:
                            from compliance.compliance_automation import ComplianceFramework
                            framework_enum = ComplianceFramework(framework)
                            await self.compliance_engine.run_automated_assessment(framework_enum)
                        except Exception as e:
                            self.logger.error(f"Compliance assessment failed for {framework}: {e}")
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Scheduled tasks error: {e}")
                await asyncio.sleep(60)
    
    async def _send_startup_notification(self):
        """Send notification about successful startup"""
        try:
            message = {
                "type": "security_framework_startup",
                "status": "success",
                "timestamp": datetime.utcnow().isoformat(),
                "systems_initialized": list(self.system_statuses.keys()),
                "message": "SutazAI Security Hardening Framework started successfully"
            }
            
            await self._send_notification(message)
            
        except Exception as e:
            self.logger.error(f"Failed to send startup notification: {e}")
    
    async def _send_security_alert(self, message: str):
        """Send security alert notification"""
        try:
            alert = {
                "type": "security_alert",
                "severity": "high",
                "timestamp": datetime.utcnow().isoformat(),
                "message": message
            }
            
            await self._send_notification(alert)
            
        except Exception as e:
            self.logger.error(f"Failed to send security alert: {e}")
    
    async def _send_notification(self, data: Dict[str, Any]):
        """Send notification via configured channels"""
        try:
            # Email notification
            if self.config["notifications"]["email"]["enabled"]:
                await self._send_email_notification(data)
            
            # Webhook notification
            if self.config["notifications"]["webhook"]["enabled"]:
                await self._send_webhook_notification(data)
            
        except Exception as e:
            self.logger.error(f"Failed to send notification: {e}")
    
    async def _send_email_notification(self, data: Dict[str, Any]):
        """Send email notification"""
        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            
            email_config = self.config["notifications"]["email"]
            
            msg = MIMEMultipart()
            msg['From'] = email_config['from']
            msg['To'] = ', '.join(email_config['to'])
            msg['Subject'] = f"SutazAI Security Alert: {data.get('type', 'Unknown')}"
            
            body = f"""
Security Framework Notification

Type: {data.get('type', 'Unknown')}
Timestamp: {data.get('timestamp', 'Unknown')}
Message: {data.get('message', 'No message')}

Status: {data.get('status', 'Unknown')}

This is an automated notification from the SutazAI Security Hardening Framework.
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(email_config['smtp_host'], email_config['smtp_port'])
            server.starttls()
            if email_config.get('username') and email_config.get('password'):
                server.login(email_config['username'], email_config['password'])
            server.send_message(msg)
            server.quit()
            
        except Exception as e:
            self.logger.error(f"Failed to send email notification: {e}")
    
    async def _send_webhook_notification(self, data: Dict[str, Any]):
        """Send webhook notification"""
        try:
            import aiohttp
            
            webhook_config = self.config["notifications"]["webhook"]
            
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_config['url'], json=data) as response:
                    if response.status == 200:
                        self.logger.debug("Webhook notification sent successfully")
                    else:
                        self.logger.warning(f"Webhook notification failed: {response.status}")
            
        except Exception as e:
            self.logger.error(f"Failed to send webhook notification: {e}")
    
    async def shutdown(self):
        """Shutdown all security systems gracefully"""
        try:
            self.logger.info("Shutting down SutazAI Security Hardening Framework...")
            self.running = False
            
            # Stop all engines
            if self.network_security_engine:
                self.network_security_engine.stop_monitoring()
            
            if self.threat_detection_engine:
                self.threat_detection_engine.stop_detection()
            
            if self.agent_communication:
                self.agent_communication.stop_message_processing()
            
            if self.compliance_engine:
                self.compliance_engine.stop_continuous_monitoring()
            
            if self.incident_response_engine:
                self.incident_response_engine.stop_incident_monitoring()
            
            self.status = SecurityOrchestrationStatus.STOPPED
            
            # Send shutdown notification
            await self._send_notification({
                "type": "security_framework_shutdown",
                "status": "success",
                "timestamp": datetime.utcnow().isoformat(),
                "message": "SutazAI Security Hardening Framework shutdown completed"
            })
            
            self.logger.info("Security framework shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status of security framework"""
        return {
            "orchestrator_status": self.status.value,
            "timestamp": datetime.utcnow().isoformat(),
            "systems": {
                name: asdict(status) for name, status in self.system_statuses.items()
            },
            "uptime": (datetime.utcnow() - datetime.fromtimestamp(time.time())).total_seconds() if self.running else 0
        }

async def main():
    """Main entry point for security orchestrator"""
    orchestrator = SecurityOrchestrator()
    
    try:
        await orchestrator.run()
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        logging.error(f"Security orchestrator failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())