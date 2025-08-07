"""
Honeypot Integration Layer for SutazAI System
Integrates honeypot infrastructure with existing security systems and provides unified management
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import threading

# Import existing security infrastructure
try:
    from app.core.security import security_manager, SecurityEvent
    SECURITY_INTEGRATION = True
except ImportError:
    SECURITY_INTEGRATION = False
    security_manager = None

# Import honeypot components
from security.honeypot_infrastructure import HoneypotOrchestrator, honeypot_orchestrator
from security.cowrie_honeypot import CowrieIntegration, cowrie_integration
from security.web_honeypot import WebHoneypotManager
from security.database_honeypot import DatabaseHoneypotManager
from security.ai_agent_honeypot import AIAgentHoneypotManager

logger = logging.getLogger(__name__)

class HoneypotSecurityBridge:
    """Bridge between honeypot system and existing security infrastructure"""
    
    def __init__(self):
        self.is_active = False
        self.alert_threshold = {
            'critical': 1,   # Alert immediately on critical events
            'high': 3,       # Alert after 3 high-severity events
            'medium': 10,    # Alert after 10 medium-severity events
            'low': 50        # Alert after 50 low-severity events
        }
        self.event_counts = {
            'critical': 0,
            'high': 0,
            'medium': 0,
            'low': 0
        }
        self.last_reset = datetime.utcnow()
        
    async def integrate_honeypot_event(self, honeypot_event) -> bool:
        """Integrate honeypot event with security system"""
        if not SECURITY_INTEGRATION or not security_manager:
            return False
            
        try:
            # Convert honeypot event to security event
            security_event_type = self._map_event_type(honeypot_event.event_type)
            
            # Determine security level
            security_level = self._map_severity_to_level(honeypot_event.severity)
            
            # Create security event details
            event_details = {
                'honeypot_id': honeypot_event.honeypot_id,
                'honeypot_type': honeypot_event.honeypot_type,
                'attack_vector': honeypot_event.attack_vector,
                'threat_indicators': honeypot_event.threat_indicators,
                'payload_preview': honeypot_event.payload[:200] if honeypot_event.payload else '',
                'session_id': honeypot_event.session_id,
                'destination_port': honeypot_event.destination_port
            }
            
            # Add credentials if present (sanitized)
            if honeypot_event.credentials:
                event_details['credentials_attempted'] = {
                    'username': honeypot_event.credentials.get('username', ''),
                    'password_length': len(honeypot_event.credentials.get('password', ''))
                }
            
            # Log to security system
            await security_manager.audit.log_event(
                security_event_type,
                security_level,
                f"honeypot_{honeypot_event.honeypot_type}",
                event_details,
                user_id=None,  # No user for honeypot events
                ip_address=honeypot_event.source_ip
            )
            
            # Check alert thresholds
            await self._check_alert_thresholds(honeypot_event)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to integrate honeypot event with security system: {e}")
            return False
    
    def _map_event_type(self, honeypot_event_type: str) -> str:
        """Map honeypot event type to security event type"""
        event_mapping = {
            # SSH events
            'ssh_connection_attempt': 'ssh_honeypot_connection',
            'ssh_auth_attempt': 'ssh_honeypot_auth_failure',
            
            # Web events
            'web_root_access': 'web_honeypot_access',
            'admin_panel_access': 'web_honeypot_admin_access',
            'admin_login_attempt': 'web_honeypot_credential_harvesting',
            'api_root_access': 'web_honeypot_api_enumeration',
            'wordpress_login_attempt': 'web_honeypot_cms_attack',
            'phpmyadmin_login_attempt': 'web_honeypot_database_attack',
            
            # Database events
            'mysql_connection_attempt': 'database_honeypot_connection',
            'mysql_auth_attempt': 'database_honeypot_auth_failure', 
            'mysql_query': 'database_honeypot_sql_injection',
            'postgresql_connection_attempt': 'database_honeypot_connection',
            'postgresql_auth_attempt': 'database_honeypot_auth_failure',
            'postgresql_query': 'database_honeypot_sql_injection',
            'redis_connection_attempt': 'database_honeypot_connection',
            'redis_command': 'database_honeypot_command_injection',
            
            # AI Agent events
            'health_check': 'ai_honeypot_reconnaissance',
            'models_list': 'ai_honeypot_enumeration',
            'coordinator_task': 'ai_honeypot_exploitation',
            'chat_completion': 'ai_honeypot_prompt_injection',
            'code_review': 'ai_honeypot_code_injection',
            'admin_config_access': 'ai_honeypot_privilege_escalation',
            'file_upload_attempt': 'ai_honeypot_file_upload_attack'
        }
        
        return event_mapping.get(honeypot_event_type, 'honeypot_unknown_event')
    
    def _map_severity_to_level(self, severity: str) -> str:
        """Map honeypot severity to security system severity"""
        severity_mapping = {
            'critical': 'critical',
            'high': 'high', 
            'medium': 'warning',
            'low': 'info',
            'info': 'info'
        }
        
        return severity_mapping.get(severity, 'info')
    
    async def _check_alert_thresholds(self, honeypot_event):
        """Check if alert thresholds are exceeded"""
        severity = honeypot_event.severity
        
        # Increment counter
        if severity in self.event_counts:
            self.event_counts[severity] += 1
        
        # Check if threshold exceeded
        if self.event_counts[severity] >= self.alert_threshold[severity]:
            await self._send_threshold_alert(severity, honeypot_event)
            
            # Reset counter for this severity
            self.event_counts[severity] = 0
    
    async def _send_threshold_alert(self, severity: str, latest_event):
        """Send alert when threshold is exceeded"""
        if not SECURITY_INTEGRATION or not security_manager:
            return
            
        alert_details = {
            'alert_type': 'honeypot_threshold_exceeded',
            'severity': severity,
            'threshold': self.alert_threshold[severity],
            'time_window': '1 hour',
            'latest_event': {
                'honeypot_id': latest_event.honeypot_id,
                'source_ip': latest_event.source_ip,
                'event_type': latest_event.event_type,
                'attack_vector': latest_event.attack_vector
            }
        }
        
        await security_manager.audit.log_event(
            'honeypot_alert_threshold_exceeded',
            'critical' if severity in ['critical', 'high'] else 'high',
            'honeypot_security_bridge',
            alert_details,
            ip_address=latest_event.source_ip
        )
        
        logger.critical(f"Honeypot alert threshold exceeded: {severity} severity")
    
    async def reset_counters(self):
        """Reset event counters (called periodically)"""
        self.event_counts = {severity: 0 for severity in self.event_counts}
        self.last_reset = datetime.utcnow()
        logger.info("Honeypot alert counters reset")

class UnifiedHoneypotManager:
    """Unified manager for all honeypot types"""
    
    def __init__(self):
        self.orchestrator = honeypot_orchestrator
        self.cowrie_integration = cowrie_integration
        self.web_manager = None
        self.database_manager = None
        self.ai_agent_manager = None
        self.security_bridge = HoneypotSecurityBridge()
        self.is_deployed = False
        self.deployment_config = {}
        
    async def initialize(self):
        """Initialize all honeypot managers"""
        try:
            # Initialize managers with shared database and intelligence engine
            database = self.orchestrator.database
            intelligence_engine = self.orchestrator.intelligence_engine
            
            self.web_manager = WebHoneypotManager(database, intelligence_engine)
            self.database_manager = DatabaseHoneypotManager(database, intelligence_engine)
            self.ai_agent_manager = AIAgentHoneypotManager(database, intelligence_engine)
            
            # Set global instances for other modules
            import security.web_honeypot
            import security.database_honeypot
            import security.ai_agent_honeypot
            
            security.web_honeypot.web_honeypot_manager = self.web_manager
            security.database_honeypot.database_honeypot_manager = self.database_manager
            security.ai_agent_honeypot.ai_agent_honeypot_manager = self.ai_agent_manager
            
            logger.info("Unified honeypot manager initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize unified honeypot manager: {e}")
            raise
    
    async def deploy_comprehensive_honeypot_infrastructure(self) -> Dict[str, Any]:
        """Deploy comprehensive honeypot infrastructure"""
        try:
            logger.info("Starting comprehensive honeypot infrastructure deployment...")
            
            deployment_results = {
                'orchestrator': False,
                'cowrie_ssh': False,
                'web_honeypots': {},
                'database_honeypots': {},
                'ai_agent_honeypots': {},
                'security_integration': False,
                'deployment_time': datetime.utcnow().isoformat()
            }
            
            # 1. Deploy main orchestrator
            try:
                await self.orchestrator.start()
                deployment_results['orchestrator'] = True
                logger.info("✓ Honeypot orchestrator deployed")
            except Exception as e:
                logger.error(f"✗ Failed to deploy orchestrator: {e}")
            
            # 2. Deploy Cowrie SSH honeypot
            try:
                cowrie_result = await self.cowrie_integration.deploy()
                deployment_results['cowrie_ssh'] = cowrie_result
                if cowrie_result:
                    logger.info("✓ Cowrie SSH honeypot deployed")
                else:
                    logger.warning("✗ Cowrie SSH honeypot deployment failed")
            except Exception as e:
                logger.error(f"✗ Cowrie deployment error: {e}")
            
            # 3. Deploy web honeypots
            try:
                # HTTP honeypot
                http_result = await self.web_manager.deploy_http_honeypot(8080)
                deployment_results['web_honeypots']['http_8080'] = http_result
                
                # HTTPS honeypot
                https_result = await self.web_manager.deploy_https_honeypot(8443)
                deployment_results['web_honeypots']['https_8443'] = https_result
                
                if http_result:
                    logger.info("✓ HTTP honeypot deployed on port 8080")
                if https_result:
                    logger.info("✓ HTTPS honeypot deployed on port 8443")
                    
            except Exception as e:
                logger.error(f"✗ Web honeypot deployment error: {e}")
            
            # 4. Deploy database honeypots
            try:
                db_results = await self.database_manager.deploy_all_database_honeypots()
                deployment_results['database_honeypots'] = db_results
                
                for db_type, result in db_results.items():
                    if result:
                        logger.info(f"✓ {db_type.upper()} honeypot deployed")
                    else:
                        logger.warning(f"✗ {db_type.upper()} honeypot deployment failed")
                        
            except Exception as e:
                logger.error(f"✗ Database honeypot deployment error: {e}")
            
            # 5. Deploy AI agent honeypots
            try:
                ai_results = await self.ai_agent_manager.deploy_multiple_ai_honeypots()
                deployment_results['ai_agent_honeypots'] = ai_results
                
                for port_config, result in ai_results.items():
                    if result:
                        logger.info(f"✓ AI agent honeypot deployed ({port_config})")
                    else:
                        logger.warning(f"✗ AI agent honeypot deployment failed ({port_config})")
                        
            except Exception as e:
                logger.error(f"✗ AI agent honeypot deployment error: {e}")
            
            # 6. Activate security integration
            try:
                self.security_bridge.is_active = True
                deployment_results['security_integration'] = True
                logger.info("✓ Security integration activated")
                
                # Start periodic counter reset
                asyncio.create_task(self._periodic_counter_reset())
                
            except Exception as e:
                logger.error(f"✗ Security integration error: {e}")
            
            # Calculate overall success
            successful_components = sum([
                deployment_results['orchestrator'],
                deployment_results['cowrie_ssh'],
                any(deployment_results['web_honeypots'].values()),
                any(deployment_results['database_honeypots'].values()),
                any(deployment_results['ai_agent_honeypots'].values()),
                deployment_results['security_integration']
            ])
            
            self.is_deployed = successful_components >= 3  # At least 3 components working
            self.deployment_config = deployment_results
            
            if self.is_deployed:
                logger.info(f"🎯 Honeypot infrastructure deployed successfully! ({successful_components}/6 components active)")
                
                # Generate deployment summary
                await self._generate_deployment_summary(deployment_results)
                
            else:
                logger.error(f"❌ Honeypot infrastructure deployment failed ({successful_components}/6 components active)")
            
            return deployment_results
            
        except Exception as e:
            logger.error(f"Comprehensive deployment failed: {e}")
            return {'error': str(e), 'deployment_time': datetime.utcnow().isoformat()}
    
    async def _generate_deployment_summary(self, results: Dict[str, Any]):
        """Generate deployment summary report"""
        try:
            active_honeypots = []
            
            if results['orchestrator']:
                orchestrator_status = await self.orchestrator.get_status()
                active_honeypots.extend([hp['id'] for hp in orchestrator_status['honeypot_details']])
            
            if results['cowrie_ssh']:
                active_honeypots.append('cowrie_ssh_2222')
            
            for hp_type, hp_results in results.get('web_honeypots', {}).items():
                if hp_results:
                    active_honeypots.append(f'web_{hp_type}')
            
            for hp_type, hp_result in results.get('database_honeypots', {}).items():
                if hp_result:
                    active_honeypots.append(f'database_{hp_type}')
            
            for hp_config, hp_result in results.get('ai_agent_honeypots', {}).items():
                if hp_result:
                    active_honeypots.append(f'ai_agent_{hp_config}')
            
            summary = {
                'deployment_time': results['deployment_time'],
                'total_honeypots': len(active_honeypots),
                'active_honeypots': active_honeypots,
                'security_integration': results['security_integration'],
                'threat_detection_ready': True,
                'intelligence_gathering_active': True
            }
            
            # Log summary to security system if available
            if SECURITY_INTEGRATION and security_manager:
                await security_manager.audit.log_event(
                    'honeypot_infrastructure_deployed',
                    'info',
                    'honeypot_unified_manager',
                    summary
                )
            
            logger.info(f"Deployment Summary: {json.dumps(summary, indent=2)}")
            
        except Exception as e:
            logger.error(f"Failed to generate deployment summary: {e}")
    
    async def _periodic_counter_reset(self):
        """Periodically reset alert counters"""
        while self.is_deployed:
            try:
                await asyncio.sleep(3600)  # Reset every hour
                await self.security_bridge.reset_counters()
            except Exception as e:
                logger.error(f"Error in periodic counter reset: {e}")
    
    async def undeploy_all(self):
        """Undeploy all honeypot infrastructure"""
        try:
            logger.info("Undeploying honeypot infrastructure...")
            
            # Stop orchestrator
            if self.orchestrator:
                await self.orchestrator.stop()
            
            # Stop Cowrie
            if self.cowrie_integration:
                await self.cowrie_integration.undeploy()
            
            # Stop web honeypots
            if self.web_manager:
                await self.web_manager.stop_all()
            
            # Stop database honeypots
            if self.database_manager:
                await self.database_manager.stop_all()
            
            # Stop AI agent honeypots
            if self.ai_agent_manager:
                await self.ai_agent_manager.stop_all()
            
            # Deactivate security integration
            self.security_bridge.is_active = False
            
            self.is_deployed = False
            logger.info("Honeypot infrastructure undeployed successfully")
            
        except Exception as e:
            logger.error(f"Error during undeployment: {e}")
    
    async def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all honeypot infrastructure"""
        try:
            status = {
                'deployment_status': 'deployed' if self.is_deployed else 'not_deployed',
                'deployment_time': self.deployment_config.get('deployment_time'),
                'security_integration': self.security_bridge.is_active,
                'components': {}
            }
            
            # Orchestrator status
            if self.orchestrator:
                status['components']['orchestrator'] = await self.orchestrator.get_status()
            
            # Cowrie status  
            if self.cowrie_integration:
                status['components']['cowrie'] = self.cowrie_integration.get_status()
            
            # Web honeypots status
            if self.web_manager:
                status['components']['web_honeypots'] = self.web_manager.get_status()
            
            # Database honeypots status
            if self.database_manager:
                status['components']['database_honeypots'] = self.database_manager.get_status()
            
            # AI agent honeypots status
            if self.ai_agent_manager:
                status['components']['ai_agent_honeypots'] = self.ai_agent_manager.get_status()
            
            # Recent activity summary
            if self.orchestrator and self.orchestrator.database:
                recent_events = self.orchestrator.database.get_events(limit=100, hours=1)
                status['recent_activity'] = {
                    'events_last_hour': len(recent_events),
                    'unique_attackers': len(set(e.source_ip for e in recent_events)),
                    'high_severity_events': len([e for e in recent_events if e.severity in ['critical', 'high']]),
                    'attack_types': list(set(e.attack_vector for e in recent_events if e.attack_vector))
                }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting comprehensive status: {e}")
            return {'error': str(e)}
    
    async def generate_threat_intelligence_report(self) -> Dict[str, Any]:
        """Generate comprehensive threat intelligence report"""
        try:
            if not self.orchestrator or not self.orchestrator.database:
                return {'error': 'Honeypot infrastructure not available'}
            
            # Get events from last 24 hours
            events = self.orchestrator.database.get_events(limit=1000, hours=24)
            
            report = {
                'report_time': datetime.utcnow().isoformat(),
                'time_period': '24 hours',
                'summary': {
                    'total_events': len(events),
                    'unique_attackers': len(set(e.source_ip for e in events)),
                    'honeypots_hit': len(set(e.honeypot_id for e in events))
                },
                'attack_analysis': {},
                'top_attackers': [],
                'attack_patterns': {},
                'honeypot_effectiveness': {},
                'threat_trends': {}
            }
            
            # Analyze attack types
            attack_vectors = {}
            severity_breakdown = {}
            honeypot_activity = {}
            
            for event in events:
                # Attack vectors
                if event.attack_vector:
                    attack_vectors[event.attack_vector] = attack_vectors.get(event.attack_vector, 0) + 1
                
                # Severity breakdown
                severity_breakdown[event.severity] = severity_breakdown.get(event.severity, 0) + 1
                
                # Honeypot activity
                honeypot_activity[event.honeypot_type] = honeypot_activity.get(event.honeypot_type, 0) + 1
            
            report['attack_analysis'] = {
                'attack_vectors': attack_vectors,
                'severity_breakdown': severity_breakdown,
                'honeypot_activity': honeypot_activity
            }
            
            # Top attackers
            attacker_counts = {}
            for event in events:
                attacker_counts[event.source_ip] = attacker_counts.get(event.source_ip, 0) + 1
            
            top_attackers = sorted(attacker_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            report['top_attackers'] = [
                {'ip': ip, 'attempts': count} for ip, count in top_attackers
            ]
            
            # Add recommendations
            report['recommendations'] = self._generate_threat_recommendations(events)
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating threat intelligence report: {e}")
            return {'error': str(e)}
    
    def _generate_threat_recommendations(self, events: List) -> List[str]:
        """Generate threat-based recommendations"""
        recommendations = []
        
        # Check for high-volume attacks
        if len(events) > 500:
            recommendations.append(
                "High attack volume detected. Consider implementing rate limiting and IP blocking."
            )
        
        # Check for SQL injection attempts
        sql_events = [e for e in events if e.attack_vector == 'sql_injection']
        if len(sql_events) > 10:
            recommendations.append(
                f"Multiple SQL injection attempts detected ({len(sql_events)} events). "
                "Ensure all database inputs are properly sanitized."
            )
        
        # Check for credential harvesting
        cred_events = [e for e in events if 'credential' in e.event_type or 'login' in e.event_type]
        if len(cred_events) > 20:
            recommendations.append(
                f"Extensive credential harvesting detected ({len(cred_events)} attempts). "
                "Consider implementing multi-factor authentication and account lockout policies."
            )
        
        # Check for AI-specific attacks
        ai_events = [e for e in events if e.honeypot_type == 'ai_agent']
        if len(ai_events) > 5:
            recommendations.append(
                f"AI service targeting detected ({len(ai_events)} events). "
                "Implement AI-specific security measures and input validation."
            )
        
        return recommendations

# Global unified honeypot manager instance
unified_honeypot_manager = UnifiedHoneypotManager()