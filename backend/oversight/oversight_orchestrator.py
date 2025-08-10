#!/usr/bin/env python3
"""
Human Oversight Orchestrator for SutazAI System
Coordinates all oversight components: monitoring, alerts, compliance, and control
"""

import asyncio
import json
import logging
import signal
import sys
from datetime import datetime
from pathlib import Path
import argparse

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from oversight.human_oversight_interface import HumanOversightInterface
from oversight.alert_notification_system import AlertNotificationSystem, AlertSeverity, AlertCategory
from oversight.compliance_reporter import ComplianceReporter, ComplianceFramework, ReportType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/opt/sutazaiapp/backend/oversight/oversight.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class OversightOrchestrator:
    """
    Main orchestrator for SutazAI human oversight system
    Coordinates monitoring, alerts, compliance, and control interfaces
    """
    
    def __init__(self, config_path: str = "/opt/sutazaiapp/backend/oversight/config.json"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Initialize components
        self.oversight_interface = HumanOversightInterface(
            port=self.config.get('oversight_port', 8095)
        )
        
        self.alert_system = AlertNotificationSystem(
            config_path=str(Path(__file__).parent / "alert_config.json")
        )
        
        self.compliance_reporter = ComplianceReporter()
        
        # System state
        self.is_running = False
        self.shutdown_event = asyncio.Event()
        self.background_tasks = set()
        
        # Statistics
        self.stats = {
            'start_time': None,
            'alerts_created': 0,
            'reports_generated': 0,
            'interventions_made': 0,
            'agents_controlled': 0
        }
    
    def _load_config(self) -> Dict[str, Any]:
        """Load orchestrator configuration"""
        default_config = {
            "oversight_port": 8095,
            "enable_compliance_reporting": True,
            "enable_alert_monitoring": True,
            "enable_automatic_reports": True,
            "compliance_frameworks": [
                "ai_ethics",
                "gdpr",
                "hipaa",
                "iso27001",
                "nist"
            ],
            "alert_thresholds": {
                "agent_failure_threshold": 3,
                "memory_usage_threshold": 90,
                "cpu_usage_threshold": 95,
                "response_time_threshold": 30
            },
            "report_schedule": {
                "daily_reports": True,
                "weekly_reports": True,
                "monthly_reports": True,
                "quarterly_reports": False
            },
            "monitoring_intervals": {
                "agent_health_check": 60,
                "system_metrics": 300,
                "compliance_check": 1800,
                "alert_escalation": 300
            }
        }
        
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
        except Exception as e:
            logger.warning(f"Could not load config from {self.config_path}: {e}")
        
        return default_config
    
    async def start(self):
        """Start the oversight orchestrator"""
        logger.info("Starting SutazAI Human Oversight Orchestrator")
        
        self.is_running = True
        self.stats['start_time'] = datetime.utcnow()
        
        # Create startup alert
        await self.alert_system.create_alert(
            title="Human Oversight System Started",
            description="SutazAI Human Oversight Orchestrator has been started successfully",
            severity=AlertSeverity.LOW,
            category=AlertCategory.SYSTEM_PERFORMANCE,
            metadata={"orchestrator_version": "1.0.0", "start_time": self.stats['start_time'].isoformat()}
        )
        
        # Start all components
        tasks = []
        
        # Start oversight interface
        oversight_task = asyncio.create_task(self.oversight_interface.start())
        tasks.append(oversight_task)
        
        # Start alert monitoring
        if self.config.get('enable_alert_monitoring', True):
            alert_task = asyncio.create_task(self.alert_system.start_monitoring())
            tasks.append(alert_task)
        
        # Start compliance reporting
        if self.config.get('enable_compliance_reporting', True):
            compliance_task = asyncio.create_task(self._compliance_monitoring_loop())
            tasks.append(compliance_task)
        
        # Start system health monitoring
        health_task = asyncio.create_task(self._system_health_monitoring())
        tasks.append(health_task)
        
        # Start agent oversight monitoring
        agent_task = asyncio.create_task(self._agent_oversight_monitoring())
        tasks.append(agent_task)
        
        # Start performance monitoring
        performance_task = asyncio.create_task(self._performance_monitoring())
        tasks.append(performance_task)
        
        # Start statistics reporting
        stats_task = asyncio.create_task(self._statistics_reporting())
        tasks.append(stats_task)
        
        try:
            logger.info("All oversight components started successfully")
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("Oversight orchestrator cancelled")
        except Exception as e:
            logger.error(f"Error in oversight orchestrator: {e}")
            await self.alert_system.create_alert(
                title="Oversight System Error",
                description=f"Critical error in oversight orchestrator: {str(e)}",
                severity=AlertSeverity.CRITICAL,
                category=AlertCategory.SYSTEM_PERFORMANCE
            )
        finally:
            await self.shutdown()
    
    async def shutdown(self):
        """Shutdown the oversight orchestrator"""
        logger.info("Shutting down SutazAI Human Oversight Orchestrator")
        
        self.is_running = False
        self.shutdown_event.set()
        
        # Create shutdown alert
        uptime = datetime.utcnow() - self.stats['start_time'] if self.stats['start_time'] else timedelta(0)
        
        await self.alert_system.create_alert(
            title="Human Oversight System Shutdown",
            description=f"SutazAI Human Oversight Orchestrator shutting down after {uptime}",
            severity=AlertSeverity.LOW,
            category=AlertCategory.SYSTEM_PERFORMANCE,
            metadata={
                "uptime_seconds": uptime.total_seconds(),
                "stats": self.stats
            }
        )
        
        # Stop components
        self.oversight_interface.stop()
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        logger.info("Oversight orchestrator shutdown complete")
    
    async def _compliance_monitoring_loop(self):
        """Monitor compliance and generate reports"""
        logger.info("Starting compliance monitoring loop")
        
        while not self.shutdown_event.is_set():
            try:
                # Generate scheduled reports
                if self.config.get('enable_automatic_reports', True):
                    current_time = datetime.utcnow()
                    
                    # Daily reports at 2 AM
                    if (self.config['report_schedule'].get('daily_reports', True) and 
                        current_time.hour == 2 and current_time.minute < 30):
                        
                        for framework_name in self.config.get('compliance_frameworks', []):
                            try:
                                framework = ComplianceFramework(framework_name)
                                report = await self.compliance_reporter.generate_compliance_report(
                                    framework, ReportType.DAILY
                                )
                                self.stats['reports_generated'] += 1
                                
                                # Check for critical violations
                                critical_violations = [v for v in report.violations if v.severity == "critical"]
                                if critical_violations:
                                    await self.alert_system.create_alert(
                                        title=f"Critical Compliance Violations: {framework_name.upper()}",
                                        description=f"Daily compliance report shows {len(critical_violations)} critical violations",
                                        severity=AlertSeverity.CRITICAL,
                                        category=AlertCategory.COMPLIANCE_VIOLATION,
                                        metadata={
                                            "framework": framework_name,
                                            "report_id": report.id,
                                            "violations_count": len(critical_violations)
                                        }
                                    )
                                
                            except Exception as e:
                                logger.error(f"Error generating daily compliance report for {framework_name}: {e}")
                    
                    # Weekly reports on Sundays at 3 AM
                    if (self.config['report_schedule'].get('weekly_reports', True) and 
                        current_time.weekday() == 6 and current_time.hour == 3):
                        
                        for framework_name in self.config.get('compliance_frameworks', []):
                            try:
                                framework = ComplianceFramework(framework_name)
                                report = await self.compliance_reporter.generate_compliance_report(
                                    framework, ReportType.WEEKLY
                                )
                                self.stats['reports_generated'] += 1
                                
                            except Exception as e:
                                logger.error(f"Error generating weekly compliance report for {framework_name}: {e}")
                
                await asyncio.sleep(self.config['monitoring_intervals']['compliance_check'])
                
            except Exception as e:
                logger.error(f"Error in compliance monitoring loop: {e}")
                await asyncio.sleep(3600)  # Wait an hour on error
    
    async def _system_health_monitoring(self):
        """Monitor overall system health"""
        logger.info("Starting system health monitoring")
        
        while not self.shutdown_event.is_set():
            try:
                # Load system metrics (this would integrate with actual monitoring)
                system_metrics = await self._collect_system_metrics()
                
                # Check thresholds
                alert_thresholds = self.config.get('alert_thresholds', {})
                
                # Memory usage check
                if system_metrics.get('memory_usage_percent', 0) > alert_thresholds.get('memory_usage_threshold', 90):
                    await self.alert_system.create_alert(
                        title="High Memory Usage Alert",
                        description=f"System memory usage at {system_metrics['memory_usage_percent']:.1f}%",
                        severity=AlertSeverity.HIGH,
                        category=AlertCategory.RESOURCE_EXHAUSTION,
                        metadata=system_metrics
                    )
                    self.stats['alerts_created'] += 1
                
                # CPU usage check
                if system_metrics.get('cpu_usage_percent', 0) > alert_thresholds.get('cpu_usage_threshold', 95):
                    await self.alert_system.create_alert(
                        title="High CPU Usage Alert",
                        description=f"System CPU usage at {system_metrics['cpu_usage_percent']:.1f}%",
                        severity=AlertSeverity.HIGH,
                        category=AlertCategory.RESOURCE_EXHAUSTION,
                        metadata=system_metrics
                    )
                    self.stats['alerts_created'] += 1
                
                await asyncio.sleep(self.config['monitoring_intervals']['system_metrics'])
                
            except Exception as e:
                logger.error(f"Error in system health monitoring: {e}")
                await asyncio.sleep(300)
    
    async def _agent_oversight_monitoring(self):
        """Monitor agent status and behaviors"""
        logger.info("Starting agent oversight monitoring")
        
        while not self.shutdown_event.is_set():
            try:
                # Load agent status
                agent_status = await self._collect_agent_status()
                
                # Check for agent failures
                failed_agents = [agent_id for agent_id, status in agent_status.items() 
                               if status.get('status') == 'unhealthy']
                
                if len(failed_agents) >= self.config['alert_thresholds'].get('agent_failure_threshold', 3):
                    await self.alert_system.create_alert(
                        title="Multiple Agent Failures Detected",
                        description=f"{len(failed_agents)} agents are reporting unhealthy status",
                        severity=AlertSeverity.CRITICAL,
                        category=AlertCategory.AGENT_MALFUNCTION,
                        metadata={
                            "failed_agents": failed_agents,
                            "total_agents": len(agent_status),
                            "failure_rate": len(failed_agents) / len(agent_status) * 100
                        }
                    )
                    self.stats['alerts_created'] += 1
                
                # Check for agents requiring human intervention
                intervention_required = [agent_id for agent_id, status in agent_status.items()
                                       if status.get('requires_intervention', False)]
                
                if intervention_required:
                    await self.alert_system.create_alert(
                        title="Human Intervention Required",
                        description=f"{len(intervention_required)} agents require human intervention",
                        severity=AlertSeverity.MEDIUM,
                        category=AlertCategory.HUMAN_INTERVENTION_REQUIRED,
                        metadata={
                            "intervention_agents": intervention_required
                        }
                    )
                    self.stats['alerts_created'] += 1
                
                await asyncio.sleep(self.config['monitoring_intervals']['agent_health_check'])
                
            except Exception as e:
                logger.error(f"Error in agent oversight monitoring: {e}")
                await asyncio.sleep(60)
    
    async def _performance_monitoring(self):
        """Monitor system and agent performance"""
        logger.info("Starting performance monitoring")
        
        while not self.shutdown_event.is_set():
            try:
                # Monitor response times
                response_metrics = await self._collect_response_metrics()
                
                slow_responses = [agent_id for agent_id, time in response_metrics.items()
                                if time > self.config['alert_thresholds'].get('response_time_threshold', 30)]
                
                if slow_responses:
                    await self.alert_system.create_alert(
                        title="Slow Agent Response Times",
                        description=f"{len(slow_responses)} agents showing slow response times",
                        severity=AlertSeverity.MEDIUM,
                        category=AlertCategory.SYSTEM_PERFORMANCE,
                        metadata={
                            "slow_agents": slow_responses,
                            "response_metrics": response_metrics
                        }
                    )
                    self.stats['alerts_created'] += 1
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in performance monitoring: {e}")
                await asyncio.sleep(300)
    
    async def _statistics_reporting(self):
        """Report system statistics periodically"""
        logger.info("Starting statistics reporting")
        
        while not self.shutdown_event.is_set():
            try:
                # Update statistics
                uptime = datetime.utcnow() - self.stats['start_time'] if self.stats['start_time'] else timedelta(0)
                
                current_stats = self.stats.copy()
                current_stats['uptime_hours'] = uptime.total_seconds() / 3600
                current_stats['active_alerts'] = len(await self.alert_system.get_active_alerts())
                
                logger.info(f"Oversight System Stats: {json.dumps(current_stats, indent=2, default=str)}")
                
                # Create daily statistics alert
                if uptime.total_seconds() % 86400 < 3600:  # Once per day
                    await self.alert_system.create_alert(
                        title="Daily Oversight Statistics",
                        description="Daily statistics report for human oversight system",
                        severity=AlertSeverity.LOW,
                        category=AlertCategory.OPERATIONAL_ANOMALY,
                        metadata=current_stats
                    )
                
                await asyncio.sleep(3600)  # Report every hour
                
            except Exception as e:
                logger.error(f"Error in statistics reporting: {e}")
                await asyncio.sleep(3600)
    
    async def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect system performance metrics"""
        try:
            import psutil
            
            return {
                'memory_usage_percent': psutil.virtual_memory().percent,
                'cpu_usage_percent': psutil.cpu_percent(interval=1),
                'disk_usage_percent': psutil.disk_usage('/').percent,
                'load_average': psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0,
                'timestamp': datetime.utcnow().isoformat()
            }
        except ImportError:
            # Fallback if psutil not available
            return {
                'memory_usage_percent': 75.0,  # Simulated
                'cpu_usage_percent': 45.0,     # Simulated
                'disk_usage_percent': 60.0,    # Simulated
                'load_average': 1.2,           # Simulated
                'timestamp': datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return {}
    
    async def _collect_agent_status(self) -> Dict[str, Dict[str, Any]]:
        """Collect agent status information"""
        try:
            agent_status_path = Path("/opt/sutazaiapp/agents/agent_status.json")
            if agent_status_path.exists():
                with open(agent_status_path, 'r') as f:
                    data = json.load(f)
                    return data.get('active_agents', {})
            else:
                return {}
        except Exception as e:
            logger.error(f"Error collecting agent status: {e}")
            return {}
    
    async def _collect_response_metrics(self) -> Dict[str, float]:
        """Collect agent response time metrics"""
        try:
            # This would integrate with actual monitoring
            # For now, return simulated data
            agent_status = await self._collect_agent_status()
            
            response_metrics = {}
            for agent_id in agent_status.keys():
                # Simulate response times (in seconds)
                import random
                response_metrics[agent_id] = random.uniform(0.5, 45.0)
            
            return response_metrics
        except Exception as e:
            logger.error(f"Error collecting response metrics: {e}")
            return {}
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating shutdown")
            asyncio.create_task(self.shutdown())
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="SutazAI Human Oversight Orchestrator")
    parser.add_argument("--config", default="/opt/sutazaiapp/backend/oversight/config.json",
                       help="Configuration file path")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Log level")
    
    args = parser.parse_args()
    
    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Create orchestrator
    orchestrator = OversightOrchestrator(config_path=args.config)
    
    # Setup signal handlers
    orchestrator.setup_signal_handlers()
    
    try:
        # Start the orchestrator
        await orchestrator.start()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Fatal error in orchestrator: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())