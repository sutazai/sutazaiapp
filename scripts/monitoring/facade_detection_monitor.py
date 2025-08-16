#!/usr/bin/env python3
"""
Facade Detection Monitor - Real-time Production Monitoring
==========================================================

This module provides real-time monitoring to detect facade implementations in production.
It continuously monitors system behavior and alerts when facade patterns are detected.

CRITICAL PURPOSE: Provide early warning system for facade implementations that might
slip through CI/CD and appear in production, preventing system degradation.
"""

import asyncio
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import schedule
from pathlib import Path
import httpx
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FacadeDetectionMonitor:
    """
    Real-time monitor for facade implementations in production.
    
    This monitor continuously watches for signs of facade implementations
    and alerts when potential issues are detected.
    """
    
    def __init__(self, config_file: str = "/opt/sutazaiapp/config/facade_monitor.json"):
        self.config_file = Path(config_file)
        self.config = self.load_config()
        self.monitoring_active = True
        self.client = None
        
        # Facade detection thresholds
        self.thresholds = {
            "empty_response_ratio": 0.3,        # 30% empty responses indicates facade
            "error_response_ratio": 0.5,        # 50% error responses indicates issues
            "response_time_degradation": 3.0,   # 3x normal response time
            "service_unavailable_ratio": 0.2,   # 20% services unavailable
            "health_check_failure_ratio": 0.4   # 40% health checks failing
        }
        
        # Monitoring intervals
        self.intervals = {
            "quick_check": 60,      # 1 minute
            "full_check": 300,      # 5 minutes
            "deep_scan": 900,       # 15 minutes
            "health_report": 3600   # 1 hour
        }
        
        # Alert configuration
        self.alert_config = self.config.get("alerts", {})
        self.notification_channels = []
        
        # Historical data storage
        self.history = {
            "facade_detections": [],
            "system_metrics": [],
            "alert_history": []
        }
        
    def load_config(self) -> Dict:
        """Load monitoring configuration."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load config: {e}")
        
        # Default configuration
        return {
            "base_url": "http://localhost:10010",
            "frontend_url": "http://localhost:10011",
            "monitoring_enabled": True,
            "alerts": {
                "email_enabled": False,
                "webhook_enabled": False,
                "log_level": "WARNING"
            },
            "retention_hours": 168  # 1 week
        }
    
    async def __aenter__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.client.aclose()
    
    async def detect_api_facades(self) -> Dict:
        """Detect facade implementations in API responses."""
        logger.info("🔍 Scanning for API facade implementations...")
        
        base_url = self.config["base_url"]
        api_endpoints = [
            "/health",
            "/api/v1/system/",
            "/api/v1/mesh/v2/services",
            "/api/v1/models/",
            "/api/v1/agents/",
            "/api/v1/hardware/",
            "/api/v1/cache/"
        ]
        
        facade_indicators = {
            "empty_responses": 0,
            "error_responses": 0,
            "slow_responses": 0,
            "total_endpoints": len(api_endpoints),
            "facade_score": 0.0,
            "suspicious_endpoints": []
        }
        
        for endpoint in api_endpoints:
            try:
                start_time = time.time()
                response = await self.client.get(f"{base_url}{endpoint}")
                response_time = time.time() - start_time
                
                endpoint_analysis = {
                    "endpoint": endpoint,
                    "status_code": response.status_code,
                    "response_time": response_time,
                    "content_length": len(response.content),
                    "facade_indicators": []
                }
                
                # Check for facade indicators
                if response.status_code >= 500:
                    facade_indicators["error_responses"] += 1
                    endpoint_analysis["facade_indicators"].append("server_error")
                
                elif response.status_code == 200:
                    try:
                        data = response.json()
                        
                        # Check for empty or minimal responses
                        if not data or len(str(data)) < 10:
                            facade_indicators["empty_responses"] += 1
                            endpoint_analysis["facade_indicators"].append("empty_response")
                        
                        # Check for obvious facade patterns
                        if isinstance(data, dict):
                            if data.get("status") == "ok" and len(data) == 1:
                                endpoint_analysis["facade_indicators"].append("minimal_ok_response")
                            
                            if "placeholder" in str(data).lower() or "mock" in str(data).lower():
                                endpoint_analysis["facade_indicators"].append("placeholder_content")
                        
                        # Check for suspiciously fast responses that should take time
                        if endpoint in ["/api/v1/models/", "/api/v1/agents/"] and response_time < 0.1:
                            endpoint_analysis["facade_indicators"].append("suspiciously_fast")
                    
                    except json.JSONDecodeError:
                        # Non-JSON response might be facade
                        if response.status_code == 200:
                            endpoint_analysis["facade_indicators"].append("non_json_200_response")
                
                # Check for slow responses
                if response_time > 5.0:
                    facade_indicators["slow_responses"] += 1
                    endpoint_analysis["facade_indicators"].append("slow_response")
                
                # Add to suspicious endpoints if indicators found
                if endpoint_analysis["facade_indicators"]:
                    facade_indicators["suspicious_endpoints"].append(endpoint_analysis)
                
            except Exception as e:
                logger.error(f"Failed to check endpoint {endpoint}: {e}")
                facade_indicators["error_responses"] += 1
                facade_indicators["suspicious_endpoints"].append({
                    "endpoint": endpoint,
                    "error": str(e),
                    "facade_indicators": ["connection_error"]
                })
        
        # Calculate facade score
        total_indicators = (
            facade_indicators["empty_responses"] + 
            facade_indicators["error_responses"] + 
            facade_indicators["slow_responses"]
        )
        facade_indicators["facade_score"] = total_indicators / facade_indicators["total_endpoints"]
        
        return facade_indicators
    
    async def detect_service_mesh_facades(self) -> Dict:
        """Detect facade implementations in service mesh."""
        logger.info("🔍 Scanning for service mesh facade implementations...")
        
        base_url = self.config["base_url"]
        
        try:
            # Check service discovery
            response = await self.client.get(f"{base_url}/api/v1/mesh/v2/services")
            
            if response.status_code != 200:
                return {
                    "service_mesh_available": False,
                    "error": f"Service discovery failed: {response.status_code}",
                    "facade_score": 1.0
                }
            
            services_data = response.json()
            services = services_data.get("services", [])
            
            facade_indicators = {
                "total_services": len(services),
                "unreachable_services": 0,
                "facade_services": 0,
                "service_mesh_healthy": True,
                "facade_score": 0.0,
                "service_details": []
            }
            
            # Test connectivity to each service
            for service in services[:10]:  # Test first 10 services
                service_name = service.get("name", "unknown")
                service_address = service.get("address", "")
                service_port = service.get("port", 0)
                
                service_detail = {
                    "name": service_name,
                    "address": service_address,
                    "port": service_port,
                    "reachable": False,
                    "facade_indicators": []
                }
                
                # Test service connectivity
                if service_address and service_port:
                    try:
                        import socket
                        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        sock.settimeout(3.0)
                        result = sock.connect_ex((service_address, service_port))
                        sock.close()
                        
                        if result == 0:
                            service_detail["reachable"] = True
                        else:
                            facade_indicators["unreachable_services"] += 1
                            service_detail["facade_indicators"].append("unreachable")
                    
                    except Exception as e:
                        facade_indicators["unreachable_services"] += 1
                        service_detail["facade_indicators"].append(f"connection_error: {e}")
                
                facade_indicators["service_details"].append(service_detail)
            
            # Calculate facade score
            if facade_indicators["total_services"] > 0:
                unreachable_ratio = facade_indicators["unreachable_services"] / facade_indicators["total_services"]
                facade_indicators["facade_score"] = unreachable_ratio
                facade_indicators["service_mesh_healthy"] = unreachable_ratio < self.thresholds["service_unavailable_ratio"]
            
            return facade_indicators
            
        except Exception as e:
            return {
                "service_mesh_available": False,
                "error": str(e),
                "facade_score": 1.0
            }
    
    async def detect_container_health_facades(self) -> Dict:
        """Detect facade implementations in container health."""
        logger.info("🔍 Scanning for container health facade implementations...")
        
        try:
            import docker
            client = docker.from_env()
            
            containers = client.containers.list(all=True)
            sutazai_containers = [c for c in containers if "sutazai" in c.name.lower()]
            
            facade_indicators = {
                "total_containers": len(sutazai_containers),
                "healthy_containers": 0,
                "unhealthy_containers": 0,
                "facade_containers": 0,
                "facade_score": 0.0,
                "container_details": []
            }
            
            for container in sutazai_containers:
                container_detail = {
                    "name": container.name,
                    "status": container.status,
                    "health": "unknown",
                    "facade_indicators": []
                }
                
                # Check Docker health status
                docker_health = container.attrs.get("State", {}).get("Health", {}).get("Status", "none")
                container_detail["health"] = docker_health
                
                # Check for facade patterns
                if container.status == "running" and docker_health in ["healthy", "none"]:
                    # Container claims to be healthy, verify with actual health check
                    try:
                        # Try to access container's expected service
                        # This is a simplified check - in practice, we'd have more specific health checks
                        if "backend" in container.name:
                            health_response = await self.client.get("http://localhost:10010/health")
                            if health_response.status_code != 200:
                                container_detail["facade_indicators"].append("health_endpoint_fails")
                                facade_indicators["facade_containers"] += 1
                            else:
                                facade_indicators["healthy_containers"] += 1
                        else:
                            facade_indicators["healthy_containers"] += 1
                    
                    except Exception:
                        container_detail["facade_indicators"].append("health_check_failed")
                        facade_indicators["facade_containers"] += 1
                
                elif container.status != "running":
                    facade_indicators["unhealthy_containers"] += 1
                    container_detail["facade_indicators"].append("not_running")
                
                facade_indicators["container_details"].append(container_detail)
            
            # Calculate facade score
            if facade_indicators["total_containers"] > 0:
                facade_ratio = facade_indicators["facade_containers"] / facade_indicators["total_containers"]
                facade_indicators["facade_score"] = facade_ratio
            
            client.close()
            return facade_indicators
            
        except Exception as e:
            return {
                "container_health_available": False,
                "error": str(e),
                "facade_score": 1.0
            }
    
    async def run_comprehensive_facade_scan(self) -> Dict:
        """Run comprehensive facade detection scan."""
        logger.info("🛡️ Running comprehensive facade detection scan...")
        
        scan_start = datetime.now()
        
        scan_result = {
            "timestamp": scan_start.isoformat(),
            "scan_type": "comprehensive_facade_detection",
            "components": {}
        }
        
        # Run all facade detection components
        components = [
            ("api_facades", self.detect_api_facades),
            ("service_mesh_facades", self.detect_service_mesh_facades),
            ("container_health_facades", self.detect_container_health_facades)
        ]
        
        total_facade_score = 0.0
        critical_issues = []
        
        for component_name, detection_method in components:
            try:
                component_result = await detection_method()
                scan_result["components"][component_name] = component_result
                
                facade_score = component_result.get("facade_score", 0.0)
                total_facade_score += facade_score
                
                # Check for critical issues
                if facade_score > 0.5:  # More than 50% facade indicators
                    critical_issues.append({
                        "component": component_name,
                        "facade_score": facade_score,
                        "severity": "critical" if facade_score > 0.8 else "warning"
                    })
                
            except Exception as e:
                logger.error(f"Facade detection failed for {component_name}: {e}")
                scan_result["components"][component_name] = {
                    "error": str(e),
                    "facade_score": 1.0  # Assume worst case on error
                }
                total_facade_score += 1.0
        
        # Calculate overall facade score
        num_components = len(components)
        overall_facade_score = total_facade_score / num_components if num_components > 0 else 0.0
        
        scan_result.update({
            "scan_duration": (datetime.now() - scan_start).total_seconds(),
            "overall_facade_score": overall_facade_score,
            "critical_issues": critical_issues,
            "system_health": "healthy" if overall_facade_score < 0.2 else "degraded" if overall_facade_score < 0.5 else "critical",
            "facade_detected": overall_facade_score > 0.3
        })
        
        # Store in history
        self.history["facade_detections"].append(scan_result)
        
        # Trigger alerts if needed
        if scan_result["facade_detected"]:
            await self.trigger_facade_alert(scan_result)
        
        return scan_result
    
    async def trigger_facade_alert(self, scan_result: Dict):
        """Trigger alerts for facade detection."""
        logger.warning(f"🚨 FACADE IMPLEMENTATION DETECTED - Score: {scan_result['overall_facade_score']:.2f}")
        
        alert = {
            "timestamp": datetime.now().isoformat(),
            "type": "facade_detection",
            "severity": "critical" if scan_result["overall_facade_score"] > 0.8 else "warning",
            "facade_score": scan_result["overall_facade_score"],
            "critical_issues": scan_result["critical_issues"],
            "scan_details": scan_result
        }
        
        # Store alert
        self.history["alert_history"].append(alert)
        
        # Send notifications
        await self.send_notifications(alert)
    
    async def send_notifications(self, alert: Dict):
        """Send notifications for facade alerts."""
        # Log alert
        severity = alert["severity"]
        if severity == "critical":
            logger.critical(f"CRITICAL FACADE ALERT: {alert}")
        else:
            logger.warning(f"FACADE WARNING: {alert}")
        
        # Email notification (if configured)
        if self.alert_config.get("email_enabled", False):
            await self.send_email_alert(alert)
        
        # Webhook notification (if configured)
        if self.alert_config.get("webhook_enabled", False):
            await self.send_webhook_alert(alert)
        
        # Write to alert file
        alert_file = Path("/opt/sutazaiapp/logs/facade_alerts.json")
        alert_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if alert_file.exists():
                with open(alert_file, 'r') as f:
                    alerts = json.load(f)
            else:
                alerts = []
            
            alerts.append(alert)
            
            # Keep only last 100 alerts
            alerts = alerts[-100:]
            
            with open(alert_file, 'w') as f:
                json.dump(alerts, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to write alert file: {e}")
    
    async def send_email_alert(self, alert: Dict):
        """Send email alert."""
        try:
            # This is a placeholder - configure with actual SMTP settings
            logger.info(f"Would send email alert: {alert['type']} - {alert['severity']}")
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
    
    async def send_webhook_alert(self, alert: Dict):
        """Send webhook alert."""
        try:
            webhook_url = self.alert_config.get("webhook_url")
            if webhook_url:
                response = await self.client.post(webhook_url, json=alert)
                logger.info(f"Webhook alert sent: {response.status_code}")
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
    
    def cleanup_history(self):
        """Clean up old history data."""
        retention_hours = self.config.get("retention_hours", 168)
        cutoff_time = datetime.now() - timedelta(hours=retention_hours)
        
        for history_type in self.history:
            self.history[history_type] = [
                item for item in self.history[history_type]
                if datetime.fromisoformat(item.get("timestamp", "1970-01-01")) > cutoff_time
            ]
    
    def get_facade_status_report(self) -> Dict:
        """Get comprehensive facade status report."""
        recent_detections = [
            detection for detection in self.history["facade_detections"]
            if datetime.fromisoformat(detection["timestamp"]) > datetime.now() - timedelta(hours=24)
        ]
        
        recent_alerts = [
            alert for alert in self.history["alert_history"]
            if datetime.fromisoformat(alert["timestamp"]) > datetime.now() - timedelta(hours=24)
        ]
        
        if recent_detections:
            latest_scan = recent_detections[-1]
            avg_facade_score = sum(d["overall_facade_score"] for d in recent_detections) / len(recent_detections)
        else:
            latest_scan = None
            avg_facade_score = 0.0
        
        return {
            "timestamp": datetime.now().isoformat(),
            "monitoring_active": self.monitoring_active,
            "latest_scan": latest_scan,
            "recent_detections_24h": len(recent_detections),
            "recent_alerts_24h": len(recent_alerts),
            "average_facade_score_24h": avg_facade_score,
            "system_status": "healthy" if avg_facade_score < 0.2 else "at_risk" if avg_facade_score < 0.5 else "critical",
            "recommendations": self.get_recommendations(avg_facade_score, recent_alerts)
        }
    
    def get_recommendations(self, facade_score: float, recent_alerts: List) -> List[str]:
        """Get recommendations based on facade detection results."""
        recommendations = []
        
        if facade_score > 0.5:
            recommendations.append("URGENT: High facade implementation detected - run comprehensive system audit")
        elif facade_score > 0.3:
            recommendations.append("WARNING: Potential facade implementations detected - investigate suspicious components")
        
        if len(recent_alerts) > 5:
            recommendations.append("Multiple facade alerts in 24h - consider system health review")
        
        if not recommendations:
            recommendations.append("System appears healthy - continue regular monitoring")
        
        return recommendations
    
    async def start_monitoring(self):
        """Start continuous facade monitoring."""
        logger.info("🛡️ Starting facade detection monitoring...")
        
        # Schedule different types of scans
        schedule.every(self.intervals["quick_check"]).seconds.do(
            lambda: asyncio.create_task(self.run_quick_facade_check())
        )
        
        schedule.every(self.intervals["full_check"]).seconds.do(
            lambda: asyncio.create_task(self.run_comprehensive_facade_scan())
        )
        
        schedule.every(self.intervals["health_report"]).seconds.do(
            lambda: asyncio.create_task(self.generate_health_report())
        )
        
        # Main monitoring loop
        while self.monitoring_active:
            try:
                schedule.run_pending()
                await asyncio.sleep(10)  # Check every 10 seconds
                
                # Cleanup old data
                if datetime.now().minute == 0:  # Once per hour
                    self.cleanup_history()
                    
            except KeyboardInterrupt:
                logger.info("Facade monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retry
    
    async def run_quick_facade_check(self):
        """Run quick facade check (API endpoints only)."""
        logger.debug("Running quick facade check...")
        api_result = await self.detect_api_facades()
        
        if api_result["facade_score"] > 0.5:
            await self.trigger_facade_alert({
                "timestamp": datetime.now().isoformat(),
                "overall_facade_score": api_result["facade_score"],
                "critical_issues": [{"component": "api", "facade_score": api_result["facade_score"]}],
                "scan_type": "quick_check"
            })
    
    async def generate_health_report(self):
        """Generate periodic health report."""
        logger.info("📊 Generating facade detection health report...")
        
        report = self.get_facade_status_report()
        
        # Write report to file
        report_file = Path("/opt/sutazaiapp/logs/facade_health_reports.json")
        report_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if report_file.exists():
                with open(report_file, 'r') as f:
                    reports = json.load(f)
            else:
                reports = []
            
            reports.append(report)
            
            # Keep only last 24 reports (24 hours)
            reports = reports[-24:]
            
            with open(report_file, 'w') as f:
                json.dump(reports, f, indent=2)
                
            logger.info(f"Health report generated: {report['system_status']}")
            
        except Exception as e:
            logger.error(f"Failed to write health report: {e}")


async def main():
    """Main entry point for facade detection monitoring."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Facade Detection Monitor")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--one-shot", action="store_true", help="Run single scan and exit")
    parser.add_argument("--report", action="store_true", help="Generate status report")
    
    args = parser.parse_args()
    
    config_file = args.config or "/opt/sutazaiapp/config/facade_monitor.json"
    
    async with FacadeDetectionMonitor(config_file) as monitor:
        if args.report:
            # Generate status report
            report = monitor.get_facade_status_report()
            print(json.dumps(report, indent=2))
            
        elif args.one_shot:
            # Run single comprehensive scan
            result = await monitor.run_comprehensive_facade_scan()
            print(json.dumps(result, indent=2))
            
            # Exit with error code if facades detected
            if result["facade_detected"]:
                exit(1)
        else:
            # Start continuous monitoring
            await monitor.start_monitoring()


if __name__ == "__main__":
    asyncio.run(main())