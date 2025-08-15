#!/usr/bin/env python3
"""
SutazAI Production Alerting Pipeline Test Suite
Tests all critical alerting functionality end-to-end
"""

import requests
import json
import time
import subprocess
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import sys
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/opt/sutazaiapp/logs/alerting_test.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class AlertingTestSuite:
    def __init__(self):
        self.prometheus_url = "http://localhost:10200"
        self.alertmanager_url = "http://localhost:10203"
        self.test_results = {
            "prometheus_connectivity": False,
            "alertmanager_connectivity": False,
            "alert_rules_loaded": False,
            "notification_channels": {},
            "test_alerts": {},
            "end_to_end_flow": False
        }
        
    def test_prometheus_connectivity(self) -> bool:
        """Test Prometheus API connectivity"""
        try:
            response = requests.get(f"{self.prometheus_url}/api/v1/status/config", timeout=10)
            if response.status_code == 200:
                logger.info("✅ Prometheus connectivity test passed")
                self.test_results["prometheus_connectivity"] = True
                return True
            else:
                logger.error(f"❌ Prometheus returned status {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"❌ Prometheus connectivity failed: {e}")
            return False
    
    def test_alertmanager_connectivity(self) -> bool:
        """Test AlertManager API connectivity"""
        try:
            response = requests.get(f"{self.alertmanager_url}/api/v2/status", timeout=10)
            if response.status_code == 200:
                logger.info("✅ AlertManager connectivity test passed")
                self.test_results["alertmanager_connectivity"] = True
                return True
            else:
                logger.error(f"❌ AlertManager returned status {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"❌ AlertManager connectivity failed: {e}")
            return False
    
    def test_alert_rules_loaded(self) -> bool:
        """Test that all alert rules are properly loaded"""
        try:
            response = requests.get(f"{self.prometheus_url}/api/v1/rules", timeout=10)
            if response.status_code != 200:
                logger.error(f"❌ Failed to fetch rules: {response.status_code}")
                return False
            
            rules_data = response.json()
            rule_groups = rules_data.get('data', {}).get('groups', [])
            
            expected_groups = [
                'sutazai_infrastructure_critical',
                'sutazai_ai_agents', 
                'sutazai_ollama_service',
                'sutazai_databases',
                'sutazai_service_mesh',
                'sutazai_backend_services',
                'sutazai_containers',
                'sutazai_security',
                'sutazai_business_metrics'
            ]
            
            loaded_groups = [group['name'] for group in rule_groups]
            missing_groups = set(expected_groups) - set(loaded_groups)
            
            if missing_groups:
                logger.error(f"❌ Missing alert rule groups: {missing_groups}")
                return False
            
            total_rules = sum(len(group.get('rules', [])) for group in rule_groups)
            logger.info(f"✅ Alert rules loaded: {len(loaded_groups)} groups, {total_rules} total rules")
            self.test_results["alert_rules_loaded"] = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to test alert rules: {e}")
            return False
    
    def test_notification_channels(self) -> Dict[str, bool]:
        """Test notification channel configurations"""
        channels = {
            "slack_default": False,
            "slack_critical": False,
            "email_critical": False,
            "pagerduty": False
        }
        
        try:
            # Test AlertManager configuration
            response = requests.get(f"{self.alertmanager_url}/api/v1/status", timeout=10)
            if response.status_code == 200:
                config_data = response.json()
                
                # Check if receivers are configured
                if 'config' in config_data.get('data', {}):
                    receivers = config_data['data']['config'].get('receivers', [])
                    receiver_names = [r['name'] for r in receivers]
                    
                    if 'default-notifications' in receiver_names:
                        channels["slack_default"] = True
                        logger.info("✅ Default Slack notifications configured")
                    
                    if 'critical-emergency' in receiver_names:
                        channels["slack_critical"] = True
                        channels["email_critical"] = True
                        channels["pagerduty"] = True
                        logger.info("✅ Critical emergency notifications configured")
                    
                    self.test_results["notification_channels"] = channels
                    return channels
            
        except Exception as e:
            logger.error(f"❌ Failed to test notification channels: {e}")
        
        return channels
    
    def create_test_alert(self, alert_name: str, labels: Dict[str, str], duration: int = 60) -> bool:
        """Create a test alert by temporarily modifying metrics"""
        try:
            # Create a test alert via AlertManager API
            alert_data = {
                "alerts": [
                    {
                        "labels": {
                            "alertname": alert_name,
                            "severity": "warning",
                            "component": "test",
                            "job": "alerting-test",
                            "instance": "test-instance",
                            **labels
                        },
                        "annotations": {
                            "summary": f"Test alert: {alert_name}",
                            "description": "This is a test alert generated by the alerting pipeline test suite",
                            "runbook_url": "https://wiki.sutazai.com/runbooks/test-alert"
                        },
                        "startsAt": datetime.utcnow().isoformat() + "Z",
                        "endsAt": (datetime.utcnow() + timedelta(seconds=duration)).isoformat() + "Z"
                    }
                ]
            }
            
            response = requests.post(
                f"{self.alertmanager_url}/api/v1/alerts",
                json=alert_data,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            if response.status_code in [200, 202]:
                logger.info(f"✅ Test alert '{alert_name}' created successfully")
                return True
            else:
                logger.error(f"❌ Failed to create test alert: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to create test alert '{alert_name}': {e}")
            return False
    
    def test_specific_alerts(self) -> Dict[str, bool]:
        """Test specific alert types"""
        test_alerts = {
            "service_down": False,
            "high_cpu": False,
            "agent_failure": False,
            "security_incident": False
        }
        
        # Test Service Down alert
        if self.create_test_alert("TestServiceDown", {"severity": "critical", "component": "infrastructure"}):
            test_alerts["service_down"] = True
        
        time.sleep(2)  # Brief pause between alerts
        
        # Test High CPU alert  
        if self.create_test_alert("TestHighCPU", {"severity": "warning", "component": "infrastructure"}):
            test_alerts["high_cpu"] = True
        
        time.sleep(2)
        
        # Test Agent Failure alert
        if self.create_test_alert("TestAgentFailure", {"severity": "critical", "component": "ai_agents"}):
            test_alerts["agent_failure"] = True
        
        time.sleep(2)
        
        # Test Security Incident alert
        if self.create_test_alert("TestSecurityIncident", {"severity": "critical", "component": "security"}):
            test_alerts["security_incident"] = True
        
        self.test_results["test_alerts"] = test_alerts
        return test_alerts
    
    def verify_alert_delivery(self) -> bool:
        """Verify alerts are being processed by AlertManager"""
        try:
            # Check active alerts
            response = requests.get(f"{self.alertmanager_url}/api/v1/alerts", timeout=10)
            if response.status_code == 200:
                alerts = response.json().get('data', [])
                test_alerts = [alert for alert in alerts if alert.get('labels', {}).get('job') == 'alerting-test']
                
                if test_alerts:
                    logger.info(f"✅ Found {len(test_alerts)} test alerts in AlertManager")
                    return True
                else:
                    logger.warning("⚠️ No test alerts found in AlertManager")
                    return False
            else:
                logger.error(f"❌ Failed to fetch alerts: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to verify alert delivery: {e}")
            return False
    
    def cleanup_test_alerts(self) -> bool:
        """Clean up test alerts"""
        try:
            # Silence all test alerts
            silence_data = {
                "matchers": [
                    {"name": "job", "value": "alerting-test", "isRegex": False}
                ],
                "startsAt": datetime.utcnow().isoformat() + "Z",
                "endsAt": (datetime.utcnow() + timedelta(hours=1)).isoformat() + "Z",
                "comment": "Cleaning up alerting pipeline test alerts",
                "createdBy": "alerting-test-suite"
            }
            
            response = requests.post(
                f"{self.alertmanager_url}/api/v1/silences",
                json=silence_data,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            if response.status_code in [200, 201, 202]:
                logger.info("✅ Test alerts cleaned up successfully")
                return True
            else:
                logger.error(f"❌ Failed to cleanup test alerts: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to cleanup test alerts: {e}")
            return False
    
    def test_service_discovery(self) -> Dict[str, bool]:
        """Test service discovery and target health"""
        services = {
            "prometheus": False,
            "alertmanager": False,
            "node-exporter": False,
            "cadvisor": False
        }
        
        try:
            response = requests.get(f"{self.prometheus_url}/api/v1/targets", timeout=10)
            if response.status_code == 200:
                targets = response.json().get('data', {}).get('activeTargets', [])
                
                for target in targets:
                    job = target.get('labels', {}).get('job', '')
                    health = target.get('health', '')
                    
                    if job in services:
                        services[job] = (health == 'up')
                        
                logger.info(f"✅ Service discovery test completed")
                return services
            else:
                logger.error(f"❌ Failed to fetch targets: {response.status_code}")
                
        except Exception as e:
            logger.error(f"❌ Service discovery test failed: {e}")
            
        return services
    
    def run_full_test_suite(self) -> Dict:
        """Run the complete alerting test suite"""
        logger.info("🚀 Starting SutazAI Alerting Pipeline Test Suite")
        logger.info("=" * 60)
        
        # Test basic connectivity
        logger.info("Phase 1: Testing Basic Connectivity")
        self.test_prometheus_connectivity()
        self.test_alertmanager_connectivity()
        
        # Test configuration
        logger.info("\nPhase 2: Testing Configuration")
        self.test_alert_rules_loaded()
        self.test_notification_channels()
        
        # Test service discovery
        logger.info("\nPhase 3: Testing Service Discovery")
        service_health = self.test_service_discovery()
        
        # Test alert creation and delivery
        logger.info("\nPhase 4: Testing Alert Creation and Delivery")
        self.test_specific_alerts()
        
        # Allow time for alert processing
        logger.info("⏳ Waiting 30 seconds for alert processing...")
        time.sleep(30)
        
        # Verify alert delivery
        logger.info("\nPhase 5: Verifying Alert Delivery")
        delivery_success = self.verify_alert_delivery()
        self.test_results["end_to_end_flow"] = delivery_success
        
        # Cleanup
        logger.info("\nPhase 6: Cleanup")
        self.cleanup_test_alerts()
        
        # Generate report
        self.generate_test_report(service_health)
        
        return self.test_results
    
    def generate_test_report(self, service_health: Dict[str, bool]):
        """Generate comprehensive test report"""
        logger.info("\n" + "=" * 60)
        logger.info("📊 SUTAZAI ALERTING PIPELINE TEST REPORT")
        logger.info("=" * 60)
        
        # Overall status
        critical_tests = [
            self.test_results["prometheus_connectivity"],
            self.test_results["alertmanager_connectivity"], 
            self.test_results["alert_rules_loaded"]
        ]
        
        overall_health = all(critical_tests)
        status_emoji = "✅" if overall_health else "❌"
        
        logger.info(f"{status_emoji} OVERALL STATUS: {'HEALTHY' if overall_health else 'ISSUES DETECTED'}")
        logger.info("")
        
        # Detailed results
        logger.info("📋 DETAILED TEST RESULTS:")
        logger.info(f"  • Prometheus Connectivity: {'✅' if self.test_results['prometheus_connectivity'] else '❌'}")
        logger.info(f"  • AlertManager Connectivity: {'✅' if self.test_results['alertmanager_connectivity'] else '❌'}")
        logger.info(f"  • Alert Rules Loaded: {'✅' if self.test_results['alert_rules_loaded'] else '❌'}")
        
        # Notification channels
        logger.info("\n📢 NOTIFICATION CHANNELS:")
        for channel, status in self.test_results["notification_channels"].items():
            logger.info(f"  • {channel}: {'✅' if status else '❌'}")
        
        # Service health
        logger.info("\n🔍 SERVICE DISCOVERY:")
        for service, healthy in service_health.items():
            logger.info(f"  • {service}: {'✅ UP' if healthy else '❌ DOWN'}")
        
        # Test alerts
        logger.info("\n🧪 TEST ALERTS:")
        for alert_type, success in self.test_results["test_alerts"].items():
            logger.info(f"  • {alert_type}: {'✅' if success else '❌'}")
        
        # End-to-end flow
        logger.info(f"\n🔄 END-TO-END FLOW: {'✅ SUCCESS' if self.test_results['end_to_end_flow'] else '❌ FAILED'}")
        
        # Recommendations
        logger.info("\n💡 RECOMMENDATIONS:")
        if not overall_health:
            logger.info("  • Fix critical connectivity issues before proceeding")
            logger.info("  • Verify AlertManager configuration")
            logger.info("  • Check Prometheus rule loading")
        else:
            logger.info("  • Alerting pipeline is operational")
            logger.info("  • Consider testing notification delivery manually")
            logger.info("  • Set up regular health checks")
        
        logger.info("\n" + "=" * 60)
        logger.info(f"🕐 Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 60)

def main():
    """Main test execution"""
    if len(sys.argv) > 1 and sys.argv[1] == '--quick':
        print("Running quick connectivity test...")
        suite = AlertingTestSuite()
        suite.test_prometheus_connectivity()
        suite.test_alertmanager_connectivity()
    else:
        print("Running full alerting pipeline test suite...")
        suite = AlertingTestSuite()
        results = suite.run_full_test_suite()
        
        # Exit with appropriate code
        critical_passed = all([
            results["prometheus_connectivity"],
            results["alertmanager_connectivity"],
            results["alert_rules_loaded"]
        ])
        
        sys.exit(0 if critical_passed else 1)

if __name__ == "__main__":
    main()