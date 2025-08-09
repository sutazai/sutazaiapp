#!/usr/bin/env python3
"""
Comprehensive AI Agent Monitoring Validation
============================================

Validates the enhanced static monitor implementation for:
1. Proper integration with all 105+ agents in the system
2. Accurate health status detection for each agent type
3. Performance metrics correctness
4. Agent registry integration
5. Production-ready error handling
6. No hardcoded values or fantasy elements

This validation ensures flawless AI agent monitoring functionality.
"""

import sys
import json
import time
import tempfile
import socket
import threading
import http.server
import socketserver
from pathlib import Path
from unittest.mock import patch, MagicMock
from typing import Dict, List, Any, Tuple

# Add the monitoring directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from static_monitor import EnhancedMonitor
except ImportError as e:
    print(f"❌ Failed to import EnhancedMonitor: {e}")
    sys.exit(1)


class MockHealthServer:
    """Mock health server for testing agent health checks"""
    
    def __init__(self, port: int, response_code: int = 200, delay: float = 0.0):
        self.port = port
        self.response_code = response_code
        self.delay = delay
        self.server = None
        self.thread = None
    
    def start(self):
        """Start mock server"""
        handler = self._create_handler()
        self.server = socketserver.TCPServer(("", self.port), handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        time.sleep(0.1)  # Give server time to start
    
    def stop(self):
        """Stop mock server"""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
    
    def _create_handler(self):
        """Create request handler with configured response"""
        delay = self.delay
        response_code = self.response_code
        
        class MockHandler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                if delay > 0:
                    time.sleep(delay)
                self.send_response(response_code)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(b'{"status": "ok"}')
            
            def log_message(self, format, *args):
                pass  # Suppress logging
        
        return MockHandler


class ComprehensiveMonitorValidator:
    """Comprehensive validation of the enhanced monitor"""
    
    def __init__(self):
        self.test_results = []
        self.validation_errors = []
        self.monitor = None
    
    def log_result(self, test_name: str, passed: bool, message: str = ""):
        """Log test result"""
        status = "✅" if passed else "❌"
        self.test_results.append({
            'test': test_name,
            'passed': passed,
            'message': message
        })
        print(f"{status} {test_name}: {message}")
        
        if not passed:
            self.validation_errors.append(f"{test_name}: {message}")
    
    def validate_agent_registry_integration(self) -> bool:
        """Validate proper agent registry integration"""
        print("\n🔍 Validating Agent Registry Integration...")
        
        self.monitor = EnhancedMonitor()
        registry = self.monitor.agent_registry
        
        # Check registry structure
        if not isinstance(registry, dict):
            self.log_result("Registry Structure", False, "Registry is not a dictionary")
            return False
        
        if 'agents' not in registry:
            self.log_result("Registry Format", False, "Missing 'agents' key in registry")
            return False
        
        agents = registry.get('agents', {})
        agent_count = len(agents)
        
        # Validate agent count (should be 105+)
        if agent_count < 105:
            self.log_result("Agent Count", False, f"Only {agent_count} agents found, expected 105+")
            return False
        
        self.log_result("Agent Count", True, f"Found {agent_count} agents")
        
        # Validate agent structure
        valid_agents = 0
        for agent_id, agent_info in agents.items():
            if self._validate_agent_structure(agent_id, agent_info):
                valid_agents += 1
        
        if valid_agents < agent_count * 0.95:  # Allow 5% tolerance
            self.log_result("Agent Structure", False, f"Only {valid_agents}/{agent_count} agents have valid structure")
            return False
        
        self.log_result("Agent Structure", True, f"{valid_agents}/{agent_count} agents have valid structure")
        return True
    
    def _validate_agent_structure(self, agent_id: str, agent_info: Dict) -> bool:
        """Validate individual agent structure"""
        required_fields = ['name', 'description', 'capabilities', 'config_path']
        
        for field in required_fields:
            if field not in agent_info:
                return False
        
        # Validate capabilities is a list
        if not isinstance(agent_info.get('capabilities'), list):
            return False
        
        # Validate no fantasy elements in description
        description = agent_info.get('description', '').lower()
        fantasy_terms = ['process', 'configurator', 'transfer', 'processing-unit', 'fantasy']
        if any(term in description for term in fantasy_terms):
            return False
        
        return True
    
    def validate_health_status_detection(self) -> bool:
        """Validate health status detection accuracy"""
        print("\n🔧 Validating Health Status Detection...")
        
        # Test different health scenarios
        scenarios = [
            {"port": 18001, "response_code": 200, "delay": 0.1, "expected": "healthy"},
            {"port": 18002, "response_code": 200, "delay": 1.5, "expected": "warning"},
            {"port": 18003, "response_code": 200, "delay": 6.0, "expected": "critical"},
            {"port": 18004, "response_code": 500, "delay": 0.1, "expected": "critical"},
        ]
        
        servers = []
        try:
            # Start mock servers
            for scenario in scenarios:
                server = MockHealthServer(
                    scenario["port"], 
                    scenario["response_code"], 
                    scenario["delay"]
                )
                server.start()
                servers.append(server)
            
            # Test health checks
            correct_detections = 0
            for scenario in scenarios:
                mock_agent_info = {"name": f"test-agent-{scenario['port']}"}
                
                # Mock the endpoint detection to return our test port
                with patch.object(self.monitor, '_get_agent_endpoint') as mock_endpoint:
                    mock_endpoint.return_value = f"http://localhost:{scenario['port']}"
                    
                    status, response_time = self.monitor._check_agent_health(
                        f"test-agent-{scenario['port']}", 
                        mock_agent_info, 
                        2  # timeout
                    )
                    
                    if status == scenario["expected"]:
                        correct_detections += 1
                    else:
                        print(f"  ⚠️  Expected {scenario['expected']}, got {status} for port {scenario['port']}")
            
            success_rate = correct_detections / len(scenarios)
            if success_rate >= 0.75:  # 75% success rate threshold
                self.log_result("Health Detection", True, f"{correct_detections}/{len(scenarios)} scenarios detected correctly")
                return True
            else:
                self.log_result("Health Detection", False, f"Only {correct_detections}/{len(scenarios)} scenarios detected correctly")
                return False
        
        finally:
            # Clean up servers
            for server in servers:
                server.stop()
    
    def validate_performance_metrics(self) -> bool:
        """Validate performance metrics calculation"""
        print("\n📊 Validating Performance Metrics...")
        
        # Test system statistics
        stats = self.monitor.get_system_stats()
        
        required_metrics = [
            'cpu_percent', 'cpu_cores', 'cpu_trend',
            'mem_percent', 'mem_used', 'mem_total', 'mem_trend',
            'disk_percent', 'disk_free',
            'network', 'connections', 'load_avg'
        ]
        
        missing_metrics = []
        invalid_metrics = []
        
        for metric in required_metrics:
            if metric not in stats:
                missing_metrics.append(metric)
            else:
                if not self._validate_metric_value(metric, stats[metric]):
                    invalid_metrics.append(metric)
        
        if missing_metrics:
            self.log_result("Required Metrics", False, f"Missing metrics: {missing_metrics}")
            return False
        
        if invalid_metrics:
            self.log_result("Metric Values", False, f"Invalid metric values: {invalid_metrics}")
            return False
        
        # Test network calculations
        if not self._validate_network_metrics(stats['network']):
            self.log_result("Network Metrics", False, "Network metrics validation failed")
            return False
        
        self.log_result("Performance Metrics", True, "All metrics calculated correctly")
        return True
    
    def _validate_metric_value(self, metric: str, value: Any) -> bool:
        """Validate individual metric value"""
        if metric in ['cpu_percent', 'mem_percent', 'disk_percent']:
            return isinstance(value, (int, float)) and 0 <= value <= 100
        
        if metric in ['cpu_cores', 'connections']:
            return isinstance(value, int) and value >= 0
        
        if metric in ['mem_used', 'mem_total', 'disk_free']:
            return isinstance(value, (int, float)) and value >= 0
        
        if metric in ['cpu_trend', 'mem_trend']:
            return value in ['↑', '↓', '→']
        
        if metric == 'network':
            return isinstance(value, dict)
        
        if metric == 'load_avg':
            return isinstance(value, (list, tuple)) and len(value) == 3
        
        return True
    
    def _validate_network_metrics(self, network_stats: Dict) -> bool:
        """Validate network metrics structure and values"""
        required_keys = ['bytes_sent', 'bytes_recv', 'bandwidth_mbps', 'upload_mbps', 'download_mbps']
        
        for key in required_keys:
            if key not in network_stats:
                return False
            
            value = network_stats[key]
            if not isinstance(value, (int, float)) or value < 0:
                return False
        
        return True
    
    def validate_error_handling(self) -> bool:
        """Validate production-ready error handling"""
        print("\n🛡️ Validating Error Handling...")
        
        error_scenarios = [
            "invalid_registry_path",
            "network_timeout",
            "invalid_config",
            "missing_dependencies"
        ]
        
        passed_scenarios = 0
        
        # Test invalid agent registry
        try:
            with patch('builtins.open', side_effect=FileNotFoundError):
                monitor = EnhancedMonitor()
                registry = monitor._load_agent_registry()
                if registry == {'agents': {}}:  # Should return empty registry
                    passed_scenarios += 1
        except Exception:
            pass  # Error handling failed
        
        # Test invalid configuration
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                f.write("invalid json content")
                f.flush()
                
                monitor = EnhancedMonitor(f.name)
                if monitor.config is not None:  # Should fallback to defaults
                    passed_scenarios += 1
        except Exception:
            pass
        
        # Test network timeout handling
        try:
            mock_agent_info = {"name": "timeout-test"}
            status, response_time = self.monitor._check_agent_health(
                "timeout-test", 
                mock_agent_info, 
                0.1  # Very short timeout
            )
            if status in ['critical', 'unknown']:  # Should handle timeout gracefully
                passed_scenarios += 1
        except Exception:
            pass
        
        # Test session cleanup
        try:
            self.monitor.cleanup()
            passed_scenarios += 1  # Should not raise exceptions
        except Exception:
            pass
        
        success_rate = passed_scenarios / len(error_scenarios)
        if success_rate >= 1.0:
            self.log_result("Error Handling", True, f"All {len(error_scenarios)} error scenarios handled correctly")
            return True
        else:
            self.log_result("Error Handling", False, f"Only {passed_scenarios}/{len(error_scenarios)} error scenarios handled correctly")
            return False
    
    def validate_configuration_flexibility(self) -> bool:
        """Validate configuration system flexibility"""
        print("\n⚙️ Validating Configuration System...")
        
        # Test custom configuration
        custom_config = {
            "refresh_rate": 3.0,
            "adaptive_refresh": False,
            "thresholds": {
                "cpu_warning": 65,
                "cpu_critical": 85,
                "memory_warning": 80,
                "memory_critical": 95,
                "response_time_warning": 1500,
                "response_time_critical": 3000
            },
            "agent_monitoring": {
                "enabled": True,
                "timeout": 3,
                "max_agents_display": 8
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(custom_config, f)
            f.flush()
            
            try:
                monitor = EnhancedMonitor(f.name)
                
                # Validate all custom settings were applied
                config_valid = (
                    monitor.config['refresh_rate'] == 3.0 and
                    monitor.config['adaptive_refresh'] == False and
                    monitor.config['thresholds']['cpu_warning'] == 65 and
                    monitor.config['agent_monitoring']['timeout'] == 3
                )
                
                if config_valid:
                    self.log_result("Configuration Flexibility", True, "Custom configuration applied correctly")
                    return True
                else:
                    self.log_result("Configuration Flexibility", False, "Custom configuration not applied correctly")
                    return False
            
            finally:
                Path(f.name).unlink()
    
    def validate_no_hardcoded_values(self) -> bool:
        """Check for hardcoded values or fantasy elements"""
        print("\n🔍 Validating No Hardcoded Values...")
        
        # Read the monitor source code
        monitor_path = Path(__file__).parent / "static_monitor.py"
        
        try:
            with open(monitor_path, 'r') as f:
                source_code = f.read()
            
            # Check for hardcoded ports (should use configuration)
            hardcoded_issues = []
            
            # Check for fantasy terms
            fantasy_terms = ['process', 'configurator', 'transfer', 'processing-unit', 'fantasy']
            for term in fantasy_terms:
                if term in source_code.lower():
                    hardcoded_issues.append(f"Fantasy term found: {term}")
            
            # Check for excessive hardcoded network ports
            import re
            port_matches = re.findall(r':\s*(\d{4,5})', source_code)
            unique_ports = set(port_matches)
            if len(unique_ports) > 10:  # Reasonable threshold
                hardcoded_issues.append(f"Too many hardcoded ports: {len(unique_ports)}")
            
            # Check configuration usage
            if 'self.config' not in source_code:
                hardcoded_issues.append("Configuration system not used")
            
            if hardcoded_issues:
                self.log_result("Hardcoded Values", False, f"Issues found: {hardcoded_issues}")
                return False
            else:
                self.log_result("Hardcoded Values", True, "No hardcoded values or fantasy elements found")
                return True
        
        except Exception as e:
            self.log_result("Hardcoded Values", False, f"Could not analyze source code: {e}")
            return False
    
    def validate_agent_type_detection(self) -> bool:
        """Validate agent type detection logic"""
        print("\n🏷️ Validating Agent Type Detection...")
        
        test_agents = [
            {"name": "backend-api", "description": "Backend API service", "expected": "BACK"},
            {"name": "frontend-ui", "description": "Frontend UI component", "expected": "FRON"},
            {"name": "ai-model", "description": "AI machine learning model", "expected": "AI"},
            {"name": "security-scanner", "description": "Security vulnerability scanner", "expected": "SECU"},
            {"name": "data-processor", "description": "Data processing pipeline", "expected": "DATA"},
            {"name": "infra-deploy", "description": "Infrastructure deployment tool", "expected": "INFR"},
        ]
        
        correct_detections = 0
        for agent in test_agents:
            detected_type = self.monitor._get_agent_type(agent)
            if detected_type == agent["expected"]:
                correct_detections += 1
            else:
                print(f"  ⚠️  Agent {agent['name']}: expected {agent['expected']}, got {detected_type}")
        
        success_rate = correct_detections / len(test_agents)
        if success_rate >= 0.8:  # 80% success rate
            self.log_result("Agent Type Detection", True, f"{correct_detections}/{len(test_agents)} types detected correctly")
            return True
        else:
            self.log_result("Agent Type Detection", False, f"Only {correct_detections}/{len(test_agents)} types detected correctly")
            return False
    
    def run_comprehensive_validation(self) -> bool:
        """Run all validation tests"""
        print("🔍 Comprehensive AI Agent Monitoring Validation")
        print("=" * 60)
        
        validation_tests = [
            self.validate_agent_registry_integration,
            self.validate_health_status_detection,
            self.validate_performance_metrics,
            self.validate_error_handling,
            self.validate_configuration_flexibility,
            self.validate_no_hardcoded_values,
            self.validate_agent_type_detection,
        ]
        
        passed_tests = 0
        total_tests = len(validation_tests)
        
        for test in validation_tests:
            try:
                if test():
                    passed_tests += 1
            except Exception as e:
                self.log_result(test.__name__, False, f"Test failed with exception: {e}")
        
        print("\n" + "=" * 60)
        print(f"📊 Validation Results: {passed_tests}/{total_tests} tests passed")
        
        if self.validation_errors:
            print("\n❌ Validation Errors:")
            for error in self.validation_errors:
                print(f"  • {error}")
        
        success_rate = passed_tests / total_tests
        if success_rate >= 0.9:  # 90% success rate required
            print(f"\n✅ VALIDATION PASSED: AI Agent Monitoring is production-ready ({success_rate:.1%} success rate)")
            return True
        else:
            print(f"\n❌ VALIDATION FAILED: AI Agent Monitoring needs improvements ({success_rate:.1%} success rate)")
            return False


def main():
    """Run comprehensive validation"""
    validator = ComprehensiveMonitorValidator()
    success = validator.run_comprehensive_validation()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())