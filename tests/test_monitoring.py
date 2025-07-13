"""Tests for the monitoring system."""
import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch
from scripts.monitoring import SystemMonitor

@pytest.fixture
def mock_config():
    return {
        'log_dir': '/tmp/sutazai/logs',
        'alert_cooldown': 3600,
        'thresholds': {
            'cpu': 90,
            'memory': 90,
            'disk': 90
        },
        'email': {
            'smtp_server': 'smtp.test.com',
            'smtp_port': 587,
            'username': 'test@test.com',
            'password': 'test_password',
            'from_addr': 'test@test.com',
            'to_addr': 'admin@test.com'
        }
    }

@pytest.fixture
def monitor(tmp_path, mock_config):
    # Create necessary directories
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Update config with test paths
    config = mock_config.copy()
    config['log_dir'] = str(log_dir)

    config_path = tmp_path / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f)

    monitor = SystemMonitor(str(config_path))
    monitor.alert_history_file = str(log_dir / "alert_history.json")
    monitor._init_alert_history()
    return monitor

def test_check_resources(monitor):
    with patch("psutil.cpu_percent") as mock_cpu, \
         patch("psutil.virtual_memory") as mock_memory, \
         patch("psutil.disk_usage") as mock_disk:

        # Mock system metrics
        mock_cpu.return_value = 95.0
        mock_memory.return_value = Mock(percent=95.0)
        mock_disk.return_value = Mock(percent=95.0)

        alerts = monitor.check_resources()
        assert len(alerts) == 3
        assert any("CPU" in alert for alert in alerts)
        assert any("memory" in alert for alert in alerts)
        assert any("disk" in alert for alert in alerts)

def test_check_services(monitor):
    with patch("os.system") as mock_system:
        # Mock service checks
        mock_system.return_value = 1  # Service not running

        alerts = monitor.check_services()
        assert len(alerts) == 2
        assert all("not running" in alert for alert in alerts)

        # Mock services running
        mock_system.return_value = 0
        alerts = monitor.check_services()
        assert len(alerts) == 0

def test_send_alert(monitor):
    with patch("smtplib.SMTP") as mock_smtp:
        mock_server = Mock()
        mock_smtp.return_value.__enter__.return_value = mock_server

        success = monitor.send_alert("Test alert message")
        assert success is True
        mock_server.send_message.assert_called_once()

        # Test failed alert
        mock_server.send_message.side_effect = Exception("SMTP error")
        success = monitor.send_alert("Test alert message")
        assert success is False

def test_should_alert(monitor, tmp_path):
    alert_type = "test_alert"

    # First alert should always be sent
    assert monitor.should_alert(alert_type) is True

    # Second alert within cooldown should not be sent
    assert monitor.should_alert(alert_type) is False

    # Modify alert history to simulate cooldown passed
    with open(monitor.alert_history_file, "r") as f:
        history = json.load(f)
    history[alert_type] = "2000-01-01T00:00:00"
    with open(monitor.alert_history_file, "w") as f:
        json.dump(history, f)

    # Alert should be sent after cooldown
    assert monitor.should_alert(alert_type) is True
