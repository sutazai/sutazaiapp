"""Tests for the system maintenance module."""
import os
import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch
from scripts.system_maintenance import SystemMaintainer
from datetime import datetime

@pytest.fixture
def maintainer(tmp_path):
    # Create necessary directories
    log_dir = tmp_path / "logs"
    backup_dir = tmp_path / "backups"
    log_dir.mkdir(parents=True, exist_ok=True)
    backup_dir.mkdir(parents=True, exist_ok=True)

    config = {
        'log_dir': str(log_dir),
        'backup_dir': str(backup_dir),
        'max_log_age_days': 7,
        'max_backup_age_days': 30,
        'backup_retention': 5
    }
    config_path = tmp_path / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f)
    return SystemMaintainer(str(config_path))

def test_check_system_health(maintainer):
    with patch("psutil.cpu_percent") as mock_cpu, \
         patch("psutil.virtual_memory") as mock_memory, \
         patch("psutil.disk_usage") as mock_disk:

        # Mock system metrics
        mock_cpu.return_value = 50.0
        mock_memory.return_value = Mock(percent=60.0)
        mock_disk.return_value = Mock(percent=70.0)

        success = maintainer.check_system_health()
        assert success is True

        # Verify metrics were saved
        metrics_file = os.path.join(maintainer.config['log_dir'], 'health_status.json')
        assert os.path.exists(metrics_file)
        with open(metrics_file) as f:
            metrics = json.load(f)
            assert metrics['cpu_usage'] == 50.0
            assert metrics['memory_usage'] == 60.0
            assert metrics['disk_usage'] == 70.0

def test_optimize_performance(maintainer):
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = Mock(returncode=0)
        success = maintainer.optimize_performance()
        assert success is True

def test_validate_security(maintainer):
    with patch("os.path.exists") as mock_exists, \
         patch("os.stat") as mock_stat, \
         patch("os.chmod") as mock_chmod, \
         patch("psutil.process_iter") as mock_process_iter:

        mock_exists.return_value = True
        mock_stat.return_value = Mock(st_mode=0o777)
        mock_process_iter.return_value = []

        success = maintainer.validate_security()
        assert success is True
        assert mock_chmod.called

def test_rotate_logs(maintainer, tmp_path):
    # Create test log files
    log_dir = tmp_path / "logs"
    log_dir.mkdir(exist_ok=True)

    # Create current log
    current_log = log_dir / "app.log"
    current_log.write_text("current log")

    # Create old logs with a very old modification time
    old_log = log_dir / "app.log.old"
    old_log.write_text("old log")

    # Set a very old modification time
    old_time = datetime(2000, 1, 1).timestamp()
    os.utime(old_log, (old_time, old_time))

    success = maintainer.rotate_logs()
    assert success is True

    # Check if old logs were removed
    assert not old_log.exists()
    assert current_log.exists()

def test_manage_backups(maintainer, tmp_path):
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = Mock(returncode=0)

        success = maintainer.manage_backups()
        assert success is True

        # Check if backup commands were called
        rsync_call = False
        pg_dump_call = False
        for call in mock_run.call_args_list:
            args = call[0][0]
            if args[0] == "rsync":
                rsync_call = True
            elif args[0] == "pg_dump":
                pg_dump_call = True

        assert rsync_call and pg_dump_call
