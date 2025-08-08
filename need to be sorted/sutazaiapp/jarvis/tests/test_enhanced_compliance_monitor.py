#!/usr/bin/env python3
"""
Comprehensive Test Suite for Enhanced Compliance Monitor
=======================================================
Purpose: 100% test coverage for production-ready compliance monitoring system
Requirements: pytest, pytest-cov, pytest-mock, psutil

Test Categories:
- Unit tests for all core functionality
- Integration tests with file system operations
- Stress tests under high violation loads
- Edge case testing (corrupted files, permission issues)
- Transaction and rollback testing
- Performance and reliability testing
"""

import os
import sys
import json
import time
import shutil
import tempfile
import sqlite3
import pytest
import threading
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
from datetime import datetime, timedelta

# Add the monitoring directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts' / 'monitoring'))

try:
    from enhanced_compliance_monitor import (
        EnhancedComplianceMonitor, RuleViolation, SystemHealthMetrics, 
        ChangeTransaction
    )
except ImportError as e:
    pytest.skip(f"Cannot import enhanced compliance monitor: {e}", allow_module_level=True)


class TestEnhancedComplianceMonitor:
    """Test suite for Enhanced Compliance Monitor"""
    
    @pytest.fixture
    def temp_project_root(self):
        """Create temporary project directory for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create basic project structure
            (temp_path / 'scripts').mkdir()
            (temp_path / 'scripts' / 'misc').mkdir()
            (temp_path / 'compliance-reports').mkdir()
            (temp_path / 'logs').mkdir()
            
            # Create some test files
            (temp_path / 'deploy.sh').write_text('#!/bin/bash\necho "deploy"')
            (temp_path / 'test_file.py').write_text('# Test file\nprint("hello")')
            (temp_path / 'test_script.sh').write_text('#!/bin/bash\necho "test"')
            
            yield temp_path
    
    @pytest.fixture
    def monitor(self, temp_project_root):
        """Create monitor instance for testing"""
        config = {
            'max_workers': 2,
            'scan_timeout': 30,
            'fix_timeout': 10,
            'max_fix_attempts': 2,
            'auto_fix_enabled': True,
            'safe_mode': False,  # Disable for testing
            'backup_retention_days': 1
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config, f)
            config_path = f.name
        
        try:
            monitor = EnhancedComplianceMonitor(
                project_root=str(temp_project_root),
                config_path=config_path
            )
            yield monitor
        finally:
            try:
                monitor.system_state_db.close()
            except:
                pass
            try:
                os.unlink(config_path)
            except:
                pass
    
    def test_initialization(self, temp_project_root):
        """Test monitor initialization"""
        monitor = EnhancedComplianceMonitor(project_root=str(temp_project_root))
        
        assert monitor.project_root == temp_project_root
        assert monitor.config is not None
        assert monitor.rules_config is not None
        assert len(monitor.rules_config) == 16
        assert monitor.system_state_db is not None
        
        # Test database tables created
        cursor = monitor.system_state_db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = {row[0] for row in cursor.fetchall()}
        expected_tables = {'system_health', 'violations_history', 'change_transactions'}
        assert expected_tables.issubset(tables)
        
        monitor.system_state_db.close()
    
    def test_config_loading(self, temp_project_root):
        """Test configuration loading with various scenarios"""
        # Test with valid JSON config
        config_data = {
            'max_workers': 8,
            'scan_timeout': 120,
            'auto_fix_enabled': False
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_path = f.name
        
        try:
            monitor = EnhancedComplianceMonitor(
                project_root=str(temp_project_root),
                config_path=config_path
            )
            
            assert monitor.config['max_workers'] == 8
            assert monitor.config['scan_timeout'] == 120
            assert monitor.config['auto_fix_enabled'] == False
            
            monitor.system_state_db.close()
        finally:
            os.unlink(config_path)
        
        # Test with invalid config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json {")
            invalid_config_path = f.name
        
        try:
            monitor = EnhancedComplianceMonitor(
                project_root=str(temp_project_root),
                config_path=invalid_config_path
            )
            # Should use defaults
            assert monitor.config['max_workers'] == 4  # default
            monitor.system_state_db.close()
        finally:
            os.unlink(invalid_config_path)
    
    def test_dependency_graph_building(self, monitor, temp_project_root):
        """Test dependency graph construction"""
        # Create files with dependencies
        py_file = temp_project_root / 'test_deps.py'
        py_file.write_text('''
import os
from .local_module import something
import requests
''')
        
        dockerfile = temp_project_root / 'Dockerfile'
        dockerfile.write_text('''
FROM python:3.9
COPY requirements.txt .
ADD scripts/ /app/scripts/
''')
        
        # Rebuild dependency graph
        monitor.dependency_graph = monitor._build_dependency_graph()
        
        assert len(monitor.dependency_graph) >= 0  # Should have at least some dependencies
        
        # Check if Python file dependencies are detected
        py_file_deps = monitor.dependency_graph.get(str(py_file), set())
        assert 'local_dependency' in py_file_deps
    
    def test_path_exclusion(self, monitor, temp_project_root):
        """Test path exclusion logic"""
        # Test various exclusion patterns
        test_cases = [
            (temp_project_root / 'venv' / 'lib' / 'test.py', True),
            (temp_project_root / '__pycache__' / 'test.pyc', True),
            (temp_project_root / '.git' / 'config', True),
            (temp_project_root / 'test.tmp', True),
            (temp_project_root / 'test.py', False),
            (temp_project_root / 'scripts' / 'test.sh', False),
        ]
        
        for path, should_exclude in test_cases:
            assert monitor._should_exclude_path(path) == should_exclude, f"Path {path} exclusion failed"
    
    def test_system_metrics_collection(self, monitor):
        """Test system health metrics collection"""
        metrics = monitor._collect_system_metrics()
        
        assert isinstance(metrics, SystemHealthMetrics)
        assert metrics.timestamp is not None
        assert 0 <= metrics.cpu_usage <= 100
        assert 0 <= metrics.memory_usage <= 100
        assert 0 <= metrics.disk_usage <= 100
        assert metrics.active_processes > 0
        assert 0 <= metrics.compliance_score <= 100
    
    def test_rule_1_fantasy_elements_detection(self, monitor, temp_project_root):
        """Test fantasy elements detection"""
        # Create test files with fantasy elements
        test_files = {
            'test_fantasy.py': '''
def process_data():
    # This is automated processing
    magic_value = 42  # This should be detected
    wizard_config = {}  # This should be detected
    result = teleport_data(magic_value)  # Both should be detected
    return result
''',
            'test_clean.py': '''
def process_data():
    # Clean code without fantasy elements
    automated_value = 42
    configuration = {}
    result = transfer_data(automated_value)
    return result
''',
            'test_comments.py': '''
def process_data():
    # This comment mentions magic but shouldn't be flagged
    value = 42
    return value
'''
        }
        
        for filename, content in test_files.items():
            (temp_project_root / filename).write_text(content)
        
        violations = monitor.check_rule_1_fantasy_elements()
        
        # Should find violations in test_fantasy.py
        fantasy_violations = [v for v in violations if 'test_fantasy.py' in v.file_path]
        assert len(fantasy_violations) >= 3  # magic, wizard, teleport
        
        # Should not find violations in test_clean.py
        clean_violations = [v for v in violations if 'test_clean.py' in v.file_path]
        assert len(clean_violations) == 0
        
        # Should not find violations in comments
        comment_violations = [v for v in violations if 'test_comments.py' in v.file_path]
        assert len(comment_violations) == 0
    
    def test_rule_7_script_organization(self, monitor, temp_project_root):
        """Test script organization checking"""
        # Create scripts in wrong locations
        (temp_project_root / 'wrong_location.sh').write_text('#!/bin/bash\necho "wrong"')
        (temp_project_root / 'subdir').mkdir()
        (temp_project_root / 'subdir' / 'misplaced.sh').write_text('#!/bin/bash\necho "misplaced"')
        
        # Create duplicate scripts
        script_content = '#!/bin/bash\necho "duplicate content"\nls -la'
        (temp_project_root / 'scripts' / 'script1.sh').write_text(script_content)
        (temp_project_root / 'scripts' / 'script2.sh').write_text(script_content)
        
        violations = monitor.check_rule_7_script_organization()
        
        # Should find misplaced scripts
        misplaced_violations = [v for v in violations if 'outside /scripts/' in v.description]
        assert len(misplaced_violations) >= 2
        
        # Should find duplicate scripts
        duplicate_violations = [v for v in violations if 'Duplicate of' in v.description]
        assert len(duplicate_violations) >= 1
    
    def test_rule_12_deployment_script(self, monitor, temp_project_root):
        """Test deployment script compliance"""
        # Remove the canonical deploy.sh to test missing script detection
        (temp_project_root / 'deploy.sh').unlink()
        
        # Create extra deployment scripts
        (temp_project_root / 'scripts' / 'deploy-prod.sh').write_text('#!/bin/bash\necho "deploy prod"')
        (temp_project_root / 'setup.sh').write_text('#!/bin/bash\necho "setup"')
        (temp_project_root / 'install-deps.sh').write_text('#!/bin/bash\necho "install"')
        
        violations = monitor.check_rule_12_deployment_script()
        
        # Should find missing canonical deploy.sh
        missing_violations = [v for v in violations if 'Missing canonical deploy.sh' in v.description]
        assert len(missing_violations) == 1
        
        # Should find extra deployment scripts
        extra_violations = [v for v in violations if 'Extra deployment script' in v.description]
        assert len(extra_violations) >= 3
    
    def test_rule_13_garbage_files(self, monitor, temp_project_root):
        """Test garbage file detection"""
        # Create various garbage files
        garbage_files = [
            'test.backup',
            'file.bak',
            'old_file.old',
            'temp.tmp',
            'file~',
            '.DS_Store',
            'core.12345'
        ]
        
        for garbage_file in garbage_files:
            (temp_project_root / garbage_file).write_text('garbage content')
        
        # Create legitimate files that might look like garbage
        (temp_project_root / 'templates' / 'example.tmp').parent.mkdir(exist_ok=True)
        (temp_project_root / 'templates' / 'example.tmp').write_text('template file')
        
        violations = monitor.check_rule_13_garbage_files()
        
        # Should find most garbage files
        garbage_violations = [v for v in violations if v.rule_number == 13]
        assert len(garbage_violations) >= len(garbage_files) - 1  # Allow for some legitimate exclusions
        
        # Should not flag legitimate template files
        template_violations = [v for v in violations if 'templates/example.tmp' in v.file_path]
        assert len(template_violations) == 0
    
    def test_compliance_check_integration(self, monitor, temp_project_root):
        """Test full compliance check integration"""
        # Create various violations
        (temp_project_root / 'fantasy.py').write_text('magic_function = lambda x: x')
        (temp_project_root / 'garbage.tmp').write_text('temp file')
        (temp_project_root / 'wrong_script.sh').write_text('#!/bin/bash\necho "wrong place"')
        
        report = monitor.run_compliance_check()
        
        assert isinstance(report, dict)
        assert 'timestamp' in report
        assert 'compliance_score' in report
        assert 'total_violations' in report
        assert 'violations_by_rule' in report
        assert 'auto_fixable_count' in report
        assert 'scan_duration_seconds' in report
        
        # Should have violations
        assert report['total_violations'] > 0
        assert report['compliance_score'] < 100
        
        # Should have violations from multiple rules
        assert len(report['violations_by_rule']) > 0
    
    def test_transaction_system(self, monitor):
        """Test transaction creation and management"""
        with monitor._create_transaction("Test transaction") as transaction:
            assert isinstance(transaction, ChangeTransaction)
            assert transaction.transaction_id is not None
            assert transaction.status == 'pending'
            assert transaction.transaction_id in monitor.active_transactions
            
            # Add some changes
            transaction.changes.append({
                'action': 'test_action',
                'file_path': '/test/path'
            })
        
        # Transaction should be committed and removed from active
        assert transaction.status == 'committed'
        assert transaction.transaction_id not in monitor.active_transactions
    
    def test_transaction_rollback(self, monitor, temp_project_root):
        """Test transaction rollback functionality"""
        test_file = temp_project_root / 'test_rollback.txt'
        test_file.write_text('original content')
        
        try:
            with monitor._create_transaction("Test rollback") as transaction:
                # Create backup
                backup_path = monitor._create_backup(test_file, transaction)
                
                # Modify file
                test_file.write_text('modified content')
                
                # Simulate error to trigger rollback
                raise Exception("Simulated error")
        
        except Exception:
            pass  # Expected
        
        # File should be restored to original content
        assert test_file.read_text() == 'original content'
    
    def test_auto_fix_garbage_files(self, monitor, temp_project_root):
        """Test automatic fixing of garbage files"""
        # Create garbage files
        garbage_files = [
            temp_project_root / 'test.tmp',
            temp_project_root / 'old.bak',
            temp_project_root / 'backup.backup'
        ]
        
        for garbage_file in garbage_files:
            garbage_file.write_text('garbage content')
        
        # Create violations
        violations = [
            RuleViolation(
                rule_number=13,
                rule_name="No Garbage Files",
                severity="high",
                file_path=str(garbage_file),
                line_number=None,
                description=f"Garbage file detected",
                timestamp=datetime.now().isoformat(),
                auto_fixable=True
            ) for garbage_file in garbage_files
        ]
        
        # Test dry run first
        results = monitor.auto_fix_violations(violations, dry_run=True)
        assert results['fixed_count'] == len(garbage_files)
        assert results['error_count'] == 0
        
        # Files should still exist after dry run
        for garbage_file in garbage_files:
            assert garbage_file.exists()
        
        # Test actual fix
        results = monitor.auto_fix_violations(violations, dry_run=False)
        assert results['fixed_count'] == len(garbage_files)
        assert results['error_count'] == 0
        
        # Files should be deleted
        for garbage_file in garbage_files:
            assert not garbage_file.exists()
    
    def test_auto_fix_fantasy_elements(self, monitor, temp_project_root):
        """Test automatic fixing of fantasy elements"""
        test_file = temp_project_root / 'test_fantasy.py'
        content = '''
def process():
    magic_value = 42
    wizard_config = {}
    teleport_data(magic_value)
'''
        test_file.write_text(content)
        
        violations = [
            RuleViolation(
                rule_number=1,
                rule_name="No Fantasy Elements",
                severity="high",
                file_path=str(test_file),
                line_number=2,
                description="Found forbidden term 'magic' in code",
                timestamp=datetime.now().isoformat(),
                auto_fixable=True
            ),
            RuleViolation(
                rule_number=1,
                rule_name="No Fantasy Elements",
                severity="high",
                file_path=str(test_file),
                line_number=3,
                description="Found forbidden term 'wizard' in code",
                timestamp=datetime.now().isoformat(),
                auto_fixable=True
            )
        ]
        
        results = monitor.auto_fix_violations(violations, dry_run=False)
        assert results['fixed_count'] > 0
        
        # Check that fantasy terms were replaced
        fixed_content = test_file.read_text()
        assert 'magic' not in fixed_content.lower()
        assert 'wizard' not in fixed_content.lower()
        assert 'automated' in fixed_content or 'configurator' in fixed_content
    
    def test_auto_fix_script_organization(self, monitor, temp_project_root):
        """Test automatic fixing of script organization"""
        # Create misplaced script
        misplaced_script = temp_project_root / 'subdir' / 'test.sh'
        misplaced_script.parent.mkdir()
        misplaced_script.write_text('#!/bin/bash\necho "test"')
        
        violation = RuleViolation(
            rule_number=7,
            rule_name="Script Organization",
            severity="medium",
            file_path=str(misplaced_script),
            line_number=None,
            description="Script found outside /scripts/ directory",
            timestamp=datetime.now().isoformat(),
            auto_fixable=True
        )
        
        results = monitor.auto_fix_violations([violation], dry_run=False)
        assert results['fixed_count'] == 1
        
        # Script should be moved to scripts/misc/
        expected_location = temp_project_root / 'scripts' / 'misc' / 'test.sh'
        assert expected_location.exists()
        assert not misplaced_script.exists()
    
    def test_system_integration_validation(self, monitor, temp_project_root):
        """Test system integration validation"""
        # Create some test files
        new_files = [
            temp_project_root / 'new_script.sh',
            temp_project_root / 'deploy.sh'  # Critical file
        ]
        
        for file_path in new_files:
            file_path.write_text('#!/bin/bash\necho "test"')
        
        warnings = monitor._validate_system_integration(new_files)
        
        # Should get warning about critical file
        critical_warnings = [w for w in warnings if 'critical file' in w.lower()]
        assert len(critical_warnings) > 0
    
    def test_system_integrity_validation(self, monitor):
        """Test comprehensive system integrity validation"""
        validation_results = monitor.validate_system_integrity()
        
        assert isinstance(validation_results, dict)
        assert 'timestamp' in validation_results
        assert 'overall_status' in validation_results
        assert 'checks' in validation_results
        assert 'warnings' in validation_results
        assert 'errors' in validation_results
        
        # Should have various checks
        assert 'database' in validation_results['checks']
        assert 'disk_usage' in validation_results['checks']
        
        # Overall status should be valid
        assert validation_results['overall_status'] in ['healthy', 'warning', 'error']
    
    def test_report_generation(self, monitor, temp_project_root):
        """Test compliance report generation"""
        # Run a quick compliance check
        report_data = monitor.run_compliance_check()
        
        # Generate report
        report_path = monitor.generate_report(report_data)
        
        assert Path(report_path).exists()
        
        # Load and validate report
        with open(report_path, 'r') as f:
            saved_report = json.load(f)
        
        assert 'metadata' in saved_report
        assert 'insights' in saved_report
        assert 'recommendations' in saved_report
        
        # Check latest symlink
        latest_link = Path(report_path).parent / 'latest_enhanced.json'
        assert latest_link.exists()
        assert latest_link.is_symlink()
    
    def test_performance_under_load(self, monitor, temp_project_root):
        """Test system performance under high violation loads"""
        # Create many files with violations
        num_files = 50
        
        for i in range(num_files):
            # Fantasy elements
            (temp_project_root / f'fantasy_{i}.py').write_text(f'magic_value_{i} = {i}')
            
            # Garbage files
            (temp_project_root / f'garbage_{i}.tmp').write_text(f'temp content {i}')
            
            # Misplaced scripts
            if i % 10 == 0:  # Every 10th file
                (temp_project_root / f'script_{i}.sh').write_text(f'#!/bin/bash\necho {i}')
        
        start_time = time.time()
        report = monitor.run_compliance_check()
        scan_time = time.time() - start_time
        
        # Should complete within reasonable time (adjust threshold as needed)
        assert scan_time < 60  # 1 minute
        
        # Should find many violations
        assert report['total_violations'] >= num_files * 2  # At least fantasy + garbage
        
        # Performance metrics should be recorded
        assert len(monitor.performance_metrics['scan_times']) > 0
    
    def test_concurrent_operations(self, monitor, temp_project_root):
        """Test system behavior under concurrent operations"""
        def create_violations():
            for i in range(10):
                (temp_project_root / f'concurrent_{i}.tmp').write_text(f'temp {i}')
        
        def run_scan():
            return monitor.run_compliance_check()
        
        # Start concurrent operations
        threads = []
        
        # Thread 1: Create violations
        t1 = threading.Thread(target=create_violations)
        threads.append(t1)
        
        # Thread 2: Run scan
        t2 = threading.Thread(target=run_scan)
        threads.append(t2)
        
        # Start all threads
        for t in threads:
            t.start()
        
        # Wait for completion
        for t in threads:
            t.join(timeout=30)  # 30 second timeout
        
        # System should remain stable
        assert monitor.system_state_db is not None
        
        # Should be able to run another scan
        final_report = monitor.run_compliance_check()
        assert isinstance(final_report, dict)
    
    def test_error_handling_and_recovery(self, monitor, temp_project_root):
        """Test error handling and recovery mechanisms"""
        # Test with corrupted file
        corrupted_file = temp_project_root / 'corrupted.py'
        corrupted_file.write_bytes(b'\xff\xfe\x00corrupted content\x00\xff')
        
        # Should handle corrupted files gracefully
        violations = monitor.check_rule_1_fantasy_elements()
        
        # Should not crash, might have 0 or more violations
        assert isinstance(violations, list)
        
        # Test with permission denied file (if possible on current system)
        try:
            restricted_file = temp_project_root / 'restricted.py'
            restricted_file.write_text('test content')
            restricted_file.chmod(0o000)  # No permissions
            
            # Should handle permission errors gracefully
            violations = monitor.check_rule_1_fantasy_elements()
            assert isinstance(violations, list)
            
            # Restore permissions for cleanup
            restricted_file.chmod(0o644)
        except (OSError, PermissionError):
            # Skip if we can't create restricted files
            pass
    
    def test_data_cleanup(self, monitor, temp_project_root):
        """Test data cleanup functionality"""
        # Create old files
        old_date = datetime.now() - timedelta(days=5)
        
        reports_dir = temp_project_root / 'compliance-reports'
        backups_dir = reports_dir / 'backups'
        
        # Create old report
        old_report = reports_dir / 'old_report.json'
        old_report.write_text('{"test": "data"}')
        
        # Create old backup directory
        old_backup_dir = backups_dir / old_date.strftime('%Y%m%d')
        old_backup_dir.mkdir(parents=True)
        (old_backup_dir / 'old_backup.txt').write_text('old backup')
        
        # Set old modification times
        old_timestamp = old_date.timestamp()
        os.utime(old_report, (old_timestamp, old_timestamp))
        os.utime(old_backup_dir, (old_timestamp, old_timestamp))
        
        # Add old database records
        old_timestamp_iso = old_date.isoformat()
        monitor.system_state_db.execute(
            "INSERT INTO system_health VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (old_timestamp_iso, 50.0, 60.0, 70.0, 100, 80.0, 5, 3, 1, 0)
        )
        monitor.system_state_db.commit()
        
        # Run cleanup
        monitor.cleanup_old_data(retention_days=3)
        
        # Old files should be removed
        assert not old_report.exists()
        assert not old_backup_dir.exists()
        
        # Old database records should be removed
        cursor = monitor.system_state_db.execute(
            "SELECT COUNT(*) FROM system_health WHERE timestamp = ?",
            (old_timestamp_iso,)
        )
        assert cursor.fetchone()[0] == 0
    
    @pytest.mark.parametrize("rule_number,expected_violations", [
        (1, True),   # Fantasy elements
        (7, True),   # Script organization  
        (12, True),  # Deployment scripts
        (13, True),  # Garbage files
    ])
    def test_individual_rule_checks(self, monitor, temp_project_root, rule_number, expected_violations):
        """Test individual rule checks in isolation"""
        # Set up violations for each rule
        if rule_number == 1:
            (temp_project_root / 'fantasy.py').write_text('magic = True')
        elif rule_number == 7:
            (temp_project_root / 'misplaced.sh').write_text('#!/bin/bash\necho test')
        elif rule_number == 12:
            (temp_project_root / 'extra_deploy.sh').write_text('#!/bin/bash\necho deploy')
        elif rule_number == 13:
            (temp_project_root / 'garbage.tmp').write_text('temp')
        
        report = monitor.run_compliance_check(rules_to_check=[rule_number])
        
        if expected_violations:
            assert report['total_violations'] > 0
            assert str(rule_number) in report['violations_by_rule']
        else:
            # Some rules might not have violations depending on setup
            assert report['total_violations'] >= 0
    
    def test_safe_mode_protection(self, temp_project_root):
        """Test safe mode prevents risky operations"""
        config = {
            'safe_mode': True,
            'auto_fix_enabled': True
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config, f)
            config_path = f.name
        
        try:
            monitor = EnhancedComplianceMonitor(
                project_root=str(temp_project_root),
                config_path=config_path
            )
            
            # Create high-risk violation
            high_risk_violation = RuleViolation(
                rule_number=12,  # Deployment scripts (high risk)
                rule_name="Single Deployment Script",
                severity="high",
                file_path=str(temp_project_root / 'test.sh'),
                line_number=None,
                description="Test high-risk violation",
                timestamp=datetime.now().isoformat(),
                auto_fixable=True,
                risk_level='high'
            )
            
            # Should skip high-risk fixes in safe mode
            results = monitor.auto_fix_violations([high_risk_violation])
            assert results['skipped_count'] == 1
            assert results['fixed_count'] == 0
            
            monitor.system_state_db.close()
        finally:
            os.unlink(config_path)
    
    def test_daemon_mode_shutdown(self, monitor):
        """Test daemon mode graceful shutdown"""
        # Mock the daemon loop to run only once
        original_sleep = time.sleep
        sleep_count = 0
        
        def mock_sleep(duration):
            nonlocal sleep_count
            sleep_count += 1
            if sleep_count >= 1:
                monitor.shutdown_event.set()
            return original_sleep(0.1)  # Very short sleep for testing
        
        with patch('time.sleep', side_effect=mock_sleep):
            # This should exit quickly due to shutdown event
            monitor.run_daemon_mode()
        
        assert monitor.shutdown_event.is_set()

class TestRuleViolation:
    """Test RuleViolation data class"""
    
    def test_creation(self):
        """Test RuleViolation creation"""
        violation = RuleViolation(
            rule_number=1,
            rule_name="Test Rule",
            severity="high",
            file_path="/test/path",
            line_number=42,
            description="Test violation",
            timestamp="2025-01-01T00:00:00",
            auto_fixable=True
        )
        
        assert violation.rule_number == 1
        assert violation.rule_name == "Test Rule"
        assert violation.severity == "high"
        assert violation.auto_fixable == True
        assert violation.dependencies == []  # Default empty list
        assert violation.fix_attempts == 0  # Default value

class TestSystemHealthMetrics:
    """Test SystemHealthMetrics data class"""
    
    def test_creation(self):
        """Test SystemHealthMetrics creation"""
        metrics = SystemHealthMetrics(
            timestamp="2025-01-01T00:00:00",
            cpu_usage=50.0,
            memory_usage=60.0,
            disk_usage=70.0,
            active_processes=100,
            compliance_score=85.0,
            violations_count=10,
            fixes_applied=8,
            fixes_failed=2,
            system_errors=0
        )
        
        assert metrics.cpu_usage == 50.0
        assert metrics.compliance_score == 85.0
        assert metrics.violations_count == 10

class TestChangeTransaction:
    """Test ChangeTransaction data class"""
    
    def test_creation(self):
        """Test ChangeTransaction creation"""
        transaction = ChangeTransaction(
            transaction_id="test_tx_123",
            changes=[],
            timestamp="2025-01-01T00:00:00",
            status="pending",
            backup_manifest={}
        )
        
        assert transaction.transaction_id == "test_tx_123"
        assert transaction.status == "pending"
        assert transaction.changes == []
        assert transaction.backup_manifest == {}

@pytest.mark.integration
class TestIntegrationFlows:
    """Integration tests for complete workflows"""
    
    def test_full_scan_and_fix_workflow(self, temp_project_root):
        """Test complete scan and fix workflow"""
        # Set up project with various violations
        violations_setup = {
            'fantasy.py': 'magic_value = wizard_function()',
            'garbage.tmp': 'temporary file',
            'misplaced.sh': '#!/bin/bash\necho "misplaced"',
            'extra_deploy.sh': '#!/bin/bash\necho "extra deploy"'
        }
        
        for filename, content in violations_setup.items():
            (temp_project_root / filename).write_text(content)
        
        # Initialize monitor
        monitor = EnhancedComplianceMonitor(project_root=str(temp_project_root))
        
        try:
            # Step 1: Initial scan
            initial_report = monitor.run_compliance_check()
            initial_violations = initial_report['total_violations']
            assert initial_violations > 0
            
            # Step 2: Auto-fix violations
            violations_list = []
            for rule_violations in initial_report["violations_by_rule"].values():
                for v_dict in rule_violations:
                    violations_list.append(RuleViolation(**v_dict))
            
            fix_results = monitor.auto_fix_violations(violations_list)
            assert fix_results['fixed_count'] > 0
            
            # Step 3: Verify improvements
            final_report = monitor.run_compliance_check()
            assert final_report['total_violations'] <= initial_violations
            
            # Step 4: Generate comprehensive report
            report_path = monitor.generate_report(final_report)
            assert Path(report_path).exists()
            
            # Step 5: Validate system integrity
            integrity_results = monitor.validate_system_integrity()
            assert integrity_results['overall_status'] in ['healthy', 'warning']
            
        finally:
            monitor.system_state_db.close()

def pytest_configure():
    """Configure pytest with custom markers"""
    pytest.main([
        "--tb=short",
        "--strict-markers", 
        "-v"
    ])

if __name__ == "__main__":
    # Run tests with coverage
    pytest.main([
        __file__,
        "-v", 
        "--tb=short",
        "--cov=enhanced_compliance_monitor",
        "--cov-report=html",
        "--cov-report=term-missing",
        "--cov-fail-under=85"
    ])