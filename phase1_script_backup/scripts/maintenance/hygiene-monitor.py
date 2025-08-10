#!/usr/bin/env python3
"""
Real-time file monitoring for codebase hygiene violations
Watches for file changes and enforces hygiene standards immediately
"""

import os
import time
import json
import re
import logging
from pathlib import Path
from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/opt/sutazaiapp/logs/hygiene-monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('HygieneMonitor')

class HygieneViolation:
    """Represents a hygiene violation"""
    def __init__(self, file_path, violation_type, message, severity='warning'):
        self.file_path = file_path
        self.violation_type = violation_type
        self.message = message
        self.severity = severity
        self.timestamp = datetime.utcnow().isoformat()

    def to_dict(self):
        return {
            'file_path': self.file_path,
            'violation_type': self.violation_type,
            'message': self.message,
            'severity': self.severity,
            'timestamp': self.timestamp
        }

class HygieneMonitor(FileSystemEventHandler):
    """Monitor file system for hygiene violations"""
    
    FORBIDDEN_PATTERNS = [
        r'.*\.backup\d*$',
        r'.*\.fantasy.*$',
        r'.*\.agi_backup$',
        r'.*\.old$',
        r'.*\.copy$',
        r'.*_backup.*$',
        r'.*_old.*$',
        r'.*_copy.*$',
    ]
    
    FORBIDDEN_DIRS = ['archive', 'Archive', 'backup', 'Backup', 'old', 'Old']
    
    IGNORE_DIRS = ['.git', '__pycache__', 'node_modules', 'venv', '.pytest_cache', 'logs']
    
    def __init__(self, root_path):
        self.root_path = Path(root_path)
        self.violations = []
        self.auto_fix = os.environ.get('HYGIENE_AUTO_FIX', 'false').lower() == 'true'
        
    def should_ignore(self, path):
        """Check if path should be ignored"""
        path_str = str(path)
        return any(ignore_dir in path_str for ignore_dir in self.IGNORE_DIRS)
    
    def check_file_violations(self, file_path):
        """Check a file for hygiene violations"""
        violations = []
        
        # Check forbidden patterns
        for pattern in self.FORBIDDEN_PATTERNS:
            if re.match(pattern, file_path.name):
                violations.append(HygieneViolation(
                    str(file_path),
                    'forbidden_file',
                    f'File matches forbidden pattern: {pattern}',
                    'error'
                ))
        
        # Check file naming conventions
        if file_path.suffix in ['.py', '.sh', '.md']:
            if not re.match(r'^[a-z0-9_-]+$', file_path.stem):
                violations.append(HygieneViolation(
                    str(file_path),
                    'naming_convention',
                    'File name should use kebab-case or snake_case',
                    'warning'
                ))
        
        # Check for large files
        try:
            size_mb = file_path.stat().st_size / (1024 * 1024)
            if size_mb > 5:
                violations.append(HygieneViolation(
                    str(file_path),
                    'large_file',
                    f'File is too large: {size_mb:.1f}MB',
                    'warning'
                ))
        except Exception as e:
            # Suppressed exception (was bare except)
            logger.debug(f"Suppressed exception: {e}")
            pass
        
        # Check Python files for specific issues
        if file_path.suffix == '.py':
            violations.extend(self.check_python_file(file_path))
        
        return violations
    
    def check_python_file(self, file_path):
        """Check Python file for hygiene violations"""
        violations = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for hardcoded secrets
            secret_patterns = [
                r'password\s*=\s*["\'][^"\']+["\']',
                r'api_key\s*=\s*["\'][^"\']+["\']',
                r'secret\s*=\s*["\'][^"\']+["\']',
            ]
            
            for pattern in secret_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    # Check if it's not using environment variables
                    if not re.search(r'os\.(environ|getenv)', content):
                        violations.append(HygieneViolation(
                            str(file_path),
                            'hardcoded_secret',
                            'Potential hardcoded secret detected',
                            'error'
                        ))
                        break
            
            # Check for TODO/FIXME comments
            todo_matches = re.findall(r'#\s*(TODO|FIXME|HACK|XXX)', content)
            if todo_matches:
                violations.append(HygieneViolation(
                    str(file_path),
                    'todo_comment',
                    f'Found {len(todo_matches)} TODO/FIXME comments',
                    'info'
                ))
            
        except Exception as e:
            logger.error(f"Error checking Python file {file_path}: {e}")
        
        return violations
    
    def handle_violation(self, violation):
        """Handle a detected violation"""
        logger.warning(f"Violation: {violation.violation_type} - {violation.file_path}: {violation.message}")
        
        self.violations.append(violation)
        
        # Auto-fix if enabled
        if self.auto_fix and violation.severity == 'error':
            self.auto_fix_violation(violation)
    
    def auto_fix_violation(self, violation):
        """Attempt to auto-fix certain violations"""
        if violation.violation_type == 'forbidden_file':
            # Delete forbidden files
            try:
                os.remove(violation.file_path)
                logger.info(f"Auto-fixed: Deleted forbidden file {violation.file_path}")
            except Exception as e:
                logger.error(f"Failed to delete {violation.file_path}: {e}")
        
    def on_created(self, event):
        """Handle file creation"""
        if event.is_directory:
            # Check for forbidden directory names
            dir_name = Path(event.src_path).name
            if dir_name in self.FORBIDDEN_DIRS:
                violation = HygieneViolation(
                    event.src_path,
                    'forbidden_directory',
                    f'Directory name "{dir_name}" is forbidden',
                    'error'
                )
                self.handle_violation(violation)
        else:
            if not self.should_ignore(event.src_path):
                file_path = Path(event.src_path)
                violations = self.check_file_violations(file_path)
                for v in violations:
                    self.handle_violation(v)
    
    def on_modified(self, event):
        """Handle file modification"""
        if not event.is_directory and not self.should_ignore(event.src_path):
            file_path = Path(event.src_path)
            if file_path.suffix in ['.py', '.js', '.json', '.yaml', '.yml']:
                violations = self.check_file_violations(file_path)
                for v in violations:
                    self.handle_violation(v)
    
    def on_moved(self, event):
        """Handle file/directory moves"""
        if not self.should_ignore(event.dest_path):
            dest_path = Path(event.dest_path)
            violations = self.check_file_violations(dest_path)
            for v in violations:
                self.handle_violation(v)
    
    def save_violations_report(self):
        """Save violations to a report file"""
        report_path = self.root_path / 'logs' / 'hygiene-violations.json'
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump([v.to_dict() for v in self.violations], f, indent=2)
        
        logger.info(f"Saved {len(self.violations)} violations to {report_path}")

def run_initial_scan(monitor, root_path):
    """Run initial scan of the codebase"""
    logger.info("Running initial hygiene scan...")
    
    for file_path in Path(root_path).rglob('*'):
        if file_path.is_file() and not monitor.should_ignore(file_path):
            violations = monitor.check_file_violations(file_path)
            for v in violations:
                monitor.handle_violation(v)
    
    logger.info(f"Initial scan complete. Found {len(monitor.violations)} violations.")

def main():
    """Main monitoring loop"""
    root_path = os.environ.get('SUTAZAI_ROOT', '/opt/sutazaiapp')
    
    logger.info(f"Starting hygiene monitor for {root_path}")
    logger.info(f"Auto-fix: {os.environ.get('HYGIENE_AUTO_FIX', 'false')}")
    
    # Create monitor
    event_handler = HygieneMonitor(root_path)
    observer = Observer()
    observer.schedule(event_handler, root_path, recursive=True)
    
    # Run initial scan
    run_initial_scan(event_handler, root_path)
    
    # Start monitoring
    observer.start()
    logger.info("Real-time monitoring started. Press Ctrl+C to stop.")
    
    try:
        while True:
            time.sleep(60)
            # Save report every minute
            if event_handler.violations:
                event_handler.save_violations_report()
    except KeyboardInterrupt:
        observer.stop()
        logger.info("Monitoring stopped by user")
    
    observer.join()
    
    # Final report
    if event_handler.violations:
        event_handler.save_violations_report()
        logger.warning(f"Total violations detected: {len(event_handler.violations)}")
    else:
        logger.info("No violations detected!")

if __name__ == "__main__":
    main()