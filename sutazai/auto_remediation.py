import os
import sys
import logging
import subprocess
import json
import smtplib
import importlib.util
import inspect
from email.mime.text import MIMEText
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta

from sutazai.error_handler import SutazAIErrorHandler
from sutazai.security_scanner import SecurityScanner, SecurityScanResult
from sutazai.config_manager import ConfigurationManager, SutazAIConfig

@dataclass
class RemediationAction:
    """
    Structured representation of a remediation action.
    """
    type: str
    description: str
    command: Optional[str] = None
    script_path: Optional[str] = None
    custom_script_module: Optional[str] = None
    custom_script_function: Optional[str] = None
    severity: str = 'low'
    status: str = 'pending'
    error_message: Optional[str] = None
    attempts: int = 0
    last_attempt_timestamp: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)

class CustomRemediationScriptLoader:
    """
    Utility class for loading and executing custom remediation scripts.
    """

    @staticmethod
    def load_script_module(script_path: str) -> Optional[Any]:
        """
        Dynamically load a Python module from a script path.

        Args:
            script_path (str): Path to the Python script

        Returns:
            Optional[Any]: Loaded module or None if loading fails
        """
        try:
            module_name = os.path.splitext(os.path.basename(script_path))[0]
            spec = importlib.util.spec_from_file_location(module_name, script_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        except Exception as e:
            logging.error(f"Could not load custom script {script_path}: {e}")
            return None

    @staticmethod
    def find_remediation_function(module: Any) -> Optional[Callable]:
        """
        Find a suitable remediation function in the module.

        Args:
            module (Any): Loaded module

        Returns:
            Optional[Callable]: Remediation function or None
        """
        for name, func in inspect.getmembers(module, inspect.isfunction):
            if name.startswith('remediate_') or 'remediation' in name.lower():
                return func
        return None

class AdvancedRemediationStrategies:
    """
    Advanced remediation strategies beyond basic command execution.
    """

    @staticmethod
    def dependency_tree_update(package: str) -> bool:
        """
        Perform a comprehensive dependency update considering dependency tree.

        Args:
            package (str): Package to update

        Returns:
            bool: Whether update was successful
        """
        try:
            # Use pip-compile or similar to resolve dependency conflicts
            subprocess.run(
                [sys.executable, '-m', 'pip', 'install', '--upgrade', package, '--upgrade-strategy', 'eager'],
                check=True
            )
            return True
        except subprocess.CalledProcessError:
            return False

    @staticmethod
    def rollback_configuration(config_path: str, backup_path: Optional[str] = None) -> bool:
        """
        Rollback configuration to a previous state.

        Args:
            config_path (str): Current configuration path
            backup_path (Optional[str]): Backup configuration path

        Returns:
            bool: Whether rollback was successful
        """
        try:
            if backup_path and os.path.exists(backup_path):
                import shutil
                shutil.copy2(backup_path, config_path)
                return True
            return False
        except Exception as e:
            logging.error(f"Configuration rollback failed: {e}")
            return False

class AutoRemediationManager:
    """
    Comprehensive automated remediation system for SutazAI.
    Provides intelligent, configurable system healing capabilities.
    """

    def __init__(
        self, 
        project_root: str = '.', 
        config: Optional[SutazAIConfig] = None
    ):
        """
        Initialize the auto-remediation manager.

        Args:
            project_root (str): Root directory of the project
            config (Optional[SutazAIConfig]): Configuration object
        """
        self.project_root = os.path.abspath(project_root)
        self.config = config or ConfigurationManager().load_config()
        
        # Setup logging and error handling
        self.error_handler = SutazAIErrorHandler()
        self.logger = logging.getLogger('AutoRemediationManager')

    def _should_remediate_action(self, action: RemediationAction) -> bool:
        """
        Determine if an action should be remediated based on configuration.

        Args:
            action (RemediationAction): Remediation action to evaluate

        Returns:
            bool: Whether the action should be remediated
        """
        if not self.config.auto_remediation_enabled:
            return False

        severity_map = {
            'high': self.config.auto_remediate_high_severity,
            'medium': self.config.auto_remediate_medium_severity,
            'low': self.config.auto_remediate_low_severity
        }

        return (
            severity_map.get(action.severity, False) and 
            action.attempts < self.config.max_remediation_attempts
        )

    def _send_remediation_notification(
        self, 
        action: RemediationAction, 
        status: str
    ) -> None:
        """
        Send email notification for remediation actions.

        Args:
            action (RemediationAction): Remediation action
            status (str): Status of the remediation
        """
        if not self.config.remediation_notification_email:
            return

        try:
            msg = MIMEText(
                f"Remediation Action {status}\n"
                f"Type: {action.type}\n"
                f"Description: {action.description}\n"
                f"Severity: {action.severity}\n"
                f"Status: {status}"
            )
            msg['Subject'] = f"SutazAI Remediation {status.capitalize()}"
            msg['From'] = "sutazai-remediation@localhost"
            msg['To'] = self.config.remediation_notification_email

            # In a real-world scenario, use proper SMTP configuration
            with smtplib.SMTP('localhost') as smtp:
                smtp.send_message(msg)
        except Exception as e:
            self.logger.error(f"Could not send remediation notification: {e}")

    def analyze_security_report(
        self, 
        security_report: SecurityScanResult
    ) -> List[RemediationAction]:
        """
        Analyze security scan results and generate remediation actions.

        Args:
            security_report (SecurityScanResult): Security scan results

        Returns:
            List[RemediationAction]: Recommended remediation actions
        """
        remediation_actions = []

        # Advanced vulnerability categorization and action generation
        vulnerability_strategies = {
            'high_vulnerabilities': {
                'severity': 'high',
                'strategies': [
                    self._generate_code_fix_action,
                    self._generate_dependency_update_action
                ]
            },
            'medium_vulnerabilities': {
                'severity': 'medium',
                'strategies': [
                    self._generate_configuration_update_action,
                    self._generate_permission_fix_action
                ]
            },
            'low_vulnerabilities': {
                'severity': 'low',
                'strategies': [
                    self._generate_logging_action,
                    self._generate_monitoring_action
                ]
            },
            'custom_vulnerabilities': {
                'severity': 'high',  # Can be configurable
                'strategies': [
                    self._generate_custom_script_action
                ]
            }
        }

        for vuln_type, strategy_config in vulnerability_strategies.items():
            vulnerabilities = getattr(security_report, vuln_type, [])
            for vuln in vulnerabilities:
                for strategy in strategy_config['strategies']:
                    action = strategy(vuln, strategy_config['severity'])
                    if action:
                        remediation_actions.append(action)

        return remediation_actions

    def _generate_code_fix_action(
        self, 
        vulnerability: Dict[str, Any], 
        severity: str
    ) -> Optional[RemediationAction]:
        """
        Generate a code fix remediation action.

        Args:
            vulnerability (Dict[str, Any]): Vulnerability details
            severity (str): Severity level

        Returns:
            Optional[RemediationAction]: Remediation action or None
        """
        if 'file' in vulnerability and 'line' in vulnerability:
            return RemediationAction(
                type='code_vulnerability',
                description=f"Code vulnerability in {vulnerability['file']} at line {vulnerability['line']}",
                severity=severity
            )
        return None

    def _generate_dependency_update_action(
        self, 
        vulnerability: Dict[str, Any], 
        severity: str
    ) -> Optional[RemediationAction]:
        """
        Generate a dependency update remediation action.

        Args:
            vulnerability (Dict[str, Any]): Vulnerability details
            severity (str): Severity level

        Returns:
            Optional[RemediationAction]: Remediation action or None
        """
        if 'package' in vulnerability:
            return RemediationAction(
                type='dependency_vulnerability',
                description=f"Dependency vulnerability in {vulnerability['package']}",
                command=f"pip install --upgrade {vulnerability['package']}",
                severity=severity
            )
        return None

    def _generate_configuration_update_action(
        self, 
        vulnerability: Dict[str, Any], 
        severity: str
    ) -> Optional[RemediationAction]:
        """
        Generate a configuration update remediation action.

        Args:
            vulnerability (Dict[str, Any]): Vulnerability details
            severity (str): Severity level

        Returns:
            Optional[RemediationAction]: Remediation action or None
        """
        if 'configuration' in vulnerability:
            return RemediationAction(
                type='configuration_vulnerability',
                description=f"Configuration vulnerability: {vulnerability['configuration']}",
                severity=severity
            )
        return None

    def _generate_permission_fix_action(
        self, 
        vulnerability: Dict[str, Any], 
        severity: str
    ) -> Optional[RemediationAction]:
        """
        Generate a file permission fix remediation action.

        Args:
            vulnerability (Dict[str, Any]): Vulnerability details
            severity (str): Severity level

        Returns:
            Optional[RemediationAction]: Remediation action or None
        """
        if 'path' in vulnerability and 'permissions' in vulnerability:
            return RemediationAction(
                type='permission_vulnerability',
                description=f"Overly permissive file: {vulnerability['path']}",
                command=f"chmod 644 {vulnerability['path']}",
                severity=severity
            )
        return None

    def _generate_logging_action(
        self, 
        vulnerability: Dict[str, Any], 
        severity: str
    ) -> Optional[RemediationAction]:
        """
        Generate a logging remediation action.

        Args:
            vulnerability (Dict[str, Any]): Vulnerability details
            severity (str): Severity level

        Returns:
            Optional[RemediationAction]: Remediation action or None
        """
        return RemediationAction(
            type='logging_vulnerability',
            description="Low-severity vulnerability detected",
            severity=severity
        )

    def _generate_monitoring_action(
        self, 
        vulnerability: Dict[str, Any], 
        severity: str
    ) -> Optional[RemediationAction]:
        """
        Generate a monitoring remediation action.

        Args:
            vulnerability (Dict[str, Any]): Vulnerability details
            severity (str): Severity level

        Returns:
            Optional[RemediationAction]: Remediation action or None
        """
        return RemediationAction(
            type='monitoring_vulnerability',
            description="Low-severity vulnerability requires monitoring",
            severity=severity
        )

    def _generate_custom_script_action(
        self, 
        vulnerability: Dict[str, Any], 
        severity: str
    ) -> Optional[RemediationAction]:
        """
        Generate a custom script remediation action.

        Args:
            vulnerability (Dict[str, Any]): Vulnerability details
            severity (str): Severity level

        Returns:
            Optional[RemediationAction]: Remediation action or None
        """
        if 'custom_script' in vulnerability:
            return RemediationAction(
                type='custom_script_vulnerability',
                description=f"Custom script remediation: {vulnerability.get('description', 'Unknown')}",
                custom_script_module=vulnerability.get('script_module'),
                custom_script_function=vulnerability.get('script_function'),
                context=vulnerability.get('context', {}),
                severity=severity
            )
        return None

    def execute_remediation_actions(
        self, 
        actions: List[RemediationAction]
    ) -> Dict[str, List[RemediationAction]]:
        """
        Execute remediation actions with intelligent handling.

        Args:
            actions (List[RemediationAction]): Actions to execute

        Returns:
            Dict[str, List[RemediationAction]]: Execution results by status
        """
        results = {
            'successful': [],
            'failed': [],
            'skipped': []
        }

        for action in actions:
            # Determine if action should be remediated
            if not self._should_remediate_action(action):
                action.status = 'skipped'
                results['skipped'].append(action)
                continue

            try:
                # Execute command-based remediation
                if action.command:
                    result = subprocess.run(
                        action.command, 
                        shell=True, 
                        check=True, 
                        capture_output=True, 
                        text=True
                    )
                    action.status = 'successful'
                    action.attempts += 1
                    action.last_attempt_timestamp = datetime.now().isoformat()
                    results['successful'].append(action)
                    self._send_remediation_notification(action, 'successful')

                # Execute script-based remediation
                elif action.script_path:
                    result = subprocess.run(
                        [sys.executable, action.script_path], 
                        check=True, 
                        capture_output=True, 
                        text=True
                    )
                    action.status = 'successful'
                    action.attempts += 1
                    action.last_attempt_timestamp = datetime.now().isoformat()
                    results['successful'].append(action)
                    self._send_remediation_notification(action, 'successful')

                # Execute custom script-based remediation
                elif action.custom_script_module and action.custom_script_function:
                    module = CustomRemediationScriptLoader.load_script_module(action.custom_script_module)
                    if module:
                        remediation_func = getattr(module, action.custom_script_function, None)
                        if remediation_func:
                            result = remediation_func(**action.context)
                            action.status = 'successful' if result else 'failed'
                            action.attempts += 1
                            action.last_attempt_timestamp = datetime.now().isoformat()
                            
                            if action.status == 'successful':
                                results['successful'].append(action)
                                self._send_remediation_notification(action, 'successful')
                            else:
                                results['failed'].append(action)
                                self._send_remediation_notification(action, 'failed')
                        else:
                            raise ValueError(f"Function {action.custom_script_function} not found")
                    else:
                        raise ValueError(f"Could not load module {action.custom_script_module}")

            except (subprocess.CalledProcessError, ValueError) as e:
                action.status = 'failed'
                action.error_message = str(e)
                action.attempts += 1
                action.last_attempt_timestamp = datetime.now().isoformat()
                results['failed'].append(action)
                self.error_handler.log_error(
                    f"Remediation action failed: {action.description}",
                    error=e
                )
                self._send_remediation_notification(action, 'failed')

        return results

    def generate_remediation_report(
        self, 
        actions_results: Dict[str, List[RemediationAction]]
    ) -> None:
        """
        Generate a comprehensive remediation report.

        Args:
            actions_results (Dict[str, List[RemediationAction]]): Remediation action results
        """
        report_path = os.path.join(
            self.project_root, 
            f'remediation_report_{os.getpid()}.json'
        )

        # Clean up old reports based on log retention configuration
        self._cleanup_old_reports()

        try:
            with open(report_path, 'w') as f:
                json.dump(
                    {k: [asdict(action) for action in v] for k, v in actions_results.items()}, 
                    f, 
                    indent=2
                )
            self.logger.info(f"Remediation report generated: {report_path}")
        except Exception as e:
            self.error_handler.log_error(
                "Could not generate remediation report", 
                error=e
            )

    def _cleanup_old_reports(self) -> None:
        """
        Clean up old remediation reports based on log retention configuration.
        """
        retention_days = self.config.remediation_log_retention_days
        cutoff_date = datetime.now() - timedelta(days=retention_days)

        for filename in os.listdir(self.project_root):
            if filename.startswith('remediation_report_') and filename.endswith('.json'):
                try:
                    file_path = os.path.join(self.project_root, filename)
                    file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                    
                    if file_mtime < cutoff_date:
                        os.remove(file_path)
                        self.logger.info(f"Removed old report: {filename}")
                except Exception as e:
                    self.logger.warning(f"Could not clean up report {filename}: {e}")

    def auto_remediate(self) -> Dict[str, List[RemediationAction]]:
        """
        Perform comprehensive automated remediation.

        Returns:
            Dict[str, List[RemediationAction]]: Remediation action results
        """
        # Run security scan
        security_scanner = SecurityScanner(project_root=self.project_root)
        security_report = security_scanner.generate_security_report()

        # Analyze security report and generate remediation actions
        remediation_actions = self.analyze_security_report(security_report)

        # Execute remediation actions
        remediation_results = self.execute_remediation_actions(remediation_actions)

        # Generate remediation report
        self.generate_remediation_report(remediation_results)

        return remediation_results

def main():
    remediation_manager = AutoRemediationManager()
    results = remediation_manager.auto_remediate()
    
    print("Remediation Results:")
    for status, actions in results.items():
        print(f"{status.capitalize()} Actions: {len(actions)}")

if __name__ == '__main__':
    main()

"""
SutazAI Automated Remediation Module

This module provides an intelligent, configurable automated remediation system
for the SutazAI project. It integrates with security scanning, error handling,
and configuration management to automatically detect, analyze, and resolve
system vulnerabilities and performance issues.

Key Features:
- Multi-level vulnerability detection and remediation
- Configurable remediation strategies
- Intelligent action selection and execution
- Comprehensive reporting and logging
- Email notifications for remediation actions

Configuration Options:
- Enable/disable auto-remediation
- Set remediation mode (standard, aggressive, conservative)
- Configure severity-based remediation
- Set maximum remediation attempts
- Enable email notifications

Remediation Strategies:
1. Code Vulnerability Fixes
2. Dependency Updates
3. Configuration Updates
4. Permission Fixes
5. Logging and Monitoring Actions

Usage:
```python
from sutazai.auto_remediation import AutoRemediationManager

# Initialize with default configuration
remediation_manager = AutoRemediationManager()

# Perform automated remediation
results = remediation_manager.auto_remediate()
```

Dependencies:
- sutazai.error_handler
- sutazai.security_scanner
- sutazai.config_manager

Author: SutazAI Development Team
Version: 1.0.0
""" 