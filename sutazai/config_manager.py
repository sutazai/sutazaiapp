#!/usr/bin/env python3

import os
import json
import base64
import hashlib
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import sys
import yaml
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict, field
from functools import lru_cache

@dataclass
class EmailNotificationConfig:
    """
    Advanced email notification configuration.
    """
    enabled: bool = False
    smtp_host: str = 'localhost'
    smtp_port: int = 25
    smtp_username: Optional[str] = None
    smtp_password: Optional[str] = None
    use_tls: bool = False
    use_ssl: bool = False
    sender_email: str = 'sutazai-remediation@localhost'
    recipient_emails: List[str] = field(default_factory=list)
    notification_levels: List[str] = field(default_factory=lambda: ['high', 'critical'])

@dataclass
class SutazAIConfig:
    """
    Centralized configuration dataclass for SutazAI.
    Provides type-safe, extensible configuration management.
    """
    # Core system settings
    project_root: str = os.getcwd()
    log_level: str = 'INFO'
    debug_mode: bool = False

    # Performance settings
    max_memory_usage_percent: float = 80.0
    max_cpu_usage_percent: float = 90.0

    # Security settings
    security_scan_frequency: int = 24  # hours
    vulnerability_threshold: str = 'medium'

    # Dependency management
    auto_update_dependencies: bool = False
    update_check_frequency: int = 7  # days

    # Advanced settings
    custom_settings: Dict[str, Any] = field(default_factory=dict)

    # Auto-remediation settings
    auto_remediation_enabled: bool = True
    remediation_mode: str = 'standard'  # Options: 'standard', 'aggressive', 'conservative'
    max_remediation_attempts: int = 3
    auto_remediate_high_severity: bool = True
    auto_remediate_medium_severity: bool = False
    auto_remediate_low_severity: bool = False
    remediation_notification_email: Optional[str] = None
    remediation_log_retention_days: int = 30

    # Enhanced email notification configuration
    email_notifications: EmailNotificationConfig = field(default_factory=EmailNotificationConfig)

class ConfigurationManager:
    """
    Advanced configuration management system for SutazAI.
    Supports multiple config formats, environment-based overrides, and validation.
    """

    def __init__(
        self, 
        config_dir: str = 'config', 
        default_config: Optional[SutazAIConfig] = None
    ):
        """
        Initialize configuration manager.

        Args:
            config_dir (str): Directory containing configuration files
            default_config (Optional[SutazAIConfig]): Default configuration
        """
        self.config_dir = os.path.abspath(config_dir)
        os.makedirs(self.config_dir, exist_ok=True)

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s: %(message)s'
        )
        self.logger = logging.getLogger('ConfigurationManager')
        
        # Default configuration
        self.default_config = default_config or SutazAIConfig()
        self.current_config = self.default_config

    @lru_cache(maxsize=1)
    def load_config(
        self, 
        config_file: Optional[str] = None, 
        config_type: str = 'json'
    ) -> SutazAIConfig:
        """
        Load configuration from file with caching and multiple format support.

        Args:
            config_file (Optional[str]): Path to config file
            config_type (str): Configuration file type (json, yaml)

        Returns:
            SutazAIConfig: Loaded configuration
        """
        if not config_file:
            config_file = self._find_config_file(config_type)

        if not config_file or not os.path.exists(config_file):
            self.logger.warning(f"No {config_type.upper()} config found. Using default.")
            return self.default_config

        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                if config_type == 'json':
                    config_data = json.load(f)
                elif config_type == 'yaml':
                    config_data = yaml.safe_load(f)
                else:
                    raise ValueError(f"Unsupported config type: {config_type}")

            # Merge loaded config with default
            merged_config = asdict(self.default_config)
            merged_config.update(config_data)

            # Create email notification manager
            email_config = merged_config.get('email_notifications', {})
            merged_config['email_notifications'] = EmailNotificationConfig(**email_config)

            return SutazAIConfig(**merged_config)

        except Exception as e:
            self.logger.error(f"Error loading {config_type} config: {e}")
            return self.default_config

    def _find_config_file(self, config_type: str) -> Optional[str]:
        """
        Find configuration file in predefined locations.

        Args:
            config_type (str): Configuration file type

        Returns:
            Optional[str]: Path to configuration file
        """
        config_files = [
            os.path.join(self.config_dir, f'sutazai_config.{config_type}'),
            os.path.join(os.getcwd(), f'sutazai_config.{config_type}'),
            os.path.join(os.path.expanduser('~'), f'.sutazai_config.{config_type}')
        ]

        for config_file in config_files:
            if os.path.exists(config_file):
                return config_file

        return None

    def save_config(
        self, 
        config: Optional[SutazAIConfig] = None, 
        config_type: str = 'json'
    ) -> bool:
        """
        Save current configuration to file.

        Args:
            config (Optional[SutazAIConfig]): Configuration to save
            config_type (str): Configuration file type

        Returns:
            bool: Whether save was successful
        """
        config = config or self.current_config
        config_file = os.path.join(
            self.config_dir, 
            f'sutazai_config.{config_type}'
        )

        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                if config_type == 'json':
                    json.dump(asdict(config), f, indent=2)
                elif config_type == 'yaml':
                    yaml.safe_dump(asdict(config), f, default_flow_style=False)
                else:
                    raise ValueError(f"Unsupported config type: {config_type}")

            self.logger.info(f"Configuration saved to {config_file}")
            return True

        except Exception as e:
            self.logger.error(f"Error saving configuration: {e}")
            return False

    def validate_config(self, config: Optional[SutazAIConfig] = None) -> bool:
        """
        Validate configuration settings.

        Args:
            config (Optional[SutazAIConfig]): Configuration to validate

        Returns:
            bool: Whether configuration is valid
        """
        config = config or self.current_config

        validations = [
            (config.max_memory_usage_percent > 0 and config.max_memory_usage_percent <= 100, 
             "Memory usage must be between 0 and 100"),
            (config.max_cpu_usage_percent > 0 and config.max_cpu_usage_percent <= 100, 
             "CPU usage must be between 0 and 100"),
            (config.security_scan_frequency > 0, 
             "Security scan frequency must be positive"),
            (config.update_check_frequency > 0, 
             "Update check frequency must be positive")
        ]

        for condition, message in validations:
            if not condition:
                self.logger.error(f"Configuration validation failed: {message}")
                return False

        # Validate email notification configuration
        email_notification_manager = EmailNotificationManager(config.email_notifications)
        if not email_notification_manager.validate_configuration():
            return False

        return True

class EmailNotificationManager:
    """
    Advanced email notification management for SutazAI.
    Provides secure and configurable email notification capabilities.
    """

    def __init__(self, config: EmailNotificationConfig):
        """
        Initialize email notification manager.

        Args:
            config (EmailNotificationConfig): Email notification configuration
        """
        self.config = config
        self.logger = logging.getLogger('EmailNotificationManager')

    def send_notification(
        self, 
        subject: str, 
        message: str, 
        severity: str = 'info'
    ) -> bool:
        """
        Send email notification with advanced configuration.

        Args:
            subject (str): Email subject
            message (str): Email body
            severity (str): Severity level of the notification

        Returns:
            bool: Whether the email was sent successfully
        """
        # Check if notifications are enabled and severity is allowed
        if (not self.config.enabled or 
            severity not in self.config.notification_levels):
            return False

        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart

            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.config.sender_email
            msg['To'] = ', '.join(self.config.recipient_emails)
            msg['Subject'] = subject

            # Attach message body
            msg.attach(MIMEText(message, 'plain'))

            # Create SMTP connection
            if self.config.use_ssl:
                smtp_class = smtplib.SMTP_SSL
            else:
                smtp_class = smtplib.SMTP

            with smtp_class(
                host=self.config.smtp_host, 
                port=self.config.smtp_port
            ) as server:
                # Start TLS if configured
                if self.config.use_tls:
                    server.starttls()

                # Authenticate if credentials provided
                if self.config.smtp_username and self.config.smtp_password:
                    server.login(
                        self.config.smtp_username, 
                        self.config.smtp_password
                    )

                # Send email
                server.sendmail(
                    self.config.sender_email, 
                    self.config.recipient_emails, 
                    msg.as_string()
                )

            self.logger.info(f"Notification sent: {subject}")
            return True

        except Exception as e:
            self.logger.error(f"Email notification failed: {e}")
            return False

    def validate_configuration(self) -> bool:
        """
        Validate email notification configuration.

        Returns:
            bool: Whether the configuration is valid
        """
        if not self.config.enabled:
            return True

        # Check required fields
        if not self.config.smtp_host:
            self.logger.error("SMTP host is required")
            return False

        if not self.config.recipient_emails:
            self.logger.error("At least one recipient email is required")
            return False

        # Optional: Additional validation for credentials
        if bool(self.config.smtp_username) != bool(self.config.smtp_password):
            self.logger.error("Both username and password must be provided for authentication")
            return False

        return True

def main():
    config_manager = ConfigurationManager()
    
    # Load configuration
    config = config_manager.load_config()
    print("Loaded Configuration:", config)

    # Validate configuration
    if config_manager.validate_config():
        print("Configuration is valid.")
    
    # Save configuration
    config_manager.save_config()

if __name__ == '__main__':
    main() 