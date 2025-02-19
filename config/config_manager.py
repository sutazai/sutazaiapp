#!/usr/bin/env python3
"""
SutazAI Comprehensive Configuration Management System

Provides intelligent, autonomous configuration 
loading, validation, and dynamic management capabilities.
"""

import os
import sys
import yaml
import logging
import json
from typing import Dict, Any, Optional
from datetime import datetime
import hashlib

class AdvancedConfigurationManager:
    """
    Ultra-Comprehensive Configuration Management Framework
    
    Key Capabilities:
    - Dynamic configuration loading
    - Configuration validation
    - Secure configuration management
    - Automatic configuration backup
    - Environment-specific configuration
    """
    
    def __init__(
        self, 
        base_dir: str = '/opt/sutazai_project/SutazAI',
        config_dir: Optional[str] = None,
        log_dir: Optional[str] = None
    ):
        """
        Initialize Advanced Configuration Manager
        
        Args:
            base_dir (str): Base project directory
            config_dir (Optional[str]): Custom configuration directory
            log_dir (Optional[str]): Custom log directory
        """
        self.base_dir = base_dir
        self.config_dir = config_dir or os.path.join(base_dir, 'config')
        self.log_dir = log_dir or os.path.join(base_dir, 'logs', 'configuration')
        
        # Ensure directories exist
        os.makedirs(self.config_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(os.path.join(self.log_dir, 'backups'), exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s: %(message)s',
            filename=os.path.join(self.log_dir, 'config_manager.log')
        )
        self.logger = logging.getLogger('SutazAI.ConfigurationManager')
        
        # Configuration cache
        self.config_cache = {}
    
    def load_configuration(
        self, 
        config_file: str, 
        environment: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Load configuration with environment-specific overrides
        
        Args:
            config_file (str): Configuration file name
            environment (Optional[str]): Specific environment (e.g., 'development', 'production')
        
        Returns:
            Loaded configuration dictionary
        """
        try:
            # Check cache first
            if config_file in self.config_cache:
                return self.config_cache[config_file]
            
            # Construct full path
            full_path = os.path.join(self.config_dir, config_file)
            
            # Load base configuration
            with open(full_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Apply environment-specific overrides
            if environment:
                env_config_path = os.path.join(
                    self.config_dir, 
                    f'{os.path.splitext(config_file)[0]}_{environment}.yaml'
                )
                
                if os.path.exists(env_config_path):
                    with open(env_config_path, 'r') as f:
                        env_config = yaml.safe_load(f)
                    
                    # Deep merge configurations
                    config = self._deep_merge(config, env_config)
            
            # Validate configuration
            self._validate_configuration(config)
            
            # Cache configuration
            self.config_cache[config_file] = config
            
            # Log configuration load
            self.logger.info(f"Loaded configuration: {config_file}")
            
            return config
        
        except Exception as e:
            self.logger.error(f"Configuration loading failed for {config_file}: {e}")
            raise
    
    def _deep_merge(self, base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively merge configuration dictionaries
        
        Args:
            base (Dict): Base configuration
            update (Dict): Configuration to merge/override
        
        Returns:
            Merged configuration dictionary
        """
        for key, value in update.items():
            if isinstance(value, dict):
                base[key] = self._deep_merge(base.get(key, {}), value)
            else:
                base[key] = value
        return base
    
    def _validate_configuration(self, config: Dict[str, Any]):
        """
        Validate configuration structure and content
        
        Args:
            config (Dict): Configuration to validate
        
        Raises:
            ValueError: If configuration is invalid
        """
        # Placeholder for comprehensive configuration validation
        # Add specific validation rules for different configuration types
        required_keys = [
            'system_version', 
            'logging', 
            'security'
        ]
        
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required configuration key: {key}")
    
    def backup_configuration(self, config_file: str):
        """
        Create a secure backup of a configuration file
        
        Args:
            config_file (str): Configuration file to backup
        """
        try:
            source_path = os.path.join(self.config_dir, config_file)
            
            # Generate unique backup filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"{os.path.splitext(config_file)[0]}_{timestamp}.yaml.bak"
            backup_path = os.path.join(self.log_dir, 'backups', backup_filename)
            
            # Copy configuration file
            with open(source_path, 'r') as source, open(backup_path, 'w') as backup:
                backup.write(source.read())
            
            # Generate and log file hash for verification
            with open(source_path, 'rb') as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
            
            self.logger.info(f"Configuration backup created: {backup_filename}")
            self.logger.info(f"Backup file hash: {file_hash}")
        
        except Exception as e:
            self.logger.error(f"Configuration backup failed: {e}")
    
    def generate_configuration_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive configuration analysis report
        
        Returns:
            Configuration report dictionary
        """
        try:
            # Scan configuration files
            config_files = [f for f in os.listdir(self.config_dir) if f.endswith('.yaml')]
            
            configuration_report = {
                'timestamp': datetime.now().isoformat(),
                'total_config_files': len(config_files),
                'configuration_details': {}
            }
            
            # Analyze each configuration file
            for config_file in config_files:
                try:
                    config = self.load_configuration(config_file)
                    configuration_report['configuration_details'][config_file] = {
                        'keys': list(config.keys()),
                        'complexity': len(config)
                    }
                except Exception as e:
                    configuration_report['configuration_details'][config_file] = {
                        'error': str(e)
                    }
            
            # Persist report
            report_path = os.path.join(
                self.log_dir, 
                f'configuration_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            )
            
            with open(report_path, 'w') as f:
                json.dump(configuration_report, f, indent=2)
            
            self.logger.info(f"Configuration analysis report generated: {report_path}")
            
            return configuration_report
        
        except Exception as e:
            self.logger.error(f"Configuration report generation failed: {e}")
            return {}

def main():
    """
    Main execution for configuration management
    """
    try:
        config_manager = AdvancedConfigurationManager()
        
        # Backup all configuration files
        config_files = [f for f in os.listdir(config_manager.config_dir) if f.endswith('.yaml')]
        for config_file in config_files:
            config_manager.backup_configuration(config_file)
        
        # Generate configuration report
        report = config_manager.generate_configuration_report()
        
        # Print key insights
        print("Configuration Management Insights:")
        print(f"Total Configuration Files: {report.get('total_config_files', 0)}")
        print("\nConfiguration Details:")
        for filename, details in report.get('configuration_details', {}).items():
            print(f"- {filename}: {len(details.get('keys', []))} keys")
    
    except Exception as e:
        print(f"Configuration management failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()