#!/usr/bin/env python3
"""
Fantasy Elements Validator Configuration Manager

This script helps manage configuration for the fantasy elements validator,
allowing easy customization of forbidden terms, scan patterns, and exclusions.

Purpose: Configure fantasy-elements-validator.py settings
Usage: python fantasy-elements-config.py [--add-term TERM] [--remove-term TERM] [--list] [--export] [--import CONFIG]
Requirements: Standard library only
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any


class FantasyElementsConfig:
    """Configuration manager for fantasy elements validator"""
    
    def __init__(self, config_file: str = "fantasy-elements-config.json"):
        self.config_file = Path(config_file)
        self.config = self._load_default_config()
        
        if self.config_file.exists():
            self._load_config()
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration"""
        return {
            "forbidden_terms": {
                "specific implementation name (e.g., emailSender, dataProcessor)": {
                    "terms": ["specific implementation name (e.g., emailSender, dataProcessor)", "automated, programmatic, or algorithmic", "automatically, programmatically", "specificHandler (e.g., authHandler, dataHandler)", "specificService (e.g., emailService, userService)", "specificFunction (e.g., validateInput, processData)"],
                    "severity": "high",
                    "suggestions": {
                        "specific implementation name (e.g., emailSender, dataProcessor)": "specific implementation name (e.g., emailSender, dataProcessor)",
                        "automated, programmatic, or algorithmic": "automated, programmatic, or algorithmic",
                        "automatically, programmatically": "automatically, programmatically",
                        "specificHandler (e.g., authHandler, dataHandler)": "specificHandler (e.g., authHandler, dataHandler)",
                        "specificService (e.g., emailService, userService)": "specificService (e.g., emailService, userService)",
                        "specificFunction (e.g., validateInput, processData)": "specificFunction (e.g., validateInput, processData)"
                    }
                },
                "assistant, helper, processor, manager": {
                    "terms": ["assistant, helper, processor, manager", "helperService, processingService", "automation, processing, computation", "processingHandler, assistantHandler"],
                    "severity": "high",
                    "suggestions": {
                        "assistant, helper, processor, manager": "assistant, helper, processor, manager",
                        "helperService, processingService": "helperService, processingService",
                        "automation, processing, computation": "automation, processing, computation",
                        "processingHandler, assistantHandler": "processingHandler, assistantHandler"
                    }
                },
                "transfer, send, transmit, copy": {
                    "terms": ["transfer, send, transmit, copy", "transferData, sendData, transmitData", "data transfer, transmission, migration", "transferring, sending, transmitting"],
                    "severity": "high",
                    "suggestions": {
                        "transfer, send, transmit, copy": "transfer, send, transmit, copy",
                        "transferData, sendData, transmitData": "transferData, sendData, transmitData",
                        "data transfer, transmission, migration": "data transfer, transmission, migration",
                        "transferring, sending, transmitting": "transferring, sending, transmitting"
                    }
                },
                "external_service, third_party_api": {
                    "terms": ["external service, third-party API, opaque system", "externalService, thirdPartyAPI", "external_service, third_party_api", "externalService, thirdPartyAPI"],
                    "severity": "medium",
                    "suggestions": {
                        "external service, third-party API, opaque system": "external service, third-party API, opaque system",
                        "externalService, thirdPartyAPI": "externalService, thirdPartyAPI",
                        "external_service, third_party_api": "external_service, third_party_api",
                        "externalService, thirdPartyAPI": "externalService, thirdPartyAPI"
                    }
                },
                "hypothetical": {
                    "terms": ["specific future version or roadmap item", "conditional logic or feature flag", "tested implementation or proof of concept", "validated approach or tested solution", "documented specification or proven concept", "concrete implementation or real example"],
                    "severity": "medium",
                    "suggestions": {
                        "specific future version or roadmap item": "specific future version or roadmap item",
                        "conditional logic or feature flag": "conditional logic or feature flag",
                        "tested implementation or proof of concept": "tested implementation or proof of concept",
                        "validated approach or tested solution": "validated approach or tested solution",
                        "documented specification or proven concept": "documented specification or proven concept",
                        "concrete implementation or real example": "concrete implementation or real example"
                    }
                }
            },
            "scan_patterns": [
                "**/*.py",
                "**/*.js",
                "**/*.ts",
                "**/*.jsx",
                "**/*.tsx",
                "**/*.go",
                "**/*.rs",
                "**/*.java",
                "**/*.cpp",
                "**/*.c",
                "**/*.h",
                "**/*.hpp",
                "**/*.md",
                "**/*.yml",
                "**/*.yaml",
                "**/*.json",
                "**/*.toml",
                "**/*.cfg",
                "**/*.ini",
                "**/Dockerfile*",
                "**/*.sh",
                "**/*.bash"
            ],
            "exclude_patterns": [
                "*.git/*",
                "*/node_modules/*",
                "*/__pycache__/*",
                "*/venv/*",
                "*/env/*",
                "*/.venv/*",
                "*/.env/*",
                "*/build/*",
                "*/dist/*",
                "*/.pytest_cache/*",
                "*/logs/*",
                "*/data/*",
                "*/archive/*",
                "*/backup*"
            ],
            "placeholder_patterns": [
                {
                    "pattern": "TODO.*specific implementation name (e.g., emailSender, dataProcessor)",
                    "description": "TODO with specific implementation name (e.g., emailSender, dataProcessor) reference"
                },
                {
                    "pattern": "TODO.*specific future version or roadmap item",
                    "description": "TODO with specific future version or roadmap item reference"
                },
                {
                    "pattern": "//.*imagine",
                    "description": "Comment with imagine"
                },
                {
                    "pattern": "#.*imagine",
                    "description": "Comment with imagine"
                },
                {
                    "pattern": "//.*TODO.*telekinesis",
                    "description": "TODO with telekinesis"
                },
                {
                    "pattern": "#.*TODO.*telekinesis",
                    "description": "TODO with telekinesis"
                },
                {
                    "pattern": "placeholder.*function",
                    "description": "Placeholder function"
                },
                {
                    "pattern": "stub.*implementation",
                    "description": "Stub implementation"
                },
                {
                    "pattern": "mock.*data.*\\(",
                    "description": "Mock data function calls"
                },
                {
                    "pattern": "fake.*api",
                    "description": "Fake API references"
                },
                {
                    "pattern": "dummy.*service",
                    "description": "Dummy service references"
                },
                {
                    "pattern": "temp.*fix",
                    "description": "Temporary fix references"
                },
                {
                    "pattern": "hack.*for.*now",
                    "description": "Hack for now"
                },
                {
                    "pattern": "quick.*dirty",
                    "description": "Quick and dirty"
                },
                {
                    "pattern": "will.*implement.*later",
                    "description": "Will implement later"
                },
                {
                    "pattern": "TODO.*implement",
                    "description": "TODO implement (often speculative)"
                }
            ]
        }
    
    def _load_config(self):
        """Load configuration from file"""
        try:
            with open(self.config_file, 'r') as f:
                saved_config = json.load(f)
                self.config.update(saved_config)
        except Exception as e:
            print(f"Warning: Could not load config file {self.config_file}: {e}")
    
    def save_config(self):
        """Save current configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            print(f"Configuration saved to {self.config_file}")
        except Exception as e:
            print(f"Error saving config: {e}")
            return False
        return True
    
    def add_forbidden_term(self, category: str, term: str, severity: str = "medium", suggestion: str = None):
        """Add a new forbidden term"""
        if category not in self.config["forbidden_terms"]:
            self.config["forbidden_terms"][category] = {
                "terms": [],
                "severity": severity,
                "suggestions": {}
            }
        
        if term not in self.config["forbidden_terms"][category]["terms"]:
            self.config["forbidden_terms"][category]["terms"].append(term)
            
            if suggestion:
                self.config["forbidden_terms"][category]["suggestions"][term] = suggestion
            
            print(f"Added forbidden term '{term}' to category '{category}'")
            return True
        else:
            print(f"Term '{term}' already exists in category '{category}'")
            return False
    
    def remove_forbidden_term(self, term: str):
        """Remove a forbidden term from all categories"""
        removed = False
        for category, config in self.config["forbidden_terms"].items():
            if term in config["terms"]:
                config["terms"].remove(term)
                if term in config["suggestions"]:
                    del config["suggestions"][term]
                print(f"Removed '{term}' from category '{category}'")
                removed = True
        
        if not removed:
            print(f"Term '{term}' not found in any category")
        
        return removed
    
    def add_scan_pattern(self, pattern: str):
        """Add a new scan pattern"""
        if pattern not in self.config["scan_patterns"]:
            self.config["scan_patterns"].append(pattern)
            print(f"Added scan pattern: {pattern}")
            return True
        else:
            print(f"Scan pattern '{pattern}' already exists")
            return False
    
    def add_exclude_pattern(self, pattern: str):
        """Add a new exclude pattern"""
        if pattern not in self.config["exclude_patterns"]:
            self.config["exclude_patterns"].append(pattern)
            print(f"Added exclude pattern: {pattern}")
            return True
        else:
            print(f"Exclude pattern '{pattern}' already exists")
            return False
    
    def add_placeholder_pattern(self, pattern: str, description: str):
        """Add a new placeholder pattern"""
        new_pattern = {"pattern": pattern, "description": description}
        
        # Check if pattern already exists
        for existing in self.config["placeholder_patterns"]:
            if existing["pattern"] == pattern:
                print(f"Placeholder pattern '{pattern}' already exists")
                return False
        
        self.config["placeholder_patterns"].append(new_pattern)
        print(f"Added placeholder pattern: {pattern}")
        return True
    
    def list_configuration(self):
        """Display current configuration"""
        print("=== Fantasy Elements Validator Configuration ===\n")
        
        print("Forbidden Terms:")
        for category, config in self.config["forbidden_terms"].items():
            print(f"  {category} ({config['severity']} severity):")
            for term in config["terms"]:
                suggestion = config["suggestions"].get(term, "No suggestion")
                print(f"    - {term} → {suggestion}")
            print()
        
        print(f"Scan Patterns ({len(self.config['scan_patterns'])}):")
        for pattern in self.config["scan_patterns"]:
            print(f"  - {pattern}")
        print()
        
        print(f"Exclude Patterns ({len(self.config['exclude_patterns'])}):")
        for pattern in self.config["exclude_patterns"]:
            print(f"  - {pattern}")
        print()
        
        print(f"Placeholder Patterns ({len(self.config['placeholder_patterns'])}):")
        for pattern_config in self.config["placeholder_patterns"]:
            print(f"  - {pattern_config['pattern']} ({pattern_config['description']})")
    
    def export_config(self, output_file: str):
        """Export configuration to specified file"""
        try:
            with open(output_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            print(f"Configuration exported to {output_file}")
            return True
        except Exception as e:
            print(f"Error exporting config: {e}")
            return False
    
    def import_config(self, input_file: str):
        """Import configuration from specified file"""
        try:
            with open(input_file, 'r') as f:
                imported_config = json.load(f)
                self.config.update(imported_config)
            print(f"Configuration imported from {input_file}")
            return True
        except Exception as e:
            print(f"Error importing config: {e}")
            return False
    
    def validate_config(self):
        """Validate current configuration"""
        errors = []
        
        # Validate forbidden terms structure
        for category, config in self.config["forbidden_terms"].items():
            if not isinstance(config.get("terms"), list):
                errors.append(f"Category '{category}' terms must be a list")
            
            if config.get("severity") not in ["high", "medium", "low"]:
                errors.append(f"Category '{category}' has invalid severity")
            
            if not isinstance(config.get("suggestions"), dict):
                errors.append(f"Category '{category}' suggestions must be a dict")
        
        # Validate patterns are lists
        if not isinstance(self.config.get("scan_patterns"), list):
            errors.append("scan_patterns must be a list")
        
        if not isinstance(self.config.get("exclude_patterns"), list):
            errors.append("exclude_patterns must be a list")
        
        if not isinstance(self.config.get("placeholder_patterns"), list):
            errors.append("placeholder_patterns must be a list")
        
        if errors:
            print("Configuration validation errors:")
            for error in errors:
                print(f"  - {error}")
            return False
        else:
            print("Configuration is valid")
            return True


def main():
    parser = argparse.ArgumentParser(
        description="Fantasy Elements Validator Configuration Manager",
        epilog="Manage forbidden terms, scan patterns, and other validation settings."
    )
    
    parser.add_argument(
        '--config-file',
        default='fantasy-elements-config.json',
        help='Configuration file path (default: fantasy-elements-config.json)'
    )
    
    parser.add_argument(
        '--add-term',
        nargs=3,
        metavar=('CATEGORY', 'TERM', 'SUGGESTION'),
        help='Add forbidden term: CATEGORY TERM SUGGESTION'
    )
    
    parser.add_argument(
        '--remove-term',
        help='Remove forbidden term from all categories'
    )
    
    parser.add_argument(
        '--add-scan-pattern',
        help='Add new file scan pattern'
    )
    
    parser.add_argument(
        '--add-exclude-pattern',
        help='Add new exclude pattern'
    )
    
    parser.add_argument(
        '--add-placeholder-pattern',
        nargs=2,
        metavar=('PATTERN', 'DESCRIPTION'),
        help='Add placeholder pattern: PATTERN DESCRIPTION'
    )
    
    parser.add_argument(
        '--list',
        action='store_true',
        help='List current configuration'
    )
    
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate current configuration'
    )
    
    parser.add_argument(
        '--export',
        help='Export configuration to file'
    )
    
    parser.add_argument(
        '--import',
        dest='import_file',
        help='Import configuration from file'
    )
    
    parser.add_argument(
        '--save',
        action='store_true',
        help='Save configuration to file'
    )
    
    args = parser.parse_args()
    
    # Initialize configuration manager
    config_mgr = FantasyElementsConfig(args.config_file)
    
    # Handle actions
    made_changes = False
    
    if args.add_term:
        category, term, suggestion = args.add_term
        if config_mgr.add_forbidden_term(category, term, suggestion=suggestion):
            made_changes = True
    
    if args.remove_term:
        if config_mgr.remove_forbidden_term(args.remove_term):
            made_changes = True
    
    if args.add_scan_pattern:
        if config_mgr.add_scan_pattern(args.add_scan_pattern):
            made_changes = True
    
    if args.add_exclude_pattern:
        if config_mgr.add_exclude_pattern(args.add_exclude_pattern):
            made_changes = True
    
    if args.add_placeholder_pattern:
        pattern, description = args.add_placeholder_pattern
        if config_mgr.add_placeholder_pattern(pattern, description):
            made_changes = True
    
    if args.import_file:
        if config_mgr.import_config(args.import_file):
            made_changes = True
    
    if args.export:
        config_mgr.export_config(args.export)
    
    if args.list:
        config_mgr.list_configuration()
    
    if args.validate:
        config_mgr.validate_config()
    
    if args.save or made_changes:
        config_mgr.save_config()
    
    if len(sys.argv) == 1:  # No arguments provided
        print("Fantasy Elements Validator Configuration Manager")
        print("Use --help for available options")
        print("Use --list to see current configuration")


if __name__ == "__main__":
    main()