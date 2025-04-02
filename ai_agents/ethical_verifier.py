#!/usr/bin/env python3
"""
Ethical Verifier Module

This module provides ethical verification of content and actions to ensure
they comply with defined constraints and policies.
"""

import os
import json
import logging
import re
from typing import Dict, List, Any
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/ethical_verifier.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("EthicalVerifier")


class EthicalVerifier:
    """
    Enforces ethical constraints on content and actions.

    This class checks text content and proposed actions against
    defined policies and constraints to ensure they comply with
    ethical guidelines.
    """

    def __init__(self, config_path: str = "config/ethics.json"):
        """
        Initialize the ethical verifier.

        Args:
            config_path: Path to the configuration file
        """
        self.config_path = config_path

        # Create default config if it doesn't exist
        if not os.path.exists(config_path):
            self._create_default_config()

        # Load configuration
        self.load_config()

        logger.info(
            f"Ethical verifier initialized with {len(self.banned_patterns)} banned patterns"
        )

    def _create_default_config(self):
        """Create a default configuration file if none exists"""
        default_config = {
            "banned_patterns": [
                r"(?i)hack\s+into",
                r"(?i)steal\s+\w+\s+data",
                r"(?i)bypass\s+security",
                r"(?i)(delete|corrupt)\s+all\s+\w+",
                r"(?i)distributed\s+denial\s+of\s+service",
                r"(?i)ransomware",
                r"(?i)exploit\s+vulnerability",
                r"(?i)illegal\s+access",
            ],
            "banned_file_paths": [
                "/etc/passwd",
                "/etc/shadow",
                "/boot/",
                "/etc/sudoers",
                "/var/www/",
                "/.ssh/",
                "~/.*_history",
                "~/.aws/credentials",
            ],
            "banned_actions": [
                "delete_system_files",
                "format_disk",
                "modify_network_config",
                "install_malware",
                "disable_firewall",
                "create_backdoor",
                "excessive_resource_usage",
            ],
            "allowed_network_domains": ["localhost", "127.0.0.1"],
            "max_resource_usage": {
                "cpu_percent": 80,
                "memory_gb": 16,
                "disk_gb": 10,
                "max_file_size_mb": 1000,
            },
            "content_policies": {
                "allow_profanity": False,
                "allow_political_content": False,
                "allow_medical_advice": False,
            },
        }

        # Ensure directory exists
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)

        # Write default config
        with open(self.config_path, "w") as f:
            json.dump(default_config, f, indent=2)

        logger.info(f"Created default ethical configuration at {self.config_path}")

    def load_config(self):
        """Load configuration from file"""
        try:
            with open(self.config_path, "r") as f:
                config = json.load(f)

                # Load configuration values
                self.banned_patterns = config.get("banned_patterns", [])
                self.banned_file_paths = config.get("banned_file_paths", [])
                self.banned_actions = config.get("banned_actions", [])
                self.allowed_network_domains = config.get("allowed_network_domains", [])
                self.max_resource_usage = config.get("max_resource_usage", {})
                self.content_policies = config.get("content_policies", {})

                # Compile patterns for efficiency
                self.compiled_patterns = [
                    re.compile(pattern) for pattern in self.banned_patterns
                ]

                logger.info(f"Loaded ethical configuration from {self.config_path}")
        except Exception as e:
            logger.error(f"Error loading ethical configuration: {str(e)}")
            raise

    def verify_content(self, content: str) -> Dict[str, Any]:
        """
        Verify text content against ethical constraints.

        Args:
            content: The text content to verify

        Returns:
            Dictionary with results of verification
        """
        if not content:
            return {"allowed": True, "message": "Empty content"}

        # Check against banned patterns
        for i, pattern in enumerate(self.compiled_patterns):
            if pattern.search(content):
                reason = f"Content contains banned pattern: {self.banned_patterns[i]}"
                logger.warning(f"Content verification failed: {reason}")
                return {
                    "allowed": False,
                    "message": "I cannot assist with that request as it appears to violate ethical guidelines. Please ensure your request doesn't involve security breaches, illegal activities, or harmful actions.",
                }

        # Check content policies
        if not self.content_policies.get("allow_profanity", False):
            # Simple profanity check
            profanity_patterns = [
                r"(?i)\b(f[*u]ck|sh[*i]t|b[*i]tch|a[*s]shole|c[*u]nt)\b"
            ]
            for pattern in profanity_patterns:
                if re.search(pattern, content):
                    logger.warning("Content verification failed: Contains profanity")
                    return {
                        "allowed": False,
                        "message": "I cannot process content containing profanity. Please rephrase your request without such language.",
                    }

        if not self.content_policies.get("allow_medical_advice", False):
            # Check for medical advice patterns
            medical_patterns = [
                r"(?i)how (to treat|to cure|to heal|do I treat|should I treat)",
                r"(?i)(diagnosis|diagnose) (my|his|her|their)",
                r"(?i)what (medication|drug|prescription|dosage)",
                r"(?i)(stop|discontinue) (taking|using) (medication|drug|prescription)",
            ]
            for pattern in medical_patterns:
                if re.search(pattern, content):
                    logger.warning(
                        "Content verification failed: Contains request for medical advice"
                    )
                    return {
                        "allowed": False,
                        "message": "I cannot provide medical advice, diagnosis, or treatment recommendations. Please consult with a qualified healthcare professional for medical concerns.",
                    }

        # All checks passed
        return {"allowed": True, "message": "Content verification passed"}

    def verify_action(
        self, action: str, parameters: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Verify a proposed action against ethical constraints.

        Args:
            action: The action to verify
            parameters: Parameters for the action

        Returns:
            Dictionary with results of verification
        """
        parameters = parameters or {}

        # Check against banned actions
        if action in self.banned_actions:
            reason = f"Action is banned: {action}"
            logger.warning(f"Action verification failed: {reason}")
            return {
                "allowed": False,
                "message": f"Action '{action}' is not permitted as it violates security policy.",
            }

        # Check file operations
        if action in ["read_file", "write_file", "delete_file", "execute_file"]:
            file_path = parameters.get("file_path", "")

            # Check against banned file paths
            for banned_path in self.banned_file_paths:
                if banned_path.endswith("/"):
                    # Directory path - check if file_path starts with it
                    if file_path.startswith(banned_path):
                        reason = f"File path not allowed: {file_path}"
                        logger.warning(f"Action verification failed: {reason}")
                        return {
                            "allowed": False,
                            "message": f"Access to files in '{banned_path}' is not permitted.",
                        }
                else:
                    # Exact file path or pattern - check exact match or regex
                    if banned_path == file_path or re.match(banned_path, file_path):
                        reason = f"File path not allowed: {file_path}"
                        logger.warning(f"Action verification failed: {reason}")
                        return {
                            "allowed": False,
                            "message": f"Access to '{file_path}' is not permitted.",
                        }

            # Additional checks for write_file
            if action == "write_file":
                content = parameters.get("content", "")

                # Verify content
                content_check = self.verify_content(content)
                if not content_check["allowed"]:
                    return content_check

                # Check file size
                max_file_size = (
                    self.max_resource_usage.get("max_file_size_mb", 1000) * 1024 * 1024
                )  # Convert to bytes
                if len(content) > max_file_size:
                    reason = f"File size too large: {len(content)} bytes (max: {max_file_size} bytes)"
                    logger.warning(f"Action verification failed: {reason}")
                    return {
                        "allowed": False,
                        "message": f"File size exceeds the maximum allowed size of {max_file_size / (1024 * 1024):.1f} MB.",
                    }

        # Check network operations
        if action in ["http_request", "connect_to", "download_file"]:
            url = parameters.get("url", "")
            host = None

            # Extract hostname from URL
            try:
                parsed_url = urlparse(url)
                host = parsed_url.netloc
            except Exception as e:
                logger.error(f"Error parsing URL: {url}. Error: {e}")
                host = url

            # Check against allowed domains
            if host and not any(
                host.endswith(domain) for domain in self.allowed_network_domains
            ):
                reason = f"Network host not allowed: {host}"
                logger.warning(f"Action verification failed: {reason}")
                return {
                    "allowed": False,
                    "message": f"Network access to {host} is not permitted.",
                }

        # Check system operations
        if action in ["execute_command", "execute_code"]:
            command = parameters.get("command", "") or parameters.get("code", "")

            # Verify command/code content
            content_check = self.verify_content(command)
            if not content_check["allowed"]:
                return content_check

            # Check for dangerous shell commands
            dangerous_commands = [
                r"rm\s+-rf",
                r"mkfs",
                r"dd\s+if=",
                r"sudo",
                r"su\s+-",
                r"/dev/sda",
                r"chmod\s+777",
                r"wget.*\|\s*bash",
                r"curl.*\|\s*bash",
            ]

            for pattern in dangerous_commands:
                if re.search(pattern, command):
                    reason = f"Command contains dangerous pattern: {pattern}"
                    logger.warning(f"Action verification failed: {reason}")
                    return {
                        "allowed": False,
                        "message": "This command contains potentially harmful operations and is not permitted.",
                    }

        # Check resource usage
        if action in ["allocate_memory", "use_cpu", "disk_operation"]:
            # Memory check
            if "memory_gb" in parameters:
                requested_memory = parameters["memory_gb"]
                max_memory = self.max_resource_usage.get("memory_gb", 16)
                if requested_memory > max_memory:
                    reason = f"Memory usage exceeds limit: {requested_memory}GB (max: {max_memory}GB)"
                    logger.warning(f"Action verification failed: {reason}")
                    return {
                        "allowed": False,
                        "message": f"Memory allocation of {requested_memory}GB exceeds the maximum allowed of {max_memory}GB.",
                    }

            # CPU check
            if "cpu_percent" in parameters:
                requested_cpu = parameters["cpu_percent"]
                max_cpu = self.max_resource_usage.get("cpu_percent", 80)
                if requested_cpu > max_cpu:
                    reason = (
                        f"CPU usage exceeds limit: {requested_cpu}% (max: {max_cpu}%)"
                    )
                    logger.warning(f"Action verification failed: {reason}")
                    return {
                        "allowed": False,
                        "message": f"CPU usage of {requested_cpu}% exceeds the maximum allowed of {max_cpu}%.",
                    }

            # Disk check
            if "disk_gb" in parameters:
                requested_disk = parameters["disk_gb"]
                max_disk = self.max_resource_usage.get("disk_gb", 10)
                if requested_disk > max_disk:
                    reason = f"Disk usage exceeds limit: {requested_disk}GB (max: {max_disk}GB)"
                    logger.warning(f"Action verification failed: {reason}")
                    return {
                        "allowed": False,
                        "message": f"Disk usage of {requested_disk}GB exceeds the maximum allowed of {max_disk}GB.",
                    }

        # All checks passed
        return {"allowed": True, "message": "Action verification passed"}

    def validate_plan(self, plan: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate a multi-step plan against ethical constraints.

        Args:
            plan: List of action dictionaries

        Returns:
            Dictionary with results of verification
        """
        for i, step in enumerate(plan):
            action = step.get("action", "")
            parameters = step.get("parameters", {})

            # Verify step
            result = self.verify_action(action, parameters)
            if not result["allowed"]:
                return {
                    "allowed": False,
                    "step": i,
                    "message": f"Step {i} failed verification: {result['message']}",
                }

        # All steps passed
        return {"allowed": True, "message": "Plan verification passed"}

    def add_banned_pattern(self, pattern: str) -> bool:
        """
        Add a new banned pattern to the configuration.

        Args:
            pattern: Regular expression pattern to ban

        Returns:
            Boolean indicating success
        """
        try:
            # Test the pattern
            re.compile(pattern)

            # Add to config
            self.banned_patterns.append(pattern)
            self.compiled_patterns.append(re.compile(pattern))

            # Save config
            with open(self.config_path, "r") as f:
                config = json.load(f)
                config["banned_patterns"] = self.banned_patterns

            with open(self.config_path, "w") as f:
                json.dump(config, f, indent=2)

            logger.info(f"Added banned pattern: {pattern}")
            return True

        except Exception as e:
            logger.error(f"Error adding banned pattern: {str(e)}")
            return False

    def remove_banned_pattern(self, pattern: str) -> bool:
        """
        Remove a banned pattern from the configuration.

        Args:
            pattern: Pattern to remove

        Returns:
            Boolean indicating success
        """
        try:
            if pattern in self.banned_patterns:
                index = self.banned_patterns.index(pattern)
                self.banned_patterns.remove(pattern)
                self.compiled_patterns.pop(index)

                # Save config
                with open(self.config_path, "r") as f:
                    config = json.load(f)
                    config["banned_patterns"] = self.banned_patterns

                with open(self.config_path, "w") as f:
                    json.dump(config, f, indent=2)

                logger.info(f"Removed banned pattern: {pattern}")
                return True
            else:
                logger.warning(f"Pattern not found: {pattern}")
                return False

        except Exception as e:
            logger.error(f"Error removing banned pattern: {str(e)}")
            return False
