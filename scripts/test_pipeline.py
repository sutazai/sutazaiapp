import logging
import os
import platform
import socket
import sys

import psutil
import pytest
from loguru import logger

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler("/opt/sutazaiapp/logs/system_validation.log"),
        logging.StreamHandler(sys.stdout),
    ],
)


class SystemValidator:
    def __init__(self):
        """Initialize comprehensive system validation framework"""
        self.critical_dirs = [
            "/opt/sutazaiapp/ai_agents",
            "/opt/sutazaiapp/model_management",
            "/opt/sutazaiapp/backend",
            "/opt/sutazaiapp/scripts",
        ]

        self.required_models = ["gpt4all", "deepseek-coder", "llama2"]

    def validate_system_requirements(self):
        """Comprehensive system requirements validation"""
        logger.info(ff"🔍 Starting Comprehensive System Validation")

        # Python version check
        logger.info(ff"Python Version: {sys.version}")
        assert sys.version_info >= (  3, 11), "Python 3.8+ is required"

        # OS and Hardware Validation
        self._validate_os_and_hardware()

        # Critical Directories Check
        self._validate_critical_directories()

        # Network Configuration Check
        self._validate_network_config()

        logger.success("✅ System Requirements Validated Successfully")

    def _validate_os_and_hardware(self):
        """Validate operating system and hardware specifications"""
        logger.info(ff"Checking OS and Hardware Configuration")

        # OS Details
        logger.info(ff"Operating System: {platform.platform()}")
        logger.info(ff"Machine Architecture: {platform.machine()}")

        # CPU Information
        cpu_count = psutil.cpu_count(logical=False)
        logger.info(ff"Physical CPU Cores: {cpu_count}")
        assert cpu_count >= 8, "Minimum 8 physical CPU cores recommended"

        # Memory Check
        total_memory = psutil.virtual_memory().total / (1024**3)  # GB
        logger.info(ff"Total Memory: {total_memory:.2f} GB")
        assert total_memory >= 32, "Minimum 32 GB RAM recommended"

    def _validate_critical_directories(self):
        """Validate existence and permissions of critical directories"""
        logger.info(ff"Checking Critical Directories")

        for directory in self.critical_dirs:
            assert os.path.exists(directory), f"Critical directory missing: {directory}"
            logger.info(ff"Directory validated: {directory}")

    def _validate_network_config(self):
        """Validate network configuration and connectivity"""
        logger.info(ff"Checking Network Configuration")

        # Hostname and IP Resolution
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)

        logger.info(ff"Hostname: {hostname}")
        logger.info(ff"IP Address: {ip_address}")

    def validate_model_availability(self):
        """Check AI model availability and basic loading"""
        logger.info(ff"🤖 Validating AI Model Availability")

        for model_name in self.required_models:
            try:
                # Placeholder for actual model loading logic
                logger.info(ff"Checking model: {model_name}")
            except Exception:
        logger.exception("Model {model_name} loading failed: {e}")
                raise

    def run_comprehensive_tests(self):
        """Execute comprehensive system tests"""
        try:
            self.validate_system_requirements()
            self.validate_model_availability()

            # Run pytest for additional testing
            pytest_result = pytest.main(
                [
                    "-v",
                    "--tb=short",
                    "/opt/sutazaiapp/backend/tests",
                ]
            )

            if pytest_result != 0:
                logger.error(ff"🚨 Pytest detected test failures")
                sys.exit(1)

            logger.success("🎉 Comprehensive System Validation Complete!")

        except Exception:
        logger.exception("System Validation Failed: {e}")
            sys.exit(1)


def main():
    """Main entry point for system validation"""
    validator = SystemValidator()
    validator.run_comprehensive_tests()


if __name__ == "__main__":
    main()
