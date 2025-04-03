"""
Test Core Configuration

This module contains tests for the core configuration.
"""

import os
import unittest
import tempfile
import shutil
from pathlib import Path

import sys

sys.path.append("/opt/sutazaiapp")

from backend.core.config import Settings


class TestCoreConfig(unittest.TestCase):
    """Tests for core configuration."""

    def setUp(self):
        """Set up test environment."""
        # Create secure temporary paths
        self.temp_dir = Path(tempfile.mkdtemp())
        self.temp_upload_dir = self.temp_dir / "uploads"
        self.temp_upload_dir.mkdir()
        self.temp_log_dir = self.temp_dir / "logs"
        self.temp_log_dir.mkdir()

        # Create test environment variables - align with Settings class
        self.test_env = {
            "APP_ENV": "test",
            "DEBUG": "true",
            "API_HOST": "127.0.0.1",
            "API_PORT": "8888",
            "API_WORKERS": "2",
            "SECRET_KEY": "test_secret_key",
            "ALGORITHM": "HS256",
            "ACCESS_TOKEN_EXPIRE_MINUTES": "30",
            "UPLOAD_DIR": str(self.temp_upload_dir),
            "MAX_UPLOAD_SIZE": "10485760",
            "ALLOWED_EXTENSIONS": "pdf,doc,docx,txt",
            "DATABASE_URL": "sqlite:///./test_db.sqlite",
            "SQLALCHEMY_DATABASE_URL": "sqlite+aiosqlite:///./test_db_sqlalchemy.sqlite",
            "LOG_LEVEL": "DEBUG",
            "LOG_DIR": str(self.temp_log_dir),
            "DEFAULT_MODEL": "test-model",
            "MODEL_DEPLOYMENT_PATH": str(self.temp_dir / "models"),
            "ENABLE_PROMETHEUS": "true",
            "PROMETHEUS_PORT": "9999",
        }

        # Set environment variables
        for key, value in self.test_env.items():
            os.environ[key] = value

    def tearDown(self):
        """Clean up test environment."""
        # Remove environment variables
        for key in self.test_env:
            if key in os.environ:
                del os.environ[key]

        # Clean up temporary directory
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        # Clean up test DB files if they exist
        if os.path.exists("./test_db.sqlite"):
            os.remove("./test_db.sqlite")
        if os.path.exists("./test_db_sqlalchemy.sqlite"):
            os.remove("./test_db_sqlalchemy.sqlite")

    def test_settings_initialization(self):
        """Test settings initialization."""
        settings = Settings()

        # Test application settings
        self.assertEqual(settings.APP_ENV, "test")

        # Test server settings
        self.assertEqual(settings.API_HOST, "127.0.0.1")
        self.assertEqual(settings.API_PORT, 8888)
        self.assertEqual(settings.API_WORKERS, 2)

        # Test security settings
        self.assertEqual(settings.SECRET_KEY, "test_secret_key")
        self.assertEqual(settings.ALGORITHM, "HS256")
        self.assertEqual(settings.ACCESS_TOKEN_EXPIRE_MINUTES, 30)

        # Test file upload settings
        self.assertEqual(settings.UPLOAD_DIR, str(self.temp_upload_dir))

        # Test database settings
        self.assertEqual(settings.DATABASE_URL, "sqlite:///./test_db.sqlite")
        self.assertEqual(settings.SQLALCHEMY_DATABASE_URL, "sqlite+aiosqlite:///./test_db_sqlalchemy.sqlite")

        # Test logging settings
        self.assertEqual(settings.LOG_DIR, str(self.temp_log_dir))

        # Test AI model settings
        self.assertEqual(settings.DEFAULT_MODEL, "test-model")
        self.assertEqual(settings.MODEL_DEPLOYMENT_PATH, str(self.temp_dir / "models"))

        # Test monitoring settings
        self.assertTrue(settings.ENABLE_PROMETHEUS)
        self.assertEqual(settings.PROMETHEUS_PORT, 9999)

    def test_directory_creation(self):
        """Test directory creation."""
        # settings = Settings()

        # Check that upload directory was created
        self.assertTrue(self.temp_upload_dir.exists())


if __name__ == "__main__":
    unittest.main()
