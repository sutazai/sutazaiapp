#!/usr/bin/env python3
"""
Tests for optional feature flags functionality
"""

import os
import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
# Path handled by pytest configuration

from app.core.config import Settings, get_settings
from app.services.code_completion.factory import code_completion_factory, reset_completion_client
from app.services.code_completion.null_client import NullCodeCompletionClient
from app.services.code_completion.tabby_client import TabbyCodeCompletionClient
from app.services.training.factory import trainer_factory, reset_trainer
from app.services.training.default_trainer import DefaultTrainer
from app.services.training.fsdp_trainer import FsdpTrainer


class TestFeatureFlags:
    """Test feature flag configuration"""
    
    def test_default_feature_flags(self):
        """Test that feature flags are disabled by default"""
        settings = Settings()
        assert settings.ENABLE_FSDP is False
        assert settings.ENABLE_TABBY is False
        assert settings.ENABLE_GPU is False
    
    def test_enable_fsdp_via_env(self):
        """Test enabling FSDP via environment variable"""
        with patch.dict(os.environ, {'ENABLE_FSDP': 'true'}):
            settings = Settings()
            assert settings.ENABLE_FSDP is True
    
    def test_enable_tabby_via_env(self):
        """Test enabling TabbyML via environment variable"""
        with patch.dict(os.environ, {'ENABLE_TABBY': 'true', 'TABBY_API_KEY': 'key'}):
            settings = Settings()
            assert settings.ENABLE_TABBY is True
            assert settings.TABBY_API_KEY == 'key'


class TestCodeCompletionFactory:
    """Test code completion service factory"""
    
    def setup_method(self):
        """Reset cached client before each test"""
        reset_completion_client()
    
    def test_null_client_when_disabled(self):
        """Test that NullCodeCompletionClient is used when TabbyML is disabled"""
        settings = Settings(ENABLE_TABBY=False)
        client = code_completion_factory(settings)
        assert isinstance(client, NullCodeCompletionClient)
    
    @patch('app.services.code_completion.factory.TabbyCodeCompletionClient')
    def test_tabby_client_when_enabled(self, mock_tabby_class):
        """Test that TabbyCodeCompletionClient is used when TabbyML is enabled"""
        mock_client = Mock()
        mock_tabby_class.return_value = mock_client
        
        settings = Settings(
            ENABLE_TABBY=True,
            TABBY_URL="http://test:8080",
            TABBY_API_KEY="key"
        )
        
        client = code_completion_factory(settings)
        
        mock_tabby_class.assert_called_once_with(
            base_url="http://test:8080",
            api_key="key"
        )
        assert client == mock_client
    
    def test_null_client_methods(self):
        """Test NullCodeCompletionClient methods return expected values"""
        settings = Settings(ENABLE_TABBY=False)
        client = code_completion_factory(settings)
        
        # Test complete method
        result = client.complete("test code", "python")
        assert result == ""
        
        # Test is_available method
        assert client.is_available() is False


class TestTrainingFactory:
    """Test training service factory"""
    
    def setup_method(self):
        """Reset cached trainer before each test"""
        reset_trainer()
    
    def test_default_trainer_when_disabled(self):
        """Test that DefaultTrainer is used when FSDP is disabled"""
        settings = Settings(ENABLE_FSDP=False)
        trainer = trainer_factory(settings)
        assert isinstance(trainer, DefaultTrainer)
    
    @patch('app.services.training.factory.FsdpTrainer')
    def test_fsdp_trainer_when_enabled(self, mock_fsdp_class):
        """Test that FsdpTrainer is used when FSDP is enabled"""
        mock_trainer = Mock()
        mock_fsdp_class.return_value = mock_trainer
        
        settings = Settings(ENABLE_FSDP=True)
        trainer = trainer_factory(settings)
        
        mock_fsdp_class.assert_called_once()
        assert trainer == mock_trainer
    
    def test_default_trainer_methods(self):
        """Test DefaultTrainer methods"""
        settings = Settings(ENABLE_FSDP=False)
        trainer = trainer_factory(settings)
        
        # Test train method
        model = Mock()
        data = Mock()
        result = trainer.train(model, data)
        assert result == model  # DefaultTrainer returns model unchanged
        
        # Test is_distributed method
        assert trainer.is_distributed() is False


class TestFeaturesEndpoint:
    """Test the /api/v1/features endpoint"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        from fastapi.testclient import TestClient
        from app.main import app
        return TestClient(app)
    
    def test_features_endpoint_default(self, client):
        """Test features endpoint with default settings"""
        response = client.get("/api/v1/features/")
        assert response.status_code == 200
        
        data = response.json()
        assert "features" in data
        assert "metadata" in data
        
        # Check default feature states
        assert data["features"]["fsdp"] is False
        assert data["features"]["tabby"] is False
        assert data["features"]["gpu"] is False
        assert data["features"]["monitoring"] is True
    
    @patch.dict(os.environ, {'ENABLE_FSDP': 'true', 'ENABLE_TABBY': 'true'})
    def test_features_endpoint_enabled(self, client):
        """Test features endpoint with features enabled"""
        # Force settings reload
        from app.core import config
        config._settings = None
        
        response = client.get("/api/v1/features/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["features"]["fsdp"] is True
        assert data["features"]["tabby"] is True
    
    def test_features_endpoint_metadata(self, client):
        """Test features endpoint metadata"""
        response = client.get("/api/v1/features/")
        assert response.status_code == 200
        
        data = response.json()
        assert "metadata" in data
        assert "environment" in data["metadata"]
        
        # When TabbyML is disabled, tabby_url should be empty
        assert data["metadata"]["tabby_url"] == ""


class TestOptionalDependencies:
    """Test optional dependency handling"""
    
    def test_fsdp_imports_optional(self):
        """Test that FSDP imports are optional"""
        # This should not raise ImportError even if fms-fsdp is not installed
        try:
            from app.services.training.fsdp_trainer import FsdpTrainer
            # If we get here, the module exists
            assert True
        except ImportError as e:
            # If fms-fsdp is not installed, we should get a graceful failure
            assert "fms-fsdp" in str(e) or "fms_fsdp" in str(e)
    
    def test_tabby_imports_optional(self):
        """Test that TabbyML imports are optional"""
        try:
            from app.services.code_completion.tabby_client import TabbyCodeCompletionClient
            # If we get here, the module exists
            assert True
        except ImportError as e:
            # If tabby-client is not installed, we should get a graceful failure
            assert "tabby" in str(e).lower()


class TestDockerComposeProfiles:
    """Test Docker Compose profile configuration"""
    
    def test_fsdp_profile_exists(self):
        """Test that FSDP service has the correct profile"""
        import yaml
        
        with open('/opt/sutazaiapp/docker-compose.yml', 'r') as f:
            compose = yaml.safe_load(f)
        
        # Check if fms-fsdp service exists and has profile
        if 'fms-fsdp' in compose['services']:
            service = compose['services']['fms-fsdp']
            assert 'profiles' in service
            assert 'fsdp' in service['profiles']
    
    def test_tabby_profile_exists(self):
        """Test that TabbyML service has the correct profile"""
        import yaml
        
        with open('/opt/sutazaiapp/docker-compose.yml', 'r') as f:
            compose = yaml.safe_load(f)
        
        # Check if tabbyml service exists and has profile
        if 'tabbyml' in compose['services']:
            service = compose['services']['tabbyml']
            assert 'profiles' in service
            assert 'tabby' in service['profiles']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
