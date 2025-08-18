"""
Tests for optional feature flags and service abstractions
"""
import os
import pytest
from unittest.Mock import Mock, patch, AsyncMock
import sys
# Path handled by pytest configuration

from backend.app.core.config import Settings
from backend.app.services.code_completion.factory import code_completion_factory, reset_completion_client
from backend.app.services.code_completion.interfaces import CompletionRequest
from backend.app.services.training.factory import trainer_factory, reset_trainer
from backend.app.services.training.interfaces import TrainingConfig, TrainingStatus

class TestFeatureFlags:
    """Test feature flag configuration"""
    
    def test_default_flags_disabled(self):
        """Test that feature flags are disabled by default"""
        settings = Settings()
        assert settings.ENABLE_FSDP == False
        assert settings.ENABLE_TABBY == False
    
    def test_flags_from_environment(self):
        """Test loading feature flags from environment"""
        with patch.dict(os.environ, {
            'ENABLE_FSDP': 'true',
            'ENABLE_TABBY': 'true',
            'TABBY_URL': 'http://custom-tabby:9000',
            'TABBY_API_KEY': 'key'
        }):
            settings = Settings()
            assert settings.ENABLE_FSDP == True
            assert settings.ENABLE_TABBY == True
            assert settings.TABBY_URL == 'http://custom-tabby:9000'
            assert settings.TABBY_API_KEY == 'test-key'

class TestCodeCompletionFactory:
    """Test code completion factory and clients"""
    
    def setup_method(self):
        """Reset factory before each test"""
        reset_completion_client()
    
    def test_null_client_when_disabled(self):
        """Test that null client is returned when feature is disabled"""
        settings = Mock(spec=Settings)
        settings.ENABLE_TABBY = False
        
        client = code_completion_factory(settings)
        assert client.__class__.__name__ == 'NullCodeCompletionClient'
        assert client.is_available() == True
    
    def test_tabby_client_when_enabled(self):
        """Test that TabbyML client is returned when feature is enabled"""
        settings = Mock(spec=Settings)
        settings.ENABLE_TABBY = True
        settings.TABBY_URL = 'http://tabby:8080'
        settings.TABBY_API_KEY = os.getenv('TEST_API_KEY', 'test-api-key-placeholder')
        
        client = code_completion_factory(settings)
        assert client.__class__.__name__ == 'TabbyCodeCompletionClient'
    
    @pytest.mark.asyncio
    async def test_null_client_returns_disabled_message(self):
        """Test that null client returns appropriate disabled message"""
        settings = Mock(spec=Settings)
        settings.ENABLE_TABBY = False
        
        client = code_completion_factory(settings)
        request = CompletionRequest(code="def hello():")
        response = await client.complete(request)
        
        assert "disabled" in response.completion.lower()
        assert response.confidence == 0.0
        assert response.metadata['enabled'] == False
    
    @pytest.mark.asyncio
    async def test_null_client_health_check(self):
        """Test null client health check always returns True"""
        settings = Mock(spec=Settings)
        settings.ENABLE_TABBY = False
        
        client = code_completion_factory(settings)
        health = await client.health_check()
        assert health == True

class TestTrainingFactory:
    """Test training factory and trainers"""
    
    def setup_method(self):
        """Reset factory before each test"""
        reset_trainer()
    
    def test_default_trainer_when_disabled(self):
        """Test that default trainer is returned when FSDP is disabled"""
        settings = Mock(spec=Settings)
        settings.ENABLE_FSDP = False
        
        trainer = trainer_factory(settings)
        assert trainer.__class__.__name__ == 'DefaultTrainer'
        assert trainer.is_available() == True
    
    def test_fsdp_trainer_when_enabled(self):
        """Test that FSDP trainer is returned when feature is enabled"""
        settings = Mock(spec=Settings)
        settings.ENABLE_FSDP = True
        
        trainer = trainer_factory(settings)
        assert trainer.__class__.__name__ == 'FsdpTrainer'
    
    @pytest.mark.asyncio
    async def test_default_trainer_train(self):
        """Test default trainer can execute training"""
        settings = Mock(spec=Settings)
        settings.ENABLE_FSDP = False
        
        trainer = trainer_factory(settings)
        config = TrainingConfig(model_name="test-model")
        result = await trainer.train(config)
        
        assert result.job_id is not None
        assert result.status in [TrainingStatus.COMPLETED, TrainingStatus.RUNNING]
    
    @pytest.mark.asyncio
    async def test_default_trainer_health_check(self):
        """Test default trainer health check"""
        settings = Mock(spec=Settings)
        settings.ENABLE_FSDP = False
        
        trainer = trainer_factory(settings)
        health = await trainer.health_check()
        assert health == True

class TestFeatureEndpoint:
    """Test the /features API endpoint"""
    
    @pytest.mark.asyncio
    async def test_features_endpoint_response(self):
        """Test that features endpoint returns correct structure"""
        from backend.app.api.v1.features import get_feature_flags
        
        with patch('backend.app.api.v1.features.get_settings') as Mock_get_settings:
            settings = Mock(spec=Settings)
            settings.ENABLE_FSDP = True
            settings.ENABLE_TABBY = False
            settings.TABBY_URL = 'http://tabby:8080'
            settings.ENABLE_GPU = False
            settings.ENABLE_MONITORING = True
            Mock_get_settings.return_value = settings
            
            response = await get_feature_flags(settings)
            
            assert response['fsdp']['enabled'] == True
            assert response['tabby']['enabled'] == False
            assert response['tabby']['url'] is None  # URL should be None when disabled
            assert response['gpu']['enabled'] == False
            assert response['monitoring']['enabled'] == True

class TestOptionalImports:
    """Test that optional imports don't break when dependencies are missing"""
    
    def test_fsdp_trainer_handles_missing_torch(self):
        """Test FSDP trainer gracefully handles missing PyTorch"""
        with patch.dict(sys.modules, {'torch': None}):
            from backend.app.services.training.fsdp_trainer import FsdpTrainer
            trainer = FsdpTrainer()
            # Should not raise ImportError during initialization
            assert trainer is not None
    
    def test_default_trainer_handles_missing_torch(self):
        """Test default trainer works without PyTorch"""
        with patch.dict(sys.modules, {'torch': None, 'transformers': None}):
            from backend.app.services.training.default_trainer import DefaultTrainer
            trainer = DefaultTrainer()
            assert trainer is not None

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
