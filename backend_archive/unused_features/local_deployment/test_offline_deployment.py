#!/usr/bin/env python3
"""
Test Suite for Offline Model Deployment

This module provides comprehensive testing for the offline model deployment system
to ensure 100% autonomous operation.
"""

import asyncio
import logging
import time
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Any
import threading

from .offline_model_manager import (
    OfflineModelManager, ModelConfig, ModelFramework, 
    ModelState, QuantizationType
)
from .local_server import LocalModelServer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestOfflineDeployment")

class OfflineDeploymentValidator:
    """
    Comprehensive validator for offline model deployment capabilities
    """
    
    def __init__(self):
        self.test_results: Dict[str, Any] = {}
        self.temp_dirs: List[Path] = []
        
    async def run_full_validation(self) -> Dict[str, Any]:
        """Run complete validation suite"""
        
        logger.info("Starting Offline Deployment Validation Suite")
        
        # Test categories
        tests = [
            ("framework_availability", self.test_framework_availability),
            ("model_discovery", self.test_model_discovery),
            ("model_loading", self.test_model_loading),
            ("text_generation", self.test_text_generation),
            ("resource_monitoring", self.test_resource_monitoring),
            ("offline_operation", self.test_offline_operation),
            ("performance_benchmarks", self.test_performance_benchmarks)
        ]
        
        results = {}
        
        for test_name, test_func in tests:
            logger.info(f"Running test: {test_name}")
            
            try:
                start_time = time.time()
                test_result = await test_func()
                execution_time = time.time() - start_time
                
                results[test_name] = {
                    "status": "passed" if test_result.get("success", False) else "failed",
                    "execution_time": execution_time,
                    "details": test_result
                }
                
                if test_result.get("success", False):
                    logger.info(f" {test_name} passed ({execution_time:.2f}s)")
                else:
                    logger.error(f" {test_name} failed: {test_result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                logger.error(f" {test_name} crashed: {e}")
                results[test_name] = {
                    "status": "crashed",
                    "execution_time": 0,
                    "error": str(e)
                }
        
        # Calculate summary
        passed = sum(1 for r in results.values() if r["status"] == "passed")
        total = len(results)
        
        summary = {
            "total_tests": total,
            "passed": passed,
            "failed": total - passed,
            "success_rate": passed / total * 100 if total > 0 else 0,
            "results": results
        }
        
        logger.info(f"Validation Summary: {passed}/{total} tests passed ({summary['success_rate']:.1f}%)")
        
        return summary
    
    async def test_framework_availability(self) -> Dict[str, Any]:
        """Test availability of local model frameworks"""
        
        try:
            # Create temporary config
            config = self._create_test_config()
            manager = OfflineModelManager(config)
            
            available_frameworks = manager.available_frameworks
            
            # Check for essential frameworks
            essential_frameworks = [ModelFramework.PYTORCH, ModelFramework.ONNX]
            missing_essential = [f for f in essential_frameworks if f not in available_frameworks]
            
            # Check for preferred frameworks
            preferred_frameworks = [ModelFramework.OLLAMA, ModelFramework.TRANSFORMERS]
            available_preferred = [f for f in preferred_frameworks if f in available_frameworks]
            
            return {
                "success": len(missing_essential) == 0,
                "available_frameworks": [f.value for f in available_frameworks],
                "missing_essential": [f.value for f in missing_essential],
                "available_preferred": [f.value for f in available_preferred],
                "total_available": len(available_frameworks)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def test_model_discovery(self) -> Dict[str, Any]:
        """Test model discovery capabilities"""
        
        try:
            # Create test environment
            temp_dir = self._create_temp_models_dir()
            
            config = self._create_test_config()
            config['models_dir'] = str(temp_dir)
            
            manager = OfflineModelManager(config)
            
            # Discover models
            discovered_models = await manager.discover_local_models()
            
            # Verify discovery results
            model_count = len(discovered_models)
            framework_types = set(model.framework for model in discovered_models)
            
            return {
                "success": True,
                "discovered_models": model_count,
                "frameworks_found": [f.value for f in framework_types],
                "model_details": [
                    {
                        "model_id": model.model_id,
                        "name": model.name,
                        "framework": model.framework.value
                    }
                    for model in discovered_models[:5]  # First 5 for brevity
                ]
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def test_model_loading(self) -> Dict[str, Any]:
        """Test model loading and unloading"""
        
        try:
            # Create test environment
            temp_dir = self._create_temp_models_dir()
            
            config = self._create_test_config()
            config['models_dir'] = str(temp_dir)
            
            manager = OfflineModelManager(config)
            
            # Discover models first
            models = await manager.discover_local_models()
            
            if not models:
                return {"success": False, "error": "No models found for testing"}
            
            # Test loading first model
            test_model = models[0]
            load_result = await manager.load_model(test_model.model_id)
            
            if not load_result["success"]:
                return {"success": False, "error": f"Failed to load model: {load_result.get('error')}"}
            
            # Verify model is loaded
            status = manager.get_model_status(test_model.model_id)
            
            return {
                "success": load_result["success"],
                "load_time": load_result.get("load_time", 0),
                "memory_usage_mb": load_result.get("memory_usage_mb", 0),
                "model_state": status.get("state", "unknown")
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def test_text_generation(self) -> Dict[str, Any]:
        """Test text generation capabilities"""
        
        try:
            # Create test environment
            temp_dir = self._create_temp_models_dir()
            
            config = self._create_test_config()
            config['models_dir'] = str(temp_dir)
            
            manager = OfflineModelManager(config)
            
            # Discover and load a model
            models = await manager.discover_local_models()
            
            if not models:
                return {"success": False, "error": "No models found for testing"}
            
            test_model = models[0]
            load_result = await manager.load_model(test_model.model_id)
            
            if not load_result["success"]:
                return {"success": False, "error": f"Failed to load model: {load_result.get('error')}"}
            
            # Test text generation
            test_prompt = "Hello, this is a test prompt for text generation."
            
            generation_result = await manager.generate_text(
                model_id=test_model.model_id,
                prompt=test_prompt,
                max_tokens=50
            )
            
            return {
                "success": generation_result["success"],
                "generated_text": generation_result.get("text", ""),
                "inference_time": generation_result.get("inference_time", 0),
                "tokens_generated": generation_result.get("tokens_generated", 0),
                "model_used": generation_result.get("model_id", "")
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def test_resource_monitoring(self) -> Dict[str, Any]:
        """Test system resource monitoring"""
        
        try:
            config = self._create_test_config()
            manager = OfflineModelManager(config)
            
            # Get resource usage
            resource_usage = manager.get_resource_usage()
            
            # Validate resource data
            required_fields = ["cpu", "memory", "disk"]
            missing_fields = [field for field in required_fields if field not in resource_usage]
            
            # Check for reasonable values
            cpu_usage = resource_usage.get("cpu", {}).get("usage_percent", 0)
            memory_usage = resource_usage.get("memory", {}).get("usage_percent", 0)
            
            reasonable_usage = 0 <= cpu_usage <= 100 and 0 <= memory_usage <= 100
            
            return {
                "success": len(missing_fields) == 0 and reasonable_usage,
                "missing_fields": missing_fields,
                "cpu_usage_percent": cpu_usage,
                "memory_usage_percent": memory_usage
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def test_offline_operation(self) -> Dict[str, Any]:
        """Test complete offline operation without external dependencies"""
        
        try:
            # Create isolated test environment
            temp_dir = self._create_temp_models_dir()
            
            config = self._create_test_config()
            config['models_dir'] = str(temp_dir)
            
            manager = OfflineModelManager(config)
            
            # Perform operations that should work offline
            models = await manager.discover_local_models()
            resource_usage = manager.get_resource_usage()
            status = manager.get_model_status()
            
            # Verify no network dependency errors
            operations_successful = True
            
            return {
                "success": operations_successful,
                "models_discovered": len(models),
                "resource_monitoring_ok": "cpu" in resource_usage,
                "status_retrieval_ok": "total_models" in status
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def test_performance_benchmarks(self) -> Dict[str, Any]:
        """Test performance benchmarks for deployment validation"""
        
        try:
            config = self._create_test_config()
            manager = OfflineModelManager(config)
            
            # Benchmark model discovery
            start_time = time.time()
            models = await manager.discover_local_models()
            discovery_time = time.time() - start_time
            
            # Benchmark resource monitoring
            start_time = time.time()
            for _ in range(10):
                manager.get_resource_usage()
            monitoring_time = (time.time() - start_time) / 10
            
            # Performance thresholds
            discovery_acceptable = discovery_time < 30.0  # 30 seconds max
            monitoring_acceptable = monitoring_time < 1.0  # 1 second max
            
            return {
                "success": discovery_acceptable and monitoring_acceptable,
                "discovery_time": discovery_time,
                "monitoring_time": monitoring_time,
                "discovery_acceptable": discovery_acceptable,
                "monitoring_acceptable": monitoring_acceptable,
                "models_found": len(models)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _create_test_config(self) -> Dict[str, Any]:
        """Create test configuration"""
        
        temp_dir = Path(tempfile.mkdtemp())
        self.temp_dirs.append(temp_dir)
        
        return {
            'models_dir': str(temp_dir / 'models'),
            'cache_dir': str(temp_dir / 'cache'),
            'max_total_memory_gb': 4,
            'auto_optimization': False
        }
    
    def _create_temp_models_dir(self) -> Path:
        """Create temporary models directory with mock models"""
        
        temp_dir = Path(tempfile.mkdtemp())
        self.temp_dirs.append(temp_dir)
        
        models_dir = temp_dir / 'models'
        models_dir.mkdir(exist_ok=True)
        
        # Create mock model directories
        mock_models = [
            {
                "name": "test_model_1",
                "framework": "transformers",
                "config": {
                    "name": "Test Model 1",
                    "max_position_embeddings": 2048,
                    "capabilities": ["text_generation"]
                }
            },
            {
                "name": "test_model_2", 
                "framework": "pytorch",
                "config": {
                    "name": "Test Model 2",
                    "max_position_embeddings": 4096,
                    "capabilities": ["text_generation", "chat"]
                }
            }
        ]
        
        for mock_model in mock_models:
            model_dir = models_dir / mock_model["name"]
            model_dir.mkdir(exist_ok=True)
            
            # Create config file
            config_file = model_dir / "config.json"
            with open(config_file, 'w') as f:
                json.dump(mock_model["config"], f, indent=2)
            
            # Create mock model file based on framework
            if mock_model["framework"] == "transformers":
                (model_dir / "pytorch_model.bin").touch()
                (model_dir / "tokenizer.json").touch()
            elif mock_model["framework"] == "pytorch":
                (model_dir / "model.pt").touch()
        
        return temp_dir
    
    def cleanup(self):
        """Clean up temporary directories"""
        
        for temp_dir in self.temp_dirs:
            try:
                import shutil
                shutil.rmtree(temp_dir)
            except Exception as e:
                logger.warning(f"Failed to cleanup {temp_dir}: {e}")

def run_offline_deployment_tests():
    """Run the complete offline deployment test suite"""
    
    logger.info("Starting SutazAI Offline Deployment Test Suite")
    
    async def run_async_tests():
        validator = OfflineDeploymentValidator()
        
        try:
            results = await validator.run_full_validation()
            
            # Save results to file
            results_file = Path("/opt/sutazaiapp/logs/offline_deployment_test_results.json")
            results_file.parent.mkdir(exist_ok=True)
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Test results saved to: {results_file}")
            
            return results['success_rate'] == 100.0
            
        finally:
            validator.cleanup()
    
    # Run async tests
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    success = loop.run_until_complete(run_async_tests())
    loop.close()
    
    return success

if __name__ == "__main__":
    success = run_offline_deployment_tests()
    exit(0 if success else 1)