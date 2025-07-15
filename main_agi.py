"""
SutazAI - Main AGI/ASI Application
Enterprise-grade orchestration and deployment of the complete AGI system

This is the main entry point for the SutazAI AGI/ASI system, integrating all components
including neural networks, code generation, knowledge management, and local models.
"""

import asyncio
import json
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import uvicorn
from fastapi import FastAPI
from contextlib import asynccontextmanager

# Import core AGI components
from core.agi_system import get_agi_system, AGITask, TaskPriority, create_agi_task
from api.agi_api import get_api_app
from models.local_model_manager import get_model_manager
from deployment.docker_deployment import create_deployment_manager, DeploymentEnvironment
from deployment.kubernetes_deployment import create_kubernetes_manager, KubernetesEnvironment

# Import enhanced enterprise components
from core.security import SecurityManager
from core.exceptions import SutazaiException
from database.manager import DatabaseManager
from performance.profiler import PerformanceProfiler
from config.settings import Settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/opt/sutazaiapp/logs/agi_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SutazAIOrchestrator:
    """
    Main orchestrator for the SutazAI AGI/ASI system
    
    This class coordinates all system components and provides a unified interface
    for system management, deployment, and operation.
    """
    
    def __init__(self):
        self.base_dir = Path("/opt/sutazaiapp")
        self.logs_dir = self.base_dir / "logs"
        self.logs_dir.mkdir(exist_ok=True)
        
        # System components
        self.agi_system = None
        self.model_manager = None
        self.security_manager = None
        self.db_manager = None
        self.performance_profiler = None
        self.settings = None
        
        # Deployment managers
        self.docker_manager = None
        self.k8s_manager = None
        
        # FastAPI app
        self.app = None
        
        # System state
        self.is_running = False
        self.initialization_complete = False
        
        logger.info("SutazAI Orchestrator initialized")
    
    async def initialize_system(self):
        """Initialize all system components"""
        try:
            logger.info("=== Initializing SutazAI AGI/ASI System ===")
            
            # Load configuration
            await self._load_configuration()
            
            # Initialize enterprise components
            await self._initialize_enterprise_components()
            
            # Initialize AGI system
            await self._initialize_agi_system()
            
            # Initialize model manager
            await self._initialize_model_manager()
            
            # Initialize deployment managers
            await self._initialize_deployment_managers()
            
            # Initialize API application
            await self._initialize_api_app()
            
            # Setup signal handlers
            self._setup_signal_handlers()
            
            # Run system validation
            await self._validate_system()
            
            self.initialization_complete = True
            logger.info("=== SutazAI System Initialization Complete ===")
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            raise SutazaiException(f"System initialization failed: {e}")
    
    async def _load_configuration(self):
        """Load system configuration"""
        try:
            logger.info("Loading system configuration...")
            self.settings = Settings()
            logger.info("Configuration loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    async def _initialize_enterprise_components(self):
        """Initialize enterprise-grade components"""
        try:
            logger.info("Initializing enterprise components...")
            
            # Initialize security manager
            self.security_manager = SecurityManager()
            logger.info("Security manager initialized")
            
            # Initialize database manager
            self.db_manager = DatabaseManager()
            logger.info("Database manager initialized")
            
            # Initialize performance profiler
            self.performance_profiler = PerformanceProfiler()
            logger.info("Performance profiler initialized")
            
            logger.info("Enterprise components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize enterprise components: {e}")
            raise
    
    async def _initialize_agi_system(self):
        """Initialize the core AGI system"""
        try:
            logger.info("Initializing AGI system...")
            self.agi_system = get_agi_system()
            
            # Submit a test task to verify system operation
            test_task = create_agi_task(
                name="system_test",
                priority=TaskPriority.HIGH,
                data={"test_type": "initialization"}
            )
            
            task_id = self.agi_system.submit_task(test_task)
            logger.info(f"AGI system test task submitted: {task_id}")
            
            # Wait for task completion
            await asyncio.sleep(2)
            
            logger.info("AGI system initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize AGI system: {e}")
            raise
    
    async def _initialize_model_manager(self):
        """Initialize the local model manager"""
        try:
            logger.info("Initializing local model manager...")
            self.model_manager = get_model_manager()
            
            # Install and load default models
            await self._setup_default_models()
            
            logger.info("Local model manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize model manager: {e}")
            raise
    
    async def _setup_default_models(self):
        """Setup default models for immediate use"""
        try:
            logger.info("Setting up default models...")
            
            # Define essential models
            essential_models = ["llama3.1", "codellama"]
            
            for model_name in essential_models:
                try:
                    # Check if model is available
                    model_info = self.model_manager.get_model_info(model_name)
                    
                    if model_info["status"] == "available":
                        logger.info(f"Installing model: {model_name}")
                        await self.model_manager.install_model(model_name)
                        
                        logger.info(f"Loading model: {model_name}")
                        await self.model_manager.load_model(model_name)
                        
                        logger.info(f"Model {model_name} ready for use")
                        
                except Exception as e:
                    logger.warning(f"Failed to setup model {model_name}: {e}")
                    continue
            
            logger.info("Default models setup complete")
            
        except Exception as e:
            logger.error(f"Failed to setup default models: {e}")
            # Don't raise - system can continue without all models
    
    async def _initialize_deployment_managers(self):
        """Initialize deployment managers"""
        try:
            logger.info("Initializing deployment managers...")
            
            # Initialize Docker deployment manager
            self.docker_manager = create_deployment_manager()
            logger.info("Docker deployment manager initialized")
            
            # Initialize Kubernetes deployment manager
            try:
                self.k8s_manager = create_kubernetes_manager()
                logger.info("Kubernetes deployment manager initialized")
            except Exception as e:
                logger.warning(f"Kubernetes not available: {e}")
                self.k8s_manager = None
            
            logger.info("Deployment managers initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize deployment managers: {e}")
            raise
    
    async def _initialize_api_app(self):
        """Initialize the FastAPI application"""
        try:
            logger.info("Initializing API application...")
            
            # Get the FastAPI app from the API module
            self.app = get_api_app()
            
            # Add additional routes for orchestrator
            self._add_orchestrator_routes()
            
            logger.info("API application initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize API application: {e}")
            raise
    
    def _add_orchestrator_routes(self):
        """Add orchestrator-specific routes to the FastAPI app"""
        
        @self.app.get("/orchestrator/status")
        async def get_orchestrator_status():
            """Get orchestrator system status"""
            try:
                return {
                    "orchestrator": {
                        "status": "running" if self.is_running else "stopped",
                        "initialization_complete": self.initialization_complete,
                        "uptime": time.time() - self.start_time if hasattr(self, 'start_time') else 0
                    },
                    "agi_system": self.agi_system.get_system_status() if self.agi_system else None,
                    "model_manager": self.model_manager.get_system_status() if self.model_manager else None,
                    "components": {
                        "security_manager": "active" if self.security_manager else "inactive",
                        "db_manager": "active" if self.db_manager else "inactive",
                        "performance_profiler": "active" if self.performance_profiler else "inactive",
                        "docker_manager": "active" if self.docker_manager else "inactive",
                        "k8s_manager": "active" if self.k8s_manager else "inactive"
                    }
                }
            except Exception as e:
                logger.error(f"Failed to get orchestrator status: {e}")
                return {"error": str(e)}
        
        @self.app.post("/orchestrator/deploy")
        async def deploy_system(environment: str = "development", platform: str = "docker"):
            """Deploy the system to specified environment"""
            try:
                if platform == "docker":
                    if not self.docker_manager:
                        raise Exception("Docker manager not available")
                    
                    env = DeploymentEnvironment(environment)
                    result = self.docker_manager.deploy(env)
                    return result
                    
                elif platform == "kubernetes":
                    if not self.k8s_manager:
                        raise Exception("Kubernetes manager not available")
                    
                    env = KubernetesEnvironment(environment)
                    result = await self.k8s_manager.deploy(env)
                    return result
                    
                else:
                    raise Exception(f"Unsupported platform: {platform}")
                    
            except Exception as e:
                logger.error(f"Deployment failed: {e}")
                return {"error": str(e)}
        
        @self.app.get("/orchestrator/models")
        async def get_model_status():
            """Get model manager status"""
            try:
                if not self.model_manager:
                    return {"error": "Model manager not available"}
                
                return {
                    "available_models": self.model_manager.get_available_models(),
                    "system_status": self.model_manager.get_system_status()
                }
            except Exception as e:
                logger.error(f"Failed to get model status: {e}")
                return {"error": str(e)}
        
        @self.app.post("/orchestrator/models/{model_name}/install")
        async def install_model(model_name: str):
            """Install a specific model"""
            try:
                if not self.model_manager:
                    return {"error": "Model manager not available"}
                
                result = await self.model_manager.install_model(model_name)
                return result
            except Exception as e:
                logger.error(f"Failed to install model {model_name}: {e}")
                return {"error": str(e)}
        
        @self.app.post("/orchestrator/models/{model_name}/load")
        async def load_model(model_name: str):
            """Load a specific model"""
            try:
                if not self.model_manager:
                    return {"error": "Model manager not available"}
                
                result = await self.model_manager.load_model(model_name)
                return result
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")
                return {"error": str(e)}
        
        @self.app.post("/orchestrator/emergency-shutdown")
        async def emergency_shutdown():
            """Emergency shutdown of the entire system"""
            try:
                logger.info("Emergency shutdown requested")
                
                # Shutdown AGI system
                if self.agi_system:
                    self.agi_system.emergency_shutdown("chrissuta01@gmail.com")
                
                # Stop system
                await self.stop_system()
                
                return {"status": "success", "message": "System shutdown initiated"}
                
            except Exception as e:
                logger.error(f"Emergency shutdown failed: {e}")
                return {"error": str(e)}
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown")
            asyncio.create_task(self.stop_system())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def _validate_system(self):
        """Validate system components are working correctly"""
        try:
            logger.info("Validating system components...")
            
            validation_results = {}
            
            # Validate AGI system
            if self.agi_system:
                status = self.agi_system.get_system_status()
                validation_results["agi_system"] = {
                    "status": "ok" if status["state"] == "ready" else "error",
                    "details": status
                }
            
            # Validate model manager
            if self.model_manager:
                status = self.model_manager.get_system_status()
                validation_results["model_manager"] = {
                    "status": "ok" if status["ollama_status"]["status"] == "running" else "warning",
                    "details": status
                }
            
            # Validate enterprise components
            validation_results["enterprise_components"] = {
                "security_manager": "ok" if self.security_manager else "error",
                "db_manager": "ok" if self.db_manager else "error",
                "performance_profiler": "ok" if self.performance_profiler else "error"
            }
            
            # Save validation results
            validation_file = self.logs_dir / "system_validation.json"
            with open(validation_file, 'w') as f:
                json.dump({
                    "timestamp": datetime.now().isoformat(),
                    "results": validation_results
                }, f, indent=2)
            
            logger.info("System validation complete")
            
        except Exception as e:
            logger.error(f"System validation failed: {e}")
            raise
    
    async def start_system(self, host: str = "0.0.0.0", port: int = 8000):
        """Start the complete system"""
        try:
            logger.info("=== Starting SutazAI AGI/ASI System ===")
            
            # Initialize system if not already done
            if not self.initialization_complete:
                await self.initialize_system()
            
            # Record start time
            self.start_time = time.time()
            self.is_running = True
            
            # Create system status file
            await self._create_system_status_file()
            
            logger.info(f"System starting on {host}:{port}")
            
            # Start the API server
            config = uvicorn.Config(
                app=self.app,
                host=host,
                port=port,
                log_level="info",
                access_log=True
            )
            
            server = uvicorn.Server(config)
            await server.serve()
            
        except Exception as e:
            logger.error(f"Failed to start system: {e}")
            raise
    
    async def stop_system(self):
        """Stop the complete system gracefully"""
        try:
            logger.info("=== Stopping SutazAI AGI/ASI System ===")
            
            self.is_running = False
            
            # Stop AGI system
            if self.agi_system:
                self.agi_system.emergency_shutdown("chrissuta01@gmail.com")
            
            # Unload models
            if self.model_manager:
                models = self.model_manager.get_available_models()
                for model in models:
                    if model["loaded"]:
                        await self.model_manager.unload_model(model["name"])
            
            # Create shutdown status file
            await self._create_shutdown_status_file()
            
            logger.info("System stopped successfully")
            
        except Exception as e:
            logger.error(f"Failed to stop system: {e}")
            raise
    
    async def _create_system_status_file(self):
        """Create system status file for monitoring"""
        try:
            status_file = self.base_dir / "SYSTEM_STATUS.json"
            
            status = {
                "timestamp": datetime.now().isoformat(),
                "status": "running",
                "initialization_complete": self.initialization_complete,
                "components": {
                    "agi_system": "active" if self.agi_system else "inactive",
                    "model_manager": "active" if self.model_manager else "inactive",
                    "security_manager": "active" if self.security_manager else "inactive",
                    "db_manager": "active" if self.db_manager else "inactive",
                    "performance_profiler": "active" if self.performance_profiler else "inactive",
                    "docker_manager": "active" if self.docker_manager else "inactive",
                    "k8s_manager": "active" if self.k8s_manager else "inactive"
                },
                "urls": {
                    "api_docs": f"http://localhost:8000/api/docs",
                    "health_check": f"http://localhost:8000/health",
                    "orchestrator_status": f"http://localhost:8000/orchestrator/status"
                }
            }
            
            with open(status_file, 'w') as f:
                json.dump(status, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to create system status file: {e}")
    
    async def _create_shutdown_status_file(self):
        """Create shutdown status file"""
        try:
            status_file = self.base_dir / "SHUTDOWN_STATUS.json"
            
            status = {
                "timestamp": datetime.now().isoformat(),
                "status": "shutdown",
                "message": "System shutdown gracefully",
                "uptime": time.time() - self.start_time if hasattr(self, 'start_time') else 0
            }
            
            with open(status_file, 'w') as f:
                json.dump(status, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to create shutdown status file: {e}")

# Global orchestrator instance
_orchestrator_instance = None

def get_orchestrator() -> SutazAIOrchestrator:
    """Get the global orchestrator instance"""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = SutazAIOrchestrator()
    return _orchestrator_instance

async def main():
    """Main entry point for the SutazAI system"""
    try:
        # Create orchestrator
        orchestrator = get_orchestrator()
        
        # Start system
        await orchestrator.start_system()
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
        if _orchestrator_instance:
            await _orchestrator_instance.stop_system()
    except Exception as e:
        logger.error(f"System error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Run the main system
    asyncio.run(main())