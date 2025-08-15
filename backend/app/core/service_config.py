"""
Service Configuration Module for SutazAI System
Handles all service URLs and networking configuration
Replaces hardcoded localhost references with environment-based configuration

Rule 2 Compliance: Environment-based configuration prevents breaking changes
"""

import os
from typing import Optional, Dict, Any
from functools import lru_cache
from logging_config import get_logger

logger = get_logger(__name__)


class ServiceConfig:
    """
    Centralized service configuration management
    Handles all service URLs, ports, and networking configuration
    """
    
    def __init__(self):
        """Initialize service configuration"""
        self.environment = os.getenv('SUTAZAI_ENV', 'development')
        self.base_host = os.getenv('SUTAZAI_HOST', 'localhost')
        self.use_https = os.getenv('SUTAZAI_USE_HTTPS', 'false').lower() == 'true'
        
        # Load custom port mappings if provided
        self._custom_ports = self._load_custom_ports()
        
        logger.info(f"ServiceConfig initialized for environment: {self.environment}")
        logger.info(f"Base host: {self.base_host}")
        logger.info(f"HTTPS enabled: {self.use_https}")
    
    def _load_custom_ports(self) -> Dict[str, int]:
        """Load custom port mappings from environment variables"""
        custom_ports = {}
        
        # Check for environment-specific port overrides
        for key in os.environ:
            if key.startswith('SUTAZAI_PORT_'):
                service_name = key[13:].lower()  # Remove 'SUTAZAI_PORT_' prefix
                try:
                    port = int(os.environ[key])
                    custom_ports[service_name] = port
                except ValueError:
                    logger.warning(f"Invalid port value for {key}: {os.environ[key]}")
        
        return custom_ports
    
    @property
    def protocol(self) -> str:
        """Get the protocol scheme (http or https)"""
        return 'https' if self.use_https else 'http'
    
    def get_port(self, service: str) -> int:
        """
        Get port for a specific service
        
        Args:
            service: Service name (e.g., 'backend', 'frontend', 'postgres')
            
        Returns:
            Port number for the service
        """
        # Check for custom port override
        if service.lower() in self._custom_ports:
            return self._custom_ports[service.lower()]
        
        # Default port mappings based on PortRegistry.md
        default_ports = {
            # Infrastructure Services (10000-10199)
            'postgres': 10000,
            'postgresql': 10000,
            'redis': 10001,
            'neo4j': 10002,
            'neo4j_http': 10003,
            'rabbitmq': 10007,
            'rabbitmq_mgmt': 10008,
            'backend': 10010,
            'api': 10010,
            'frontend': 10011,
            'streamlit': 10011,
            
            # Vector & AI Services (10100-10199)
            'chromadb': 10100,
            'qdrant': 10101,
            'qdrant_grpc': 10102,
            'faiss': 10103,
            'ollama': 10104,
            
            # Monitoring Stack (10200-10299)
            'prometheus': 10200,
            'grafana': 10201,
            'loki': 10202,
            'alertmanager': 10203,
            'blackbox_exporter': 10204,
            'node_exporter': 10205,
            'cadvisor': 10206,
            'postgres_exporter': 10207,
            'redis_exporter': 10208,
            'jaeger': 10210,
            
            # AI Agents (11000+)
            'hardware_optimizer': 8116,  # Special case for existing agent
            'jarvis': 11001,
            'task_coordinator': 11002,
            'resource_arbitrator': 11003,
            'ai_orchestrator': 11004,
            'ollama_integration': 11005,
            'jarvis_hardware': 11006,
            'ultra_frontend': 11007,
        }
        
        port = default_ports.get(service.lower())
        if port is None:
            logger.warning(f"Unknown service: {service}, using default port 8000")
            return 8000
        
        return port
    
    def get_url(self, service: str, path: str = '') -> str:
        """
        Get complete URL for a service
        
        Args:
            service: Service name
            path: Optional path to append (should start with /)
            
        Returns:
            Complete URL for the service
        """
        port = self.get_port(service)
        base_url = f"{self.protocol}://{self.base_host}:{port}"
        
        if path and not path.startswith('/'):
            path = '/' + path
        
        return base_url + path
    
    def get_database_url(self, 
                        db_type: str = 'postgresql', 
                        username: Optional[str] = None,
                        password: Optional[str] = None,
                        database: Optional[str] = None,
                        async_driver: bool = False) -> str:
        """
        Get database connection URL
        
        Args:
            db_type: Database type ('postgresql', 'redis', 'neo4j')
            username: Database username (from env if not provided)
            password: Database password (from env if not provided)
            database: Database name (from env if not provided)
            async_driver: Whether to use async driver
            
        Returns:
            Database connection URL
        """
        if db_type.lower() == 'postgresql':
            username = username or os.getenv('POSTGRES_USER', 'sutazai')
            password = password or os.getenv('POSTGRES_PASSWORD', 'sutazai123')
            database = database or os.getenv('POSTGRES_DB', 'sutazai')
            port = self.get_port('postgres')
            
            driver = 'postgresql+asyncpg' if async_driver else 'postgresql'
            return f"{driver}://{username}:{password}@{self.base_host}:{port}/{database}"
        
        elif db_type.lower() == 'redis':
            password = password or os.getenv('REDIS_PASSWORD', '')
            port = self.get_port('redis')
            
            auth = f":{password}@" if password else ""
            return f"redis://{auth}{self.base_host}:{port}/0"
        
        elif db_type.lower() == 'neo4j':
            username = username or os.getenv('NEO4J_USER', 'neo4j')
            password = password or os.getenv('NEO4J_PASSWORD', 'sutazai123')
            port = self.get_port('neo4j')
            
            return f"bolt://{username}:{password}@{self.base_host}:{port}"
        
        else:
            raise ValueError(f"Unsupported database type: {db_type}")
    
    def get_service_discovery_config(self) -> Dict[str, Any]:
        """
        Get service discovery configuration
        
        Returns:
            Dictionary with all service endpoints
        """
        services = {}
        
        # Core services
        services.update({
            'backend': self.get_url('backend'),
            'frontend': self.get_url('frontend'),
            'api': self.get_url('backend', '/api/v1'),
            'health': self.get_url('backend', '/health'),
            'metrics': self.get_url('backend', '/metrics'),
        })
        
        # Database services
        services.update({
            'postgres': self.get_database_url('postgresql'),
            'postgres_async': self.get_database_url('postgresql', async_driver=True),
            'redis': self.get_database_url('redis'),
            'neo4j': self.get_database_url('neo4j'),
        })
        
        # Vector databases
        services.update({
            'chromadb': self.get_url('chromadb'),
            'qdrant': self.get_url('qdrant'),
            'faiss': self.get_url('faiss'),
        })
        
        # AI services
        services.update({
            'ollama': self.get_url('ollama'),
            'ollama_api': self.get_url('ollama', '/api'),
        })
        
        # Monitoring services
        services.update({
            'prometheus': self.get_url('prometheus'),
            'grafana': self.get_url('grafana'),
            'loki': self.get_url('loki'),
        })
        
        return services
    
    def validate_connectivity(self) -> Dict[str, bool]:
        """
        Validate connectivity to all configured services
        
        Returns:
            Dictionary with service names and connectivity status
        """
        import requests
        from urllib.parse import urlparse
        
        results = {}
        services = self.get_service_discovery_config()
        
        for service_name, url in services.items():
            if service_name.startswith(('postgres', 'redis', 'neo4j')):
                # Skip database URLs as they require special handling
                continue
                
            try:
                parsed_url = urlparse(url)
                if parsed_url.scheme in ('http', 'https'):
                    response = requests.get(f"{url}/health", timeout=5)
                    results[service_name] = response.status_code < 500
                else:
                    results[service_name] = None  # Cannot test non-HTTP services
            except Exception as e:
                logger.debug(f"Connectivity test failed for {service_name}: {e}")
                results[service_name] = False
        
        return results
    
    def export_config(self, mask_passwords: bool = True) -> Dict[str, Any]:
        """
        Export configuration for debugging/monitoring
        
        Args:
            mask_passwords: Whether to mask sensitive information
            
        Returns:
            Configuration dictionary
        """
        config = {
            'environment': self.environment,
            'base_host': self.base_host,
            'protocol': self.protocol,
            'use_https': self.use_https,
            'custom_ports': self._custom_ports,
            'services': self.get_service_discovery_config()
        }
        
        if mask_passwords:
            # Mask passwords in database URLs
            for key, value in config['services'].items():
                if isinstance(value, str) and '://' in value and '@' in value:
                    # Mask password in URL
                    parts = value.split('@')
                    if len(parts) == 2:
                        auth_part = parts[0]
                        if ':' in auth_part:
                            protocol_user = auth_part.rsplit(':', 1)[0]
                            config['services'][key] = f"{protocol_user}:***@{parts[1]}"
        
        return config


# Global configuration instance
@lru_cache(maxsize=1)
def get_service_config() -> ServiceConfig:
    """Get the global service configuration instance"""
    return ServiceConfig()


# Convenience functions for common use cases
def get_service_url(service: str, path: str = '') -> str:
    """Get URL for a service"""
    return get_service_config().get_url(service, path)


def get_backend_url(path: str = '') -> str:
    """Get backend API URL"""
    return get_service_config().get_url('backend', path)


def get_database_url(db_type: str = 'postgresql', async_driver: bool = False) -> str:
    """Get database connection URL"""
    return get_service_config().get_database_url(db_type, async_driver=async_driver)


def get_api_base_url() -> str:
    """Get base API URL"""
    return get_service_config().get_url('backend', '/api/v1')


def validate_all_services() -> Dict[str, bool]:
    """Validate connectivity to all services"""
    return get_service_config().validate_connectivity()