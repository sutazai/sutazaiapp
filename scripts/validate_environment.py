#!/usr/bin/env python3
"""
Environment Validation Script
Validates environment variables, database connectivity, and dependencies before service start
"""

import os
import sys
import asyncio
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import socket
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("env_validator")


@dataclass
class ValidationResult:
    """Result of a validation check"""
    check_name: str
    passed: bool
    message: str
    required: bool = True


class EnvironmentValidator:
    """Validates environment configuration before service startup"""
    
    def __init__(self, service_name: str = "sutazai"):
        self.service_name = service_name
        self.results: List[ValidationResult] = []
        
    def check_env_var(
        self,
        var_name: str,
        required: bool = True,
        expected_values: Optional[List[str]] = None
    ) -> ValidationResult:
        """Check if environment variable exists and optionally validate value"""
        value = os.getenv(var_name)
        
        if value is None:
            return ValidationResult(
                check_name=f"ENV: {var_name}",
                passed=not required,
                message=f"Missing {'required' if required else 'optional'} environment variable",
                required=required
            )
            
        if expected_values and value not in expected_values:
            return ValidationResult(
                check_name=f"ENV: {var_name}",
                passed=False,
                message=f"Invalid value '{value}', expected one of: {expected_values}",
                required=required
            )
            
        return ValidationResult(
            check_name=f"ENV: {var_name}",
            passed=True,
            message=f"Set to '{value[:50]}...' " if len(value) > 50 else f"Set to '{value}'",
            required=required
        )
        
    def check_tcp_connection(
        self,
        host: str,
        port: int,
        timeout: int = 5,
        required: bool = True
    ) -> ValidationResult:
        """Check if TCP connection to host:port is possible"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((host, port))
            sock.close()
            
            if result == 0:
                return ValidationResult(
                    check_name=f"TCP: {host}:{port}",
                    passed=True,
                    message="Connection successful",
                    required=required
                )
            else:
                return ValidationResult(
                    check_name=f"TCP: {host}:{port}",
                    passed=False,
                    message=f"Connection failed (error code: {result})",
                    required=required
                )
        except Exception as e:
            return ValidationResult(
                check_name=f"TCP: {host}:{port}",
                passed=False,
                message=f"Connection error: {str(e)}",
                required=required
            )
            
    async def check_postgres(
        self,
        host: str,
        port: int,
        database: str,
        user: str,
        password: str,
        required: bool = True
    ) -> ValidationResult:
        """Check PostgreSQL connectivity and database existence"""
        try:
            import asyncpg
            
            conn = await asyncpg.connect(
                host=host,
                port=port,
                database=database,
                user=user,
                password=password,
                timeout=5
            )
            
            version = await conn.fetchval('SELECT version()')
            await conn.close()
            
            return ValidationResult(
                check_name=f"PostgreSQL: {host}:{port}/{database}",
                passed=True,
                message=f"Connected successfully ({version.split()[1]})",
                required=required
            )
        except Exception as e:
            return ValidationResult(
                check_name=f"PostgreSQL: {host}:{port}/{database}",
                passed=False,
                message=f"Connection failed: {str(e)}",
                required=required
            )
            
    async def check_redis(
        self,
        host: str,
        port: int,
        required: bool = True
    ) -> ValidationResult:
        """Check Redis connectivity"""
        try:
            import redis.asyncio as redis
            
            client = redis.Redis(
                host=host,
                port=port,
                socket_connect_timeout=5,
                decode_responses=True
            )
            
            await client.ping()
            info = await client.info('server')
            version = info.get('redis_version', 'unknown')
            await client.close()
            
            return ValidationResult(
                check_name=f"Redis: {host}:{port}",
                passed=True,
                message=f"Connected successfully (v{version})",
                required=required
            )
        except Exception as e:
            return ValidationResult(
                check_name=f"Redis: {host}:{port}",
                passed=False,
                message=f"Connection failed: {str(e)}",
                required=required
            )
            
    def check_file_exists(
        self,
        file_path: str,
        required: bool = True
    ) -> ValidationResult:
        """Check if required file exists"""
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            return ValidationResult(
                check_name=f"File: {file_path}",
                passed=True,
                message=f"Exists ({size} bytes)",
                required=required
            )
        else:
            return ValidationResult(
                check_name=f"File: {file_path}",
                passed=False,
                message="File not found",
                required=required
            )
            
    def check_directory_writable(
        self,
        dir_path: str,
        required: bool = True
    ) -> ValidationResult:
        """Check if directory exists and is writable"""
        if not os.path.exists(dir_path):
            return ValidationResult(
                check_name=f"Directory: {dir_path}",
                passed=False,
                message="Directory does not exist",
                required=required
            )
            
        if not os.access(dir_path, os.W_OK):
            return ValidationResult(
                check_name=f"Directory: {dir_path}",
                passed=False,
                message="Directory is not writable",
                required=required
            )
            
        return ValidationResult(
            check_name=f"Directory: {dir_path}",
            passed=True,
            message="Exists and writable",
            required=required
        )
        
    def add_result(self, result: ValidationResult):
        """Add validation result to list"""
        self.results.append(result)
        
        # Log result
        if result.passed:
            logger.info(f"✓ {result.check_name}: {result.message}")
        else:
            if result.required:
                logger.error(f"✗ {result.check_name}: {result.message} [REQUIRED]")
            else:
                logger.warning(f"⚠ {result.check_name}: {result.message} [OPTIONAL]")
                
    def validate_common_env(self):
        """Validate common environment variables"""
        logger.info("Validating environment variables...")
        
        # Core service configuration
        self.add_result(self.check_env_var("ENVIRONMENT", required=False))
        self.add_result(self.check_env_var("LOG_LEVEL", required=False))
        
    async def validate_backend_env(self):
        """Validate backend-specific environment"""
        logger.info("Validating backend environment...")
        
        # Database
        self.add_result(self.check_env_var("DATABASE_URL", required=True))
        
        # JWT/Security
        self.add_result(self.check_env_var("SECRET_KEY", required=True))
        self.add_result(self.check_env_var("JWT_SECRET", required=False))
        
        # Redis
        self.add_result(self.check_env_var("REDIS_HOST", required=False))
        self.add_result(self.check_env_var("REDIS_PORT", required=False))
        
        # Check database connectivity
        db_host = os.getenv("DB_HOST", "sutazai-postgres")
        db_port = int(os.getenv("DB_PORT", "5432"))
        db_name = os.getenv("DB_NAME", "jarvis_ai")
        db_user = os.getenv("DB_USER", "jarvis")
        db_pass = os.getenv("DB_PASSWORD", "")
        
        self.add_result(self.check_tcp_connection(db_host, db_port, required=True))
        
        if db_pass:  # Only try full connection if password provided
            self.add_result(await self.check_postgres(
                db_host, db_port, db_name, db_user, db_pass, required=True
            ))
            
        # Check Redis
        redis_host = os.getenv("REDIS_HOST", "sutazai-redis")
        redis_port = int(os.getenv("REDIS_PORT", "6379"))
        
        self.add_result(self.check_tcp_connection(redis_host, redis_port, required=False))
        
    def get_summary(self) -> Tuple[int, int, int]:
        """Get validation summary (passed, failed_required, failed_optional)"""
        passed = sum(1 for r in self.results if r.passed)
        failed_required = sum(1 for r in self.results if not r.passed and r.required)
        failed_optional = sum(1 for r in self.results if not r.passed and not r.required)
        
        return passed, failed_required, failed_optional
        
    def print_summary(self):
        """Print validation summary"""
        passed, failed_required, failed_optional = self.get_summary()
        total = len(self.results)
        
        logger.info("=" * 60)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total checks:     {total}")
        logger.info(f"Passed:           {passed} ✓")
        logger.info(f"Failed (required): {failed_required} ✗")
        logger.info(f"Failed (optional): {failed_optional} ⚠")
        logger.info("=" * 60)
        
        if failed_required > 0:
            logger.error(f"Validation FAILED: {failed_required} required checks failed")
            return False
        else:
            logger.info("Validation PASSED: All required checks successful")
            return True


async def validate_environment(service_type: str = "backend") -> bool:
    """
    Main validation function
    
    Args:
        service_type: Type of service to validate (backend, frontend, agent)
        
    Returns:
        True if all required checks pass, False otherwise
    """
    validator = EnvironmentValidator(service_name=service_type)
    
    # Common validation
    validator.validate_common_env()
    
    # Service-specific validation
    if service_type == "backend":
        await validator.validate_backend_env()
    # Add more service types as needed
    
    # Print summary and return result
    return validator.print_summary()


if __name__ == "__main__":
    # Parse command line arguments
    service_type = sys.argv[1] if len(sys.argv) > 1 else "backend"
    
    logger.info(f"Starting environment validation for {service_type} service...")
    
    # Run validation
    success = asyncio.run(validate_environment(service_type))
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)
