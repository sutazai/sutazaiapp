#!/usr/bin/env python3
"""
Secure Backend Startup Script
Validates security requirements before starting the application
"""

import os
import sys
import secrets
import logging
from pathlib import Path
from typing import Dict, List, Optional

# Add backend to path
sys.path.insert(0, '/opt/sutazaiapp/backend')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SecurityRequirementsValidator:
    """Validates security requirements before application startup"""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
        
    def validate_database_security(self) -> bool:
        """Validate database security configuration"""
        logger.info("🔍 Validating database security...")
        
        db_password = os.getenv('POSTGRES_PASSWORD')
        if not db_password:
            self.errors.append("POSTGRES_PASSWORD environment variable is required")
            return False
            
        # Check password strength
        if len(db_password) < 12:
            self.warnings.append("Database password should be at least 12 characters long")
            
        if db_password in ['password', '123456', 'sutazai123', 'admin']:
            self.errors.append("Database password is using a common/weak password")
            return False
            
        logger.info("✅ Database security validation passed")
        return True
        
    def validate_jwt_security(self) -> bool:
        """Validate JWT security configuration"""
        logger.info("🔍 Validating JWT security...")
        
        jwt_secret = os.getenv('JWT_SECRET_KEY') or os.getenv('JWT_SECRET')
        if not jwt_secret:
            self.errors.append("JWT_SECRET_KEY environment variable is required")
            return False
            
        # Check secret strength
        if len(jwt_secret) < 32:
            self.errors.append("JWT secret must be at least 32 characters long")
            return False
            
        if jwt_secret in ['your_secret_key_here', 'secret', 'jwt_secret']:
            self.errors.append("JWT secret is using a default/weak value")
            return False
            
        # Check if secret looks randomly generated
        if not any(c.isdigit() for c in jwt_secret) or not any(c.isalpha() for c in jwt_secret):
            self.warnings.append("JWT secret should contain both letters and numbers")
            
        logger.info("✅ JWT security validation passed")
        return True
        
    def validate_environment_security(self) -> bool:
        """Validate environment security settings"""
        logger.info("🔍 Validating environment security...")
        
        environment = os.getenv('SUTAZAI_ENV', 'production')
        
        # Check if running in production with debug settings
        if environment == 'production':
            if os.getenv('DEBUG', '').lower() == 'true':
                self.errors.append("DEBUG mode should not be enabled in production")
                return False
                
            if os.getenv('LOG_LEVEL', 'INFO').upper() == 'DEBUG':
                self.warnings.append("DEBUG log level in production may expose sensitive information")
                
        # Check for test credentials in production
        if environment == 'production':
            test_indicators = ['test', 'dev', 'localhost', '127.0.0.1']
            db_host = os.getenv('POSTGRES_HOST', '')
            
            if any(indicator in db_host.lower() for indicator in test_indicators):
                self.warnings.append(f"Database host '{db_host}' looks like a development setting")
                
        logger.info("✅ Environment security validation passed")
        return True
        
    def validate_network_security(self) -> bool:
        """Validate network security settings"""
        logger.info("🔍 Validating network security...")
        
        # Check CORS origins
        allowed_origins = os.getenv('ALLOWED_ORIGINS', '')
        if '*' in allowed_origins:
            self.errors.append("CORS allows all origins (*) - this is a security risk")
            return False
            
        # Check if localhost is allowed in production
        environment = os.getenv('SUTAZAI_ENV', 'production')
        if environment == 'production' and 'localhost' in allowed_origins:
            self.warnings.append("Localhost in CORS origins for production environment")
            
        logger.info("✅ Network security validation passed")
        return True
        
    def validate_api_keys(self) -> bool:
        """Validate API key security"""
        logger.info("🔍 Validating API key security...")
        
        api_keys = os.getenv('VALID_API_KEYS', '')
        if not api_keys:
            self.warnings.append("No API keys configured - service-to-service auth will be disabled")
            return True
            
        keys = [key.strip() for key in api_keys.split(',') if key.strip()]
        
        for key in keys:
            if len(key) < 20:
                self.warnings.append(f"API key '{key[:8]}...' is shorter than recommended (20+ chars)")
                
            if key.lower() in ['apikey', 'key123', 'secret']:
                self.errors.append(f"API key '{key[:8]}...' is using a weak/default value")
                return False
                
        logger.info("✅ API key security validation passed")
        return True
        
    def generate_secure_secrets(self) -> Dict[str, str]:
        """Generate secure secrets for missing credentials"""
        secrets_generated = {}
        
        # Generate JWT secret if missing
        if not os.getenv('JWT_SECRET_KEY'):
            jwt_secret = secrets.token_urlsafe(64)  # 64 bytes = 512 bits of entropy
            secrets_generated['JWT_SECRET_KEY'] = jwt_secret
            logger.info("🔑 Generated secure JWT secret")
            
        # Generate API key if missing
        if not os.getenv('VALID_API_KEYS'):
            api_key = secrets.token_urlsafe(32)
            secrets_generated['VALID_API_KEYS'] = api_key
            logger.info("🔑 Generated secure API key")
            
        return secrets_generated
        
    def run_validation(self, generate_missing: bool = False) -> bool:
        """Run all security validations"""
        logger.info("=" * 60)
        logger.info("🔒 SECURITY VALIDATION STARTING")
        logger.info("=" * 60)
        
        # Generate missing secrets if requested
        if generate_missing:
            generated = self.generate_secure_secrets()
            for key, value in generated.items():
                os.environ[key] = value
                logger.info(f"🔑 Set {key} with generated secure value")
        
        # Run all validations
        validations = [
            ("Database Security", self.validate_database_security),
            ("JWT Security", self.validate_jwt_security),
            ("Environment Security", self.validate_environment_security),
            ("Network Security", self.validate_network_security),
            ("API Key Security", self.validate_api_keys),
        ]
        
        all_passed = True
        for name, validator in validations:
            try:
                if not validator():
                    all_passed = False
            except Exception as e:
                logger.error(f"❌ {name} validation failed with error: {e}")
                self.errors.append(f"{name}: {str(e)}")
                all_passed = False
        
        # Report results
        logger.info("=" * 60)
        logger.info("🔒 SECURITY VALIDATION RESULTS")
        logger.info("=" * 60)
        
        if self.warnings:
            logger.warning("⚠️ WARNINGS:")
            for warning in self.warnings:
                logger.warning(f"  • {warning}")
                
        if self.errors:
            logger.error("❌ ERRORS:")
            for error in self.errors:
                logger.error(f"  • {error}")
                
        if all_passed:
            if self.warnings:
                logger.info("✅ VALIDATION PASSED (with warnings)")
            else:
                logger.info("✅ VALIDATION PASSED - All security requirements met")
        else:
            logger.error("❌ VALIDATION FAILED - Security requirements not met")
            logger.error("Fix the errors above before starting the application")
            
        return all_passed


def main():
    """Main validation and startup"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Secure backend startup with validation")
    parser.add_argument("--generate-secrets", action="store_true", 
                       help="Generate missing secrets automatically")
    parser.add_argument("--start-server", action="store_true",
                       help="Start the server after validation")
    parser.add_argument("--port", type=int, default=10010,
                       help="Port to run the server on")
    
    args = parser.parse_args()
    
    # Run security validation
    validator = SecurityRequirementsValidator()
    validation_passed = validator.run_validation(generate_missing=args.generate_secrets)
    
    if not validation_passed:
        logger.critical("❌ STARTUP ABORTED - Security validation failed")
        logger.critical("Fix security issues and try again")
        sys.exit(1)
        
    logger.info("🚀 Security validation completed successfully")
    
    if args.start_server:
        logger.info(f"🌐 Starting secure server on port {args.port}...")
        
        # Import and start uvicorn
        try:
            import uvicorn
            uvicorn.run(
                "app.main:app",
                host="0.0.0.0",
                port=args.port,
                log_level="info",
                access_log=True,
                server_header=False,  # Security: Hide server version
                date_header=False,    # Security: Hide date header
            )
        except ImportError:
            logger.error("uvicorn not installed. Install with: pip install uvicorn[standard]")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            sys.exit(1)
    else:
        logger.info("✅ Validation complete. Use --start-server to start the application")


if __name__ == "__main__":
    main()