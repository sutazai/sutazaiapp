#!/usr/bin/env python3
"""
SutazAI Security Secrets Generator
Generates cryptographically secure passwords and secrets for production deployment
"""

import os
import secrets
import string
import hashlib
from pathlib import Path


def generate_password(length=32, include_symbols=True):
    """Generate a cryptographically secure password"""
    alphabet = string.ascii_letters + string.digits
    if include_symbols:
        alphabet += "!@#$%^&*"
    
    return ''.join(secrets.choice(alphabet) for _ in range(length))


def generate_hex_key(length=64):
    """Generate a cryptographically secure hex key"""
    return secrets.token_hex(length // 2)


def generate_jwt_secret():
    """Generate a cryptographically secure JWT secret"""
    return secrets.token_urlsafe(64)


def generate_api_key():
    """Generate a secure API key"""
    return secrets.token_urlsafe(32)


def main():
    """Generate all required secrets and create .env file"""
    
    # Define secure secrets
    secrets_config = {
        'POSTGRES_PASSWORD': generate_password(24),
        'REDIS_PASSWORD': generate_password(24),
        'NEO4J_PASSWORD': generate_password(24),
        'RABBITMQ_DEFAULT_PASS': generate_password(24),
        'SECRET_KEY': generate_hex_key(64),
        'JWT_SECRET': generate_jwt_secret(),
        'GRAFANA_PASSWORD': generate_password(16, False),  # No symbols for Grafana
        'PROMETHEUS_PASSWORD': generate_password(16, False),
        'CHROMADB_API_KEY': generate_api_key(),
        'PORCUPINE_ACCESS_KEY': generate_api_key(),
        'VAULT_TOKEN': generate_hex_key(32),
        'KEYCLOAK_CLIENT_SECRET': generate_hex_key(32),
        'KEYCLOAK_ADMIN_PASSWORD': generate_password(16, False),
        'TABBY_API_KEY': generate_api_key(),
    }
    
    # Read the template
    template_path = Path(__file__).parent.parent / '.env.secure.template'
    output_path = Path(__file__).parent.parent / '.env.production.secure'
    
    with open(template_path, 'r') as f:
        content = f.read()
    
    # Replace empty values with generated secrets
    for key, value in secrets_config.items():
        content = content.replace(f'{key}=', f'{key}={value}')
    
    # Write the production environment file
    with open(output_path, 'w') as f:
        f.write(content)
    
    # Set secure permissions
    os.chmod(output_path, 0o600)
    
    print("🔐 Secure secrets generated successfully!")
    print(f"📁 Production environment file: {output_path}")
    print("⚠️  CRITICAL: Store this file securely and restrict access")
    print("📋 Next steps:")
    print("   1. Copy .env.production.secure to .env in project root")
    print("   2. Secure the secrets file with proper file permissions")
    print("   3. Add .env to .gitignore to prevent accidental commits")
    print("   4. Use a secrets management system in production")
    
    # Generate secrets directory structure
    secrets_dir = Path(__file__).parent.parent / 'secrets_secure'
    secrets_dir.mkdir(exist_ok=True)
    
    # Write individual secret files
    for key, value in secrets_config.items():
        secret_file = secrets_dir / f"{key.lower()}.txt"
        with open(secret_file, 'w') as f:
            f.write(value)
        os.chmod(secret_file, 0o600)
    
    print(f"📂 Individual secret files created in: {secrets_dir}")


if __name__ == '__main__':
    main()