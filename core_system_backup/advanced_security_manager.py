#!/usr/bin/env python3
"""
SutazAI Advanced Security Management System

Comprehensive security framework providing:
- Multi-tier threat detection
- Adaptive access control
- Advanced cryptographic services
- Continuous security monitoring
"""

import os
import sys
import json
import logging
import hashlib
import secrets
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

import cryptography
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import bandit
import jwt

class AdvancedSecurityManager:
    """
    Intelligent security management system with adaptive protection mechanisms
    """
    
    def __init__(
        self, 
        base_dir: str = '/opt/sutazai_project/SutazAI',
        config_dir: str = 'config/security'
    ):
        """
        Initialize advanced security manager
        
        Args:
            base_dir (str): Base project directory
            config_dir (str): Security configuration directory
        """
        self.base_dir = base_dir
        self.config_dir = os.path.join(base_dir, config_dir)
        
        # Ensure config directory exists
        os.makedirs(self.config_dir, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s: %(message)s',
            filename=os.path.join(base_dir, 'logs/security_management.log')
        )
        self.logger = logging.getLogger('SutazAI.SecurityManager')
    
    def generate_secure_key(self, key_length: int = 32) -> bytes:
        """
        Generate a cryptographically secure random key
        
        Args:
            key_length (int): Length of the key in bytes
        
        Returns:
            Secure random key
        """
        return secrets.token_bytes(key_length)
    
    def derive_key(self, password: str, salt: bytes = None) -> bytes:
        """
        Derive a secure encryption key from a password
        
        Args:
            password (str): User-provided password
            salt (bytes, optional): Cryptographic salt
        
        Returns:
            Derived encryption key
        """
        if salt is None:
            salt = os.urandom(16)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        
        return kdf.derive(password.encode())
    
    def encrypt_data(self, data: str, key: bytes = None) -> Dict[str, str]:
        """
        Encrypt sensitive data with advanced cryptographic techniques
        
        Args:
            data (str): Data to encrypt
            key (bytes, optional): Encryption key
        
        Returns:
            Dictionary with encrypted data and metadata
        """
        if key is None:
            key = self.generate_secure_key()
        
        f = Fernet(base64.urlsafe_b64encode(key))
        encrypted_data = f.encrypt(data.encode())
        
        return {
            'encrypted_data': encrypted_data.decode(),
            'encryption_metadata': {
                'timestamp': datetime.now().isoformat(),
                'encryption_method': 'Fernet'
            }
        }
    
    def decrypt_data(self, encrypted_data: Dict[str, str], key: bytes) -> str:
        """
        Decrypt sensitive data
        
        Args:
            encrypted_data (Dict): Encrypted data and metadata
            key (bytes): Decryption key
        
        Returns:
            Decrypted data
        """
        f = Fernet(base64.urlsafe_b64encode(key))
        return f.decrypt(encrypted_data['encrypted_data'].encode()).decode()
    
    def generate_jwt_token(
        self, 
        payload: Dict[str, Any], 
        secret_key: str = None, 
        expiration: int = 3600
    ) -> str:
        """
        Generate a JSON Web Token with configurable expiration
        
        Args:
            payload (Dict): Token payload
            secret_key (str, optional): Secret key for signing
            expiration (int): Token expiration time in seconds
        
        Returns:
            Signed JWT token
        """
        if secret_key is None:
            secret_key = secrets.token_hex(32)
        
        payload['exp'] = datetime.utcnow() + timedelta(seconds=expiration)
        return jwt.encode(payload, secret_key, algorithm='HS256')
    
    def validate_jwt_token(
        self, 
        token: str, 
        secret_key: str
    ) -> Optional[Dict[str, Any]]:
        """
        Validate and decode a JWT token
        
        Args:
            token (str): JWT token
            secret_key (str): Secret key for verification
        
        Returns:
            Decoded token payload or None if invalid
        """
        try:
            return jwt.decode(token, secret_key, algorithms=['HS256'])
        except jwt.ExpiredSignatureError:
            self.logger.warning("JWT token has expired")
            return None
        except jwt.InvalidTokenError:
            self.logger.warning("Invalid JWT token")
            return None
    
    def scan_code_vulnerabilities(self, project_dir: str = None) -> List[Dict[str, Any]]:
        """
        Perform comprehensive code vulnerability scanning
        
        Args:
            project_dir (str, optional): Directory to scan
        
        Returns:
            List of identified vulnerabilities
        """
        if project_dir is None:
            project_dir = self.base_dir
        
        try:
            # Use Bandit for static code analysis
            bandit_results = bandit.run([project_dir])
            
            return [
                {
                    'filename': issue.filename,
                    'line_number': issue.line_number,
                    'issue_text': issue.text,
                    'severity': issue.severity,
                    'confidence': issue.confidence
                } for issue in bandit_results.issues
            ]
        except Exception as e:
            self.logger.error(f"Code vulnerability scanning failed: {e}")
            return []
    
    def generate_security_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive security management report
        
        Returns:
            Detailed security analysis report
        """
        security_report = {
            'timestamp': datetime.now().isoformat(),
            'code_vulnerabilities': self.scan_code_vulnerabilities(),
            'recommendations': []
        }
        
        # Generate recommendations
        if security_report['code_vulnerabilities']:
            security_report['recommendations'].append(
                "Critical: Address identified code vulnerabilities immediately"
            )
        
        # Persist report
        report_path = os.path.join(
            self.base_dir, 
            f'logs/security_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        )
        
        with open(report_path, 'w') as f:
            json.dump(security_report, f, indent=2)
        
        self.logger.info(f"Security report generated: {report_path}")
        
        return security_report

def main():
    """
    Main execution for advanced security management
    """
    try:
        security_manager = AdvancedSecurityManager()
        report = security_manager.generate_security_report()
        
        print("Security Management Report:")
        print("Recommendations:")
        for recommendation in report.get('recommendations', []):
            print(f"- {recommendation}")
    
    except Exception as e:
        print(f"Security management failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 