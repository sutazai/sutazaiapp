"""
SutazAI Security Manager
Enterprise-grade security framework for input validation, threat detection, and data protection
"""

import hashlib
import hmac
import secrets
import re
import json
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import base64

logger = logging.getLogger(__name__)

class ThreatLevel(Enum):
    """Threat level classification"""
    LOW = 0.1
    MEDIUM = 0.5
    HIGH = 0.8
    CRITICAL = 1.0

@dataclass
class SecurityContext:
    """Security context for operations"""
    user_id: str
    session_id: str
    ip_address: str
    user_agent: str
    timestamp: datetime
    permissions: List[str]
    threat_level: float = 0.0

class SecurityManager:
    """Enterprise security manager"""
    
    def __init__(self):
        self.authorized_user = "chrissuta01@gmail.com"
        self.secret_key = secrets.token_urlsafe(32)
        self.threat_patterns = self._load_threat_patterns()
        self.security_log = []
        
    def _load_threat_patterns(self) -> Dict[str, List[str]]:
        """Load security threat patterns"""
        return {
            "xss": [
                r"<script[^>]*>.*?</script>",
                r"javascript:",
                r"on\w+\s*=",
                r"eval\s*\(",
                r"document\.cookie"
            ],
            "sql_injection": [
                r"'\s*(or|and)\s*'",
                r"union\s+select",
                r"drop\s+table",
                r"delete\s+from",
                r"insert\s+into",
                r"update\s+set"
            ],
            "path_traversal": [
                r"\.\./",
                r"\.\.\\",
                r"/etc/passwd",
                r"/windows/system32"
            ],
            "command_injection": [
                r";\s*rm\s+",
                r";\s*cat\s+",
                r";\s*ls\s+",
                r"\|\s*nc\s+",
                r"`.*`"
            ]
        }
    
    def validate_input(self, data: Dict[str, Any]) -> bool:
        """Validate input data for security threats"""
        try:
            if not isinstance(data, dict):
                return False
            
            # Convert to string for pattern matching
            data_str = json.dumps(data).lower()
            
            # Check against threat patterns
            for threat_type, patterns in self.threat_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, data_str, re.IGNORECASE):
                        self._log_security_event(
                            event_type="threat_detected",
                            threat_type=threat_type,
                            pattern=pattern,
                            data=data_str[:100]
                        )
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Input validation failed: {e}")
            return False
    
    def assess_threat_level(self, data: Dict[str, Any]) -> float:
        """Assess threat level of input data"""
        try:
            threat_score = 0.0
            data_str = json.dumps(data).lower()
            
            for threat_type, patterns in self.threat_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, data_str, re.IGNORECASE):
                        threat_score += 0.2
            
            return min(threat_score, 1.0)
            
        except Exception as e:
            logger.error(f"Threat assessment failed: {e}")
            return 0.5
    
    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        try:
            # Simple encryption for demo - in production use proper encryption
            encoded = base64.b64encode(data.encode()).decode()
            return f"encrypted_{encoded}"
        except Exception as e:
            logger.error(f"Data encryption failed: {e}")
            return data
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        try:
            if encrypted_data.startswith("encrypted_"):
                encoded = encrypted_data[10:]  # Remove "encrypted_" prefix
                return base64.b64decode(encoded).decode()
            return encrypted_data
        except Exception as e:
            logger.error(f"Data decryption failed: {e}")
            return encrypted_data
    
    def process_security_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process security-related request"""
        try:
            operation = request.get("operation", "validate")
            parameters = request.get("parameters", {})
            
            if operation == "validate":
                is_valid = self.validate_input(parameters)
                threat_level = self.assess_threat_level(parameters)
                
                return {
                    "operation": operation,
                    "is_valid": is_valid,
                    "threat_level": threat_level,
                    "timestamp": datetime.now().isoformat()
                }
            
            elif operation == "encrypt":
                data = parameters.get("data", "")
                encrypted = self.encrypt_data(data)
                
                return {
                    "operation": operation,
                    "encrypted_data": encrypted,
                    "timestamp": datetime.now().isoformat()
                }
            
            elif operation == "decrypt":
                encrypted_data = parameters.get("encrypted_data", "")
                decrypted = self.decrypt_data(encrypted_data)
                
                return {
                    "operation": operation,
                    "decrypted_data": decrypted,
                    "timestamp": datetime.now().isoformat()
                }
            
            else:
                return {
                    "operation": operation,
                    "error": "Unknown operation",
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Security request processing failed: {e}")
            return {
                "operation": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _log_security_event(self, event_type: str, **kwargs):
        """Log security event"""
        event = {
            "event_type": event_type,
            "timestamp": datetime.now().isoformat(),
            **kwargs
        }
        self.security_log.append(event)
        logger.warning(f"Security event: {event_type} - {kwargs}")
    
    def get_security_log(self) -> List[Dict[str, Any]]:
        """Get security event log"""
        return self.security_log.copy()
    
    def is_authorized(self, user_email: str) -> bool:
        """Check if user is authorized"""
        return user_email == self.authorized_user