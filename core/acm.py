"""
Authorization Control Module
Secure authorization and access control
"""

import logging
from typing import Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)

class AuthorizationControlModule:
    """Authorization Control Module for secure access control"""
    
    def __init__(self):
        self.initialized = True
        self.authorized_user = "chrissuta01@gmail.com"
        self.access_log = []
        
    def is_authorized(self, user_email: str) -> bool:
        """Check if user is authorized"""
        authorized = user_email == self.authorized_user
        
        # Log access attempt
        self.access_log.append({
            "user": user_email,
            "authorized": authorized,
            "timestamp": datetime.now().isoformat()
        })
        
        if not authorized:
            logger.warning(f"Unauthorized access attempt by: {user_email}")
        
        return authorized
    
    def emergency_shutdown(self, user_email: str) -> bool:
        """Emergency shutdown - only authorized user"""
        if not self.is_authorized(user_email):
            return False
        
        logger.info(f"Emergency shutdown initiated by: {user_email}")
        return True
    
    def get_access_log(self) -> List[Dict[str, Any]]:
        """Get access log"""
        return self.access_log.copy()
    
    def get_authorization_stats(self) -> Dict[str, Any]:
        """Get authorization statistics"""
        return {
            "authorized_user": self.authorized_user,
            "total_access_attempts": len(self.access_log),
            "timestamp": datetime.now().isoformat()
        }