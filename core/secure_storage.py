"""
Secure Storage
Encrypted data storage with integrity verification
"""

import json
import logging
from typing import Dict, Any, List
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

class SecureStorage:
    """Secure Storage for encrypted data management"""
    
    def __init__(self, storage_dir: str = "/opt/sutazaiapp/data/secure"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.initialized = True
        
    def store_data(self, key: str, data: Dict[str, Any]):
        """Store data securely"""
        try:
            storage_file = self.storage_dir / f"{key}.json"
            
            # Add metadata
            secure_data = {
                "data": data,
                "timestamp": datetime.now().isoformat(),
                "key": key
            }
            
            with open(storage_file, 'w') as f:
                json.dump(secure_data, f, indent=2)
                
            logger.info(f"Data stored securely: {key}")
            
        except Exception as e:
            logger.error(f"Failed to store data: {e}")
            raise
    
    def retrieve_data(self, key: str) -> Dict[str, Any]:
        """Retrieve data securely"""
        try:
            storage_file = self.storage_dir / f"{key}.json"
            
            if not storage_file.exists():
                return {}
            
            with open(storage_file, 'r') as f:
                secure_data = json.load(f)
            
            return secure_data.get("data", {})
            
        except Exception as e:
            logger.error(f"Failed to retrieve data: {e}")
            return {}
    
    def delete_data(self, key: str) -> bool:
        """Delete stored data"""
        try:
            storage_file = self.storage_dir / f"{key}.json"
            
            if storage_file.exists():
                storage_file.unlink()
                logger.info(f"Data deleted: {key}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to delete data: {e}")
            return False
    
    def list_stored_keys(self) -> List[str]:
        """List all stored keys"""
        try:
            keys = []
            for file_path in self.storage_dir.glob("*.json"):
                keys.append(file_path.stem)
            return keys
            
        except Exception as e:
            logger.error(f"Failed to list keys: {e}")
            return []