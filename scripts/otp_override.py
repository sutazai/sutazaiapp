import os
import sys
import functools
import time
import logging
import pyotp
import json
from typing import Callable, Any, Dict, Optional

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from otp_manager import OTPManager
from cryptography.fernet import Fernet

class OTPValidator:
    def __init__(self, config_path="/opt/sutazaiapp/config/external_calls.toml"):
        """
        Initialize OTP validator with configuration
        
        Args:
            config_path (str): Path to external calls configuration
        """
        self.otp_manager = OTPManager()
        self.config_path = config_path
        self.external_calls_enabled = self._load_config()
    
    def _load_config(self) -> bool:
        """
        Load external calls configuration
        
        Returns:
            bool: Whether external calls are enabled
        """
        try:
            import toml
            config = toml.load(self.config_path)
            return config.get('security', {}).get('external_calls_enabled', False)
        except Exception as e:
            print(f"Config load error: {e}")
            return False
    
    def validate_external_call(self, func: Callable) -> Callable:
        """
        Decorator to validate external calls with OTP
        
        Args:
            func (Callable): Function to be decorated
        
        Returns:
            Callable: Decorated function with OTP validation
        """
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # If external calls disabled, block
            if not self.external_calls_enabled:
                raise PermissionError("External calls are disabled")
            
            # Prompt for OTP
            print("External call requires OTP validation.")
            user_otp = input("Enter OTP: ")
            
            # Validate OTP
            last_otp_info = {}  # Replace with secure OTP retrieval
            if not self.otp_manager.validate_otp(user_otp, last_otp_info):
                raise ValueError("Invalid OTP. External call blocked.")
            
            return func(*args, **kwargs)
        
        return wrapper

class OTPManager:
    """
    Comprehensive OTP management system with enhanced security
    """
    
    def __init__(
        self, 
        secret_dir: str = "/opt/sutazaiapp/secrets",
        log_dir: str = "/var/log/sutazaiapp",
        validity_window: int = 60
    ):
        """
        Initialize OTP Manager
        
        Args:
            secret_dir (str): Directory to store encryption keys
            log_dir (str): Directory for logging OTP attempts
            validity_window (int): OTP validity window in seconds
        """
        self.secret_dir = secret_dir
        self.log_dir = log_dir
        self.validity_window = validity_window
        
        # Ensure directories exist
        os.makedirs(secret_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            filename=os.path.join(log_dir, "otp_validation.log"),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s: %(message)s'
        )
        
        # Load or generate encryption key
        self._load_or_generate_key()
        
        # Load or generate TOTP secret
        self._load_or_generate_totp_secret()
    
    def _load_or_generate_key(self):
        """
        Load or generate encryption key for sensitive data
        """
        key_path = os.path.join(self.secret_dir, "encryption.key")
        
        try:
            if os.path.exists(key_path):
                with open(key_path, 'rb') as f:
                    self.encryption_key = f.read()
            else:
                self.encryption_key = Fernet.generate_key()
                with open(key_path, 'wb') as f:
                    f.write(self.encryption_key)
            
            self.cipher_suite = Fernet(self.encryption_key)
        except Exception as e:
            logging.error(f"Encryption key generation error: {e}")
            raise
    
    def _load_or_generate_totp_secret(self):
        """
        Load or generate TOTP secret
        """
        secret_path = os.path.join(self.secret_dir, "totp_secret.json")
        
        try:
            if os.path.exists(secret_path):
                with open(secret_path, 'rb') as f:
                    encrypted_data = f.read()
                    decrypted_data = self.cipher_suite.decrypt(encrypted_data)
                    secret_data = json.loads(decrypted_data)
                    self.totp_secret = secret_data['secret']
            else:
                # Generate new TOTP secret
                self.totp_secret = pyotp.random_base32()
                secret_data = {'secret': self.totp_secret}
                
                # Encrypt and save
                encrypted_data = self.cipher_suite.encrypt(
                    json.dumps(secret_data).encode()
                )
                with open(secret_path, 'wb') as f:
                    f.write(encrypted_data)
            
            self.totp = pyotp.TOTP(self.totp_secret)
        except Exception as e:
            logging.error(f"TOTP secret generation error: {e}")
            raise
    
    def generate_otp(self) -> str:
        """
        Generate a new OTP
        
        Returns:
            str: Generated OTP
        """
        otp = self.totp.now()
        logging.info("OTP generated")
        return otp
    
    def validate_otp(self, user_otp: str) -> bool:
        """
        Validate OTP with time-based window
        
        Args:
            user_otp (str): OTP provided by user
        
        Returns:
            bool: Whether OTP is valid
        """
        try:
            # Validate OTP with a time window
            is_valid = self.totp.verify(
                user_otp, 
                valid_window=self.validity_window // 30
            )
            
            # Log validation attempt
            log_entry = {
                'timestamp': time.time(),
                'otp_valid': is_valid
            }
            
            with open(os.path.join(self.log_dir, "otp_attempts.json"), 'a') as f:
                json.dump(log_entry, f)
                f.write('\n')
            
            return is_valid
        
        except Exception as e:
            logging.error(f"OTP validation error: {e}")
            return False
    
    def get_provisioning_uri(self, account_name: str = "SutazAI") -> str:
        """
        Generate provisioning URI for authenticator apps
        
        Args:
            account_name (str): Account name for QR code
        
        Returns:
            str: Provisioning URI
        """
        return self.totp.provisioning_uri(
            name=account_name, 
            issuer_name="SutazAI Orchestrator"
        )

def main():
    """
    Example usage and testing
    """
    otp_manager = OTPManager()
    
    # Generate OTP
    current_otp = otp_manager.generate_otp()
    print(f"Current OTP: {current_otp}")
    
    # Validate OTP
    print("Validation Result:", otp_manager.validate_otp(current_otp))
    
    # Get provisioning URI (for QR code)
    print("Provisioning URI:", otp_manager.get_provisioning_uri())

if __name__ == "__main__":
    main() 