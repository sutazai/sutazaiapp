#!/usr/bin/env python3
"""
OTP (One-Time Password) Handler for SutazAI
Manages OTP generation, validation, and email delivery
"""

import secrets
import string
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
import redis
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib
from backend.config import Config
from loguru import logger

config = Config()

# Redis client for OTP storage
try:
    redis_client = redis.Redis(
        host=config.redis.host,
        port=config.redis.port,
        password=config.redis.password,
        db=config.redis.db,
        decode_responses=True
    )
except Exception as e:
    logger.warning(f"Redis connection failed, using in-memory storage: {e}")
    redis_client = None

# In-memory fallback for OTP storage
_otp_storage: Dict[str, Dict[str, Any]] = {}


def generate_otp(length: int = 6) -> str:
    """Generate a random OTP"""
    if length < 4 or length > 10:
        length = 6
    
    # Generate numeric OTP
    digits = string.digits
    otp = ''.join(secrets.choice(digits) for _ in range(length))
    
    logger.info(f"Generated OTP of length {length}")
    return otp


def store_otp(identifier: str, otp: str, expires_in_minutes: int = 10) -> bool:
    """Store OTP with expiration"""
    try:
        expiry_time = datetime.now(timezone.utc) + timedelta(minutes=expires_in_minutes)
        
        otp_data = {
            "otp": otp,
            "expires_at": expiry_time.isoformat(),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "attempts": 0
        }
        
        if redis_client:
            # Store in Redis with TTL
            key = f"otp:{identifier}"
            redis_client.hset(key, mapping=otp_data)
            redis_client.expire(key, expires_in_minutes * 60)
        else:
            # Store in memory
            _otp_storage[identifier] = otp_data
        
        logger.info(f"Stored OTP for {identifier}, expires in {expires_in_minutes} minutes")
        return True
    
    except Exception as e:
        logger.error(f"Failed to store OTP for {identifier}: {e}")
        return False


def get_otp_data(identifier: str) -> Optional[Dict[str, Any]]:
    """Retrieve OTP data"""
    try:
        if redis_client:
            key = f"otp:{identifier}"
            data = redis_client.hgetall(key)
            if not data:
                return None
            return data
        else:
            return _otp_storage.get(identifier)
    
    except Exception as e:
        logger.error(f"Failed to retrieve OTP data for {identifier}: {e}")
        return None


def delete_otp(identifier: str) -> bool:
    """Delete OTP data"""
    try:
        if redis_client:
            key = f"otp:{identifier}"
            redis_client.delete(key)
        else:
            _otp_storage.pop(identifier, None)
        
        logger.info(f"Deleted OTP for {identifier}")
        return True
    
    except Exception as e:
        logger.error(f"Failed to delete OTP for {identifier}: {e}")
        return False


def validate_otp(identifier: str, provided_otp: str, max_attempts: int = 3) -> Dict[str, Any]:
    """
    Validate OTP
    Returns dict with 'valid', 'message', and optional 'remaining_attempts'
    """
    otp_data = get_otp_data(identifier)
    
    if not otp_data:
        return {
            "valid": False,
            "message": "OTP not found or expired",
            "code": "OTP_NOT_FOUND"
        }
    
    try:
        # Check expiration
        expires_at = datetime.fromisoformat(otp_data["expires_at"])
        if datetime.now(timezone.utc) > expires_at:
            delete_otp(identifier)
            return {
                "valid": False,
                "message": "OTP has expired",
                "code": "OTP_EXPIRED"
            }
        
        # Check attempts
        attempts = int(otp_data.get("attempts", 0))
        if attempts >= max_attempts:
            delete_otp(identifier)
            return {
                "valid": False,
                "message": "Too many failed attempts",
                "code": "TOO_MANY_ATTEMPTS"
            }
        
        # Validate OTP
        stored_otp = otp_data["otp"]
        if provided_otp == stored_otp:
            delete_otp(identifier)
            logger.info(f"OTP validated successfully for {identifier}")
            return {
                "valid": True,
                "message": "OTP is valid",
                "code": "OTP_VALID"
            }
        else:
            # Increment attempts
            attempts += 1
            otp_data["attempts"] = str(attempts)
            
            if redis_client:
                key = f"otp:{identifier}"
                redis_client.hset(key, "attempts", attempts)
            else:
                _otp_storage[identifier] = otp_data
            
            remaining_attempts = max_attempts - attempts
            logger.warning(f"Invalid OTP for {identifier}, {remaining_attempts} attempts remaining")
            
            return {
                "valid": False,
                "message": f"Invalid OTP. {remaining_attempts} attempts remaining",
                "code": "OTP_INVALID",
                "remaining_attempts": remaining_attempts
            }
    
    except Exception as e:
        logger.error(f"Error validating OTP for {identifier}: {e}")
        return {
            "valid": False,
            "message": "Error validating OTP",
            "code": "VALIDATION_ERROR"
        }


def send_otp_email(email: str, otp: str, purpose: str = "verification") -> bool:
    """Send OTP via email"""
    try:
        if not hasattr(config, 'email') or not config.email.smtp_host:
            logger.warning("Email configuration not found, cannot send OTP")
            return False
        
        # Create message
        msg = MIMEMultipart()
        msg['From'] = config.email.from_address
        msg['To'] = email
        msg['Subject'] = f"SutazAI {purpose.title()} Code"
        
        # Email body
        body = f"""
        <html>
        <body>
        <h2>SutazAI {purpose.title()} Code</h2>
        <p>Your verification code is:</p>
        <h1 style="color: #007bff; font-family: monospace; letter-spacing: 3px;">{otp}</h1>
        <p>This code will expire in 10 minutes.</p>
        <p>If you did not request this code, please ignore this email.</p>
        <br>
        <p>Best regards,<br>SutazAI Team</p>
        </body>
        </html>
        """
        
        msg.attach(MIMEText(body, 'html'))
        
        # Send email
        server = smtplib.SMTP(config.email.smtp_host, config.email.smtp_port)
        
        if config.email.use_tls:
            server.starttls()
        
        if config.email.username and config.email.password:
            server.login(config.email.username, config.email.password)
        
        server.send_message(msg)
        server.quit()
        
        logger.info(f"OTP sent to {email}")
        return True
    
    except Exception as e:
        logger.error(f"Failed to send OTP email to {email}: {e}")
        return False


def send_otp_sms(phone: str, otp: str, purpose: str = "verification") -> bool:
    """Send OTP via SMS (placeholder implementation)"""
    # TODO: Implement SMS sending using services like Twilio, AWS SNS, etc.
    logger.info(f"SMS OTP would be sent to {phone}: {otp}")
    return True


def generate_and_send_otp(
    identifier: str,
    email: str = None,
    phone: str = None,
    purpose: str = "verification",
    method: str = "email"
) -> Dict[str, Any]:
    """Generate OTP and send it via specified method"""
    try:
        # Generate OTP
        otp = generate_otp()
        
        # Store OTP
        if not store_otp(identifier, otp):
            return {
                "success": False,
                "message": "Failed to store OTP",
                "code": "STORAGE_ERROR"
            }
        
        # Send OTP
        sent = False
        if method == "email" and email:
            sent = send_otp_email(email, otp, purpose)
        elif method == "sms" and phone:
            sent = send_otp_sms(phone, otp, purpose)
        
        if sent:
            return {
                "success": True,
                "message": f"OTP sent via {method}",
                "code": "OTP_SENT"
            }
        else:
            delete_otp(identifier)
            return {
                "success": False,
                "message": f"Failed to send OTP via {method}",
                "code": "SEND_ERROR"
            }
    
    except Exception as e:
        logger.error(f"Error generating and sending OTP: {e}")
        return {
            "success": False,
            "message": "Error processing OTP request",
            "code": "PROCESSING_ERROR"
        }


def cleanup_expired_otps() -> int:
    """Cleanup expired OTPs from in-memory storage"""
    if redis_client:
        return 0  # Redis handles TTL automatically
    
    expired_count = 0
    current_time = datetime.now(timezone.utc)
    
    for identifier in list(_otp_storage.keys()):
        otp_data = _otp_storage[identifier]
        expires_at = datetime.fromisoformat(otp_data["expires_at"])
        
        if current_time > expires_at:
            del _otp_storage[identifier]
            expired_count += 1
    
    if expired_count > 0:
        logger.info(f"Cleaned up {expired_count} expired OTPs")
    
    return expired_count