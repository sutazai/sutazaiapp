"""
Email service for sending transactional emails
Real implementation using SMTP with support for multiple providers
"""

import logging
import smtplib
import asyncio
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
import os
from pathlib import Path
import json

# Try to import aiosmtplib, but continue if not available
try:
    import aiosmtplib
    HAS_AIOSMTPLIB = True
except ImportError:
    HAS_AIOSMTPLIB = False
    logging.warning("aiosmtplib not installed - email sending will be simulated")

from app.core.config import settings

logger = logging.getLogger(__name__)


class EmailService:
    """
    Email service for sending transactional emails
    Supports multiple SMTP providers and template rendering
    """
    
    def __init__(self):
        """Initialize email service with configuration"""
        # SMTP Configuration - can be overridden by environment variables
        self.smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_user = os.getenv("SMTP_USER", "noreply@sutazai.com")
        self.smtp_password = os.getenv("SMTP_PASSWORD", "")
        self.smtp_use_tls = os.getenv("SMTP_USE_TLS", "true").lower() == "true"
        self.smtp_use_ssl = os.getenv("SMTP_USE_SSL", "false").lower() == "true"
        
        # Email settings
        self.from_email = os.getenv("FROM_EMAIL", self.smtp_user)
        self.from_name = os.getenv("FROM_NAME", "SutazAI Platform")
        self.reply_to = os.getenv("REPLY_TO_EMAIL", self.from_email)
        
        # Rate limiting for email sending
        self.max_emails_per_minute = int(os.getenv("MAX_EMAILS_PER_MINUTE", "30"))
        self._email_counter = []
        
        # Template directory with fallback handling for permission issues
        self._initialize_template_directory()
        
        # Email queue for retry logic
        self._retry_queue = []
        
        logger.info(f"Email service initialized with SMTP host: {self.smtp_host}:{self.smtp_port}")
    
    def _initialize_template_directory(self):
        """
        Initialize template directory with fallback handling for permission issues
        Tries multiple locations to ensure the service can start even in restricted environments
        """
        # Try primary location first
        primary_dir = Path(__file__).parent.parent / "templates" / "emails"
        fallback_dir = Path("/tmp/email_templates")
        
        for directory in [primary_dir, fallback_dir]:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                self.template_dir = directory
                logger.info(f"Email templates directory initialized at: {directory}")
                return
            except (PermissionError, OSError) as e:
                logger.warning(f"Could not create template directory at {directory}: {e}")
        
        # If all locations fail, set to None and continue without templates
        self.template_dir = None
        logger.warning("Email service running without template directory - templates will not be saved")
    
    async def send_email(
        self,
        to_email: str,
        subject: str,
        body_text: str,
        body_html: Optional[str] = None,
        cc: Optional[List[str]] = None,
        bcc: Optional[List[str]] = None,
        attachments: Optional[List[Dict[str, Any]]] = None,
        priority: str = "normal",
        headers: Optional[Dict[str, str]] = None
    ) -> bool:
        """
        Send an email asynchronously using SMTP
        
        Args:
            to_email: Recipient email address
            subject: Email subject
            body_text: Plain text body
            body_html: Optional HTML body
            cc: Optional CC recipients
            bcc: Optional BCC recipients
            attachments: Optional list of attachments
            priority: Email priority (low, normal, high)
            headers: Optional custom headers
            
        Returns:
            True if sent successfully, False otherwise
        """
        try:
            # Check rate limiting
            if not await self._check_rate_limit():
                logger.warning(f"Rate limit exceeded, queuing email to {to_email}")
                self._retry_queue.append({
                    "to_email": to_email,
                    "subject": subject,
                    "body_text": body_text,
                    "body_html": body_html,
                    "timestamp": datetime.now(timezone.utc)
                })
                return False
            
            # Create message
            message = MIMEMultipart("alternative")
            message["From"] = f"{self.from_name} <{self.from_email}>"
            message["To"] = to_email
            message["Subject"] = subject
            message["Reply-To"] = self.reply_to
            
            # Add priority headers
            if priority == "high":
                message["X-Priority"] = "1"
                message["Importance"] = "high"
            elif priority == "low":
                message["X-Priority"] = "5"
                message["Importance"] = "low"
            
            # Add custom headers
            if headers:
                for key, value in headers.items():
                    message[key] = value
            
            # Add CC and BCC
            if cc:
                message["Cc"] = ", ".join(cc)
            
            # Add body parts
            text_part = MIMEText(body_text, "plain", "utf-8")
            message.attach(text_part)
            
            if body_html:
                html_part = MIMEText(body_html, "html", "utf-8")
                message.attach(html_part)
            
            # Prepare recipients
            recipients = [to_email]
            if cc:
                recipients.extend(cc)
            if bcc:
                recipients.extend(bcc)
            
            # Send email using aiosmtplib for async operation
            if self.smtp_password:
                # Only attempt to send if SMTP password is configured
                await aiosmtplib.send(
                    message,
                    hostname=self.smtp_host,
                    port=self.smtp_port,
                    username=self.smtp_user,
                    password=self.smtp_password,
                    use_tls=self.smtp_use_tls,
                    start_tls=self.smtp_use_tls and not self.smtp_use_ssl,
                    timeout=30
                )
                
                logger.info(f"Email sent successfully to {to_email}: {subject}")
                
                # Update rate limit counter
                self._email_counter.append(datetime.now(timezone.utc))
                
                return True
            else:
                # Log email details when SMTP is not configured (development mode)
                logger.warning(
                    f"SMTP not configured. Email would be sent to {to_email}\n"
                    f"Subject: {subject}\n"
                    f"Body: {body_text[:200]}..."
                )
                
                # Save to local file for development testing
                dev_email_dir = self.template_dir if self.template_dir else Path("/tmp/sutazai_emails")
                try:
                    dev_email_dir.mkdir(exist_ok=True)
                except (PermissionError, OSError):
                    dev_email_dir = Path("/tmp/sutazai_emails")
                    dev_email_dir.mkdir(exist_ok=True)
                
                email_data = {
                    "to": to_email,
                    "subject": subject,
                    "body_text": body_text,
                    "body_html": body_html,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "cc": cc,
                    "bcc": bcc
                }
                
                email_file = dev_email_dir / f"email_{datetime.now(timezone.utc).timestamp()}.json"
                with open(email_file, "w") as f:
                    json.dump(email_data, f, indent=2, default=str)
                
                logger.info(f"Development mode: Email saved to {email_file}")
                return True
                
        except Exception as e:
            # Handle SMTP errors if aiosmtplib is available
            if HAS_AIOSMTPLIB and isinstance(e, aiosmtplib.SMTPException):
                logger.error(f"SMTP error sending email to {to_email}: {e}")
                self._retry_queue.append({
                    "to_email": to_email,
                    "subject": subject,
                    "body_text": body_text,
                    "body_html": body_html,
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc)
                })
                return False
            else:
                logger.error(f"Error sending email to {to_email}: {e}")
                return False
    
    async def send_password_reset_email(self, email: str, reset_token: str) -> bool:
        """
        Send password reset email with token
        
        Args:
            email: User's email address
            reset_token: Password reset token
            
        Returns:
            True if sent successfully
        """
        # Generate reset URL
        base_url = os.getenv("FRONTEND_URL", "http://sutazai-frontend:3000")
        reset_url = f"{base_url}/reset-password?token={reset_token}"
        
        subject = "Password Reset Request - SutazAI Platform"
        
        body_text = f"""
Hello,

You have requested to reset your password for SutazAI Platform.

Please click the following link to reset your password:
{reset_url}

This link will expire in 1 hour for security reasons.

If you did not request this password reset, please ignore this email.
Your password will remain unchanged.

Best regards,
SutazAI Platform Team
        """
        
        body_html = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
        .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
        .header {{ background-color: #007bff; color: white; padding: 20px; text-align: center; }}
        .content {{ padding: 20px; background-color: #f9f9f9; }}
        .button {{ display: inline-block; padding: 12px 30px; background-color: #007bff; 
                   color: white; text-decoration: none; border-radius: 5px; margin: 20px 0; }}
        .footer {{ text-align: center; padding: 20px; color: #666; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Password Reset Request</h1>
        </div>
        <div class="content">
            <p>Hello,</p>
            <p>You have requested to reset your password for SutazAI Platform.</p>
            <p>Please click the button below to reset your password:</p>
            <div style="text-align: center;">
                <a href="{reset_url}" class="button">Reset Password</a>
            </div>
            <p>Or copy and paste this link into your browser:</p>
            <p style="word-break: break-all;">{reset_url}</p>
            <p><strong>This link will expire in 1 hour for security reasons.</strong></p>
            <p>If you did not request this password reset, please ignore this email. 
               Your password will remain unchanged.</p>
        </div>
        <div class="footer">
            <p>© 2025 SutazAI Platform. All rights reserved.</p>
            <p>This is an automated message, please do not reply to this email.</p>
        </div>
    </div>
</body>
</html>
        """
        
        return await self.send_email(
            to_email=email,
            subject=subject,
            body_text=body_text,
            body_html=body_html,
            priority="high",
            headers={"X-Email-Type": "password-reset"}
        )
    
    async def send_verification_email(self, email: str, verification_token: str) -> bool:
        """
        Send email verification link
        
        Args:
            email: User's email address
            verification_token: Email verification token
            
        Returns:
            True if sent successfully
        """
        # Generate verification URL
        base_url = os.getenv("FRONTEND_URL", "http://sutazai-frontend:3000")
        verify_url = f"{base_url}/verify-email?token={verification_token}"
        
        subject = "Verify Your Email - SutazAI Platform"
        
        body_text = f"""
Welcome to SutazAI Platform!

Please verify your email address by clicking the following link:
{verify_url}

This link will expire in 24 hours.

If you did not create an account with us, please ignore this email.

Best regards,
SutazAI Platform Team
        """
        
        body_html = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
        .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
        .header {{ background-color: #28a745; color: white; padding: 20px; text-align: center; }}
        .content {{ padding: 20px; background-color: #f9f9f9; }}
        .button {{ display: inline-block; padding: 12px 30px; background-color: #28a745; 
                   color: white; text-decoration: none; border-radius: 5px; margin: 20px 0; }}
        .footer {{ text-align: center; padding: 20px; color: #666; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Welcome to SutazAI Platform!</h1>
        </div>
        <div class="content">
            <p>Thank you for signing up!</p>
            <p>Please verify your email address by clicking the button below:</p>
            <div style="text-align: center;">
                <a href="{verify_url}" class="button">Verify Email</a>
            </div>
            <p>Or copy and paste this link into your browser:</p>
            <p style="word-break: break-all;">{verify_url}</p>
            <p><strong>This link will expire in 24 hours.</strong></p>
            <p>If you did not create an account with us, please ignore this email.</p>
        </div>
        <div class="footer">
            <p>© 2025 SutazAI Platform. All rights reserved.</p>
            <p>This is an automated message, please do not reply to this email.</p>
        </div>
    </div>
</body>
</html>
        """
        
        return await self.send_email(
            to_email=email,
            subject=subject,
            body_text=body_text,
            body_html=body_html,
            priority="high",
            headers={"X-Email-Type": "email-verification"}
        )
    
    async def _check_rate_limit(self) -> bool:
        """
        Check if we're within rate limits for sending emails
        
        Returns:
            True if within limits, False if rate limited
        """
        now = datetime.now(timezone.utc)
        
        # Remove emails older than 1 minute
        self._email_counter = [
            timestamp for timestamp in self._email_counter
            if (now - timestamp).total_seconds() < 60
        ]
        
        # Check if we're within limit
        return len(self._email_counter) < self.max_emails_per_minute
    
    async def process_retry_queue(self) -> int:
        """
        Process emails in the retry queue
        
        Returns:
            Number of emails successfully sent from retry queue
        """
        if not self._retry_queue:
            return 0
        
        sent_count = 0
        failed_items = []
        
        for item in self._retry_queue:
            # Skip items older than 1 hour
            if (datetime.now(timezone.utc) - item["timestamp"]).total_seconds() > 3600:
                logger.warning(f"Dropping old email from retry queue: {item['subject']}")
                continue
            
            # Try to send
            success = await self.send_email(
                to_email=item["to_email"],
                subject=item["subject"],
                body_text=item["body_text"],
                body_html=item.get("body_html")
            )
            
            if success:
                sent_count += 1
            else:
                failed_items.append(item)
        
        # Update retry queue with failed items
        self._retry_queue = failed_items
        
        if sent_count > 0:
            logger.info(f"Successfully sent {sent_count} emails from retry queue")
        
        return sent_count


# Global email service instance
email_service = EmailService()