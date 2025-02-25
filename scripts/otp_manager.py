import logging
import os
import secrets
import smtplib
from datetime import datetime, timedelta
from email.mime.text import MIMEText

from cryptography.fernet import Fernet


class OTPManager:
    def __init__(
        self, root_email="chrissuta01@gmail.com", otp_expiry_minutes=15
    ):
        """
        Initialize OTP Management System

        Args:
            root_email (str): Root user's email for OTP communication
            otp_expiry_minutes (int): OTP validity duration
        """
        self.root_email = root_email
        self.otp_expiry_minutes = otp_expiry_minutes

        # Secure key generation and storage
        self.encryption_key = self._generate_encryption_key()
        self.cipher_suite = Fernet(self.encryption_key)

        # Logging configuration
        logging.basicConfig(
            filename="/opt/sutazai_project/SutazAI/logs/otp_manager.log",
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s: %(message)s",
        )

    def _generate_encryption_key(self):
        """Generate a secure encryption key"""

        if os.path.exists(key_path):
            with open(key_path, "rb") as key_file:
                return key_file.read()

        key = Fernet.generate_key()
        with open(key_path, "wb") as key_file:
            key_file.write(key)

        os.chmod(key_path, 0o600)  # Restrict file permissions
        return key

    def generate_otp(self):
        """
        Generate a secure, time-limited OTP

        Returns:
            dict: OTP details with encrypted OTP and generation timestamp
        """
        otp = secrets.token_urlsafe(6)[:6].upper()
        timestamp = datetime.now()

        # Encrypt OTP for secure storage
        encrypted_otp = self.cipher_suite.encrypt(otp.encode()).decode()

        logging.info(f"OTP generated for root user: {self.root_email}")

        return {"encrypted_otp": encrypted_otp, "timestamp": timestamp}

    def validate_otp(self, user_otp, stored_otp_info):
        """
        Validate the provided OTP

        Args:
            user_otp (str): User-provided OTP
            stored_otp_info (dict): Previously generated OTP information

        Returns:
            bool: OTP validation result
        """
        try:
            # Decrypt stored OTP
            decrypted_otp = self.cipher_suite.decrypt(
                stored_otp_info["encrypted_otp"].encode()
            ).decode()

            # Check OTP expiry
            current_time = datetime.now()
            otp_generation_time = stored_otp_info["timestamp"]

            if (current_time - otp_generation_time) > timedelta(
                minutes=self.otp_expiry_minutes
            ):
                logging.warning("OTP expired")
                return False

            # Compare OTPs
            is_valid = secrets.compare_digest(user_otp.upper(), decrypted_otp)

            if is_valid:
                logging.info("OTP successfully validated")
            else:
                logging.warning("Invalid OTP attempt")

            return is_valid

        except Exception as e:
            logging.error(f"OTP validation error: {e}")
            return False

    def send_otp_email(self, otp):
        """
        Send OTP via email

        Args:
            otp (str): Generated OTP
        """
        try:
            msg = MIMEText(f"Your SutazAI Network Access OTP is: {otp}")
            msg["Subject"] = "SutazAI Network Access OTP"
            msg["From"] = "sutazai@system.com"
            msg["To"] = self.root_email

            # Email configuration (replace with actual SMTP details)
            smtp_server = "localhost"
            smtp_port = 25

            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.sendmail(msg["From"], [msg["To"]], msg.as_string())

            logging.info(f"OTP email sent to {self.root_email}")

        except Exception as e:
            logging.error(f"Email sending failed: {e}")


def main():
    """Demonstration of OTP management workflow"""
    otp_manager = OTPManager()

    # Generate OTP
    otp_info = otp_manager.generate_otp()
    decrypted_otp = otp_manager.cipher_suite.decrypt(
        otp_info["encrypted_otp"].encode()
    ).decode()

    # Send OTP
    otp_manager.send_otp_email(decrypted_otp)

    # Simulate validation (replace with actual user input)
    validation_result = otp_manager.validate_otp(decrypted_otp, otp_info)

    print(f"OTP Validation Result: {validation_result}")


if __name__ == "__main__":
    main()
