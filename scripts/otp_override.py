#!/usr/bin/env python3
"""
otp_override.py - Script for overriding OTP verification in development/testing environments
"""

import argparse
import os
import sys
import json
import hashlib
import time
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("otp_override")

# Constants
DEFAULT_OTP_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".sutazai", "otp_cache")
DEFAULT_OTP_CODE = "123456"  # Default OTP for development


def setup_cache_dir(cache_dir):
    """Create the OTP cache directory if it doesn't exist."""
    try:
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f"OTP cache directory: {cache_dir}")
        return True
    except Exception as e:
        logger.error(f"Failed to create OTP cache directory: {e}")
        return False


def generate_user_hash(user_id):
    """Generate a deterministic hash for a user ID."""
    return hashlib.sha256(user_id.encode()).hexdigest()


def store_otp(cache_dir, user_id, otp_code, expires_in=300):
    """Store OTP code for a user with expiration time."""
    user_hash = generate_user_hash(user_id)
    cache_file = os.path.join(cache_dir, f"{user_hash}.json")

    otp_data = {
        "user_id": user_id,
        "otp_code": otp_code,
        "created_at": int(time.time()),
        "expires_at": int(time.time()) + expires_in,
    }

    try:
        with open(cache_file, "w") as f:
            json.dump(otp_data, f)
        logger.info(f"Stored OTP code for user: {user_id}")
        logger.info(f"OTP code: {otp_code}")
        return True
    except Exception as e:
        logger.error(f"Failed to store OTP: {e}")
        return False


def verify_otp(cache_dir, user_id, otp_code):
    """Verify the OTP code for a user."""
    user_hash = generate_user_hash(user_id)
    cache_file = os.path.join(cache_dir, f"{user_hash}.json")

    if not os.path.exists(cache_file):
        logger.error(f"No OTP found for user: {user_id}")
        return False

    try:
        with open(cache_file, "r") as f:
            otp_data = json.load(f)

        # Check if OTP has expired
        current_time = int(time.time())
        if current_time > otp_data["expires_at"]:
            logger.error(f"OTP has expired for user: {user_id}")
            os.remove(cache_file)
            return False

        # Check if OTP matches
        if otp_data["otp_code"] == otp_code:
            logger.info(f"OTP verified successfully for user: {user_id}")
            os.remove(cache_file)
            return True
        else:
            logger.error(f"Invalid OTP for user: {user_id}")
            return False

    except Exception as e:
        logger.error(f"Failed to verify OTP: {e}")
        return False


def generate_otp(user_id, override_code=None):
    """Generate or return override OTP code."""
    if override_code:
        return override_code

    # In development/testing mode, use a fixed OTP
    if os.environ.get("ENVIRONMENT", "development").lower() in [
        "development",
        "testing",
    ]:
        return DEFAULT_OTP_CODE

    # For production, use a more secure method or integrate with an actual OTP service
    # This is just a placeholder for demonstration
    import random

    return str(random.randint(100000, 999999))  # nosec B311 - Test OTP, not security critical


def main():
    """Main function to parse arguments and execute commands."""
    parser = argparse.ArgumentParser(
        description="OTP Override Tool for Development and Testing"
    )
    parser.add_argument(
        "--cache-dir", default=DEFAULT_OTP_CACHE_DIR, help="OTP cache directory"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Generate OTP command
    generate_parser = subparsers.add_parser("generate", help="Generate OTP code")
    generate_parser.add_argument("user_id", help="User ID or email")
    generate_parser.add_argument("--code", help="Override OTP code")
    generate_parser.add_argument(
        "--expires", type=int, default=300, help="Expiration time in seconds"
    )

    # Verify OTP command
    verify_parser = subparsers.add_parser("verify", help="Verify OTP code")
    verify_parser.add_argument("user_id", help="User ID or email")
    verify_parser.add_argument("otp_code", help="OTP code to verify")

    # Clear OTP cache command
    clear_parser = subparsers.add_parser("clear", help="Clear OTP cache")
    clear_parser.add_argument("--user-id", help="User ID to clear (optional)")

    args = parser.parse_args()

    # Create cache directory
    if not setup_cache_dir(args.cache_dir):
        sys.exit(1)

    # Execute commands
    if args.command == "generate":
        otp_code = generate_otp(args.user_id, args.code)
        if store_otp(args.cache_dir, args.user_id, otp_code, args.expires):
            sys.exit(0)
        else:
            sys.exit(1)

    elif args.command == "verify":
        if verify_otp(args.cache_dir, args.user_id, args.otp_code):
            sys.exit(0)
        else:
            sys.exit(1)

    elif args.command == "clear":
        if args.user_id:
            user_hash = generate_user_hash(args.user_id)
            cache_file = os.path.join(args.cache_dir, f"{user_hash}.json")
            if os.path.exists(cache_file):
                os.remove(cache_file)
                logger.info(f"Cleared OTP for user: {args.user_id}")
            else:
                logger.warning(f"No OTP found for user: {args.user_id}")
        else:
            for file in Path(args.cache_dir).glob("*.json"):
                os.remove(file)
            logger.info("Cleared all OTP cache files")
        sys.exit(0)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
