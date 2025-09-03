#!/usr/bin/env python3
"""
Test email service functionality
"""
import asyncio
import sys
import os
sys.path.insert(0, '/opt/sutazaiapp/backend')

from app.services.email import email_service

async def test_email_service():
    """Test email sending"""
    print("Testing Email Service")
    print("=" * 50)
    
    # Test verification email
    print("\n1. Testing verification email...")
    result = await email_service.send_verification_email(
        email="test@example.com",
        verification_token="test-verification-token-123"
    )
    print(f"Verification email result: {result}")
    
    # Test password reset email
    print("\n2. Testing password reset email...")
    result = await email_service.send_password_reset_email(
        email="test@example.com",
        reset_token="test-reset-token-456"
    )
    print(f"Password reset email result: {result}")
    
    # Check if emails were saved locally
    print("\n3. Checking saved emails...")
    email_dir = "/tmp/sutazai_emails"
    if os.path.exists(email_dir):
        files = os.listdir(email_dir)
        print(f"Found {len(files)} saved email(s)")
        for file in files[:5]:  # Show first 5
            print(f"  - {file}")
    else:
        print("No saved emails directory found")

if __name__ == "__main__":
    asyncio.run(test_email_service())