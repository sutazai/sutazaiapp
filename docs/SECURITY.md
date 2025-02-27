# Sutazaiapp Security Architecture

## OTP Gating System

### Overview
- Time-based One-Time Password (TOTP) validation
- Offline-first security approach
- Comprehensive external call protection

### Key Components
1. **OTP Generation**
   - Uses `pyotp` for secure TOTP generation
   - Encrypted secret storage
   - Configurable validity window

2. **External Call Validation**
   - All deployment and critical operations require OTP
   - Decorator-based validation mechanism
   - Logging of validation attempts

### SSH Key Management
- Ed25519 keys recommended
- Minimum 4096-bit RSA as alternative
- Strict permissions (600 for private keys)

### Deployment Security
- OTP required for each deployment stage
- Automatic rollback on unauthorized access
- Comprehensive logging of blocked attempts

## Best Practices

### Key Rotation
```bash
# Generate new OTP secret
python scripts/otp_override.py --rotate-secret

# Regenerate SSH keys annually
ssh-keygen -t ed25519 -f ~/.ssh/sutazaiapp_key
```

### Permissions Management
```bash
# Secure key permissions
chmod 600 ~/.ssh/sutazaiapp_key
chmod 644 ~/.ssh/sutazaiapp_key.pub
```

## Threat Mitigation
- Rate-limited OTP attempts
- Encrypted sensitive configurations
- Automatic blocking of repeated failures

## Monitoring
- `/var/log/sutazaiapp/otp_attempts.log`
- `/var/log/sutazaiapp/blocked_attempts.log`

## Emergency Procedures
1. Revoke compromised OTP secrets
2. Regenerate SSH keys
3. Update deployment configurations 