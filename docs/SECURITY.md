# SutazAI Security Guide

## Security Overview

SutazAI implements enterprise-grade security measures designed to protect against threats while maintaining system performance and usability. This guide covers all aspects of the security architecture.

## Security Architecture

### Defense in Depth Strategy

SutazAI employs a multi-layered security approach:

1. **Perimeter Security**: Network-level protection
2. **Application Security**: Input validation and secure coding
3. **Data Security**: Encryption and access controls
4. **Infrastructure Security**: Container and system hardening
5. **Monitoring**: Real-time threat detection

### Security Components

```
┌─────────────────────────────────────────────┐
│                User Layer                   │
├─────────────────────────────────────────────┤
│          Authentication & Authorization     │
├─────────────────────────────────────────────┤
│              Application Layer              │
├─────────────────────────────────────────────┤
│               Data Encryption               │
├─────────────────────────────────────────────┤
│             System Hardening               │
├─────────────────────────────────────────────┤
│           Infrastructure Security           │
└─────────────────────────────────────────────┘
```

## Authentication and Authorization

### Authorization Control Module (ACM)

**Location**: `sutazai/core/acm.py`

The ACM provides centralized security management:

```python
from sutazai.core.acm import AuthorizationControl

# Initialize ACM
acm = AuthorizationControl()

# Authenticate user
if acm.authenticate_user("chrissuta01@gmail.com"):
    # User is authorized
    acm.log_access("login_success")
else:
    # Access denied
    acm.log_access("login_failed")
```

### Hardcoded Authorization

The system implements hardcoded authorization for the primary user:

```python
AUTHORIZED_USERS = {
    "chrissuta01@gmail.com": {
        "role": "admin",
        "permissions": ["shutdown", "configure", "monitor"],
        "can_authorize_others": True
    }
}
```

### Multi-Factor Authentication (MFA)

#### Email-Based MFA
```python
# Send verification code
acm.send_verification_code("chrissuta01@gmail.com")

# Verify code
if acm.verify_code("chrissuta01@gmail.com", "123456"):
    # Grant access
    session = acm.create_session(user_id)
```

#### Time-Based OTP (TOTP)
```python
import pyotp

# Generate TOTP secret
secret = pyotp.random_base32()
totp = pyotp.TOTP(secret)

# Verify TOTP token
if totp.verify(user_token):
    # Token is valid
    pass
```

## Data Security

### Encryption at Rest

All sensitive data is encrypted using AES-256:

```python
from sutazai.core.secure_storage import SecureStorage

storage = SecureStorage()

# Encrypt and store data
storage.store_encrypted("user_data", sensitive_data)

# Retrieve and decrypt data
data = storage.retrieve_encrypted("user_data")
```

### Encryption in Transit

#### TLS Configuration
```nginx
# Nginx TLS configuration
server {
    listen 443 ssl http2;
    ssl_certificate /path/to/certificate.crt;
    ssl_certificate_key /path/to/private.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA384:ECDHE-RSA-CHACHA20-POLY1305;
}
```

#### API Security Headers
```python
# FastAPI security headers
@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000"
    return response
```

### Database Security

#### Connection Security
```python
# Secure database connection
DATABASE_CONFIG = {
    "encryption": True,
    "ssl_mode": "require",
    "connection_timeout": 30,
    "prepared_statements": True
}
```

#### Query Protection
```python
# Parameterized queries to prevent SQL injection
def get_user_data(user_id: str):
    query = "SELECT * FROM users WHERE id = ?"
    return db.execute(query, (user_id,))
```

## Input Validation and Sanitization

### Request Validation
```python
from pydantic import BaseModel, validator

class UserInput(BaseModel):
    prompt: str
    max_length: int = 1000
    
    @validator('prompt')
    def validate_prompt(cls, v):
        if len(v) > 10000:
            raise ValueError('Prompt too long')
        if any(char in v for char in ['<script>', 'javascript:']):
            raise ValueError('Invalid content detected')
        return v
```

### Content Filtering
```python
import re

def sanitize_input(text: str) -> str:
    # Remove potentially dangerous content
    text = re.sub(r'<script.*?</script>', '', text, flags=re.IGNORECASE)
    text = re.sub(r'javascript:', '', text, flags=re.IGNORECASE)
    text = re.sub(r'on\w+\s*=', '', text, flags=re.IGNORECASE)
    return text
```

## Network Security

### Firewall Configuration
```bash
# UFW firewall rules
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw allow 8000/tcp  # SutazAI API
sudo ufw enable
```

### Rate Limiting
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.get("/api/v1/generate")
@limiter.limit("10/minute")
async def generate_code(request: Request):
    # API endpoint with rate limiting
    pass
```

### IP Whitelisting
```python
ALLOWED_IPS = [
    "192.168.1.0/24",  # Local network
    "10.0.0.0/8",      # Private network
    "127.0.0.1"        # Localhost
]

@app.middleware("http")
async def ip_whitelist(request: Request, call_next):
    client_ip = request.client.host
    if not any(ipaddress.ip_address(client_ip) in ipaddress.ip_network(allowed) 
               for allowed in ALLOWED_IPS):
        raise HTTPException(status_code=403, detail="Access denied")
    return await call_next(request)
```

## Container Security

### Docker Security Best Practices

#### Dockerfile Security
```dockerfile
# Use non-root user
FROM python:3.9-slim
RUN adduser --disabled-password --gecos '' sutazai
USER sutazai

# Copy with proper permissions
COPY --chown=sutazai:sutazai . /app
WORKDIR /app

# Security scanning
RUN pip install --no-cache-dir safety
RUN safety check
```

#### Container Hardening
```bash
# Run container with security options
docker run -d   --name sutazai   --read-only   --tmpfs /tmp   --cap-drop ALL   --cap-add NET_BIND_SERVICE   --security-opt no-new-privileges   --user 1000:1000   sutazai:latest
```

### Kubernetes Security
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: sutazai
spec:
  securityContext:
    runAsNonRoot: true
    runAsUser: 1000
    fsGroup: 1000
  containers:
  - name: sutazai
    image: sutazai:latest
    securityContext:
      allowPrivilegeEscalation: false
      readOnlyRootFilesystem: true
      capabilities:
        drop:
        - ALL
```

## Audit Logging

### Comprehensive Audit Trail
```python
import logging
import json
from datetime import datetime

class AuditLogger:
    def __init__(self):
        self.logger = logging.getLogger('audit')
        
    def log_access(self, user_id: str, action: str, resource: str, 
                   success: bool, details: dict = None):
        audit_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "action": action,
            "resource": resource,
            "success": success,
            "details": details or {},
            "ip_address": self.get_client_ip(),
            "user_agent": self.get_user_agent()
        }
        
        self.logger.info(json.dumps(audit_entry))
```

### Log Analysis
```bash
# Search for failed login attempts
grep "login_failed" /opt/sutazaiapp/logs/audit.log

# Monitor unusual access patterns
tail -f /opt/sutazaiapp/logs/audit.log | grep "SUSPICIOUS"

# Generate security reports
python3 scripts/security_report.py --days 7
```

## Vulnerability Management

### Dependency Scanning
```bash
# Scan Python dependencies
pip install safety
safety check

# Scan for known vulnerabilities
pip install pip-audit
pip-audit

# Update dependencies
pip list --outdated
pip install --upgrade package_name
```

### Code Security Analysis
```bash
# Static code analysis
pip install bandit
bandit -r sutazai/

# Security linting
pip install semgrep
semgrep --config=security sutazai/
```

### Container Scanning
```bash
# Scan Docker images
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock   aquasec/trivy image sutazai:latest

# Kubernetes security scanning
kubectl run kube-bench --image=aquasec/kube-bench:latest
```

## Incident Response

### Security Incident Workflow

1. **Detection**: Automated monitoring alerts
2. **Assessment**: Evaluate threat severity
3. **Containment**: Isolate affected systems
4. **Investigation**: Root cause analysis
5. **Recovery**: Restore normal operations
6. **Lessons Learned**: Update security measures

### Emergency Procedures

#### System Shutdown
```python
# Emergency shutdown (authorized users only)
from sutazai.core.acm import AuthorizationControl

acm = AuthorizationControl()
if acm.verify_emergency_authorization("chrissuta01@gmail.com"):
    acm.emergency_shutdown()
```

#### Incident Reporting
```python
class IncidentReporter:
    def report_incident(self, severity: str, description: str, 
                       affected_systems: list):
        incident = {
            "id": self.generate_incident_id(),
            "timestamp": datetime.utcnow(),
            "severity": severity,
            "description": description,
            "affected_systems": affected_systems,
            "reporter": self.get_current_user(),
            "status": "open"
        }
        
        self.store_incident(incident)
        self.notify_security_team(incident)
```

## Security Monitoring

### Real-time Monitoring
```python
class SecurityMonitor:
    def __init__(self):
        self.metrics = SecurityMetrics()
        
    def monitor_threats(self):
        # Monitor for suspicious patterns
        failed_logins = self.get_failed_login_count()
        if failed_logins > 10:
            self.alert_security_team("Multiple failed logins detected")
            
        # Monitor system resources
        cpu_usage = psutil.cpu_percent()
        if cpu_usage > 90:
            self.alert_security_team("High CPU usage - potential DoS attack")
```

### Security Metrics
```python
# Security KPIs
SECURITY_METRICS = {
    "authentication_success_rate": 99.5,
    "failed_login_attempts": 15,
    "vulnerability_scan_score": 95,
    "security_incidents": 0,
    "patch_level": "current"
}
```

## Compliance and Standards

### Security Standards Compliance
- **ISO 27001**: Information Security Management
- **SOC 2 Type II**: Security and Availability
- **NIST Cybersecurity Framework**: Risk Management
- **OWASP Top 10**: Web Application Security

### Data Protection
- **GDPR Compliance**: EU data protection regulation
- **CCPA Compliance**: California consumer privacy
- **Data Retention**: Automated data lifecycle management
- **Right to Erasure**: User data deletion capabilities

## Security Configuration

### Environment Variables
```bash
# Security configuration
export ENCRYPTION_KEY="your-256-bit-encryption-key"
export JWT_SECRET="your-jwt-secret-key"
export SESSION_TIMEOUT=3600
export MAX_LOGIN_ATTEMPTS=5
export ENABLE_MFA=true
export AUDIT_LOGGING=true
```

### Security Policies
```python
SECURITY_POLICIES = {
    "password_policy": {
        "min_length": 12,
        "require_uppercase": True,
        "require_lowercase": True,
        "require_numbers": True,
        "require_special_chars": True
    },
    "session_policy": {
        "timeout": 3600,
        "extend_on_activity": True,
        "single_session": False
    },
    "access_policy": {
        "max_failed_attempts": 5,
        "lockout_duration": 900,
        "require_mfa": True
    }
}
```

## Security Testing

### Penetration Testing
```bash
# Network security testing
nmap -sS -O target_ip

# Web application testing
nikto -h http://localhost:8000

# SSL/TLS testing
sslyze localhost:443
```

### Security Test Suite
```python
# Security unit tests
class TestSecurity:
    def test_authentication(self):
        # Test authentication mechanisms
        pass
        
    def test_authorization(self):
        # Test access controls
        pass
        
    def test_input_validation(self):
        # Test input sanitization
        pass
        
    def test_encryption(self):
        # Test data encryption
        pass
```

## Security Maintenance

### Regular Security Tasks
- Weekly vulnerability scans
- Monthly security reviews
- Quarterly penetration testing
- Annual security audits

### Update Procedures
```bash
# Security update script
#!/bin/bash
pip install --upgrade safety
safety check
pip install --upgrade package_name
python3 scripts/security_audit.py
```

## Contact and Support

### Security Team
- **Security Email**: security@sutazai.com
- **Emergency Contact**: +1-xxx-xxx-xxxx
- **PGP Key**: Available at keybase.io/sutazai

### Vulnerability Reporting
If you discover a security vulnerability:

1. **DO NOT** disclose publicly
2. Email security@sutazai.com with details
3. Include proof-of-concept if possible
4. Allow reasonable time for remediation

We appreciate responsible disclosure and will acknowledge security researchers who help improve SutazAI's security.

---

**Security is a shared responsibility.** Stay vigilant and follow security best practices.
