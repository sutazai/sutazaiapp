# SutazAI Security Guide

**Last Updated:** August 8, 2025  
**Document Status:** AUTHORITATIVE  
**Security Posture:** CRITICAL - Multiple High-Priority Issues Identified

## Table of Contents

1. [Security Architecture Overview](#1-security-architecture-overview)
2. [Current Security Issues](#2-current-security-issues)
3. [Authentication & Authorization](#3-authentication--authorization)
4. [Network Security](#4-network-security)
5. [Data Security](#5-data-security)
6. [Container Security](#6-container-security)
7. [Application Security](#7-application-security)
8. [Compliance & Governance](#8-compliance--governance)
9. [Security Monitoring](#9-security-monitoring)
10. [Vulnerability Management](#10-vulnerability-management)
11. [Security Hardening Checklist](#11-security-hardening-checklist)
12. [Incident Response](#12-incident-response)

---

## 1. Security Architecture Overview

### Current Security Posture Assessment

**Overall Risk Level: HIGH**

The SutazAI system currently operates with multiple critical security vulnerabilities that require immediate attention. Based on system analysis and CLAUDE.md reality check:

- **28 containers running** with **59 ports exposed** (10000-11200 range)
- **No authentication** on most services
- **Hardcoded secrets** in configuration files
- **No rate limiting** or DDoS protection
- **Containers running as root** user
- **No vulnerability scanning** in CI/CD pipeline
- **0% test coverage** including security tests

### Security Boundaries and Zones

```
┌─────────────────────────────────────────────────────────────┐
│                     Internet (Untrusted)                     │
└───────────────┬─────────────────────────────────────────────┘
                │
    ┌───────────▼───────────┐
    │   Perimeter Zone       │ ⚠️ MISSING
    │   - WAF                │
    │   - DDoS Protection    │
    │   - Rate Limiting       │
    └───────────┬───────────┘
                │
    ┌───────────▼───────────┐
    │   DMZ/Public Zone      │ ⚠️ EXPOSED
    │   - Kong Gateway:10005 │ No auth configured
    │   - Frontend:10011     │ No login system
    │   - Grafana:10201      │ Default credentials
    └───────────┬───────────┘
                │
    ┌───────────▼───────────┐
    │   Application Zone     │ ⚠️ VULNERABLE
    │   - Backend API:10010  │ JWT partially implemented
    │   - Agent Services     │ All unauthenticated
    │   - Ollama:10104       │ Direct access
    └───────────┬───────────┘
                │
    ┌───────────▼───────────┐
    │   Data Zone            │ ⚠️ UNPROTECTED
    │   - PostgreSQL:10000   │ No encryption at rest
    │   - Redis:10001        │ No ACLs configured
    │   - Neo4j:10002/10003  │ Default credentials
    └────────────────────────┘
```

### Threat Model (STRIDE Analysis)

| Threat | Current Risk | Affected Components | Mitigation Priority |
|--------|--------------|--------------------|--------------------|
| **Spoofing** | HIGH | All services (no auth) | P0 - Immediate |
| **Tampering** | HIGH | Databases, APIs | P0 - Immediate |
| **Repudiation** | HIGH | No audit logging | P1 - Week 1 |
| **Information Disclosure** | CRITICAL | Hardcoded secrets, open ports | P0 - Immediate |
| **Denial of Service** | HIGH | No rate limiting | P1 - Week 1 |
| **Elevation of Privilege** | CRITICAL | Root containers, no RBAC | P0 - Immediate |

### Risk Assessment Matrix

| Risk Category | Likelihood | Impact | Risk Score | Priority |
|---------------|------------|--------|------------|----------|
| Unauthorized Access | Very High (5) | Critical (5) | 25 | P0 |
| Data Breach | High (4) | Critical (5) | 20 | P0 |
| Service Disruption | High (4) | High (4) | 16 | P1 |
| Compliance Violation | Medium (3) | High (4) | 12 | P1 |
| Insider Threat | Medium (3) | Medium (3) | 9 | P2 |

---

## 2. Current Security Issues

### Critical Vulnerabilities (P0 - Fix Immediately)

#### 1. No Authentication on Services
- **Impact:** Complete system compromise possible
- **Affected:** 90% of services
- **Evidence:** Direct curl access to all ports succeeds
- **CVSS Score:** 10.0 (Critical)

#### 2. Hardcoded Secrets (ISSUE-0008)
- **Location:** `/opt/sutazaiapp/IMPORTANT/init_db.sql`
- **Impact:** Credential exposure
- **Evidence:** Database passwords in plaintext
- **CVSS Score:** 9.8 (Critical)

#### 3. Root Container Execution
- **Impact:** Container escape, host compromise
- **Affected:** All containers
- **Evidence:** `docker ps` shows UID 0
- **CVSS Score:** 9.0 (Critical)

#### 4. Exposed Internal Services
- **Impact:** Direct database access possible
- **Affected:** PostgreSQL:10000, Redis:10001, Neo4j:10002
- **Evidence:** Ports bound to 0.0.0.0
- **CVSS Score:** 9.1 (Critical)

### High Priority Issues (P1 - Week 1)

#### 5. Missing JWT Implementation (ISSUE-0005)
- **Status:** Partially implemented in backend
- **Impact:** No user session management
- **Required:** Complete implementation with refresh tokens

#### 6. No Rate Limiting
- **Impact:** DDoS vulnerability
- **Affected:** All API endpoints
- **Required:** Kong configuration or middleware

#### 7. Missing Security Headers
- **Impact:** XSS, clickjacking vulnerabilities
- **Affected:** Frontend and API responses
- **Required:** CSP, HSTS, X-Frame-Options

#### 8. No Audit Logging
- **Impact:** Cannot detect or investigate breaches
- **Affected:** All security events
- **Required:** Centralized security logging

---

## 3. Authentication & Authorization

### Current State

```python
# Backend JWT implementation exists but incomplete
/backend/app/auth/
├── jwt_handler.py     # ✅ Basic JWT creation
├── password.py        # ✅ Bcrypt hashing
├── router.py          # ⚠️  Endpoints not integrated
├── dependencies.py    # ⚠️  Not enforced globally
└── migrations.sql     # ❌ Not executed
```

### JWT Implementation Plan

#### Phase 1: Backend Completion (Week 1)
```python
# Required modifications to /backend/app/main.py

from app.auth.dependencies import get_current_user, require_admin
from app.core.middleware import JWTMiddleware

# Add global JWT middleware
app.add_middleware(JWTMiddleware)

# Protect endpoints
@app.get("/api/v1/protected")
async def protected_route(user = Depends(get_current_user)):
    return {"user": user.username}
```

#### Phase 2: Frontend Integration (Week 1)
```python
# /frontend/auth_handler.py - NEW FILE
import streamlit as st
import requests
from datetime import datetime, timedelta

class AuthHandler:
    def __init__(self):
        self.api_base = "http://backend:8000"
        
    def login(self, username: str, password: str):
        response = requests.post(
            f"{self.api_base}/auth/login",
            json={"username": username, "password": password}
        )
        if response.status_code == 200:
            tokens = response.json()
            st.session_state.access_token = tokens["access_token"]
            st.session_state.refresh_token = tokens["refresh_token"]
            return True
        return False
```

### API Key Management

```yaml
# Kong Gateway Configuration
services:
  - name: backend-api
    url: http://backend:8000
    plugins:
      - name: key-auth
        config:
          key_names: ["X-API-Key"]
      - name: rate-limiting
        config:
          minute: 100
          hour: 1000
```

### Role-Based Access Control (RBAC)

```python
# Role hierarchy
class Role(Enum):
    SYSTEM = "system"      # Internal services only
    ADMIN = "admin"        # Full access
    OPERATOR = "operator"  # Manage agents
    DEVELOPER = "developer" # Deploy/modify
    USER = "user"          # Basic access
    VIEWER = "viewer"      # Read-only

# Permission matrix
PERMISSIONS = {
    "system": ["*"],
    "admin": ["users:*", "agents:*", "config:*", "logs:*"],
    "operator": ["agents:*", "logs:read", "metrics:*"],
    "developer": ["agents:deploy", "config:read", "logs:read"],
    "user": ["agents:execute", "results:read"],
    "viewer": ["*:read"]
}
```

### Service-to-Service Authentication

```python
# Implement mutual TLS for internal services
# /backend/app/core/service_auth.py

import jwt
from cryptography import x509

class ServiceAuthenticator:
    def __init__(self):
        self.trusted_services = {
            "agent-orchestrator": "cert-fingerprint-1",
            "task-coordinator": "cert-fingerprint-2",
            "ollama-service": "cert-fingerprint-3"
        }
    
    def generate_service_token(self, service_name: str) -> str:
        """Generate short-lived token for service"""
        payload = {
            "service": service_name,
            "iat": datetime.utcnow(),
            "exp": datetime.utcnow() + timedelta(minutes=5)
        }
        return jwt.encode(payload, self.service_secret, "HS256")
```

---

## 4. Network Security

### Docker Network Isolation

```yaml
# docker-compose.yml modifications needed
networks:
  public:
    driver: bridge
  application:
    driver: bridge
    internal: true
  data:
    driver: bridge
    internal: true

services:
  kong:
    networks:
      - public
      - application
      
  backend:
    networks:
      - application
      - data
    # Remove direct port exposure
    # ports:
    #   - "10010:8000"
    
  postgres:
    networks:
      - data
    # Remove public exposure
    # ports:
    #   - "10000:5432"
```

### Port Exposure Analysis

| Service | Current Port | Risk | Remediation |
|---------|--------------|------|-------------|
| PostgreSQL | 10000 (public) | CRITICAL | Internal only |
| Redis | 10001 (public) | CRITICAL | Internal only |
| Neo4j | 10002-10003 (public) | HIGH | VPN access only |
| Backend API | 10010 (public) | MEDIUM | Via Kong only |
| Ollama | 10104 (public) | HIGH | Internal only |
| Grafana | 10201 (public) | MEDIUM | Add auth proxy |
| Prometheus | 10200 (public) | HIGH | Internal only |

### TLS/SSL Implementation

```nginx
# Kong Gateway TLS configuration
server {
    listen 443 ssl http2;
    ssl_certificate /etc/ssl/certs/sutazai.crt;
    ssl_certificate_key /etc/ssl/private/sutazai.key;
    
    # Modern TLS configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256;
    ssl_prefer_server_ciphers off;
    
    # Security headers
    add_header Strict-Transport-Security "max-age=63072000" always;
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
}
```

### API Gateway Security (Kong)

```lua
-- Kong security plugins to enable
plugins = {
  "rate-limiting",
  "bot-detection",
  "cors",
  "ip-restriction",
  "request-size-limiting",
  "response-transformer"
}

-- Rate limiting configuration
config.rate_limiting = {
  second = 10,
  minute = 100,
  hour = 1000,
  day = 10000,
  policy = "redis",
  redis_host = "redis",
  redis_port = 6379
}
```

### Service Mesh Security (Consul)

```hcl
# Consul ACL configuration
acl = {
  enabled = true
  default_policy = "deny"
  
  tokens = {
    master = "secure-token-here"
    agent = "agent-token-here"
  }
}

# Enable TLS
ports {
  https = 8501
}

verify_incoming = true
verify_outgoing = true
verify_server_hostname = true
```

---

## 5. Data Security

### Encryption at Rest

#### PostgreSQL
```sql
-- Enable Transparent Data Encryption (TDE)
ALTER SYSTEM SET ssl = on;
ALTER SYSTEM SET ssl_cert_file = 'server.crt';
ALTER SYSTEM SET ssl_key_file = 'server.key';

-- Column-level encryption for sensitive data
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Encrypt sensitive columns
ALTER TABLE users 
  ALTER COLUMN password_hash TYPE bytea 
  USING pgp_sym_encrypt(password_hash::text, 'encryption-key');
```

#### Redis
```conf
# redis.conf
requirepass "strong-redis-password"
# Enable TLS
tls-port 6380
port 0
tls-cert-file /etc/redis/tls/redis.crt
tls-key-file /etc/redis/tls/redis.key
tls-ca-cert-file /etc/redis/tls/ca.crt
```

#### Neo4j
```conf
# neo4j.conf
dbms.ssl.policy.default.enabled=true
dbms.ssl.policy.default.base_directory=certificates/default
dbms.ssl.policy.default.private_key=private.key
dbms.ssl.policy.default.public_certificate=public.crt
```

### Encryption in Transit

```python
# Enforce TLS for all database connections
DATABASE_URL = "postgresql+ssl://user:pass@postgres:5432/sutazai?sslmode=require"
REDIS_URL = "rediss://default:password@redis:6380/0"
NEO4J_URL = "neo4j+s://neo4j:7687"
```

### Secrets Management

#### Current State (INSECURE)
```bash
# Environment variables in docker-compose.yml
environment:
  - DB_PASSWORD=plaintext_password  # ❌ INSECURE
```

#### Target State with HashiCorp Vault
```python
# /backend/app/core/secrets.py
import hvac

class VaultSecretManager:
    def __init__(self):
        self.client = hvac.Client(
            url='http://vault:8200',
            token=os.environ.get('VAULT_TOKEN')
        )
    
    def get_secret(self, path: str) -> dict:
        """Retrieve secret from Vault"""
        response = self.client.secrets.kv.v2.read_secret_version(
            path=path,
            mount_point='secret'
        )
        return response['data']['data']
    
    def rotate_database_credentials(self):
        """Automatically rotate database passwords"""
        self.client.write(
            'database/rotate-root/postgresql',
            name='postgresql'
        )
```

#### Immediate Fix (Docker Secrets)
```yaml
# docker-compose.yml
secrets:
  db_password:
    file: ./secrets/db_password.txt
  jwt_secret:
    file: ./secrets/jwt_secret.txt

services:
  backend:
    secrets:
      - db_password
      - jwt_secret
    environment:
      - DB_PASSWORD_FILE=/run/secrets/db_password
```

### PII Handling

```python
# /backend/app/core/pii_handler.py
import re
from typing import Any, Dict

class PIIHandler:
    """Handle Personal Identifiable Information"""
    
    PII_PATTERNS = {
        'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
        'credit_card': r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
    }
    
    def mask_pii(self, text: str) -> str:
        """Mask PII in text"""
        for pii_type, pattern in self.PII_PATTERNS.items():
            text = re.sub(pattern, f'[{pii_type.upper()}_MASKED]', text)
        return text
    
    def encrypt_pii(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt PII fields"""
        pii_fields = ['email', 'phone', 'ssn', 'credit_card']
        for field in pii_fields:
            if field in data:
                data[field] = self.encrypt(data[field])
        return data
```

### Data Retention Policies

```python
# /backend/app/core/data_retention.py
from datetime import datetime, timedelta

class DataRetentionPolicy:
    """Enforce data retention policies"""
    
    RETENTION_PERIODS = {
        'user_activity': timedelta(days=90),
        'system_logs': timedelta(days=30),
        'audit_logs': timedelta(days=365),
        'temporary_data': timedelta(days=7),
        'backups': timedelta(days=30)
    }
    
    async def cleanup_expired_data(self):
        """Remove data past retention period"""
        for data_type, retention_period in self.RETENTION_PERIODS.items():
            cutoff_date = datetime.utcnow() - retention_period
            await self.delete_data_before(data_type, cutoff_date)
```

---

## 6. Container Security

### Docker Security Best Practices

#### Non-Root User Implementation
```dockerfile
# Dockerfile.secure
FROM python:3.11-slim

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Install dependencies as root
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Switch to non-root user
USER appuser

# Copy application files
COPY --chown=appuser:appuser . /app
WORKDIR /app

# Run as non-root
CMD ["python", "main.py"]
```

### Image Scanning with Trivy

```yaml
# .github/workflows/security-scan.yml
name: Security Scan

on:
  push:
    branches: [main, develop]
  pull_request:

jobs:
  trivy-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Build Docker image
        run: docker build -t sutazai:${{ github.sha }} .
      
      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: sutazai:${{ github.sha }}
          format: 'sarif'
          output: 'trivy-results.sarif'
          severity: 'CRITICAL,HIGH'
          
      - name: Upload Trivy results to GitHub Security
        uses: github/codeql-action/upload-sarif@v2
        with:
          sarif_file: 'trivy-results.sarif'
```

### Container Runtime Security

```yaml
# docker-compose.yml security additions
services:
  backend:
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL
    cap_add:
      - NET_BIND_SERVICE
    read_only: true
    tmpfs:
      - /tmp
    user: "1000:1000"
```

### Resource Limits and Quotas

```yaml
# docker-compose.yml resource limits
services:
  backend:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 2G
        reservations:
          cpus: '0.5'
          memory: 512M
    ulimits:
      nproc: 65535
      nofile:
        soft: 20000
        hard: 40000
```

---

## 7. Application Security

### Input Validation

```python
# /backend/app/core/validation.py
from pydantic import BaseModel, validator, constr, EmailStr
import bleach
import re

class SecureInputValidator:
    """Validate and sanitize all user inputs"""
    
    @staticmethod
    def sanitize_html(text: str) -> str:
        """Remove dangerous HTML/JS"""
        allowed_tags = ['b', 'i', 'u', 'em', 'strong', 'p', 'br']
        return bleach.clean(text, tags=allowed_tags, strip=True)
    
    @staticmethod
    def validate_sql_safe(text: str) -> bool:
        """Check for SQL injection patterns"""
        sql_patterns = [
            r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|ALTER)\b)",
            r"(--|#|\/\*|\*\/)",
            r"(\bOR\b.*=.*)",
            r"(\bAND\b.*=.*)"
        ]
        for pattern in sql_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                raise ValueError("Potential SQL injection detected")
        return True

class UserInput(BaseModel):
    username: constr(regex='^[a-zA-Z0-9_]{3,20}$')
    email: EmailStr
    message: str
    
    @validator('message')
    def sanitize_message(cls, v):
        return SecureInputValidator.sanitize_html(v)
```

### SQL Injection Prevention

```python
# Use parameterized queries ALWAYS
from sqlalchemy import text

# ❌ NEVER DO THIS
query = f"SELECT * FROM users WHERE username = '{username}'"

# ✅ ALWAYS DO THIS
query = text("SELECT * FROM users WHERE username = :username")
result = db.execute(query, {"username": username})

# With SQLAlchemy ORM
user = db.query(User).filter(User.username == username).first()
```

### XSS Protection

```python
# /backend/app/core/middleware.py
from fastapi import Response

class SecurityHeadersMiddleware:
    """Add security headers to all responses"""
    
    async def __call__(self, request, call_next):
        response = await call_next(request)
        
        # XSS Protection
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["X-Content-Type-Options"] = "nosniff"
        
        # Content Security Policy
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self' data:; "
            "connect-src 'self'"
        )
        
        return response
```

### CSRF Protection

```python
# /backend/app/core/csrf.py
import secrets
from fastapi import HTTPException

class CSRFProtection:
    """CSRF token validation"""
    
    def generate_token(self) -> str:
        """Generate CSRF token"""
        return secrets.token_urlsafe(32)
    
    def validate_token(self, token: str, session_token: str) -> bool:
        """Validate CSRF token"""
        if not token or token != session_token:
            raise HTTPException(status_code=403, detail="CSRF validation failed")
        return True
```

### Security Headers

```python
# Complete security headers configuration
SECURITY_HEADERS = {
    # HSTS
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains; preload",
    
    # Prevent clickjacking
    "X-Frame-Options": "SAMEORIGIN",
    
    # XSS Protection
    "X-XSS-Protection": "1; mode=block",
    
    # Prevent MIME sniffing
    "X-Content-Type-Options": "nosniff",
    
    # Referrer Policy
    "Referrer-Policy": "strict-origin-when-cross-origin",
    
    # Permissions Policy
    "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
    
    # CSP
    "Content-Security-Policy": "default-src 'self'"
}
```

---

## 8. Compliance & Governance

### GDPR Considerations

```python
# /backend/app/compliance/gdpr.py
class GDPRCompliance:
    """GDPR compliance implementation"""
    
    async def handle_data_request(self, user_id: str, request_type: str):
        """Handle GDPR data requests"""
        if request_type == "access":
            return await self.export_user_data(user_id)
        elif request_type == "deletion":
            return await self.delete_user_data(user_id)
        elif request_type == "portability":
            return await self.export_portable_data(user_id)
    
    async def get_consent(self, user_id: str, purpose: str) -> bool:
        """Verify user consent for data processing"""
        consent = await db.get_consent(user_id, purpose)
        return consent and consent.is_active
```

### SOC2 Requirements

| Control Family | Requirement | Implementation Status | Priority |
|----------------|-------------|----------------------|----------|
| CC1 - Control Environment | Risk assessment | ❌ Not implemented | P1 |
| CC2 - Communication | Security training | ❌ Not implemented | P2 |
| CC3 - Risk Assessment | Threat modeling | ⚠️ Partial (this doc) | P1 |
| CC4 - Monitoring | Security monitoring | ⚠️ Partial (Prometheus) | P1 |
| CC5 - Control Activities | Access control | ❌ Not implemented | P0 |
| CC6 - Logical Access | Authentication | ⚠️ Partial (JWT) | P0 |
| CC7 - System Operations | Change management | ❌ Not implemented | P2 |
| CC8 - Change Management | Code review | ❌ Not implemented | P1 |
| CC9 - Risk Mitigation | Incident response | ❌ Not implemented | P1 |

### Security Policies

```markdown
## Access Control Policy
- All access must be authenticated and authorized
- Principle of least privilege enforced
- Regular access reviews (quarterly)
- Immediate revocation upon termination

## Data Classification Policy
- PUBLIC: No restrictions
- INTERNAL: Company use only
- CONFIDENTIAL: Need-to-know basis
- RESTRICTED: Executive approval required

## Incident Response Policy
- 15-minute response time for P0 incidents
- 1-hour response time for P1 incidents
- All incidents logged and reviewed
- Post-mortem for P0/P1 incidents
```

### Audit Logging

```python
# /backend/app/core/audit.py
from datetime import datetime
import json

class AuditLogger:
    """Comprehensive audit logging"""
    
    SECURITY_EVENTS = [
        "login_attempt",
        "login_success",
        "login_failure",
        "logout",
        "permission_denied",
        "data_access",
        "data_modification",
        "config_change",
        "user_created",
        "user_deleted",
        "role_changed"
    ]
    
    async def log_event(self, event_type: str, user_id: str, details: dict):
        """Log security event"""
        if event_type not in self.SECURITY_EVENTS:
            raise ValueError(f"Unknown event type: {event_type}")
        
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "user_id": user_id,
            "details": details,
            "ip_address": details.get("ip_address"),
            "user_agent": details.get("user_agent"),
            "session_id": details.get("session_id")
        }
        
        # Store in database
        await db.audit_logs.insert(event)
        
        # Send to SIEM
        await self.send_to_siem(event)
```

---

## 9. Security Monitoring

### Security Event Logging

```python
# /backend/app/monitoring/security_events.py
from prometheus_client import Counter, Histogram

# Security metrics
auth_attempts = Counter('auth_attempts_total', 'Authentication attempts', ['result'])
auth_failures = Counter('auth_failures_total', 'Failed authentications', ['reason'])
suspicious_requests = Counter('suspicious_requests_total', 'Suspicious request patterns')
data_access = Counter('data_access_total', 'Data access events', ['resource', 'action'])
```

### Intrusion Detection

```python
# /backend/app/security/ids.py
class IntrusionDetectionSystem:
    """Basic IDS implementation"""
    
    ATTACK_PATTERNS = {
        'sql_injection': [
            r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP)\b)",
            r"(--|#|\/\*|\*\/)"
        ],
        'xss': [
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"on\w+\s*="
        ],
        'path_traversal': [
            r"\.\./",
            r"\.\.\\",
            r"%2e%2e%2f"
        ],
        'command_injection': [
            r"[;&|]",
            r"\$\(",
            r"`"
        ]
    }
    
    def detect_attack(self, request_data: str) -> list:
        """Detect potential attacks"""
        detected = []
        for attack_type, patterns in self.ATTACK_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, request_data, re.IGNORECASE):
                    detected.append(attack_type)
                    break
        return detected
```

### Anomaly Detection

```python
# /backend/app/security/anomaly.py
from collections import defaultdict
from datetime import datetime, timedelta

class AnomalyDetector:
    """Detect anomalous behavior"""
    
    def __init__(self):
        self.baseline = defaultdict(lambda: {"count": 0, "last_seen": None})
        self.thresholds = {
            "login_attempts": 5,  # per minute
            "api_calls": 100,     # per minute
            "data_downloads": 10  # per hour
        }
    
    async def check_anomaly(self, user_id: str, action: str) -> bool:
        """Check if action is anomalous"""
        key = f"{user_id}:{action}"
        now = datetime.utcnow()
        
        if self.baseline[key]["last_seen"]:
            time_diff = now - self.baseline[key]["last_seen"]
            if time_diff < timedelta(minutes=1):
                self.baseline[key]["count"] += 1
                if self.baseline[key]["count"] > self.thresholds.get(action, 100):
                    await self.alert_security_team(user_id, action)
                    return True
            else:
                self.baseline[key]["count"] = 1
        
        self.baseline[key]["last_seen"] = now
        return False
```

### Security Dashboards in Grafana

```json
{
  "dashboard": {
    "title": "Security Monitoring",
    "panels": [
      {
        "title": "Authentication Attempts",
        "targets": [
          {
            "expr": "rate(auth_attempts_total[5m])"
          }
        ]
      },
      {
        "title": "Failed Logins",
        "targets": [
          {
            "expr": "rate(auth_failures_total[5m])"
          }
        ]
      },
      {
        "title": "Suspicious Activities",
        "targets": [
          {
            "expr": "rate(suspicious_requests_total[5m])"
          }
        ]
      },
      {
        "title": "Data Access Patterns",
        "targets": [
          {
            "expr": "sum by (resource) (rate(data_access_total[5m]))"
          }
        ]
      }
    ]
  }
}
```

### Alert Rules for Security Events

```yaml
# prometheus/alerts/security.yml
groups:
  - name: security
    rules:
      - alert: HighFailedLoginRate
        expr: rate(auth_failures_total[5m]) > 10
        labels:
          severity: warning
        annotations:
          summary: High rate of failed login attempts
          
      - alert: SuspiciousActivity
        expr: rate(suspicious_requests_total[5m]) > 5
        labels:
          severity: critical
        annotations:
          summary: Suspicious activity detected
          
      - alert: UnauthorizedDataAccess
        expr: increase(data_access_total{result="denied"}[5m]) > 0
        labels:
          severity: critical
        annotations:
          summary: Unauthorized data access attempt
```

---

## 10. Vulnerability Management

### Dependency Scanning

```yaml
# .github/workflows/dependency-scan.yml
name: Dependency Security Scan

on:
  schedule:
    - cron: '0 0 * * *'  # Daily
  push:
    paths:
      - 'requirements.txt'
      - 'package.json'
      - 'Pipfile'

jobs:
  python-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Safety check
        run: |
          pip install safety
          safety check --json > safety-report.json
          
      - name: Bandit security check
        run: |
          pip install bandit
          bandit -r backend/ -f json -o bandit-report.json
```

### Container Image Scanning

```bash
#!/bin/bash
# /scripts/security/scan-images.sh

# Scan all running containers
for image in $(docker ps --format "{{.Image}}"); do
  echo "Scanning $image..."
  trivy image --severity HIGH,CRITICAL "$image"
done

# Scan all images in registry
for image in $(docker images --format "{{.Repository}}:{{.Tag}}"); do
  if [[ "$image" != *"<none>"* ]]; then
    grype "$image"
  fi
done
```

### Code Security Analysis

```yaml
# .github/workflows/code-security.yml
name: Code Security Analysis

on: [push, pull_request]

jobs:
  codeql:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Initialize CodeQL
        uses: github/codeql-action/init@v2
        with:
          languages: python, javascript
          
      - name: Perform CodeQL Analysis
        uses: github/codeql-action/analyze@v2
  
  sonarqube:
    runs-on: ubuntu-latest
    steps:
      - uses: sonarsource/sonarqube-scan-action@master
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          SONAR_TOKEN: ${{ secrets.SONAR_TOKEN }}
```

### Penetration Testing Schedule

| Test Type | Frequency | Scope | Provider |
|-----------|-----------|-------|----------|
| External Pentest | Quarterly | Internet-facing services | External vendor |
| Internal Pentest | Semi-annual | Internal network | Security team |
| API Security Test | Monthly | All API endpoints | Automated + Manual |
| Container Security | Weekly | All containers | Automated scanning |
| Code Review | Per PR | All code changes | Peer review + tools |

### Security Patch Management

```python
# /scripts/security/patch_manager.py
import subprocess
import json
from datetime import datetime

class SecurityPatchManager:
    """Manage security patches"""
    
    def check_updates(self):
        """Check for security updates"""
        updates = {
            "os": self.check_os_updates(),
            "python": self.check_python_updates(),
            "node": self.check_node_updates(),
            "docker": self.check_docker_updates()
        }
        return updates
    
    def apply_patches(self, patch_list: list, test_first: bool = True):
        """Apply security patches"""
        if test_first:
            self.run_tests()
        
        for patch in patch_list:
            self.apply_patch(patch)
            self.verify_patch(patch)
        
        self.run_tests()
        self.update_patch_log()
```

---

## 11. Security Hardening Checklist

### Immediate Fixes (P0 - Do Now)

- [ ] **Enable authentication on all services**
  ```bash
  # Priority services:
  - PostgreSQL: Add password authentication
  - Redis: Enable AUTH command
  - Neo4j: Change default credentials
  - Grafana: Change admin/admin
  ```

- [ ] **Remove hardcoded secrets**
  ```bash
  # Move to environment files:
  grep -r "password\|secret\|key" docker-compose.yml
  # Create .env file with secrets
  # Update docker-compose.yml to use ${VAR_NAME}
  ```

- [ ] **Restrict port exposure**
  ```yaml
  # Change all services to internal networks
  # Only expose Kong gateway on 443
  ```

- [ ] **Implement rate limiting**
  ```python
  # Add to backend/app/main.py
  from slowapi import Limiter
  limiter = Limiter(key_func=get_remote_address)
  app.state.limiter = limiter
  ```

### Short-term Improvements (P1 - Week 1)

- [ ] **Complete JWT implementation**
  - [ ] Backend middleware enforcement
  - [ ] Frontend login UI
  - [ ] Token refresh mechanism
  - [ ] Session management

- [ ] **Add security headers**
  - [ ] CSP policy
  - [ ] HSTS header
  - [ ] X-Frame-Options
  - [ ] X-Content-Type-Options

- [ ] **Implement audit logging**
  - [ ] Authentication events
  - [ ] Data access logs
  - [ ] Configuration changes
  - [ ] Error tracking

- [ ] **Setup vulnerability scanning**
  - [ ] Container scanning with Trivy
  - [ ] Dependency scanning with Safety
  - [ ] Code scanning with Bandit

### Long-term Enhancements (P2 - Month 1)

- [ ] **Implement Zero Trust Architecture**
  - [ ] Service mesh with mTLS
  - [ ] Network segmentation
  - [ ] Microsegmentation
  - [ ] Identity-based access

- [ ] **Advanced Threat Detection**
  - [ ] SIEM integration
  - [ ] Behavioral analytics
  - [ ] Machine learning anomaly detection
  - [ ] Threat intelligence feeds

- [ ] **Compliance Automation**
  - [ ] Automated compliance checks
  - [ ] Policy as code
  - [ ] Continuous compliance monitoring
  - [ ] Audit report generation

### Security Gates for Deployment

```yaml
# Deployment criteria that must pass
security_gates:
  pre_deploy:
    - vulnerability_scan: "HIGH and CRITICAL == 0"
    - dependency_check: "No known vulnerabilities"
    - secrets_scan: "No hardcoded secrets"
    - container_scan: "No root containers"
    
  post_deploy:
    - health_check: "All services healthy"
    - auth_test: "Authentication working"
    - ssl_test: "TLS properly configured"
    - penetration_test: "No critical findings"
```

### Security Review Process

```mermaid
graph TD
    A[Code Change] --> B{Security Review Required?}
    B -->|Yes| C[Automated Scans]
    B -->|No| D[Standard Review]
    C --> E{Vulnerabilities Found?}
    E -->|Yes| F[Fix Required]
    E -->|No| G[Manual Review]
    F --> C
    G --> H{Approved?}
    H -->|Yes| I[Merge]
    H -->|No| F
    D --> I
```

---

## 12. Incident Response

### Security Incident Classification

| Severity | Description | Response Time | Examples |
|----------|-------------|---------------|----------|
| P0 - Critical | System compromise or data breach | 15 minutes | RCE, data leak, ransomware |
| P1 - High | Potential compromise | 1 hour | Auth bypass, SQLi attempt |
| P2 - Medium | Security weakness | 4 hours | XSS, information disclosure |
| P3 - Low | Minor issue | 24 hours | Missing headers, weak config |

### Response Procedures

```python
# /backend/app/security/incident_response.py
class IncidentResponse:
    """Incident response coordination"""
    
    async def handle_incident(self, incident_type: str, severity: str):
        """Coordinate incident response"""
        incident_id = self.create_incident_record()
        
        # Immediate actions
        if severity in ["P0", "P1"]:
            await self.isolate_affected_systems()
            await self.preserve_evidence()
            await self.notify_security_team()
        
        # Investigation
        evidence = await self.collect_evidence()
        analysis = await self.analyze_incident(evidence)
        
        # Containment
        await self.contain_threat(analysis)
        
        # Eradication
        await self.remove_threat()
        
        # Recovery
        await self.restore_services()
        
        # Lessons learned
        await self.create_post_mortem(incident_id)
```

### Forensics and Investigation

```bash
#!/bin/bash
# /scripts/security/incident-forensics.sh

# Capture system state
echo "Capturing system state..."
docker ps -a > /tmp/incident/docker-ps.log
docker logs $(docker ps -q) > /tmp/incident/container-logs.log
netstat -tulpn > /tmp/incident/network.log
ps aux > /tmp/incident/processes.log

# Capture database state
docker exec sutazai-postgres pg_dump -U sutazai > /tmp/incident/database-backup.sql

# Create timeline
echo "Creating incident timeline..."
grep -h ERROR /var/log/*.log | sort -k1,2 > /tmp/incident/error-timeline.log

# Package evidence
tar -czf /tmp/incident-$(date +%Y%m%d-%H%M%S).tar.gz /tmp/incident/
```

### Communication Plan

```markdown
## Incident Communication Matrix

### Internal Communications
- P0: CEO, CTO, Security Team, Legal (Immediate)
- P1: CTO, Security Team, DevOps (Within 1 hour)
- P2: Security Team, DevOps (Within 4 hours)
- P3: Security Team (Within 24 hours)

### External Communications
- Customers: If data affected (Within 72 hours - GDPR)
- Authorities: If breach confirmed (As required by law)
- Partners: If integration affected (Within 24 hours)
- Public: Via security advisory (After mitigation)

### Communication Templates
- Initial notification
- Status updates (every 2 hours for P0/P1)
- Resolution notification
- Post-mortem summary
```

### Post-Incident Review

```markdown
## Post-Mortem Template

### Incident Summary
- Incident ID:
- Date/Time:
- Duration:
- Severity:
- Impact:

### Timeline
- Detection:
- Response initiated:
- Contained:
- Resolved:

### Root Cause Analysis
- What happened:
- Why it happened:
- How it was detected:
- How it was resolved:

### Lessons Learned
- What went well:
- What went poorly:
- Where we got lucky:

### Action Items
- [ ] Immediate fixes
- [ ] Short-term improvements
- [ ] Long-term changes
- [ ] Process updates
```

---

## Implementation Timeline (8-Week MVP Aligned)

### Week 1: Critical Security Fixes
- Enable authentication on all services
- Remove hardcoded secrets
- Implement basic JWT auth
- Add rate limiting

### Week 2: Network Security
- Configure Docker networks
- Setup TLS/SSL
- Implement security headers
- Configure Kong gateway

### Week 3: Testing & Monitoring
- Add security tests (ISSUE-0015)
- Setup vulnerability scanning
- Configure security monitoring
- Implement audit logging

### Week 4: Data Protection
- Encrypt sensitive data
- Implement secrets management
- Add PII handling
- Configure backups

### Week 5: Container Hardening
- Non-root containers
- Resource limits
- Security scanning
- Runtime protection

### Week 6: Application Security
- Input validation
- XSS/CSRF protection
- SQL injection prevention
- Error handling

### Week 7: Compliance & Documentation
- GDPR compliance
- SOC2 controls
- Security policies
- Incident response plan

### Week 8: Testing & Validation
- Penetration testing
- Security audit
- Compliance review
- Final hardening

---

## Appendix A: Security Tools and Resources

### Recommended Tools
- **Scanning:** Trivy, Grype, Clair
- **SAST:** Bandit, Semgrep, SonarQube
- **DAST:** OWASP ZAP, Burp Suite
- **Secrets:** HashiCorp Vault, Sealed Secrets
- **Monitoring:** Falco, Wazuh, OSSEC
- **SIEM:** ELK Stack, Splunk, Graylog

### Security References
- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [CIS Docker Benchmark](https://www.cisecurity.org/benchmark/docker)
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)
- [SANS Security Controls](https://www.sans.org/critical-security-controls/)

### Compliance Resources
- [GDPR Compliance Checklist](https://gdpr.eu/checklist/)
- [SOC2 Trust Principles](https://www.aicpa.org/interestareas/frc/assuranceadvisoryservices/sorhome)
- [ISO 27001 Controls](https://www.iso.org/isoiec-27001-information-security.html)

---

## Appendix B: Emergency Contacts

```yaml
security_contacts:
  security_team:
    email: security@sutazai.com
    phone: "+1-555-SEC-RITY"
    oncall: "https://oncall.sutazai.com"
  
  incident_response:
    email: incident@sutazai.com
    phone: "+1-555-INC-DENT"
    slack: "#security-incidents"
  
  external:
    legal: legal@company.com
    pr: pr@company.com
    insurance: cyber@insurance.com
```

---

**Document Status:** This security guide represents the current state and required improvements for the SutazAI system. It should be reviewed weekly and updated as security measures are implemented.

**Next Review Date:** Week of August 15, 2025

**Document Owner:** Security Team

**Last Security Audit:** Never performed (scheduled for Week 8)