# SutazAI Backend API - Complete Documentation

**Version:** 1.0.0  
**Last Updated:** 2025-11-15 18:00:00 UTC  
**Base URL:** `http://localhost:10200/api/v1`

## Table of Contents

1. [Overview](#overview)
2. [Authentication](#authentication)
3. [API Endpoints](#api-endpoints)
4. [Error Codes](#error-codes)
5. [Rate Limiting](#rate-limiting)
6. [Security](#security)
7. [Examples](#examples)

---

## Overview

The SutazAI Backend API provides comprehensive authentication, user management, and AI integration capabilities. All endpoints follow REST conventions and return JSON responses.

### Base URL
```
Production: https://api.sutazai.com/api/v1
Development: http://localhost:10200/api/v1
```

### API Version
Current version: `v1`

### Content Type
All requests and responses use `application/json` content type.

---

## Authentication

### JWT Bearer Token

The API uses JWT (JSON Web Tokens) for authentication. Include the token in the Authorization header:

```http
Authorization: Bearer <your_access_token>
```

### Token Types

1. **Access Token**
   - Short-lived (30 minutes)
   - Used for API authentication
   - Required for protected endpoints

2. **Refresh Token**
   - Long-lived (7 days)
   - Used to obtain new access tokens
   - Stored securely, invalidated on logout

---

## API Endpoints

### 1. User Registration

**Endpoint:** `POST /auth/register`

**Description:** Register a new user account

**Authentication:** Not required

**Request Body:**
```json
{
  "email": "user@example.com",
  "username": "johndoe",
  "password": "SecureP@ssw0rd123!",
  "full_name": "John Doe" // optional
}
```

**Response (201 Created):**
```json
{
  "id": 1,
  "email": "user@example.com",
  "username": "johndoe",
  "full_name": "John Doe",
  "is_active": true,
  "is_verified": false,
  "is_superuser": false,
  "created_at": "2025-11-15T18:00:00Z"
}
```

**Validation Rules:**
- Email: Valid email format, unique
- Username: 3-20 characters, alphanumeric + underscore, unique
- Password: Minimum 8 characters, must include:
  - At least one uppercase letter
  - At least one lowercase letter
  - At least one number
  - At least one special character

**Error Responses:**
- `400 Bad Request`: Invalid data, duplicate email/username, weak password
- `422 Unprocessable Entity`: Validation errors

---

### 2. User Login

**Endpoint:** `POST /auth/login`

**Description:** Authenticate user and receive JWT tokens

**Authentication:** Not required

**Request Body (Form Data):**
```
username=user@example.com
password=SecureP@ssw0rd123!
```

**Response (200 OK):**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 1800
}
```

**Notes:**
- Username field accepts both username and email
- Account locks after 5 failed attempts for 30 minutes
- Successful login resets failed attempt counter

**Error Responses:**
- `401 Unauthorized`: Invalid credentials
- `403 Forbidden`: Account locked
- `422 Unprocessable Entity`: Missing fields

---

### 3. Token Refresh

**Endpoint:** `POST /auth/refresh`

**Description:** Obtain new access token using refresh token

**Authentication:** Not required (uses refresh token)

**Request Body:**
```json
{
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

**Response (200 OK):**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 1800
}
```

**Error Responses:**
- `401 Unauthorized`: Invalid or expired refresh token

---

### 4. Logout

**Endpoint:** `POST /auth/logout`

**Description:** Invalidate refresh token and logout

**Authentication:** Required (Bearer token)

**Response (200 OK):**
```json
{
  "message": "Successfully logged out"
}
```

---

### 5. Get Current User

**Endpoint:** `GET /auth/me`

**Description:** Retrieve authenticated user's information

**Authentication:** Required (Bearer token)

**Response (200 OK):**
```json
{
  "id": 1,
  "email": "user@example.com",
  "username": "johndoe",
  "full_name": "John Doe",
  "is_active": true,
  "is_verified": true,
  "is_superuser": false,
  "created_at": "2025-11-15T18:00:00Z",
  "last_login": "2025-11-15T18:30:00Z"
}
```

**Error Responses:**
- `401 Unauthorized`: Invalid or missing token

---

### 6. Request Password Reset

**Endpoint:** `POST /auth/password-reset`

**Description:** Request password reset email

**Authentication:** Not required

**Rate Limit:** 5 requests per minute (strict)

**Request Body:**
```json
{
  "email": "user@example.com"
}
```

**Response (200 OK):**
```json
{
  "message": "If the email exists, a password reset link has been sent"
}
```

**Notes:**
- Always returns success to prevent email enumeration
- Reset token expires in 1 hour
- Email contains reset link: `https://app.sutazai.com/reset-password?token=...`

**Error Responses:**
- `429 Too Many Requests`: Rate limit exceeded

---

### 7. Confirm Password Reset

**Endpoint:** `POST /auth/password-reset/confirm`

**Description:** Reset password using reset token

**Authentication:** Not required

**Request Body:**
```json
{
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "new_password": "NewSecureP@ssw0rd123!"
}
```

**Response (200 OK):**
```json
{
  "message": "Password successfully reset"
}
```

**Error Responses:**
- `400 Bad Request`: Invalid or expired token
- `404 Not Found`: User not found

---

### 8. Verify Email

**Endpoint:** `POST /auth/verify-email/{token}`

**Description:** Verify user's email address

**Authentication:** Not required

**Path Parameters:**
- `token`: Email verification token (string)

**Response (200 OK):**
```json
{
  "message": "Email successfully verified"
}
```

**Error Responses:**
- `400 Bad Request`: Invalid or expired token
- `404 Not Found`: User not found

---

### 9. Health Check

**Endpoint:** `GET /health`

**Description:** Basic health check

**Authentication:** Not required

**Response (200 OK):**
```json
{
  "status": "healthy",
  "app": "SutazAI Backend API"
}
```

---

### 10. Detailed Health Check

**Endpoint:** `GET /health/detailed`

**Description:** Comprehensive health check for all services

**Authentication:** Not required

**Response (200 OK):**
```json
{
  "status": "healthy",
  "app": "SutazAI Backend API",
  "version": "1.0.0",
  "services": {
    "redis": true,
    "rabbitmq": true,
    "neo4j": true,
    "chromadb": true,
    "qdrant": true,
    "faiss": true,
    "consul": true,
    "kong": true,
    "ollama": true
  },
  "healthy_count": 9,
  "total_services": 9
}
```

**Response (503 Service Unavailable):**
```json
{
  "status": "degraded",
  "services": { ... },
  "healthy_count": 6,
  "total_services": 9
}
```

---

### 11. Prometheus Metrics

**Endpoint:** `GET /metrics`

**Description:** Prometheus-compatible metrics endpoint

**Authentication:** Not required

**Content-Type:** `text/plain; version=0.0.4`

**Response Example:**
```
# HELP http_requests_total Total HTTP requests
# TYPE http_requests_total counter
http_requests_total{method="GET",endpoint="/health",status_code="200"} 1234.0

# HELP http_request_duration_seconds HTTP request latency
# TYPE http_request_duration_seconds histogram
http_request_duration_seconds_bucket{method="GET",endpoint="/health",le="0.005"} 1000.0
http_request_duration_seconds_bucket{method="GET",endpoint="/health",le="0.01"} 1200.0

# HELP auth_login_total Total login attempts
# TYPE auth_login_total counter
auth_login_total{status="success"} 500.0
auth_login_total{status="failure"} 25.0

# HELP db_connection_pool_size Database connection pool size
# TYPE db_connection_pool_size gauge
db_connection_pool_size 10.0
```

**Metrics Categories:**
- HTTP metrics (requests, duration, size, errors)
- Authentication metrics (logins, lockouts, resets)
- Database metrics (queries, connections, errors)
- Cache metrics (hits, misses, operations)
- RabbitMQ metrics (messages published/consumed)
- Vector DB metrics (operations, duration)
- External API metrics (calls, errors)

---

## Error Codes

### HTTP Status Codes

| Code | Meaning | Description |
|------|---------|-------------|
| 200 | OK | Request successful |
| 201 | Created | Resource created successfully |
| 400 | Bad Request | Invalid request data |
| 401 | Unauthorized | Authentication required or invalid |
| 403 | Forbidden | Insufficient permissions or account locked |
| 404 | Not Found | Resource not found |
| 422 | Unprocessable Entity | Validation errors |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Server error |
| 503 | Service Unavailable | Service temporarily unavailable |

### Error Response Format

```json
{
  "detail": "Error message describing what went wrong"
}
```

### Common Errors

**Invalid Token:**
```json
{
  "detail": "Could not validate credentials"
}
```

**Account Locked:**
```json
{
  "detail": "Account locked due to multiple failed login attempts. Try again in 30 minutes."
}
```

**Rate Limit Exceeded:**
```json
{
  "detail": "Too many requests. Retry after 60 seconds."
}
```

**Validation Error:**
```json
{
  "detail": [
    {
      "loc": ["body", "password"],
      "msg": "Password must be at least 8 characters",
      "type": "value_error"
    }
  ]
}
```

---

## Rate Limiting

### Global Rate Limits

| Endpoint Pattern | Limit | Window |
|-----------------|-------|--------|
| General API | 100 requests | 1 minute |
| Auth - Login | 10 requests | 1 minute |
| Auth - Password Reset | 5 requests | 1 minute |
| Auth - Registration | 10 requests | 1 hour |

### Rate Limit Headers

Responses include rate limit information:

```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1700000000
Retry-After: 60
```

### Rate Limit Response

```json
{
  "detail": "Rate limit exceeded. Too many requests."
}
```

---

## Security

### Password Requirements

- **Minimum Length:** 8 characters
- **Maximum Length:** 100 characters
- **Required Characters:**
  - At least one uppercase letter
  - At least one lowercase letter
  - At least one number
  - At least one special character

### Account Security

1. **Account Lockout**
   - Triggers after 5 failed login attempts
   - Duration: 30 minutes
   - Automatically unlocks after lockout period

2. **Token Expiration**
   - Access tokens: 30 minutes
   - Refresh tokens: 7 days
   - Password reset tokens: 1 hour
   - Email verification tokens: 24 hours

3. **Password Hashing**
   - Algorithm: bcrypt
   - Work factor: 12 rounds
   - Salted automatically

### Security Headers

All responses include security headers:

```http
X-Frame-Options: DENY
X-Content-Type-Options: nosniff
X-XSS-Protection: 1; mode=block
Strict-Transport-Security: max-age=31536000; includeSubDomains
Content-Security-Policy: default-src 'self'
```

### Request Correlation

Each request receives a unique correlation ID for tracking:

```http
X-Correlation-ID: 550e8400-e29b-41d4-a716-446655440000
X-Process-Time: 0.145
```

---

## Examples

### Complete Authentication Flow

#### 1. Register New User

```bash
curl -X POST http://localhost:10200/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "john@example.com",
    "username": "johndoe",
    "password": "SecureP@ssw0rd123!",
    "full_name": "John Doe"
  }'
```

#### 2. Login

```bash
curl -X POST http://localhost:10200/api/v1/auth/login \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=johndoe&password=SecureP@ssw0rd123!"
```

Response:
```json
{
  "access_token": "eyJhbGci...",
  "refresh_token": "eyJhbGci...",
  "token_type": "bearer",
  "expires_in": 1800
}
```

#### 3. Access Protected Endpoint

```bash
curl -X GET http://localhost:10200/api/v1/auth/me \
  -H "Authorization: Bearer eyJhbGci..."
```

#### 4. Refresh Token

```bash
curl -X POST http://localhost:10200/api/v1/auth/refresh \
  -H "Content-Type: application/json" \
  -d '{
    "refresh_token": "eyJhbGci..."
  }'
```

#### 5. Logout

```bash
curl -X POST http://localhost:10200/api/v1/auth/logout \
  -H "Authorization: Bearer eyJhbGci..."
```

### Password Reset Flow

#### 1. Request Password Reset

```bash
curl -X POST http://localhost:10200/api/v1/auth/password-reset \
  -H "Content-Type: application/json" \
  -d '{
    "email": "john@example.com"
  }'
```

#### 2. Confirm Password Reset (using token from email)

```bash
curl -X POST http://localhost:10200/api/v1/auth/password-reset/confirm \
  -H "Content-Type: application/json" \
  -d '{
    "token": "eyJhbGci...",
    "new_password": "NewSecureP@ssw0rd456!"
  }'
```

### Email Verification

```bash
curl -X POST http://localhost:10200/api/v1/auth/verify-email/eyJhbGci...
```

### Health Monitoring

```bash
# Basic health check
curl http://localhost:10200/health

# Detailed service health
curl http://localhost:10200/health/detailed

# Prometheus metrics
curl http://localhost:10200/metrics
```

---

## SDK Examples

### Python

```python
import httpx

class SutazAIClient:
    def __init__(self, base_url="http://localhost:10200"):
        self.base_url = base_url
        self.access_token = None
        self.refresh_token = None
    
    async def register(self, email, username, password, full_name=None):
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/v1/auth/register",
                json={
                    "email": email,
                    "username": username,
                    "password": password,
                    "full_name": full_name
                }
            )
            return response.json()
    
    async def login(self, username, password):
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/v1/auth/login",
                data={"username": username, "password": password}
            )
            data = response.json()
            self.access_token = data["access_token"]
            self.refresh_token = data["refresh_token"]
            return data
    
    async def get_me(self):
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/api/v1/auth/me",
                headers={"Authorization": f"Bearer {self.access_token}"}
            )
            return response.json()
```

### JavaScript/TypeScript

```typescript
class SutazAIClient {
  private baseUrl: string;
  private accessToken?: string;
  private refreshToken?: string;

  constructor(baseUrl = "http://localhost:10200") {
    this.baseUrl = baseUrl;
  }

  async register(email: string, username: string, password: string, fullName?: string) {
    const response = await fetch(`${this.baseUrl}/api/v1/auth/register`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, username, password, full_name: fullName })
    });
    return response.json();
  }

  async login(username: string, password: string) {
    const response = await fetch(`${this.baseUrl}/api/v1/auth/login`, {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: new URLSearchParams({ username, password })
    });
    const data = await response.json();
    this.accessToken = data.access_token;
    this.refreshToken = data.refresh_token;
    return data;
  }

  async getMe() {
    const response = await fetch(`${this.baseUrl}/api/v1/auth/me`, {
      headers: { "Authorization": `Bearer ${this.accessToken}` }
    });
    return response.json();
  }
}
```

---

## Support

For issues or questions:
- **GitHub Issues:** https://github.com/sutazai/sutazaiapp/issues
- **Documentation:** https://deepwiki.com/sutazai/sutazaiapp
- **Email:** support@sutazai.com

---

**Document Version:** 1.0.0  
**API Version:** v1  
**Last Updated:** 2025-11-15 18:00:00 UTC
