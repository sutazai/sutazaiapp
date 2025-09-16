# JWT Authentication API Documentation

## Overview
The SutazAI backend implements JWT (JSON Web Token) based authentication with access and refresh tokens. The system is fully operational and running on port 10200.

## Base URL
```
http://localhost:10200/api/v1
```

## Authentication Endpoints

### 1. User Registration
**Endpoint:** `POST /api/v1/auth/register`  
**Content-Type:** `application/json`

**Request Body:**
```json
{
  "email": "user@example.com",
  "username": "johndoe",
  "password": "SecurePassword123!",
  "full_name": "John Doe"  // optional
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
  "is_superuser": false,
  "is_verified": false,
  "created_at": "2025-08-28T20:00:00Z",
  "updated_at": null,
  "last_login": null
}
```

**Error Responses:**
- `400 Bad Request`: Email already registered or username taken
- `422 Unprocessable Entity`: Invalid input data

### 2. User Login (OAuth2)
**Endpoint:** `POST /api/v1/auth/login`  
**Content-Type:** `application/x-www-form-urlencoded`  
**Note:** This endpoint uses OAuth2 form data, NOT JSON!

**Request Body (Form Data):**
```
username=johndoe&password=SecurePassword123!
```

**cURL Example:**
```bash
curl -X POST http://localhost:10200/api/v1/auth/login \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=johndoe&password=SecurePassword123!"
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
- `401 Unauthorized`: Invalid credentials
- `403 Forbidden`: Account locked or inactive

### 3. Get Current User
**Endpoint:** `GET /api/v1/auth/me`  
**Authorization:** Required (Bearer token)

**Request Headers:**
```
Authorization: Bearer <access_token>
```

**cURL Example:**
```bash
curl -X GET http://localhost:10200/api/v1/auth/me \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
```

**Response (200 OK):**
```json
{
  "id": 1,
  "email": "user@example.com",
  "username": "johndoe",
  "full_name": "John Doe",
  "is_active": true,
  "is_superuser": false,
  "is_verified": false,
  "created_at": "2025-08-28T20:00:00Z",
  "updated_at": "2025-08-28T20:05:00Z",
  "last_login": "2025-08-28T20:05:00Z"
}
```

**Error Responses:**
- `401 Unauthorized`: Invalid or expired token
- `403 Forbidden`: User account inactive

### 4. Refresh Access Token
**Endpoint:** `POST /api/v1/auth/refresh`  
**Content-Type:** `application/json`

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

### 5. Logout
**Endpoint:** `POST /api/v1/auth/logout`  
**Authorization:** Required (Bearer token)

**Request Headers:**
```
Authorization: Bearer <access_token>
```

**Response (200 OK):**
```json
{
  "message": "Successfully logged out"
}
```

### 6. Password Reset Request
**Endpoint:** `POST /api/v1/auth/password-reset`  
**Content-Type:** `application/json`

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

**Note:** Always returns success to prevent email enumeration

### 7. Confirm Password Reset
**Endpoint:** `POST /api/v1/auth/password-reset/confirm`  
**Content-Type:** `application/json`

**Request Body:**
```json
{
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "new_password": "NewSecurePassword123!"
}
```

**Response (200 OK):**
```json
{
  "message": "Password successfully reset"
}
```

**Error Responses:**
- `400 Bad Request`: Invalid or expired reset token
- `404 Not Found`: User not found

## Token Information

### Access Token
- **Expiration:** 30 minutes (1800 seconds)
- **Type:** Bearer token
- **Usage:** Include in Authorization header for protected endpoints
- **Format:** `Authorization: Bearer <access_token>`

### Refresh Token
- **Expiration:** 7 days
- **Type:** JWT token
- **Usage:** Used to obtain new access tokens without re-authentication
- **Storage:** Should be stored securely (HttpOnly cookie recommended)

## Security Features

1. **Password Requirements:**
   - Minimum 8 characters
   - Maximum 100 characters
   - Should include uppercase, lowercase, numbers, and special characters

2. **Account Lockout:**
   - Account locks after 5 failed login attempts
   - Lockout duration: 30 minutes
   - Counter resets on successful login

3. **Token Security:**
   - Tokens are signed with a secure secret key
   - Tokens include user ID and email for validation
   - Refresh tokens can be invalidated on logout

4. **Rate Limiting:**
   - General endpoints: 100 requests per minute
   - Sensitive endpoints (password reset): 5 requests per minute

## Common Issues and Solutions

### Issue: 422 Unprocessable Entity on Registration
**Solution:** Ensure JSON is properly formatted with correct escaping:
```bash
# Correct
echo '{"email":"test@example.com","username":"testuser","password":"TestPassword123"}' | curl -X POST http://localhost:10200/api/v1/auth/register -H "Content-Type: application/json" -d @-

# Incorrect (shell escaping issues)
curl -X POST http://localhost:10200/api/v1/auth/register -H "Content-Type: application/json" -d '{"email":"test@example.com","username":"testuser","password":"TestPassword123!"}'
```

### Issue: 401 Unauthorized on Login
**Solution:** Login endpoint uses OAuth2 form data, not JSON:
```bash
# Correct (form data)
curl -X POST http://localhost:10200/api/v1/auth/login \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=testuser&password=TestPassword123"

# Incorrect (JSON)
curl -X POST http://localhost:10200/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"testuser","password":"TestPassword123"}'
```

### Issue: 401 on Protected Endpoints
**Solution:** Include the Bearer token in the Authorization header:
```bash
# Correct
curl -X GET http://localhost:10200/api/v1/auth/me \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."

# Incorrect (missing Bearer prefix)
curl -X GET http://localhost:10200/api/v1/auth/me \
  -H "Authorization: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
```

## Testing the Authentication System

A comprehensive test script is available at `/opt/sutazaiapp/backend/test_auth.py`:

```bash
cd /opt/sutazaiapp/backend
python3 test_auth.py
```

This script tests all authentication endpoints and provides detailed feedback on the system status.

## Integration Examples

### Python (requests)
```python
import requests

# Register
response = requests.post(
    "http://localhost:10200/api/v1/auth/register",
    json={
        "email": "user@example.com",
        "username": "johndoe",
        "password": "SecurePassword123!"
    }
)

# Login
response = requests.post(
    "http://localhost:10200/api/v1/auth/login",
    data={
        "username": "johndoe",
        "password": "SecurePassword123!"
    }
)
tokens = response.json()

# Use authenticated endpoint
response = requests.get(
    "http://localhost:10200/api/v1/auth/me",
    headers={
        "Authorization": f"Bearer {tokens['access_token']}"
    }
)
```

### JavaScript (fetch)
```javascript
// Register
const registerResponse = await fetch('http://localhost:10200/api/v1/auth/register', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    email: 'user@example.com',
    username: 'johndoe',
    password: 'SecurePassword123!'
  })
});

// Login
const formData = new URLSearchParams();
formData.append('username', 'johndoe');
formData.append('password', 'SecurePassword123!');

const loginResponse = await fetch('http://localhost:10200/api/v1/auth/login', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/x-www-form-urlencoded'
  },
  body: formData
});

const tokens = await loginResponse.json();

// Use authenticated endpoint
const userResponse = await fetch('http://localhost:10200/api/v1/auth/me', {
  headers: {
    'Authorization': `Bearer ${tokens.access_token}`
  }
});
```

## System Status
✅ **All JWT authentication endpoints are fully operational**
- Backend running on port 10200
- Database connection established
- Token generation and validation working
- All security features active