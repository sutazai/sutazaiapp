from flask import request, jsonify, g
from functools import wraps
import jwt
import datetime
from .models import User
import os
import logging
from .secure_config import get_auth_secret, get_jwt_secret

logger = logging.getLogger("SecureAuth")

# SECURITY FIX: Use secure configuration management
# Removed hardcoded fallback secret for production security
try:
    SECRET_KEY = get_auth_secret()
    logger.info("Loaded secure authentication secret")
except Exception as e:
    logger.error(f"Failed to load secure secret: {e}")
    if os.environ.get("SUTAZAI_ENV", "production") == "development":
        SECRET_KEY = "dev-only-insecure-key-do-not-use-in-production"
        logger.warning("Using development-only secret key")
    else:
        raise RuntimeError("Cannot start without secure authentication secret in production")

TOKEN_EXPIRATION = 24 * 60 * 60  # 24 hours in seconds


def generate_token(user_id):
    """
    Generate a JWT token for a user
    """
    payload = {
        "exp": datetime.datetime.utcnow()
        + datetime.timedelta(seconds=TOKEN_EXPIRATION),
        "iat": datetime.datetime.utcnow(),
        "sub": user_id,
    }
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")


def decode_token(token):
    """
    Decode a JWT token
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload["sub"]
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


def token_required(f):
    """
    Decorator for views that require authentication
    """

    @wraps(f)
    def decorated(*args, **kwargs):
        token = None

        # Get token from Authorization header
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]

        if not token:
            return jsonify({"message": "Authentication token is missing"}), 401

        user_id = decode_token(token)
        if not user_id:
            return jsonify({"message": "Invalid or expired token"}), 401

        # Get the user from the database
        user = User.query.get(user_id)
        if not user:
            return jsonify({"message": "User not found"}), 401

        # Store the user in the request context
        g.current_user = user

        return f(*args, **kwargs)

    return decorated


def admin_required(f):
    """
    Decorator for views that require admin privileges
    """

    @wraps(f)
    def decorated(*args, **kwargs):
        # First verify the token
        auth_decorator = token_required(lambda *a, **kw: None)
        result = auth_decorator()
        if result is not None:
            return result

        # Then check if user is admin
        if not g.current_user.is_admin:
            return jsonify({"message": "Admin privileges required"}), 403

        return f(*args, **kwargs)

    return decorated
