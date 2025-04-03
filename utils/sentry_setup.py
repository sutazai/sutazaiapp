"""
Sentry integration module for SutazAI

This module provides standardized Sentry setup for error tracking and performance monitoring
within the SutazAI FastAPI application.
"""

import os
import logging
from typing import Dict, Any, Optional

# Import Sentry if available
try:
    import sentry_sdk
    from sentry_sdk.integrations.fastapi import FastApiIntegration
    from sentry_sdk.integrations.starlette import StarletteIntegration
    from sentry_sdk.integrations.logging import LoggingIntegration
    from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration
    from sentry_sdk.integrations.redis import RedisIntegration

    SENTRY_AVAILABLE = True
except ImportError:
    SENTRY_AVAILABLE = False

# Get environment settings
ENV = os.environ.get("ENVIRONMENT", "development")
SENTRY_DSN = os.environ.get("SENTRY_DSN", None)
TRACES_SAMPLE_RATE = float(os.environ.get("SENTRY_TRACES_SAMPLE_RATE", "0.1"))
PROFILES_SAMPLE_RATE = float(os.environ.get("SENTRY_PROFILES_SAMPLE_RATE", "0.1"))
SEND_DEFAULT_PII = os.environ.get("SENTRY_SEND_DEFAULT_PII", "false").lower() == "true"

logger = logging.getLogger(__name__)


def before_send(
    event: Dict[str, Any], hint: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Filter sensitive information from Sentry events before sending.

    Args:
        event: The event to be sent
        hint: Contextual information about the event

    Returns:
        Modified event or None if the event should be discarded
    """
    # Exclude certain errors if needed
    if "exc_info" in hint:
        exc_type, exc_value, tb = hint["exc_info"]
        if isinstance(exc_value, (KeyboardInterrupt, SystemExit)):
            return None

    # Sanitize request data if present
    if "request" in event and "headers" in event["request"]:
        headers = event["request"]["headers"]
        for sensitive_header in ["authorization", "cookie", "x-api-key"]:
            if sensitive_header in headers:
                headers[sensitive_header] = "[Filtered]"

    return event


def setup_sentry(app=None):
    """
    Initialize Sentry SDK for the FastAPI application.

    Args:
        app: FastAPI application instance (optional)
    """
    if not SENTRY_AVAILABLE:
        logger.warning("Sentry SDK not available. Skipping Sentry setup.")
        return

    if not SENTRY_DSN:
        logger.info("Sentry DSN not configured. Skipping Sentry setup.")
        return

    # Set up logging integration
    logging_integration = LoggingIntegration(
        level=logging.INFO,  # Capture info and above as breadcrumbs
        event_level=logging.ERROR,  # Send errors as events
    )

    # Initialize Sentry
    logger.info(
        f"Initializing Sentry in {ENV} environment with traces_sample_rate={TRACES_SAMPLE_RATE}"
    )
    sentry_sdk.init(
        dsn=SENTRY_DSN,
        environment=ENV,
        before_send=before_send,
        traces_sample_rate=TRACES_SAMPLE_RATE,
        profiles_sample_rate=PROFILES_SAMPLE_RATE,
        send_default_pii=SEND_DEFAULT_PII,
        integrations=[
            FastApiIntegration(),
            StarletteIntegration(),
            logging_integration,
            SqlalchemyIntegration(),
            RedisIntegration(),
        ],
        # Set maximum length of serialized values
        max_value_length=1024,
    )

    # Add Sentry middleware to FastAPI app if provided
    if app is not None:
        logger.info("Adding Sentry middleware to FastAPI application")
        # The FastAPI integration automatically adds middleware, no extra steps needed

        # Setup performance monitoring route for testing
        @app.get("/sentry-debug", include_in_schema=False)
        async def trigger_error():
            division_by_zero = 1 / 0
            return {"result": division_by_zero}


def capture_message(
    message: str, level: str = "info", tags: Optional[Dict[str, str]] = None
):
    """
    Capture a message in Sentry.

    Args:
        message: The message to capture
        level: The log level ('info', 'warning', 'error', 'fatal')
        tags: Optional tags to attach to the message
    """
    if not SENTRY_AVAILABLE or not SENTRY_DSN:
        return

    with sentry_sdk.configure_scope() as scope:
        if tags:
            for key, value in tags.items():
                scope.set_tag(key, value)

        sentry_sdk.capture_message(message, level=level)


def set_user(user_id: str, email: Optional[str] = None, username: Optional[str] = None):
    """
    Set the current user for Sentry events.

    Args:
        user_id: User ID
        email: User email
        username: Username
    """
    if not SENTRY_AVAILABLE or not SENTRY_DSN:
        return

    sentry_sdk.set_user({"id": user_id, "email": email, "username": username})


def set_tag(key: str, value: str):
    """
    Set a tag for all future events in the current scope.

    Args:
        key: Tag key
        value: Tag value
    """
    if not SENTRY_AVAILABLE or not SENTRY_DSN:
        return

    sentry_sdk.set_tag(key, value)


def start_transaction(name: str, op: str):
    """
    Start a new transaction for performance monitoring.

    Args:
        name: Transaction name
        op: Operation name

    Returns:
        A transaction object that can be used as a context manager
    """
    if not SENTRY_AVAILABLE or not SENTRY_DSN:

        class NoopTransaction:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                pass

            def set_tag(self, key, value):
                pass

        return NoopTransaction()

    return sentry_sdk.start_transaction(name=name, op=op)


def capture_exception(exception: Exception, context: Optional[Dict[str, Any]] = None):
    """
    Capture an exception in Sentry.

    Args:
        exception: The exception to capture
        context: Additional context for the exception
    """
    if not SENTRY_AVAILABLE or not SENTRY_DSN:
        return

    with sentry_sdk.configure_scope() as scope:
        if context:
            for key, value in context.items():
                scope.set_extra(key, value)

        sentry_sdk.capture_exception(exception)
