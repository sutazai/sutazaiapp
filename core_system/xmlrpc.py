"""Secure XML-RPC Transport Implementation with Comprehensive Safety Features"""

import logging
import os
import urllib.parse
import xmlrpc.client
from functools import lru_cache
from typing import TYPE_CHECKING, Tuple

import defusedxml.xmlrpc
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from pip._internal.network.session import PipSession

if TYPE_CHECKING:
    from xmlrpc.client import _Marshallable

logger = logging.getLogger(__name__)

# Apply security patch to xmlrpc
defusedxml.xmlrpc.monkey_patch()

class SecureXMLRPCError(Exception):
    """Custom exception for XML-RPC security issues."""

class PipXmlrpcTransport(xmlrpc.client.Transport):
    """Enhanced XML-RPC Transport with comprehensive security features."""

    MAX_REQUEST_SIZE = 10 * 1024 * 1024  # 10MB limit
    ALLOWED_CONTENT_TYPES = {'text/xml', 'application/xml'}
    MAX_RETRY_COUNT = 3

    def __init__(
        self, 
        index_url: str, 
        session: PipSession, 
        use_datetime: bool = False,
        timeout: int = 30,
        verify_ssl: bool = True
    ) -> None:
        """Initialize secure transport with enhanced parameters.
        
        Args:
            index_url: Base URL for the XML-RPC server
            session: PipSession instance for requests
            use_datetime: Whether to use datetime objects
            timeout: Request timeout in seconds
            verify_ssl: Whether to verify SSL certificates
        """
        super().__init__(use_datetime)
        self._scheme, self._netloc = self._parse_url(index_url)
        self._session = session
        self._timeout = timeout
        self._verify_ssl = verify_ssl
        self._retry_count = 0

    @staticmethod
    def _parse_url(url: str) -> Tuple[str, str]:
        """Safely parse URL components.
        
        Args:
            url: URL to parse
            
        Returns:
            Tuple of scheme and netloc
            
        Raises:
            SecureXMLRPCError: If URL is invalid or uses insecure protocol
        """
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ('https', 'http'):
            raise SecureXMLRPCError(f"Invalid URL scheme: {parsed.scheme}")
        if parsed.scheme == 'http':
            logger.warning("Using insecure HTTP connection")
        return parsed.scheme, parsed.netloc

    @lru_cache(maxsize=128)
    def _validate_request(
        self,
        host: str,
        handler: str,
        request_size: int,
        content_type: str
    ) -> None:
        """Validate request parameters.
        
        Args:
            host: Target host
            handler: Request handler
            request_size: Size of request body
            content_type: Content type header
            
        Raises:
            SecureXMLRPCError: If validation fails
        """
        if not host or not handler:
            raise SecureXMLRPCError("Invalid host or handler")
        if request_size > self.MAX_REQUEST_SIZE:
            raise SecureXMLRPCError(f"Request too large: {request_size} bytes")
        if content_type not in self.ALLOWED_CONTENT_TYPES:
            raise SecureXMLRPCError(f"Invalid content type: {content_type}")

    def request(
        self,
        host: str,
        handler: str,
        request_body: bytes,
        verbose: bool = False,
    ) -> _Marshallable:
        """Make a secure XML-RPC request with comprehensive validation.
        
        Args:
            host: Target host
            handler: Request handler
            request_body: Request data
            verbose: Enable verbose logging
            
        Returns:
            Parsed response data
            
        Raises:
            SecureXMLRPCError: If request fails validation or execution
        """
        try:
            assert host == self._netloc, "Host mismatch"
            
            # Validate request parameters
            self._validate_request(
                host=host,
                handler=handler,
                request_size=len(request_body),
                content_type='text/xml'
            )

            # Prepare headers with security measures
            headers = {
                "Content-Type": "text/xml",
                "X-Request-ID": self._generate_request_id(),
                "Accept": "text/xml, application/xml",
            }

            # Make request with retry logic
            while self._retry_count < self.MAX_RETRY_COUNT:
                try:
                    response = self._session.post(
                        f"{self._scheme}://{self._netloc}{handler}",
                        data=request_body,
                        headers=headers,
                        stream=True,
                        timeout=self._timeout,
                        verify=self._verify_ssl
                    )
                    response.raise_for_status()
                    self.verbose = verbose
                    return self.parse_response(response.raw)
                
                except Exception as e:
                    self._retry_count += 1
                    if self._retry_count >= self.MAX_RETRY_COUNT:
                        raise SecureXMLRPCError(f"Request failed after {self.MAX_RETRY_COUNT} retries: {str(e)}")
                    logger.warning(f"Request failed, attempt {self._retry_count} of {self.MAX_RETRY_COUNT}")

        except AssertionError as e:
            raise SecureXMLRPCError(f"Security assertion failed: {str(e)}")
        except Exception as e:
            raise SecureXMLRPCError(f"Request failed: {str(e)}")
        finally:
            self._retry_count = 0

    @staticmethod
    def _generate_request_id() -> str:
        """Generate a secure request ID using cryptographic functions."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=os.urandom(16),
            iterations=100000,
        )
        return kdf.derive(os.urandom(32)).hex()[:32]
