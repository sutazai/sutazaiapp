"""
NTLM authenticating pool, contributed by erikcederstran

Issue #10, see: http://code.google.com/p/urllib3/issues/detail?id=10
"""

from __future__ import absolute_import

import warnings
from logging import getLogger
from typing import Any, Dict, Optional, Tuple, Union

# Try to import ntlm if available
try:
    from ntlm import ntlm
except ImportError:
    ntlm = None


# Define base classes that will be used regardless of import success
class BaseHTTPSConnection:
    """Base HTTPS Connection class with minimal required functionality."""

    def __init__(
        self, host: str, port: Optional[int] = None, **kwargs: Any
    ) -> None:
        self.host = host
        self.port = port
        self.headers: Dict[str, str] = {}

    def request(
        self,
        method: str,
        url: str,
        body: Optional[bytes] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        self.headers = headers or {}

    def getresponse(self) -> Any:
        pass


class BaseHTTPSConnectionPool:
    """Base HTTPS Connection Pool class with minimal required functionality."""

    def __init__(
        self, host: str, port: Optional[int] = None, **kwargs: Any
    ) -> None:
        self.host = host
        self.port = port
        self.num_connections = 0
        self.headers: Dict[str, str] = {}


# Try to import from urllib3, fall back to base classes if not available
try:
    from urllib3 import HTTPSConnectionPool
    from urllib3.packages.six.moves.http_client import HTTPSConnection
except ImportError:
    HTTPSConnectionPool = BaseHTTPSConnectionPool
    HTTPSConnection = BaseHTTPSConnection

warnings.warn(
    "The 'urllib3.contrib.ntlmpool' module is deprecated and will be removed "
    "in urllib3 v2.0 release, urllib3 is not able to support it properly due "
    "to reasons listed in issue: https://github.com/urllib3/urllib3/issues/2282. "
    "If you are a user of this module please comment in the mentioned issue.",
    DeprecationWarning,
)

log = getLogger(__name__)


class NTLMConnectionPool(HTTPSConnectionPool):
    """
    HTTPS Connection pool with NTLM authentication

    Supports NTLM authentication: http://en.wikipedia.org/wiki/NT_LAN_Manager
    """

    scheme = "https"

    def __init__(
        self, user: str, pw: str, authurl: str, *args: Any, **kwargs: Any
    ) -> None:
        """
        authurl is a random URL on the server that is protected by NTLM.
        user is the Windows user, probably in the DOMAIN\\User format.
        pw is the password for the user.
        """
        super(NTLMConnectionPool, self).__init__(*args, **kwargs)
        self.authurl = authurl
        self.rawuser = user
        self.rawpw = pw

        # Construct the domain and username
        if "\\" in user:
            self.domain, self.user = user.split("\\", 1)
        else:
            self.domain = None
            self.user = user

        # Create the authentication manager
        self.manager = None
        if ntlm is not None:
            self.manager = ntlm

    def _new_conn(self) -> HTTPSConnection:
        # Performs the NTLM handshake that secures the connection
        self.num_connections += 1

        log.debug(
            "Starting NTLM HTTPS connection no. %d: https://%s:%s",
            self.num_connections,
            self.host,
            self.port,
        )

        conn = HTTPSConnection(self.host, self.port)

        if self.manager is None:
            warnings.warn(
                "NTLM authentication requires the 'ntlm' package to be installed",
                ImportWarning,
            )
            return conn

        # Send negotiation message
        headers = {"Connection": "Keep-Alive"}
        req_header = self.manager.create_NTLM_negotiate_message(self.rawuser)
        headers["Authorization"] = "NTLM %s" % req_header.decode("ascii")
        conn.request("GET", self.authurl, None, headers)
        res = conn.getresponse()

        # Parse challenge message
        try:
            auth_header_values = res.headers.get("www-authenticate", "")
            if auth_header_values:
                auth_header_value = None
                for h in auth_header_values.split(","):
                    if h.startswith("NTLM"):
                        auth_header_value = h.strip().split(" ")[1]
                if auth_header_value:
                    server_challenge = auth_header_value

                    # Send response
                    headers = {"Connection": "Keep-Alive"}
                    auth_msg = self.manager.create_NTLM_authenticate_message(
                        server_challenge, self.user, self.domain, self.rawpw
                    )
                    headers["Authorization"] = "NTLM %s" % auth_msg.decode(
                        "ascii"
                    )
                    conn.request("GET", self.authurl, None, headers)
                    res = conn.getresponse()
                    res.read()

                    log.debug("NTLM HTTPS connection established")
                    return conn
        except Exception as e:
            log.error("NTLM handshake failed: %s", e)

        log.warning("NTLM handshake failed")
        return conn

    def urlopen(
        self,
        method,
        url,
        body=None,
        headers=None,
        retries=3,
        redirect=True,
        assert_same_host=True,
    ):
        if headers is None:
            headers = {}
        headers["Connection"] = "Keep-Alive"
        return super(NTLMConnectionPool, self).urlopen(
            method, url, body, headers, retries, redirect, assert_same_host
        )
