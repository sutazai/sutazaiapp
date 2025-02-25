"""
SutazAI API Client

Robust API client with advanced error handling, retries, and logging.
"""

import logging
import time
from typing import Any, Dict, Optional, Union

import requests
from requests.adapters import HTTPAdapter
from requests.exceptions import (
    ConnectionError,
    HTTPError,
    RequestException,
    Timeout,
    TooManyRedirects,
)
from urllib3.util.retry import Retry

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler(
    "/opt/sutazai_project/SutazAI/logs/api_client.log"
)
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(handler)


class APIClient:
    """Robust API client with advanced error handling and retries."""

    def __init__(
        self,
        base_url: str,
        timeout: int = 30,
        max_retries: int = 3,
        backoff_factor: float = 0.5,
        verify_ssl: bool = True,
    ):
        """
        Initialize API client.

        Args:
            base_url: Base URL for API requests
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
            backoff_factor: Backoff factor between retries
            verify_ssl: Whether to verify SSL certificates
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.verify_ssl = verify_ssl

        # Configure session with retries
        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
    ) -> Union[Dict[str, Any], None]:
        """
        Make an API request with error handling and retries.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            params: Query parameters
            data: Request body data
            headers: Request headers
            timeout: Request timeout (overrides default)

        Returns:
            Response data or None if request failed
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        timeout = timeout or self.timeout
        start_time = time.time()

        try:
            logger.info(f"Making {method} request to {url}")
            response = self.session.request(
                method=method,
                url=url,
                params=params,
                json=data,
                headers=headers,
                timeout=timeout,
                verify=self.verify_ssl,
            )
            response.raise_for_status()

            elapsed = time.time() - start_time
            logger.info(
                f"Request successful: {method} {url} "
                f"(Status: {response.status_code}, Time: {elapsed:.2f}s)"
            )

            return response.json()

        except HTTPError as e:
            logger.error(
                f"HTTP error occurred: {e} "
                f"(Status: {e.response.status_code}, URL: {url})"
            )
            if e.response.status_code == 429:
                retry_after = e.response.headers.get("Retry-After", "60")
                logger.warning(
                    f"Rate limited. Retry after {retry_after} seconds"
                )
            return None

        except ConnectionError as e:
            logger.error(f"Connection error: {e} (URL: {url})")
            return None

        except Timeout as e:
            logger.error(
                f"Request timed out after {timeout}s: {e} (URL: {url})"
            )
            return None

        except TooManyRedirects as e:
            logger.error(f"Too many redirects: {e} (URL: {url})")
            return None

        except RequestException as e:
            logger.error(f"Request failed: {e} (URL: {url})")
            return None

        except Exception as e:
            logger.error(f"Unexpected error: {e} (URL: {url})")
            return None

    def get(
        self, endpoint: str, params: Optional[Dict[str, Any]] = None, **kwargs
    ) -> Union[Dict[str, Any], None]:
        """Make GET request."""
        return self.request("GET", endpoint, params=params, **kwargs)

    def post(
        self, endpoint: str, data: Optional[Dict[str, Any]] = None, **kwargs
    ) -> Union[Dict[str, Any], None]:
        """Make POST request."""
        return self.request("POST", endpoint, data=data, **kwargs)

    def put(
        self, endpoint: str, data: Optional[Dict[str, Any]] = None, **kwargs
    ) -> Union[Dict[str, Any], None]:
        """Make PUT request."""
        return self.request("PUT", endpoint, data=data, **kwargs)

    def delete(self, endpoint: str, **kwargs) -> Union[Dict[str, Any], None]:
        """Make DELETE request."""
        return self.request("DELETE", endpoint, **kwargs)

    def patch(
        self, endpoint: str, data: Optional[Dict[str, Any]] = None, **kwargs
    ) -> Union[Dict[str, Any], None]:
        """Make PATCH request."""
        return self.request("PATCH", endpoint, data=data, **kwargs)
