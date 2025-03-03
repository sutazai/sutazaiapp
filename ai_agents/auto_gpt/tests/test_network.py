#!/usr/bin/env python3.11
"""Tests for the network module."""

import pytest
import json
from unittest.mock import Mock, patch
from typing import Dict, Any, List
from datetime import datetime

from ai_agents.auto_gpt.src.network import (
    HTTPClient,
    HTTPRequest,
    HTTPResponse,
    HTTPError,
    NetworkError,
    TimeoutError,
    AuthenticationError,
    RateLimitError,
    RetryStrategy,
    CircuitBreaker,
    RequestInterceptor,
    ResponseInterceptor,
)


@pytest.fixture
def http_client():
    """Create a test HTTP client."""
    return HTTPClient(
        base_url="https://api.example.com",
        timeout=30,
        retry_strategy=RetryStrategy(max_retries=3),
        circuit_breaker=CircuitBreaker(
            failure_threshold=5,
            reset_timeout=60,
        ),
    )


@pytest.fixture
def mock_response():
    """Create a mock HTTP response."""
    response = Mock()
    response.status_code = 200
    response.headers = {
        "Content-Type": "application/json",
        "X-RateLimit-Limit": "100",
        "X-RateLimit-Remaining": "95",
        "X-RateLimit-Reset": "3600",
    }
    response.json.return_value = {"key": "value"}
    response.text = json.dumps({"key": "value"})
    return response


def test_http_request():
    """Test HTTP request creation."""
    # Test basic request
    request = HTTPRequest(
        method="GET",
        url="/test",
        headers={"Authorization": "Bearer token"},
    )
    assert request.method == "GET"
    assert request.url == "/test"
    assert request.headers == {"Authorization": "Bearer token"}
    assert request.params is None
    assert request.data is None
    assert request.json is None
    
    # Test request with parameters
    request = HTTPRequest(
        method="POST",
        url="/test",
        params={"key": "value"},
        data={"field": "data"},
        json={"json": "data"},
    )
    assert request.params == {"key": "value"}
    assert request.data == {"field": "data"}
    assert request.json == {"json": "data"}


def test_http_response():
    """Test HTTP response handling."""
    # Test successful response
    response = HTTPResponse(
        status_code=200,
        headers={"Content-Type": "application/json"},
        content='{"key": "value"}',
    )
    assert response.status_code == 200
    assert response.headers == {"Content-Type": "application/json"}
    assert response.json() == {"key": "value"}
    assert response.text == '{"key": "value"}'
    
    # Test error response
    response = HTTPResponse(
        status_code=404,
        headers={},
        content="Not Found",
    )
    assert response.status_code == 404
    assert response.text == "Not Found"
    with pytest.raises(HTTPError):
        response.raise_for_status()


def test_http_client_initialization(http_client):
    """Test HTTP client initialization."""
    assert http_client.base_url == "https://api.example.com"
    assert http_client.timeout == 30
    assert isinstance(http_client.retry_strategy, RetryStrategy)
    assert isinstance(http_client.circuit_breaker, CircuitBreaker)
    assert http_client.interceptors == []


def test_http_client_request(http_client, mock_response):
    """Test HTTP client request handling."""
    with patch("requests.request", return_value=mock_response) as mock_request:
        # Test GET request
        response = http_client.get("/test")
        assert response.status_code == 200
        assert response.json() == {"key": "value"}
        mock_request.assert_called_once()
        
        # Test POST request
        response = http_client.post("/test", json={"data": "test"})
        assert response.status_code == 200
        mock_request.assert_called()
        
        # Test PUT request
        response = http_client.put("/test", data={"field": "value"})
        assert response.status_code == 200
        mock_request.assert_called()
        
        # Test DELETE request
        response = http_client.delete("/test")
        assert response.status_code == 200
        mock_request.assert_called()


def test_http_client_error_handling(http_client):
    """Test HTTP client error handling."""
    with patch("requests.request", side_effect=Exception("Network error")):
        # Test network error
        with pytest.raises(NetworkError):
            http_client.get("/test")
        
        # Test timeout error
        with patch("requests.request", side_effect=TimeoutError("Timeout")):
            with pytest.raises(TimeoutError):
                http_client.get("/test")
        
        # Test authentication error
        with patch("requests.request", side_effect=AuthenticationError("Auth failed")):
            with pytest.raises(AuthenticationError):
                http_client.get("/test")
        
        # Test rate limit error
        with patch("requests.request", side_effect=RateLimitError("Rate limit exceeded")):
            with pytest.raises(RateLimitError):
                http_client.get("/test")


def test_retry_strategy():
    """Test retry strategy functionality."""
    strategy = RetryStrategy(
        max_retries=3,
        backoff_factor=2,
        status_codes=[500, 502, 503, 504],
    )
    
    # Test retry decision
    assert strategy.should_retry(500) is True
    assert strategy.should_retry(404) is False
    assert strategy.should_retry(200) is False
    
    # Test retry count
    assert strategy.retry_count == 0
    strategy.increment_retry()
    assert strategy.retry_count == 1
    
    # Test max retries
    for _ in range(3):
        strategy.increment_retry()
    assert strategy.should_retry(500) is False


def test_circuit_breaker():
    """Test circuit breaker functionality."""
    breaker = CircuitBreaker(
        failure_threshold=3,
        reset_timeout=1,
    )
    
    # Test initial state
    assert breaker.is_open() is False
    
    # Test failure handling
    for _ in range(3):
        breaker.record_failure()
    assert breaker.is_open() is True
    
    # Test reset
    breaker.reset()
    assert breaker.is_open() is False


def test_request_interceptor():
    """Test request interceptor functionality."""
    interceptor = RequestInterceptor()
    
    # Test request modification
    request = HTTPRequest(
        method="GET",
        url="/test",
        headers={},
    )
    modified = interceptor.intercept(request)
    assert modified == request
    
    # Test custom interceptor
    class CustomInterceptor(RequestInterceptor):
        def intercept(self, request: HTTPRequest) -> HTTPRequest:
            request.headers["Custom-Header"] = "value"
            return request
    
    interceptor = CustomInterceptor()
    modified = interceptor.intercept(request)
    assert modified.headers["Custom-Header"] == "value"


def test_response_interceptor():
    """Test response interceptor functionality."""
    interceptor = ResponseInterceptor()
    
    # Test response modification
    response = HTTPResponse(
        status_code=200,
        headers={},
        content="{}",
    )
    modified = interceptor.intercept(response)
    assert modified == response
    
    # Test custom interceptor
    class CustomInterceptor(ResponseInterceptor):
        def intercept(self, response: HTTPResponse) -> HTTPResponse:
            response.headers["Custom-Header"] = "value"
            return response
    
    interceptor = CustomInterceptor()
    modified = interceptor.intercept(response)
    assert modified.headers["Custom-Header"] == "value"


def test_http_client_interceptors(http_client):
    """Test HTTP client interceptor handling."""
    # Add request interceptor
    class RequestHeaderInterceptor(RequestInterceptor):
        def intercept(self, request: HTTPRequest) -> HTTPRequest:
            request.headers["X-Test-Header"] = "test"
            return request
    
    # Add response interceptor
    class ResponseHeaderInterceptor(ResponseInterceptor):
        def intercept(self, response: HTTPResponse) -> HTTPResponse:
            response.headers["X-Processed-By"] = "test"
            return response
    
    http_client.add_interceptor(RequestHeaderInterceptor())
    http_client.add_interceptor(ResponseHeaderInterceptor())
    
    with patch("requests.request", return_value=Mock()) as mock_request:
        http_client.get("/test")
        call_args = mock_request.call_args[1]
        assert call_args["headers"]["X-Test-Header"] == "test" 