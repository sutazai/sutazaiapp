import json
import statistics
from typing import Any, Dict, List

from locust import HttpUser, between, task


class SutazAIPerformanceTest(HttpUser):
    wait_time = between(1, 5)  # Wait 1-5 seconds between tasks
    performance_data: List[Dict[str, Any]] = []

    def on_start(self):
        """Simulate user login or initialization"""
        login_response = self.client.post("/auth/login", json={
            "username": "test_user",
            "password": "test_password"
        })
        assert login_response.status_code == 200, "Login failed"
        self.token = login_response.json().get('access_token')

    @task(3)  # Most common task
    def health_check(self):
        """Simulate health check endpoint"""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
                self.performance_data.append({
                    "endpoint": "/health",
                    "response_time": response.elapsed.total_seconds(),
                    "status": "success"
                })
            else:
                response.failure("Health check failed")

    @task(2)
    def test_core_api(self):
        """Test core API functionality with advanced tracking"""
        headers = {"Authorization": f"Bearer {self.token}"}
        with self.client.get("/api/v1/core", headers=headers, catch_response=True) as response:
            if response.status_code == 200 and "data" in response.json():
                response.success()
                self.performance_data.append({
                    "endpoint": "/api/v1/core",
                    "response_time": response.elapsed.total_seconds(),
                    "payload_size": len(response.content),
                    "status": "success"
                })
            else:
                response.failure("Core API endpoint failed")

    @task(1)
    def test_complex_query(self):
        """Simulate a more complex query with detailed tracking"""
        headers = {"Authorization": f"Bearer {self.token}"}
        payload = {
            "query": "complex_analysis",
            "parameters": {
                "depth": 5,
                "complexity": "high"
            }
        }
        with self.client.post("/api/v1/analyze", json=payload, headers=headers, catch_response=True) as response:
            if response.status_code in [200, 202]:
                response.success()
                self.performance_data.append({
                    "endpoint": "/api/v1/analyze",
                    "response_time": response.elapsed.total_seconds(),
                    "payload_size": len(response.content),
                    "status": "success"
                })
            else:
                response.failure("Complex query failed")

    def on_stop(self):
        """Logout and process performance data"""
        self.client.post("/auth/logout")
        self._analyze_performance_data()

    def _analyze_performance_data(self):
        """Comprehensive performance data analysis"""
        if not self.performance_data:
            return

        response_times = [data['response_time'] for data in self.performance_data]
        
        performance_summary = {
            "total_requests": len(self.performance_data),
            "successful_requests": len([d for d in self.performance_data if d['status'] == 'success']),
            "avg_response_time": statistics.mean(response_times),
            "median_response_time": statistics.median(response_times),
            "min_response_time": min(response_times),
            "max_response_time": max(response_times),
            "response_time_std_dev": statistics.stdev(response_times) if len(response_times) > 1 else 0
        }

        print("Performance Summary:", json.dumps(performance_summary, indent=2))

# Performance test configuration
def test_performance_thresholds(performance_data):
    """Advanced performance threshold validation"""
    assert performance_data['total_requests'] > 10, "Insufficient test coverage"
    assert performance_data['avg_response_time'] < 0.5, "High average response time"
    assert performance_data['successful_requests'] / performance_data['total_requests'] > 0.95, "High failure rate"
    assert performance_data['max_response_time'] < 2.0, "Extreme response time outliers"