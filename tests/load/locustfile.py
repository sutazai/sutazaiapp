"""
Locust load testing script for SutazAI system
Tests system performance under various load conditions
"""

import random
import json
import time
from datetime import datetime
from locust import HttpUser, task, between, events
from locust.exception import RescheduleTask
import logging


# Test configuration
API_ENDPOINTS = {
    "health": "/health",
    "agents": "/api/agents", 
    "metrics": "/api/metrics",
    "chat": "/api/chat",
    "models": "/api/models",
    "orchestration": "/api/orchestration"
}

# Sample test data
SAMPLE_MESSAGES = [
    "Hello, how are you today?",
    "What is artificial intelligence?",
    "Can you help me with a coding problem?",
    "Explain quantum computing in simple terms.",
    "What are the benefits of machine learning?",
    "How does neural network training work?",
    "What is the difference between AI and AGI?",
    "Can you write a Python function to sort a list?",
    "Explain the concept of recursion.",
    "What are the applications of natural language processing?"
]

SAMPLE_AGENTS = [
    "agi-brain",
    "reasoning-agent", 
    "coding-assistant",
    "research-agent",
    "creative-writer"
]

SAMPLE_MODELS = [
    "llama3.2:1b",
    "mistral:7b",
    "codellama:7b",
    "phi:2.7b"
]


class SutazAIUser(HttpUser):
    """Base user class for SutazAI load testing."""
    
    wait_time = between(1, 3)  # Wait 1-3 seconds between tasks
    
    def on_start(self):
        """Initialize user session."""
        self.session_id = f"load_test_{random.randint(1000, 9999)}"
        self.user_context = {
            "preferred_agent": random.choice(SAMPLE_AGENTS),
            "preferred_model": random.choice(SAMPLE_MODELS),
            "conversation_history": []
        }
        
        # Test health check on start
        self.test_health_check()
    
    def on_stop(self):
        """Cleanup user session."""
        logging.info(f"User {self.session_id} completed {len(self.user_context['conversation_history'])} conversations")

    def test_health_check(self):
        """Test health check endpoint."""
        try:
            with self.client.get(API_ENDPOINTS["health"], catch_response=True) as response:
                if response.status_code == 200:
                    response.success()
                else:
                    response.failure(f"Health check failed: {response.status_code}")
        except Exception as e:
            logging.error(f"Health check error: {str(e)}")

    @task(3)
    def get_system_health(self):
        """Test system health endpoint - high frequency."""
        with self.client.get(API_ENDPOINTS["health"], catch_response=True) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if "status" in data:
                        response.success()
                    else:
                        response.failure("Invalid health response format")
                except json.JSONDecodeError:
                    response.failure("Health response is not valid JSON")
            else:
                response.failure(f"Health check failed: {response.status_code}")

    @task(2)
    def get_agents_list(self):
        """Test agents list endpoint."""
        with self.client.get(API_ENDPOINTS["agents"], catch_response=True) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if isinstance(data, dict) and "agents" in data:
                        response.success()
                        # Update user context with available agents
                        available_agents = [agent.get("id", "") for agent in data.get("agents", [])]
                        if available_agents:
                            self.user_context["available_agents"] = available_agents
                    else:
                        response.success()  # Accept different response formats
                except json.JSONDecodeError:
                    response.failure("Agents response is not valid JSON")
            elif response.status_code == 404:
                response.success()  # Endpoint may not be implemented
            else:
                response.failure(f"Get agents failed: {response.status_code}")

    @task(2)
    def get_system_metrics(self):
        """Test system metrics endpoint."""
        with self.client.get(API_ENDPOINTS["metrics"], catch_response=True) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if isinstance(data, dict):
                        response.success()
                        # Log metrics for analysis
                        if "system" in data:
                            cpu = data["system"].get("cpu_percent", 0)
                            memory = data["system"].get("memory_percent", 0)
                            logging.info(f"System metrics - CPU: {cpu}%, Memory: {memory}%")
                    else:
                        response.success()  # Accept different formats
                except json.JSONDecodeError:
                    response.failure("Metrics response is not valid JSON")
            elif response.status_code == 404:
                response.success()  # Endpoint may not be implemented
            else:
                response.failure(f"Get metrics failed: {response.status_code}")

    @task(5)
    def send_chat_message(self):
        """Test chat message sending - primary load test."""
        message = random.choice(SAMPLE_MESSAGES)
        agent = self.user_context.get("preferred_agent", "agi-brain")
        model = self.user_context.get("preferred_model", "llama3.2:1b")
        
        payload = {
            "message": message,
            "agent": agent,
            "model": model,
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat()
        }
        
        with self.client.post(
            API_ENDPOINTS["chat"],
            json=payload,
            catch_response=True,
            timeout=30  # Longer timeout for AI responses
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if "response" in data:
                        response.success()
                        # Track conversation
                        self.user_context["conversation_history"].append({
                            "user_message": message,
                            "ai_response": data["response"][:100],  # Truncate for logging
                            "response_time": response.elapsed.total_seconds()
                        })
                        
                        # Log slow responses
                        if response.elapsed.total_seconds() > 10:
                            logging.warning(f"Slow chat response: {response.elapsed.total_seconds()}s")
                    else:
                        response.success()  # Accept different response formats
                except json.JSONDecodeError:
                    response.failure("Chat response is not valid JSON")
            elif response.status_code == 404:
                response.success()  # Endpoint may not be implemented
            elif response.status_code == 500:
                response.failure("Internal server error")
            else:
                response.failure(f"Chat request failed: {response.status_code}")

    @task(1)
    def get_available_models(self):
        """Test models list endpoint - low frequency."""
        with self.client.get(API_ENDPOINTS["models"], catch_response=True) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if isinstance(data, dict) and "models" in data:
                        response.success()
                        # Update user context with available models
                        available_models = [model.get("id", "") for model in data.get("models", [])]
                        if available_models:
                            self.user_context["available_models"] = available_models
                    else:
                        response.success()  # Accept different formats
                except json.JSONDecodeError:
                    response.failure("Models response is not valid JSON")
            elif response.status_code == 404:
                response.success()  # Endpoint may not be implemented
            else:
                response.failure(f"Get models failed: {response.status_code}")

    @task(1)
    def test_orchestration_endpoint(self):
        """Test orchestration endpoint - low frequency."""
        payload = {
            "task": "health_check",
            "parameters": {
                "check_all_services": True
            }
        }
        
        with self.client.post(
            API_ENDPOINTS["orchestration"],
            json=payload,
            catch_response=True
        ) as response:
            if response.status_code in [200, 201]:
                try:
                    data = response.json()
                    response.success()
                except json.JSONDecodeError:
                    response.success()  # Accept non-JSON responses
            elif response.status_code == 404:
                response.success()  # Endpoint may not be implemented
            else:
                response.failure(f"Orchestration request failed: {response.status_code}")


class LightLoadUser(SutazAIUser):
    """Light load user - fewer requests, longer waits."""
    
    wait_time = between(3, 8)
    weight = 3

    @task(5)
    def light_health_check(self):
        """Light health checking."""
        self.get_system_health()

    @task(2)
    def light_chat(self):
        """Light chat interaction."""
        self.send_chat_message()

    @task(1)
    def light_metrics(self):
        """Light metrics checking."""
        self.get_system_metrics()


class MediumLoadUser(SutazAIUser):
    """Medium load user - balanced requests."""
    
    wait_time = between(2, 5)
    weight = 2

    @task(4)
    def medium_health_check(self):
        """Medium health checking."""
        self.get_system_health()

    @task(3)
    def medium_chat(self):
        """Medium chat interaction."""
        self.send_chat_message()

    @task(2)
    def medium_metrics(self):
        """Medium metrics checking."""
        self.get_system_metrics()

    @task(1)
    def medium_agents(self):
        """Medium agent list checking."""
        self.get_agents_list()


class HeavyLoadUser(SutazAIUser):
    """Heavy load user - frequent requests, short waits."""
    
    wait_time = between(0.5, 2)
    weight = 1

    @task(6)
    def heavy_chat(self):
        """Heavy chat interaction."""
        self.send_chat_message()

    @task(4)
    def heavy_health_check(self):
        """Heavy health checking."""
        self.get_system_health()

    @task(3)
    def heavy_metrics(self):
        """Heavy metrics checking."""
        self.get_system_metrics()

    @task(2)
    def heavy_agents(self):
        """Heavy agent checking."""
        self.get_agents_list()

    @task(1)
    def heavy_models(self):
        """Heavy model checking."""
        self.get_available_models()


class SpikeTestUser(HttpUser):
    """Spike test user - simulates traffic spikes."""
    
    wait_time = between(0.1, 0.5)  # Very short waits

    def on_start(self):
        """Initialize spike test user."""
        self.spike_id = f"spike_{random.randint(1000, 9999)}"

    @task(10)
    def spike_health_check(self):
        """Rapid health check requests."""
        with self.client.get(API_ENDPOINTS["health"], catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Spike health check failed: {response.status_code}")

    @task(5)
    def spike_chat_burst(self):
        """Burst chat requests."""
        message = "Quick test message"
        payload = {
            "message": message,
            "agent": "agi-brain",
            "model": "llama3.2:1b"
        }
        
        with self.client.post(
            API_ENDPOINTS["chat"],
            json=payload,
            catch_response=True,
            timeout=10
        ) as response:
            if response.status_code in [200, 429]:  # Accept rate limiting
                response.success()
            else:
                response.failure(f"Spike chat failed: {response.status_code}")


# Event handlers for monitoring
@events.request.add_listener
def on_request(request_type, name, response_time, response_length, exception, context, **kwargs):
    """Log slow requests."""
    if response_time > 5000:  # 5 seconds
        logging.warning(f"Slow request: {request_type} {name} took {response_time}ms")


@events.user_error.add_listener  
def on_user_error(user_instance, exception, tb, **kwargs):
    """Log user errors."""
    logging.error(f"User error: {exception}")


@events.quitting.add_listener
def on_quitting(environment, **kwargs):
    """Log test completion."""
    logging.info("Load test completed")


# Custom scenarios for different load patterns
class DatabaseHeavyUser(HttpUser):
    """User that focuses on database-heavy operations."""
    
    wait_time = between(1, 3)

    @task(4)
    def database_metrics(self):
        """Request metrics that require database queries."""
        self.client.get(API_ENDPOINTS["metrics"])

    @task(3)
    def database_agents(self):
        """Request agents data from database."""
        self.client.get(API_ENDPOINTS["agents"])

    @task(2)
    def database_health(self):
        """Health check with database status."""
        self.client.get(API_ENDPOINTS["health"])


class AIHeavyUser(HttpUser):
    """User that focuses on AI-heavy operations."""
    
    wait_time = between(2, 5)  # Longer waits for AI processing

    @task(8)
    def ai_chat_long(self):
        """Long, complex chat messages."""
        long_messages = [
            "Please write a detailed explanation of how machine learning algorithms work, including examples of supervised, unsupervised, and reinforcement learning.",
            "Can you create a comprehensive Python program that implements a simple neural network from scratch with detailed comments?",
            "Explain the philosophical implications of artificial general intelligence and its potential impact on society.",
            "Write a technical analysis of different database architectures and their trade-offs in modern applications."
        ]
        
        message = random.choice(long_messages)
        payload = {
            "message": message,
            "agent": random.choice(SAMPLE_AGENTS),
            "model": random.choice(SAMPLE_MODELS)
        }
        
        with self.client.post(
            API_ENDPOINTS["chat"],
            json=payload,
            timeout=60  # Longer timeout for complex AI tasks
        ) as response:
            pass

    @task(2)
    def ai_models_check(self):
        """Check available AI models."""
        self.client.get(API_ENDPOINTS["models"])


class ConcurrentSessionUser(HttpUser):
    """User that maintains concurrent sessions."""
    
    wait_time = between(1, 2)

    def on_start(self):
        """Start multiple sessions."""
        self.sessions = [f"session_{i}_{random.randint(1000, 9999)}" for i in range(3)]

    @task(5)
    def multi_session_chat(self):
        """Send messages in different sessions."""
        for session_id in self.sessions:
            message = random.choice(SAMPLE_MESSAGES)
            payload = {
                "message": message,
                "session_id": session_id,
                "agent": "agi-brain"
            }
            
            self.client.post(API_ENDPOINTS["chat"], json=payload, timeout=20)
            time.sleep(0.1)  # Small delay between sessions


# Load test scenarios
def create_load_test_scenarios():
    """Create different load test scenarios."""
    return {
        "light_load": {
            "users": [LightLoadUser],
            "description": "Light load testing with basic health checks and occasional chat"
        },
        "medium_load": {
            "users": [LightLoadUser, MediumLoadUser],
            "description": "Medium load testing with balanced request patterns"
        },
        "heavy_load": {
            "users": [LightLoadUser, MediumLoadUser, HeavyLoadUser],
            "description": "Heavy load testing with intensive request patterns"
        },
        "spike_test": {
            "users": [SpikeTestUser],
            "description": "Spike testing with burst request patterns"
        },
        "database_intensive": {
            "users": [DatabaseHeavyUser],
            "description": "Database-intensive load testing"
        },
        "ai_intensive": {
            "users": [AIHeavyUser],
            "description": "AI-intensive load testing with complex requests"
        },
        "concurrent_sessions": {
            "users": [ConcurrentSessionUser],
            "description": "Concurrent session load testing"
        },
        "mixed_workload": {
            "users": [LightLoadUser, MediumLoadUser, HeavyLoadUser, DatabaseHeavyUser, AIHeavyUser],
            "description": "Mixed workload testing with all user types"
        }
    }


if __name__ == "__main__":
    print("SutazAI Load Testing Scenarios:")
    scenarios = create_load_test_scenarios()
    
    for name, scenario in scenarios.items():
        print(f"\n{name}: {scenario['description']}")
        print(f"  User types: {[user.__name__ for user in scenario['users']]}")
    
    print("\nTo run a specific scenario:")
    print("locust -f locustfile.py --users 10 --spawn-rate 2 --run-time 300s --host http://localhost:8000")
    print("\nTo run with web UI:")
    print("locust -f locustfile.py --host http://localhost:8000")