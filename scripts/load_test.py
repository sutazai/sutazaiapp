import time
from locust import HttpUser, task, between

class SutazAIUser(HttpUser):
    wait_time = between(1, 5)

    @task
    def chat(self):
        self.client.post("/api/chat", json={"message": "Hello, world!", "model": "llama3.2:1b"})
