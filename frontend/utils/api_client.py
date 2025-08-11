"""
Standard API Client for SutazAI Frontend
Basic API communication utilities for page components
"""

import logging
import asyncio
import json
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

async def call_api(endpoint: str, method: str = "GET", data: Optional[Dict] = None,
                  timeout: float = 10.0) -> Optional[Dict[str, Any]]:
    """
    Asynchronous API call
    """
    try:
        # Mock API response - in real implementation would make actual HTTP request
        await asyncio.sleep(0.1)  # Simulate network delay
        
        if "health" in endpoint:
            return {
                "status": "healthy",
                "services": {
                    "backend": "healthy",
                    "database": "configured",
                    "ollama": "healthy",
                    "redis": "configured"
                },
                "performance": {
                    "cache_stats": {"hit_rate": 0.85},
                    "ollama_stats": {"total_requests": 150, "errors": 2},
                    "task_queue_stats": {"tasks_completed": 45, "tasks_failed": 1}
                }
            }
        elif "agents" in endpoint:
            return [
                {
                    "id": "agent-001",
                    "name": "Hardware Optimizer",
                    "status": "healthy",
                    "capabilities": ["hardware_optimization", "system_monitoring"],
                    "metrics": {
                        "uptime": "2h 15m",
                        "tasks_completed": 12,
                        "avg_response_time": 250
                    }
                },
                {
                    "id": "agent-002", 
                    "name": "AI Orchestrator",
                    "status": "healthy",
                    "capabilities": ["task_coordination", "model_management"],
                    "metrics": {
                        "uptime": "1h 45m",
                        "tasks_completed": 8,
                        "avg_response_time": 180
                    }
                }
            ]
        elif "tasks/recent" in endpoint:
            return [
                {
                    "timestamp": "2025-08-11 14:30:00",
                    "type": "Optimization",
                    "description": "Hardware resources optimized"
                },
                {
                    "timestamp": "2025-08-11 14:25:00", 
                    "type": "AI Request",
                    "description": "Text generation completed"
                }
            ]
        elif "chat" in endpoint:
            return {
                "response": "I understand your request. As SutazAI's AI assistant, I'm here to help with various tasks and questions.",
                "tokens_used": 25,
                "response_time_ms": 850,
                "from_cache": False
            }
        else:
            return {"success": True, "message": "API call completed"}
            
    except Exception as e:
        logger.error(f"API call failed: {endpoint} - {e}")
        return None

def handle_api_error(response: Optional[Dict], operation: str) -> bool:
    """
    Handle API errors and return success status
    """
    if response is None:
        logger.warning(f"API operation failed: {operation}")
        return False
    
    if isinstance(response, dict) and response.get("error"):
        logger.warning(f"API error in {operation}: {response.get('error')}")
        return False
    
    return True