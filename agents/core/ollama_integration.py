#!/usr/bin/env python3
"""
Purpose: Ollama integration for all agents with TinyLlama default
Usage: Imported by all agent implementations
Requirements: ollama, httpx, asyncio
"""

import os
import httpx
import asyncio
import json
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)

class OllamaIntegration:
    """Handles all Ollama interactions for agents"""
    
    def __init__(self, 
                 base_url: str = "http://localhost:11434",
                 default_model: str = "tinyllama",
                 timeout: int = 300):
        self.base_url = base_url
        self.default_model = default_model
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout)
        
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
        
    async def ensure_model_available(self, model: str = None) -> bool:
        """Ensure the model is available locally"""
        model = model or self.default_model
        
        try:
            # Check if model exists
            response = await self.client.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "").split(":")[0] for m in models]
                
                if model not in model_names:
                    logger.info(f"Model {model} not found, pulling...")
                    await self.pull_model(model)
                    
                return True
        except Exception as e:
            logger.error(f"Error checking model availability: {e}")
            return False
            
    async def pull_model(self, model: str) -> bool:
        """Pull a model from Ollama registry"""
        try:
            response = await self.client.post(
                f"{self.base_url}/api/pull",
                json={"name": model}
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Error pulling model {model}: {e}")
            return False
            
    async def generate(self, 
                      prompt: str,
                      model: str = None,
                      system: str = None,
                      temperature: float = 0.7,
                      max_tokens: int = 2048) -> Optional[str]:
        """Generate a response using Ollama"""
        model = model or self.default_model
        
        # Ensure model is available
        if not await self.ensure_model_available(model):
            logger.error(f"Model {model} not available")
            return None
            
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }
            
            if system:
                payload["system"] = system
                
            response = await self.client.post(
                f"{self.base_url}/api/generate",
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "")
            else:
                logger.error(f"Ollama generate error: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return None
            
    async def chat(self,
                   messages: List[Dict[str, str]],
                   model: str = None,
                   temperature: float = 0.7,
                   max_tokens: int = 2048) -> Optional[str]:
        """Chat completion using Ollama"""
        model = model or self.default_model
        
        # Ensure model is available
        if not await self.ensure_model_available(model):
            logger.error(f"Model {model} not available")
            return None
            
        try:
            payload = {
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }
            
            response = await self.client.post(
                f"{self.base_url}/api/chat",
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("message", {}).get("content", "")
            else:
                logger.error(f"Ollama chat error: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            return None
            
    async def embeddings(self,
                        text: str,
                        model: str = None) -> Optional[List[float]]:
        """Generate embeddings using Ollama"""
        model = model or self.default_model
        
        try:
            payload = {
                "model": model,
                "prompt": text
            }
            
            response = await self.client.post(
                f"{self.base_url}/api/embeddings",
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("embedding", [])
            else:
                logger.error(f"Ollama embeddings error: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return None

class OllamaConfig:
    """Configuration for Ollama models by agent type"""
    
    # Model assignments based on complexity
    OPUS_MODEL = "deepseek-r1:8b"  # For complex reasoning
    SONNET_MODEL = "qwen2.5-coder:7b"  # For balanced tasks
    DEFAULT_MODEL = "tinyllama"  # For simple tasks
    
    # Agent to model mapping
    AGENT_MODELS = {
        # Opus agents - complex reasoning
        "adversarial-attack-detector": OPUS_MODEL,
        "agent-creator": OPUS_MODEL,
        "ai-senior-full-stack-developer": OPUS_MODEL,
        "ai-system-architect": OPUS_MODEL,
        "bias-and-fairness-auditor": OPUS_MODEL,
        "causal-inference-expert": OPUS_MODEL,
        "cicd-pipeline-orchestrator": OPUS_MODEL,
        "code-quality-gateway-sonarqube": OPUS_MODEL,
        "cognitive-architecture-designer": OPUS_MODEL,
        "complex-problem-solver": OPUS_MODEL,
        "container-orchestrator-k3s": OPUS_MODEL,
        "deep-learning-brain-architect": OPUS_MODEL,
        "deep-learning-brain-manager": OPUS_MODEL,
        "deep-local-brain-builder": OPUS_MODEL,
        "distributed-computing-architect": OPUS_MODEL,
        "distributed-tracing-analyzer-jaeger": OPUS_MODEL,
        "ethical-governor": OPUS_MODEL,
        "evolution-strategy-trainer": OPUS_MODEL,
        "explainable-ai-specialist": OPUS_MODEL,
        "genetic-algorithm-tuner": OPUS_MODEL,
        "goal-setting-and-planning-agent": OPUS_MODEL,
        "knowledge-distillation-expert": OPUS_MODEL,
        "meta-learning-specialist": OPUS_MODEL,
        "neural-architecture-search": OPUS_MODEL,
        "neuromorphic-computing-expert": OPUS_MODEL,
        "product-strategy-architect": OPUS_MODEL,
        "quantum-ai-researcher": OPUS_MODEL,
        "reinforcement-learning-trainer": OPUS_MODEL,
        "resource-arbitration-agent": OPUS_MODEL,
        "runtime-behavior-anomaly-detector": OPUS_MODEL,
        "senior-full-stack-developer": OPUS_MODEL,
        "symbolic-reasoning-engine": OPUS_MODEL,
        "system-architect": OPUS_MODEL,
        
        # Sonnet agents - balanced performance
        "agentzero-coordinator": SONNET_MODEL,
        "ai-agent-debugger": SONNET_MODEL,
        "ai-agent-orchestrator": SONNET_MODEL,
        "ai-product-manager": SONNET_MODEL,
        "ai-scrum-master": SONNET_MODEL,
        "autonomous-system-controller": SONNET_MODEL,
        "browser-automation-orchestrator": SONNET_MODEL,
        "codebase-team-lead": SONNET_MODEL,
        "code-generation-improver": SONNET_MODEL,
        "context-optimization-engineer": SONNET_MODEL,
        "data-analysis-engineer": SONNET_MODEL,
        "data-pipeline-engineer": SONNET_MODEL,
        "data-version-controller-dvc": SONNET_MODEL,
        "deploy-automation-master": SONNET_MODEL,
        "deployment-automation-master": SONNET_MODEL,
        "dify-automation-specialist": SONNET_MODEL,
        "document-knowledge-manager": SONNET_MODEL,
        "edge-computing-optimizer": SONNET_MODEL,
        "episodic-memory-engineer": SONNET_MODEL,
        "federated-learning-coordinator": SONNET_MODEL,
        "financial-analysis-specialist": SONNET_MODEL,
        "flowiseai-flow-manager": SONNET_MODEL,
        "hardware-resource-optimizer": SONNET_MODEL,
        "infrastructure-devops-manager": SONNET_MODEL,
        "intelligence-optimization-monitor": SONNET_MODEL,
        "jarvis-voice-interface": SONNET_MODEL,
        "kali-security-specialist": SONNET_MODEL,
        "kali-hacker": SONNET_MODEL,
        "knowledge-graph-builder": SONNET_MODEL,
        "langflow-workflow-designer": SONNET_MODEL,
        "localagi-orchestration-manager": SONNET_MODEL,
        "mega-code-auditor": SONNET_MODEL,
        "memory-persistence-manager": SONNET_MODEL,
        "ml-experiment-tracker-mlflow": SONNET_MODEL,
        "model-training-specialist": SONNET_MODEL,
        "multi-modal-fusion-coordinator": SONNET_MODEL,
        "observability-dashboard-manager-grafana": SONNET_MODEL,
        "observability-monitoring-engineer": SONNET_MODEL,
        "ollama-integration-specialist": SONNET_MODEL,
        "opendevin-code-generator": SONNET_MODEL,
        "private-data-analyst": SONNET_MODEL,
        "private-registry-manager-harbor": SONNET_MODEL,
        "secrets-vault-manager-vault": SONNET_MODEL,
        "security-pentesting-specialist": SONNET_MODEL,
        "self-healing-orchestrator": SONNET_MODEL,
        "semgrep-security-analyzer": SONNET_MODEL,
        "senior-ai-engineer": SONNET_MODEL,
        "senior-backend-developer": SONNET_MODEL,
        "senior-frontend-developer": SONNET_MODEL,
        "shell-automation-specialist": SONNET_MODEL,
        "synthetic-data-generator": SONNET_MODEL,
        "system-knowledge-curator": SONNET_MODEL,
        "system-optimizer-reorganizer": SONNET_MODEL,
        "system-performance-forecaster": SONNET_MODEL,
        "system-validator": SONNET_MODEL,
        "task-assignment-coordinator": SONNET_MODEL,
        "testing-qa-validator": SONNET_MODEL,
        "transformers-migration-specialist": SONNET_MODEL,
        
        # Additional monitoring/utility agents - default model
        "attention-optimizer": DEFAULT_MODEL,
        "automated-incident-responder": DEFAULT_MODEL,
        "autonomous-task-executor": DEFAULT_MODEL,
        "cognitive-load-monitor": DEFAULT_MODEL,
        "compute-scheduler-and-optimizer": DEFAULT_MODEL,
        "container-vulnerability-scanner-trivy": DEFAULT_MODEL,
        "cpu-only-hardware-optimizer": DEFAULT_MODEL,
        "data-drift-detector": DEFAULT_MODEL,
        "data-lifecycle-manager": DEFAULT_MODEL,
        "edge-inference-proxy": DEFAULT_MODEL,
        "emergency-shutdown-coordinator": DEFAULT_MODEL,
        "energy-consumption-optimize": DEFAULT_MODEL,
        "experiment-tracker": DEFAULT_MODEL,
        "explainability-and-transparency-agent": DEFAULT_MODEL,
        "garbage-collector": DEFAULT_MODEL,
        "garbage-collector-coordinator": DEFAULT_MODEL,
        "gpu-hardware-optimizer": DEFAULT_MODEL,
        "gradient-compression-specialist": DEFAULT_MODEL,
        "honeypot-deployment-agent": DEFAULT_MODEL,
        "human-oversight-interface-agent": DEFAULT_MODEL,
        "log-aggregator-loki": DEFAULT_MODEL,
        "metrics-collector-prometheus": DEFAULT_MODEL,
        "prompt-injection-guard": DEFAULT_MODEL,
        "ram-hardware-optimizer": DEFAULT_MODEL,
        "resource-visualiser": DEFAULT_MODEL,
    }
    
    @classmethod
    def get_model_for_agent(cls, agent_name: str) -> str:
        """Get the appropriate model for an agent"""
        return cls.AGENT_MODELS.get(agent_name, cls.DEFAULT_MODEL)
        
    @classmethod
    def get_model_config(cls, agent_name: str) -> Dict[str, Any]:
        """Get complete model configuration for an agent"""
        model = cls.get_model_for_agent(agent_name)
        
        # Base configuration
        config = {
            "model": model,
            "temperature": 0.7,
            "max_tokens": 2048,
            "top_p": 0.9,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0
        }
        
        # Adjust parameters based on model type
        if model == cls.OPUS_MODEL:
            # More creative/reasoning focused
            config["temperature"] = 0.8
            config["max_tokens"] = 4096
            config["top_p"] = 0.95
        elif model == cls.SONNET_MODEL:
            # Balanced
            config["temperature"] = 0.7
            config["max_tokens"] = 2048
            config["top_p"] = 0.9
        else:
            # Conservative for simple tasks
            config["temperature"] = 0.5
            config["max_tokens"] = 1024
            config["top_p"] = 0.8
            
        return config