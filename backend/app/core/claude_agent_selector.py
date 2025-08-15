#!/usr/bin/env python3
"""
Claude Agent Selector - Intelligent agent selection based on task requirements
This is the REAL implementation that actually works with the unified registry
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

from app.core.unified_agent_registry import get_registry, UnifiedAgent

logger = logging.getLogger(__name__)

@dataclass
class TaskAnalysis:
    """Analysis of a task to determine requirements"""
    primary_domain: str
    required_capabilities: List[str]
    complexity_level: str  # simple, moderate, complex
    keywords: List[str]
    suggested_agents: List[str]
    confidence_score: float

class ClaudeAgentSelector:
    """Intelligent selector for Claude agents based on task analysis"""
    
    def __init__(self):
        self.registry = get_registry()
        
        # Domain keywords mapping
        self.domain_keywords = {
            "orchestration": ["orchestrat", "coordinat", "multi-agent", "workflow", "pipeline"],
            "code_generation": ["code", "implement", "develop", "program", "function", "class"],
            "testing": ["test", "qa", "quality", "validat", "check", "assert"],
            "deployment": ["deploy", "release", "production", "rollout", "launch"],
            "security": ["security", "vulnerab", "pentest", "audit", "threat", "protect"],
            "optimization": ["optim", "performance", "speed", "efficien", "improve"],
            "monitoring": ["monitor", "observ", "metric", "log", "alert", "track"],
            "infrastructure": ["docker", "container", "kubernetes", "devops", "infra"],
            "frontend": ["frontend", "ui", "ux", "react", "streamlit", "interface"],
            "backend": ["backend", "api", "server", "fastapi", "endpoint"],
            "data": ["data", "database", "sql", "etl", "pipeline", "analys"],
            "ai_ml": ["ai", "ml", "model", "train", "neural", "learn"],
            "documentation": ["document", "docs", "readme", "guide", "manual"]
        }
        
        # Agent expertise mapping
        self.agent_expertise = {
            "ai-agent-orchestrator": ["orchestration", "coordination", "multi-agent"],
            "complex-problem-solver": ["analysis", "research", "problem-solving"],
            "senior-backend-developer": ["backend", "api", "server"],
            "senior-frontend-developer": ["frontend", "ui", "react"],
            "testing-qa-validator": ["testing", "validation", "quality"],
            "deployment-automation-master": ["deployment", "release", "production"],
            "infrastructure-devops-manager": ["infrastructure", "docker", "devops"],
            "security-pentesting-specialist": ["security", "pentesting", "audit"],
            "hardware-resource-optimizer": ["optimization", "performance", "resources"],
            "document-knowledge-manager": ["documentation", "knowledge", "rag"],
            "senior-ai-engineer": ["ai_ml", "model", "training"],
            "task-assignment-coordinator": ["coordination", "routing", "assignment"]
        }
        
    def analyze_task(self, task_description: str) -> TaskAnalysis:
        """Analyze a task to determine requirements and suggest agents"""
        
        task_lower = task_description.lower()
        
        # Extract keywords
        keywords = self._extract_keywords(task_lower)
        
        # Determine primary domain
        domain_scores = {}
        for domain, domain_keywords in self.domain_keywords.items():
            score = sum(1 for kw in domain_keywords if kw in task_lower)
            if score > 0:
                domain_scores[domain] = score
                
        primary_domain = max(domain_scores, key=domain_scores.get) if domain_scores else "general"
        
        # Determine required capabilities
        required_capabilities = []
        for domain, score in domain_scores.items():
            if score > 0:
                required_capabilities.append(domain)
                
        # Determine complexity
        complexity = self._assess_complexity(task_description)
        
        # Suggest agents
        suggested_agents = self._suggest_agents(primary_domain, required_capabilities)
        
        # Calculate confidence
        confidence = min(1.0, len(domain_scores) * 0.2 + len(suggested_agents) * 0.1)
        
        return TaskAnalysis(
            primary_domain=primary_domain,
            required_capabilities=required_capabilities,
            complexity_level=complexity,
            keywords=keywords,
            suggested_agents=suggested_agents,
            confidence_score=confidence
        )
        
    def select_best_agent(self, task_description: str, 
                         required_capabilities: List[str] = None) -> Optional[UnifiedAgent]:
        """Select the best agent for a task"""
        
        # Analyze the task
        analysis = self.analyze_task(task_description)
        
        # Merge required capabilities
        if required_capabilities:
            analysis.required_capabilities.extend(required_capabilities)
            
        # Find best matching agent
        best_agent = None
        best_score = 0
        
        for agent in self.registry.list_agents(agent_type="claude"):
            score = self._score_agent(agent, analysis)
            if score > best_score:
                best_score = score
                best_agent = agent
                
        # If no Claude agent found, try container agents
        if not best_agent:
            for agent in self.registry.list_agents(agent_type="container"):
                score = self._score_agent(agent, analysis)
                if score > best_score:
                    best_score = score
                    best_agent = agent
                    
        return best_agent
        
    def select_multiple_agents(self, task_description: str, 
                              max_agents: int = 3) -> List[Tuple[UnifiedAgent, float]]:
        """Select multiple agents that could handle the task"""
        
        analysis = self.analyze_task(task_description)
        
        # Score all agents
        agent_scores = []
        for agent in self.registry.list_agents():
            score = self._score_agent(agent, analysis)
            if score > 0:
                agent_scores.append((agent, score))
                
        # Sort by score and return top N
        agent_scores.sort(key=lambda x: x[1], reverse=True)
        return agent_scores[:max_agents]
        
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text"""
        # Simple keyword extraction - can be enhanced with NLP
        words = re.findall(r'\b\w+\b', text)
        
        # Filter common words
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
                       'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was',
                       'are', 'were', 'been', 'be', 'have', 'has', 'had', 'do',
                       'does', 'did', 'will', 'would', 'could', 'should', 'may',
                       'might', 'must', 'can', 'need', 'want', 'i', 'me', 'you',
                       'we', 'they', 'this', 'that', 'these', 'those'}
        
        keywords = [w for w in words if w not in common_words and len(w) > 3]
        
        # Return unique keywords
        return list(set(keywords))[:20]
        
    def _assess_complexity(self, task_description: str) -> str:
        """Assess task complexity"""
        
        # Simple heuristics for complexity
        word_count = len(task_description.split())
        
        complexity_indicators = {
            "complex": ["multi", "integrat", "orchestrat", "architect", "design", 
                       "implement", "system", "framework", "comprehensive"],
            "moderate": ["create", "build", "develop", "setup", "configure", 
                        "optimize", "analyze", "process"],
            "simple": ["fix", "update", "check", "test", "run", "execute", 
                      "get", "list", "show"]
        }
        
        # Check for complexity indicators
        task_lower = task_description.lower()
        
        for level, indicators in complexity_indicators.items():
            if any(ind in task_lower for ind in indicators):
                return level
                
        # Default based on length
        if word_count > 50:
            return "complex"
        elif word_count > 20:
            return "moderate"
        else:
            return "simple"
            
    def _suggest_agents(self, primary_domain: str, 
                       capabilities: List[str]) -> List[str]:
        """Suggest specific agents based on domain and capabilities"""
        
        suggestions = []
        
        # Add agents that match the primary domain
        for agent_name, expertise in self.agent_expertise.items():
            if primary_domain in expertise:
                suggestions.append(agent_name)
                
        # Add agents that match required capabilities
        for cap in capabilities:
            for agent_name, expertise in self.agent_expertise.items():
                if cap in expertise and agent_name not in suggestions:
                    suggestions.append(agent_name)
                    
        # Add general-purpose agents if needed
        if not suggestions:
            suggestions = ["ai-agent-orchestrator", "complex-problem-solver"]
            
        return suggestions[:5]
        
    def _score_agent(self, agent: UnifiedAgent, analysis: TaskAnalysis) -> float:
        """Score an agent based on task analysis"""
        
        score = 0.0
        
        # Check if agent name is in suggestions
        if agent.name in analysis.suggested_agents:
            score += 5.0
            
        # Check capability matches
        for cap in analysis.required_capabilities:
            if cap in agent.capabilities:
                score += 2.0
                
        # Check keyword matches in description
        agent_desc_lower = agent.description.lower() if agent.description else ""
        for keyword in analysis.keywords:
            if keyword.lower() in agent_desc_lower:
                score += 0.5
                
        # Check domain match
        if analysis.primary_domain in agent.capabilities:
            score += 3.0
            
        # Prefer Claude agents
        if agent.type == "claude":
            score += 1.0
            
        # Complexity matching
        if analysis.complexity_level == "complex":
            if "orchestrat" in agent.name or "architect" in agent.name:
                score += 2.0
        elif analysis.complexity_level == "simple":
            if "simple" in agent.name or "basic" in agent.name:
                score += 1.0
                
        return score
        
    def get_agent_recommendations(self, task_description: str) -> Dict[str, Any]:
        """Get comprehensive agent recommendations for a task"""
        
        analysis = self.analyze_task(task_description)
        best_agent = self.select_best_agent(task_description)
        alternatives = self.select_multiple_agents(task_description, max_agents=5)
        
        return {
            "task_analysis": {
                "primary_domain": analysis.primary_domain,
                "required_capabilities": analysis.required_capabilities,
                "complexity": analysis.complexity_level,
                "keywords": analysis.keywords[:10],
                "confidence": analysis.confidence_score
            },
            "recommended_agent": {
                "id": best_agent.id if best_agent else None,
                "name": best_agent.name if best_agent else None,
                "type": best_agent.type if best_agent else None,
                "reason": f"Best match for {analysis.primary_domain} domain with required capabilities"
            } if best_agent else None,
            "alternative_agents": [
                {
                    "id": agent.id,
                    "name": agent.name,
                    "type": agent.type,
                    "score": score
                }
                for agent, score in alternatives
            ]
        }

# Singleton instance
_selector_instance = None

def get_selector() -> ClaudeAgentSelector:
    """Get singleton selector instance"""
    global _selector_instance
    if _selector_instance is None:
        _selector_instance = ClaudeAgentSelector()
    return _selector_instance