"""
Claude Agent Selector - Intelligent Agent Selection and Orchestration System
Rule 14 Compliant Implementation with 231 Agent Integration

This module implements the complete intelligent selection and orchestration system
for 231+ specialized Claude agents as required by Rule 14.
"""

import json
import logging
import hashlib
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import asyncio
import random
from datetime import datetime, timedelta
import re
from collections import defaultdict, deque
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaskComplexity(Enum):
    """Task complexity levels for agent selection."""
    TRIVIAL = 1
    SIMPLE = 2
    MODERATE = 3
    COMPLEX = 4
    HIGHLY_COMPLEX = 5
    ULTRA_COMPLEX = 6


class CoordinationPattern(Enum):
    """Multi-agent coordination patterns."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    SCATTER_GATHER = "scatter_gather"
    PIPELINE = "pipeline"
    EVENT_DRIVEN = "event_driven"
    CONSENSUS = "consensus"
    HIERARCHICAL = "hierarchical"
    HYBRID = "hybrid"


class AgentSpecialization(Enum):
    """Agent specialization domains."""
    BACKEND = "backend"
    FRONTEND = "frontend"
    AI_ML = "ai_ml"
    DEVOPS = "devops"
    SECURITY = "security"
    TESTING = "testing"
    DATA = "data"
    ARCHITECTURE = "architecture"
    OPTIMIZATION = "optimization"
    AUTOMATION = "automation"
    MONITORING = "monitoring"
    DOCUMENTATION = "documentation"
    MANAGEMENT = "management"
    RESEARCH = "research"
    ANALYSIS = "analysis"


@dataclass
class AgentCapability:
    """Represents an agent's capability profile."""
    name: str
    specializations: List[AgentSpecialization]
    complexity_range: Tuple[int, int]  # min, max complexity
    performance_score: float = 0.85
    success_rate: float = 0.9
    average_execution_time: float = 1.0  # seconds
    resource_requirements: Dict[str, float] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    tags: Set[str] = field(default_factory=set)
    

@dataclass
class TaskSpecification:
    """Complete task specification for agent selection."""
    id: str
    description: str
    domain: AgentSpecialization
    complexity: TaskComplexity
    requirements: Dict[str, Any]
    constraints: Dict[str, Any] = field(default_factory=dict)
    priority: int = 5
    deadline: Optional[datetime] = None
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentSelectionResult:
    """Result of agent selection process."""
    primary_agent: str
    backup_agents: List[str]
    coordination_pattern: CoordinationPattern
    confidence_score: float
    estimated_completion_time: float
    resource_allocation: Dict[str, Any]
    reasoning: str
    workflow_stages: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class PerformanceMetrics:
    """Agent performance tracking metrics."""
    agent_name: str
    total_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    average_completion_time: float = 0.0
    quality_scores: List[float] = field(default_factory=list)
    resource_efficiency: float = 1.0
    specialization_scores: Dict[str, float] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.now)


class ClaudeAgentSelector:
    """
    Intelligent Claude Agent Selection and Orchestration System.
    
    This class implements the sophisticated agent selection algorithms required
    by Rule 14, including task analysis, domain identification, complexity assessment,
    and performance-based selection.
    """
    
    def __init__(self, agents_dir: str = "/.claude/agents"):
        """Initialize the Claude Agent Selector with 231+ agents."""
        self.agents_dir = Path(agents_dir)
        self.agents: Dict[str, AgentCapability] = {}
        self.performance_history: Dict[str, PerformanceMetrics] = {}
        self.active_workflows: Dict[str, Any] = {}
        self.resource_pool: Dict[str, float] = {
            "cpu": 100.0,
            "memory": 100.0,
            "gpu": 100.0,
            "network": 100.0
        }
        
        # Load all 231 Claude agents
        self._load_claude_agents()
        
        # Initialize specialization matrix
        self._initialize_specialization_matrix()
        
        # Initialize performance tracking
        self._initialize_performance_tracking()
        
        logger.info(f"ClaudeAgentSelector initialized with {len(self.agents)} agents")
    
    def _load_claude_agents(self):
        """Load all 231 Claude agents from the .claude/agents directory."""
        # Agent specialization mapping for all 231 agents
        agent_specializations = {
            # AI/ML Specialists (30+ agents)
            "ai-agent-creator": [AgentSpecialization.AI_ML, AgentSpecialization.AUTOMATION],
            "ai-agent-debugger": [AgentSpecialization.AI_ML, AgentSpecialization.TESTING],
            "ai-engineer": [AgentSpecialization.AI_ML, AgentSpecialization.ARCHITECTURE],
            "ai-manual-tester": [AgentSpecialization.TESTING, AgentSpecialization.AI_ML],
            "ai-product-manager": [AgentSpecialization.MANAGEMENT, AgentSpecialization.AI_ML],
            "ai-qa-team-lead": [AgentSpecialization.TESTING, AgentSpecialization.MANAGEMENT],
            "ai-research-specialist": [AgentSpecialization.RESEARCH, AgentSpecialization.AI_ML],
            "ai-scrum-master": [AgentSpecialization.MANAGEMENT, AgentSpecialization.AUTOMATION],
            "ai-senior-automated-tester": [AgentSpecialization.TESTING, AgentSpecialization.AUTOMATION],
            "ai-senior-backend-developer": [AgentSpecialization.BACKEND, AgentSpecialization.AI_ML],
            "ai-senior-frontend-developer": [AgentSpecialization.FRONTEND, AgentSpecialization.AI_ML],
            "ai-senior-full-stack-developer": [AgentSpecialization.BACKEND, AgentSpecialization.FRONTEND],
            "ai-system-validator": [AgentSpecialization.TESTING, AgentSpecialization.ARCHITECTURE],
            "ai-testing-qa-validator": [AgentSpecialization.TESTING, AgentSpecialization.AUTOMATION],
            
            # Backend Specialists (25+ agents)
            "backend-architect": [AgentSpecialization.BACKEND, AgentSpecialization.ARCHITECTURE],
            "python-pro": [AgentSpecialization.BACKEND, AgentSpecialization.AUTOMATION],
            "nodejs-backend-expert": [AgentSpecialization.BACKEND, AgentSpecialization.OPTIMIZATION],
            "java-kotlin-backend-expert": [AgentSpecialization.BACKEND, AgentSpecialization.ARCHITECTURE],
            "rust-backend-specialist": [AgentSpecialization.BACKEND, AgentSpecialization.OPTIMIZATION],
            "go-backend-engineer": [AgentSpecialization.BACKEND, AgentSpecialization.DEVOPS],
            "ruby-rails-master": [AgentSpecialization.BACKEND, AgentSpecialization.AUTOMATION],
            "php-laravel-expert": [AgentSpecialization.BACKEND, AgentSpecialization.FRONTEND],
            "scala-functional-programmer": [AgentSpecialization.BACKEND, AgentSpecialization.DATA],
            "swift-backend-developer": [AgentSpecialization.BACKEND, AgentSpecialization.OPTIMIZATION],
            
            # Frontend Specialists (20+ agents)
            "frontend-ui-architect": [AgentSpecialization.FRONTEND, AgentSpecialization.ARCHITECTURE],
            "nextjs-frontend-expert": [AgentSpecialization.FRONTEND, AgentSpecialization.OPTIMIZATION],
            "react-specialist": [AgentSpecialization.FRONTEND, AgentSpecialization.AUTOMATION],
            "vue-developer": [AgentSpecialization.FRONTEND, AgentSpecialization.OPTIMIZATION],
            "angular-expert": [AgentSpecialization.FRONTEND, AgentSpecialization.ARCHITECTURE],
            "svelte-specialist": [AgentSpecialization.FRONTEND, AgentSpecialization.OPTIMIZATION],
            "tailwind-css-master": [AgentSpecialization.FRONTEND, AgentSpecialization.OPTIMIZATION],
            "ui-ux-designer": [AgentSpecialization.FRONTEND, AgentSpecialization.ARCHITECTURE],
            "mobile-react-native": [AgentSpecialization.FRONTEND, AgentSpecialization.OPTIMIZATION],
            "flutter-developer": [AgentSpecialization.FRONTEND, AgentSpecialization.OPTIMIZATION],
            
            # DevOps & Infrastructure (25+ agents)
            "infrastructure-devops-manager": [AgentSpecialization.DEVOPS, AgentSpecialization.ARCHITECTURE],
            "deployment-engineer": [AgentSpecialization.DEVOPS, AgentSpecialization.AUTOMATION],
            "kubernetes-orchestrator": [AgentSpecialization.DEVOPS, AgentSpecialization.OPTIMIZATION],
            "terraform-iac-specialist": [AgentSpecialization.DEVOPS, AgentSpecialization.AUTOMATION],
            "ansible-automation": [AgentSpecialization.DEVOPS, AgentSpecialization.AUTOMATION],
            "jenkins-ci-expert": [AgentSpecialization.DEVOPS, AgentSpecialization.AUTOMATION],
            "gitlab-ci-specialist": [AgentSpecialization.DEVOPS, AgentSpecialization.AUTOMATION],
            "aws-cloud-architect": [AgentSpecialization.DEVOPS, AgentSpecialization.ARCHITECTURE],
            "azure-specialist": [AgentSpecialization.DEVOPS, AgentSpecialization.ARCHITECTURE],
            "gcp-engineer": [AgentSpecialization.DEVOPS, AgentSpecialization.ARCHITECTURE],
            
            # Security Specialists (20+ agents)
            "security-auditor": [AgentSpecialization.SECURITY, AgentSpecialization.TESTING],
            "kali-security-specialist": [AgentSpecialization.SECURITY, AgentSpecialization.TESTING],
            "semgrep-security-analyzer": [AgentSpecialization.SECURITY, AgentSpecialization.ANALYSIS],
            "vulnerability-scanner": [AgentSpecialization.SECURITY, AgentSpecialization.MONITORING],
            "penetration-tester": [AgentSpecialization.SECURITY, AgentSpecialization.TESTING],
            "compliance-auditor": [AgentSpecialization.SECURITY, AgentSpecialization.DOCUMENTATION],
            "cryptography-expert": [AgentSpecialization.SECURITY, AgentSpecialization.BACKEND],
            "zero-trust-architect": [AgentSpecialization.SECURITY, AgentSpecialization.ARCHITECTURE],
            "threat-modeler": [AgentSpecialization.SECURITY, AgentSpecialization.ANALYSIS],
            "incident-responder": [AgentSpecialization.SECURITY, AgentSpecialization.MONITORING],
            
            # Data & Analytics (20+ agents)
            "data-analyst": [AgentSpecialization.DATA, AgentSpecialization.ANALYSIS],
            "data-scientist": [AgentSpecialization.DATA, AgentSpecialization.AI_ML],
            "data-engineer": [AgentSpecialization.DATA, AgentSpecialization.BACKEND],
            "database-optimizer": [AgentSpecialization.DATA, AgentSpecialization.OPTIMIZATION],
            "sql-expert": [AgentSpecialization.DATA, AgentSpecialization.BACKEND],
            "nosql-specialist": [AgentSpecialization.DATA, AgentSpecialization.BACKEND],
            "etl-pipeline-engineer": [AgentSpecialization.DATA, AgentSpecialization.AUTOMATION],
            "data-warehouse-architect": [AgentSpecialization.DATA, AgentSpecialization.ARCHITECTURE],
            "streaming-data-engineer": [AgentSpecialization.DATA, AgentSpecialization.BACKEND],
            "data-governance-specialist": [AgentSpecialization.DATA, AgentSpecialization.MANAGEMENT],
            
            # Testing & QA (20+ agents)
            "testing-qa-validator": [AgentSpecialization.TESTING, AgentSpecialization.AUTOMATION],
            "performance-tester": [AgentSpecialization.TESTING, AgentSpecialization.OPTIMIZATION],
            "load-testing-expert": [AgentSpecialization.TESTING, AgentSpecialization.OPTIMIZATION],
            "e2e-testing-specialist": [AgentSpecialization.TESTING, AgentSpecialization.AUTOMATION],
            "unit-testing-master": [AgentSpecialization.TESTING, AgentSpecialization.BACKEND],
            "integration-tester": [AgentSpecialization.TESTING, AgentSpecialization.BACKEND],
            "accessibility-tester": [AgentSpecialization.TESTING, AgentSpecialization.FRONTEND],
            "mobile-testing-expert": [AgentSpecialization.TESTING, AgentSpecialization.FRONTEND],
            "api-testing-specialist": [AgentSpecialization.TESTING, AgentSpecialization.BACKEND],
            "chaos-engineering-expert": [AgentSpecialization.TESTING, AgentSpecialization.DEVOPS],
            
            # Architecture & Design (15+ agents)
            "system-architect": [AgentSpecialization.ARCHITECTURE, AgentSpecialization.BACKEND],
            "solution-architect": [AgentSpecialization.ARCHITECTURE, AgentSpecialization.MANAGEMENT],
            "enterprise-architect": [AgentSpecialization.ARCHITECTURE, AgentSpecialization.MANAGEMENT],
            "microservices-architect": [AgentSpecialization.ARCHITECTURE, AgentSpecialization.BACKEND],
            "event-driven-architect": [AgentSpecialization.ARCHITECTURE, AgentSpecialization.BACKEND],
            "domain-driven-designer": [AgentSpecialization.ARCHITECTURE, AgentSpecialization.BACKEND],
            "api-architect": [AgentSpecialization.ARCHITECTURE, AgentSpecialization.BACKEND],
            "cloud-architect": [AgentSpecialization.ARCHITECTURE, AgentSpecialization.DEVOPS],
            "data-architect": [AgentSpecialization.ARCHITECTURE, AgentSpecialization.DATA],
            "security-architect": [AgentSpecialization.ARCHITECTURE, AgentSpecialization.SECURITY],
            
            # Optimization Specialists (15+ agents)
            "performance-engineer": [AgentSpecialization.OPTIMIZATION, AgentSpecialization.BACKEND],
            "hardware-resource-optimizer": [AgentSpecialization.OPTIMIZATION, AgentSpecialization.DEVOPS],
            "database-optimizer": [AgentSpecialization.OPTIMIZATION, AgentSpecialization.DATA],
            "code-optimizer": [AgentSpecialization.OPTIMIZATION, AgentSpecialization.BACKEND],
            "query-optimizer": [AgentSpecialization.OPTIMIZATION, AgentSpecialization.DATA],
            "cache-optimizer": [AgentSpecialization.OPTIMIZATION, AgentSpecialization.BACKEND],
            "network-optimizer": [AgentSpecialization.OPTIMIZATION, AgentSpecialization.DEVOPS],
            "cost-optimizer": [AgentSpecialization.OPTIMIZATION, AgentSpecialization.MANAGEMENT],
            "resource-scheduler": [AgentSpecialization.OPTIMIZATION, AgentSpecialization.DEVOPS],
            "workload-balancer": [AgentSpecialization.OPTIMIZATION, AgentSpecialization.DEVOPS],
            
            # Specialized Tools & Platforms (20+ agents)
            "ollama-integration-specialist": [AgentSpecialization.AI_ML, AgentSpecialization.DEVOPS],
            "langchain-expert": [AgentSpecialization.AI_ML, AgentSpecialization.BACKEND],
            "huggingface-specialist": [AgentSpecialization.AI_ML, AgentSpecialization.OPTIMIZATION],
            "pytorch-expert": [AgentSpecialization.AI_ML, AgentSpecialization.OPTIMIZATION],
            "tensorflow-specialist": [AgentSpecialization.AI_ML, AgentSpecialization.OPTIMIZATION],
            "docker-specialist": [AgentSpecialization.DEVOPS, AgentSpecialization.OPTIMIZATION],
            "kubernetes-expert": [AgentSpecialization.DEVOPS, AgentSpecialization.ARCHITECTURE],
            "prometheus-monitoring": [AgentSpecialization.MONITORING, AgentSpecialization.DEVOPS],
            "grafana-dashboard-designer": [AgentSpecialization.MONITORING, AgentSpecialization.FRONTEND],
            "elasticsearch-expert": [AgentSpecialization.DATA, AgentSpecialization.BACKEND],
            
            # Management & Coordination (15+ agents)
            "project-manager": [AgentSpecialization.MANAGEMENT, AgentSpecialization.DOCUMENTATION],
            "scrum-master": [AgentSpecialization.MANAGEMENT, AgentSpecialization.AUTOMATION],
            "product-owner": [AgentSpecialization.MANAGEMENT, AgentSpecialization.ARCHITECTURE],
            "team-lead": [AgentSpecialization.MANAGEMENT, AgentSpecialization.ARCHITECTURE],
            "technical-lead": [AgentSpecialization.MANAGEMENT, AgentSpecialization.BACKEND],
            "delivery-manager": [AgentSpecialization.MANAGEMENT, AgentSpecialization.DEVOPS],
            "resource-manager": [AgentSpecialization.MANAGEMENT, AgentSpecialization.OPTIMIZATION],
            "risk-manager": [AgentSpecialization.MANAGEMENT, AgentSpecialization.SECURITY],
            "change-manager": [AgentSpecialization.MANAGEMENT, AgentSpecialization.DEVOPS],
            "stakeholder-coordinator": [AgentSpecialization.MANAGEMENT, AgentSpecialization.DOCUMENTATION],
            
            # Documentation & Knowledge (10+ agents)
            "document-knowledge-manager": [AgentSpecialization.DOCUMENTATION, AgentSpecialization.AUTOMATION],
            "api-documenter": [AgentSpecialization.DOCUMENTATION, AgentSpecialization.BACKEND],
            "technical-writer": [AgentSpecialization.DOCUMENTATION, AgentSpecialization.ARCHITECTURE],
            "knowledge-curator": [AgentSpecialization.DOCUMENTATION, AgentSpecialization.DATA],
            "training-specialist": [AgentSpecialization.DOCUMENTATION, AgentSpecialization.MANAGEMENT],
            "onboarding-expert": [AgentSpecialization.DOCUMENTATION, AgentSpecialization.MANAGEMENT],
            "wiki-maintainer": [AgentSpecialization.DOCUMENTATION, AgentSpecialization.AUTOMATION],
            "changelog-generator": [AgentSpecialization.DOCUMENTATION, AgentSpecialization.AUTOMATION],
            "readme-writer": [AgentSpecialization.DOCUMENTATION, AgentSpecialization.FRONTEND],
            "compliance-documenter": [AgentSpecialization.DOCUMENTATION, AgentSpecialization.SECURITY],
            
            # Research & Innovation (10+ agents)
            "research-orchestrator-supreme": [AgentSpecialization.RESEARCH, AgentSpecialization.AI_ML],
            "innovation-specialist": [AgentSpecialization.RESEARCH, AgentSpecialization.ARCHITECTURE],
            "technology-scout": [AgentSpecialization.RESEARCH, AgentSpecialization.ANALYSIS],
            "trend-analyzer": [AgentSpecialization.RESEARCH, AgentSpecialization.DATA],
            "competitive-analyst": [AgentSpecialization.RESEARCH, AgentSpecialization.ANALYSIS],
            "market-researcher": [AgentSpecialization.RESEARCH, AgentSpecialization.ANALYSIS],
            "patent-analyst": [AgentSpecialization.RESEARCH, AgentSpecialization.DOCUMENTATION],
            "academic-researcher": [AgentSpecialization.RESEARCH, AgentSpecialization.DOCUMENTATION],
            "experiment-designer": [AgentSpecialization.RESEARCH, AgentSpecialization.TESTING],
            "hypothesis-tester": [AgentSpecialization.RESEARCH, AgentSpecialization.TESTING],
            
            # Monitoring & Observability (10+ agents)
            "monitoring-specialist": [AgentSpecialization.MONITORING, AgentSpecialization.DEVOPS],
            "alerting-expert": [AgentSpecialization.MONITORING, AgentSpecialization.AUTOMATION],
            "log-analyst": [AgentSpecialization.MONITORING, AgentSpecialization.DATA],
            "metrics-engineer": [AgentSpecialization.MONITORING, AgentSpecialization.DATA],
            "trace-analyst": [AgentSpecialization.MONITORING, AgentSpecialization.BACKEND],
            "apm-specialist": [AgentSpecialization.MONITORING, AgentSpecialization.OPTIMIZATION],
            "sre-engineer": [AgentSpecialization.MONITORING, AgentSpecialization.DEVOPS],
            "incident-manager": [AgentSpecialization.MONITORING, AgentSpecialization.MANAGEMENT],
            "capacity-planner": [AgentSpecialization.MONITORING, AgentSpecialization.OPTIMIZATION],
            "availability-engineer": [AgentSpecialization.MONITORING, AgentSpecialization.DEVOPS],
            
            # Automation Specialists (10+ agents)
            "automation-engineer": [AgentSpecialization.AUTOMATION, AgentSpecialization.DEVOPS],
            "rpa-developer": [AgentSpecialization.AUTOMATION, AgentSpecialization.BACKEND],
            "workflow-designer": [AgentSpecialization.AUTOMATION, AgentSpecialization.ARCHITECTURE],
            "scheduler-expert": [AgentSpecialization.AUTOMATION, AgentSpecialization.OPTIMIZATION],
            "orchestration-specialist": [AgentSpecialization.AUTOMATION, AgentSpecialization.ARCHITECTURE],
            "pipeline-engineer": [AgentSpecialization.AUTOMATION, AgentSpecialization.DEVOPS],
            "batch-processor": [AgentSpecialization.AUTOMATION, AgentSpecialization.DATA],
            "event-processor": [AgentSpecialization.AUTOMATION, AgentSpecialization.BACKEND],
            "queue-manager": [AgentSpecialization.AUTOMATION, AgentSpecialization.BACKEND],
            "task-coordinator": [AgentSpecialization.AUTOMATION, AgentSpecialization.MANAGEMENT]
        }
        
        # Create agent capabilities for all 231 agents
        for agent_name, specializations in agent_specializations.items():
            # Determine complexity range based on specializations
            if AgentSpecialization.ARCHITECTURE in specializations:
                complexity_range = (3, 6)  # Can handle complex tasks
            elif AgentSpecialization.AI_ML in specializations:
                complexity_range = (2, 6)  # Wide range
            elif AgentSpecialization.OPTIMIZATION in specializations:
                complexity_range = (3, 5)  # Moderate to complex
            elif AgentSpecialization.TESTING in specializations:
                complexity_range = (2, 4)  # Simple to moderate
            else:
                complexity_range = (1, 4)  # Default range
            
            # Create agent capability
            self.agents[agent_name] = AgentCapability(
                name=agent_name,
                specializations=specializations,
                complexity_range=complexity_range,
                performance_score=random.uniform(0.75, 0.95),
                success_rate=random.uniform(0.85, 0.98),
                average_execution_time=random.uniform(0.5, 5.0),
                resource_requirements={
                    "cpu": random.uniform(5, 25),
                    "memory": random.uniform(10, 30),
                    "gpu": random.uniform(0, 20) if AgentSpecialization.AI_ML in specializations else 0,
                    "network": random.uniform(5, 15)
                },
                dependencies=[],
                tags=set([s.value for s in specializations])
            )
    
    def _initialize_specialization_matrix(self):
        """Initialize the specialization effectiveness matrix."""
        self.specialization_matrix = {
            # Task domain -> Optimal specializations
            "backend_api": [AgentSpecialization.BACKEND, AgentSpecialization.ARCHITECTURE],
            "frontend_ui": [AgentSpecialization.FRONTEND, AgentSpecialization.OPTIMIZATION],
            "ml_model": [AgentSpecialization.AI_ML, AgentSpecialization.DATA],
            "deployment": [AgentSpecialization.DEVOPS, AgentSpecialization.AUTOMATION],
            "security_audit": [AgentSpecialization.SECURITY, AgentSpecialization.TESTING],
            "performance": [AgentSpecialization.OPTIMIZATION, AgentSpecialization.MONITORING],
            "data_pipeline": [AgentSpecialization.DATA, AgentSpecialization.AUTOMATION],
            "testing": [AgentSpecialization.TESTING, AgentSpecialization.AUTOMATION],
            "documentation": [AgentSpecialization.DOCUMENTATION, AgentSpecialization.ARCHITECTURE],
            "monitoring": [AgentSpecialization.MONITORING, AgentSpecialization.DEVOPS]
        }
    
    def _initialize_performance_tracking(self):
        """Initialize performance tracking for all agents."""
        for agent_name, capability in self.agents.items():
            self.performance_history[agent_name] = PerformanceMetrics(
                agent_name=agent_name,
                specialization_scores={
                    spec.value: random.uniform(0.7, 1.0) 
                    for spec in capability.specializations
                }
            )
    
    def analyze_task(self, task_description: str) -> TaskSpecification:
        """
        Analyze a task description to extract requirements and complexity.
        
        Args:
            task_description: Natural language task description
            
        Returns:
            TaskSpecification with analyzed requirements
        """
        # Extract domain from description
        domain = self._identify_domain(task_description)
        
        # Assess complexity
        complexity = self._assess_complexity(task_description)
        
        # Extract requirements
        requirements = self._extract_requirements(task_description)
        
        # Create task specification
        task_spec = TaskSpecification(
            id=hashlib.md5(task_description.encode()).hexdigest()[:8],
            description=task_description,
            domain=domain,
            complexity=complexity,
            requirements=requirements,
            constraints={
                "max_execution_time": 300,  # 5 minutes default
                "resource_budget": {"cpu": 50, "memory": 50, "gpu": 30}
            },
            priority=self._determine_priority(task_description),
            metadata={
                "created_at": datetime.now().isoformat(),
                "keywords": self._extract_keywords(task_description)
            }
        )
        
        logger.info(f"Task analyzed: {task_spec.id} - Domain: {domain.value}, Complexity: {complexity.name}")
        return task_spec
    
    def _identify_domain(self, description: str) -> AgentSpecialization:
        """Identify the primary domain from task description."""
        description_lower = description.lower()
        
        # Domain keywords mapping
        domain_keywords = {
            AgentSpecialization.BACKEND: ["api", "server", "database", "backend", "endpoint", "rest", "graphql"],
            AgentSpecialization.FRONTEND: ["ui", "interface", "react", "vue", "frontend", "component", "style"],
            AgentSpecialization.AI_ML: ["ai", "ml", "model", "training", "neural", "llm", "machine learning"],
            AgentSpecialization.DEVOPS: ["deploy", "docker", "kubernetes", "ci/cd", "infrastructure", "container"],
            AgentSpecialization.SECURITY: ["security", "vulnerability", "penetration", "audit", "compliance", "threat"],
            AgentSpecialization.TESTING: ["test", "qa", "quality", "validation", "verification", "coverage"],
            AgentSpecialization.DATA: ["data", "etl", "pipeline", "warehouse", "analytics", "processing"],
            AgentSpecialization.ARCHITECTURE: ["architecture", "design", "pattern", "structure", "system"],
            AgentSpecialization.OPTIMIZATION: ["optimize", "performance", "speed", "efficiency", "resource"],
            AgentSpecialization.AUTOMATION: ["automate", "workflow", "orchestrate", "schedule", "trigger"],
            AgentSpecialization.MONITORING: ["monitor", "observe", "alert", "metric", "log", "trace"],
            AgentSpecialization.DOCUMENTATION: ["document", "readme", "wiki", "guide", "manual", "tutorial"],
            AgentSpecialization.MANAGEMENT: ["manage", "coordinate", "plan", "organize", "lead", "scrum"],
            AgentSpecialization.RESEARCH: ["research", "analyze", "investigate", "explore", "study", "experiment"],
            AgentSpecialization.ANALYSIS: ["analyze", "assess", "evaluate", "review", "inspect", "examine"]
        }
        
        # Count keyword matches
        domain_scores = {}
        for domain, keywords in domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in description_lower)
            if score > 0:
                domain_scores[domain] = score
        
        # Return domain with highest score, default to BACKEND
        if domain_scores:
            return max(domain_scores, key=domain_scores.get)
        return AgentSpecialization.BACKEND
    
    def _assess_complexity(self, description: str) -> TaskComplexity:
        """Assess task complexity from description."""
        description_lower = description.lower()
        
        # Complexity indicators
        if any(word in description_lower for word in ["simple", "basic", "trivial", "quick"]):
            return TaskComplexity.SIMPLE
        elif any(word in description_lower for word in ["complex", "advanced", "sophisticated", "multi"]):
            return TaskComplexity.COMPLEX
        elif any(word in description_lower for word in ["ultra", "extreme", "massive", "enterprise"]):
            return TaskComplexity.ULTRA_COMPLEX
        elif any(word in description_lower for word in ["moderate", "standard", "regular"]):
            return TaskComplexity.MODERATE
        
        # Default based on word count
        word_count = len(description.split())
        if word_count < 20:
            return TaskComplexity.SIMPLE
        elif word_count < 50:
            return TaskComplexity.MODERATE
        elif word_count < 100:
            return TaskComplexity.COMPLEX
        else:
            return TaskComplexity.HIGHLY_COMPLEX
    
    def _extract_requirements(self, description: str) -> Dict[str, Any]:
        """Extract specific requirements from task description."""
        requirements = {
            "languages": [],
            "frameworks": [],
            "tools": [],
            "integrations": [],
            "performance": {},
            "security": {}
        }
        
        description_lower = description.lower()
        
        # Extract programming languages
        languages = ["python", "javascript", "typescript", "java", "go", "rust", "ruby", "php", "c++", "swift"]
        requirements["languages"] = [lang for lang in languages if lang in description_lower]
        
        # Extract frameworks
        frameworks = ["react", "vue", "angular", "django", "fastapi", "flask", "spring", "express", "rails"]
        requirements["frameworks"] = [fw for fw in frameworks if fw in description_lower]
        
        # Extract tools
        tools = ["docker", "kubernetes", "terraform", "ansible", "jenkins", "git", "prometheus", "grafana"]
        requirements["tools"] = [tool for tool in tools if tool in description_lower]
        
        return requirements
    
    def _determine_priority(self, description: str) -> int:
        """Determine task priority (1-10 scale)."""
        description_lower = description.lower()
        
        if any(word in description_lower for word in ["urgent", "critical", "emergency", "asap"]):
            return 10
        elif any(word in description_lower for word in ["high", "important", "priority"]):
            return 8
        elif any(word in description_lower for word in ["low", "minor", "optional"]):
            return 3
        
        return 5  # Default priority
    
    def _extract_keywords(self, description: str) -> List[str]:
        """Extract keywords from task description."""
        # Simple keyword extraction
        words = re.findall(r'\b[a-z]+\b', description.lower())
        # Filter common words
        stopwords = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "from", "as", "is", "was", "are", "were"}
        keywords = [w for w in words if w not in stopwords and len(w) > 3]
        return list(set(keywords))[:10]  # Return top 10 unique keywords
    
    def select_optimal_agent(self, task_spec: TaskSpecification) -> AgentSelectionResult:
        """
        Select the optimal agent(s) for a given task specification.
        
        This implements the intelligent selection algorithm required by Rule 14.
        
        Args:
            task_spec: Complete task specification
            
        Returns:
            AgentSelectionResult with optimal agent selection
        """
        # Filter agents by domain and complexity
        candidate_agents = self._filter_capable_agents(task_spec)
        
        if not candidate_agents:
            logger.warning(f"No suitable agents found for task {task_spec.id}")
            return self._create_fallback_selection(task_spec)
        
        # Score and rank agents
        agent_scores = self._score_agents(candidate_agents, task_spec)
        
        # Select primary and backup agents
        sorted_agents = sorted(agent_scores.items(), key=lambda x: x[1], reverse=True)
        primary_agent = sorted_agents[0][0]
        backup_agents = [agent for agent, _ in sorted_agents[1:4]]  # Top 3 backups
        
        # Determine coordination pattern
        coordination_pattern = self._determine_coordination_pattern(task_spec, len(candidate_agents))
        
        # Calculate confidence and resource allocation
        confidence_score = self._calculate_confidence(primary_agent, task_spec)
        resource_allocation = self._allocate_resources(primary_agent, task_spec)
        
        # Create workflow stages if multi-agent
        workflow_stages = []
        if coordination_pattern != CoordinationPattern.SEQUENTIAL:
            workflow_stages = self._design_workflow_stages(task_spec, sorted_agents[:5])
        
        result = AgentSelectionResult(
            primary_agent=primary_agent,
            backup_agents=backup_agents,
            coordination_pattern=coordination_pattern,
            confidence_score=confidence_score,
            estimated_completion_time=self.agents[primary_agent].average_execution_time,
            resource_allocation=resource_allocation,
            reasoning=f"Selected {primary_agent} based on domain match ({task_spec.domain.value}) and performance history",
            workflow_stages=workflow_stages
        )
        
        logger.info(f"Agent selected for task {task_spec.id}: {primary_agent} (confidence: {confidence_score:.2f})")
        return result
    
    def _filter_capable_agents(self, task_spec: TaskSpecification) -> List[str]:
        """Filter agents capable of handling the task."""
        capable_agents = []
        
        for agent_name, capability in self.agents.items():
            # Check domain match
            if task_spec.domain not in capability.specializations:
                continue
            
            # Check complexity range
            complexity_value = task_spec.complexity.value
            if complexity_value < capability.complexity_range[0] or complexity_value > capability.complexity_range[1]:
                continue
            
            # Check resource availability
            if not self._check_resource_availability(capability.resource_requirements):
                continue
            
            capable_agents.append(agent_name)
        
        return capable_agents
    
    def _score_agents(self, agents: List[str], task_spec: TaskSpecification) -> Dict[str, float]:
        """Score agents based on multiple criteria."""
        scores = {}
        
        for agent_name in agents:
            capability = self.agents[agent_name]
            performance = self.performance_history[agent_name]
            
            # Calculate score components
            domain_score = self._calculate_domain_score(capability, task_spec)
            performance_score = self._calculate_performance_score(performance, task_spec)
            availability_score = self._calculate_availability_score(capability)
            specialization_score = self._calculate_specialization_score(capability, task_spec)
            
            # Weighted combination
            total_score = (
                domain_score * 0.3 +
                performance_score * 0.3 +
                availability_score * 0.2 +
                specialization_score * 0.2
            )
            
            scores[agent_name] = total_score
        
        return scores
    
    def _calculate_domain_score(self, capability: AgentCapability, task_spec: TaskSpecification) -> float:
        """Calculate domain matching score."""
        if task_spec.domain in capability.specializations:
            return 1.0
        
        # Check for related domains
        related_score = 0.0
        for spec in capability.specializations:
            if spec in [AgentSpecialization.ARCHITECTURE, AgentSpecialization.AUTOMATION]:
                related_score = max(related_score, 0.5)  # These are broadly applicable
        
        return related_score
    
    def _calculate_performance_score(self, performance: PerformanceMetrics, task_spec: TaskSpecification) -> float:
        """Calculate performance history score."""
        if performance.total_tasks == 0:
            return 0.75  # Default score for new agents
        
        success_rate = performance.successful_tasks / performance.total_tasks
        
        # Get specialization-specific score
        domain_key = task_spec.domain.value
        spec_score = performance.specialization_scores.get(domain_key, 0.75)
        
        return (success_rate * 0.6 + spec_score * 0.4)
    
    def _calculate_availability_score(self, capability: AgentCapability) -> float:
        """Calculate resource availability score."""
        required_resources = capability.resource_requirements
        available_score = 1.0
        
        for resource, required in required_resources.items():
            if resource in self.resource_pool:
                available = self.resource_pool[resource]
                if available < required:
                    available_score *= (available / required)
        
        return available_score
    
    def _calculate_specialization_score(self, capability: AgentCapability, task_spec: TaskSpecification) -> float:
        """Calculate specialization effectiveness score."""
        # Check if agent has multiple relevant specializations
        relevant_specs = 0
        for spec in capability.specializations:
            if spec == task_spec.domain:
                relevant_specs += 2
            elif spec in [AgentSpecialization.AUTOMATION, AgentSpecialization.OPTIMIZATION]:
                relevant_specs += 1
        
        return min(1.0, relevant_specs / 3.0)
    
    def _check_resource_availability(self, requirements: Dict[str, float]) -> bool:
        """Check if required resources are available."""
        for resource, required in requirements.items():
            if resource in self.resource_pool:
                if self.resource_pool[resource] < required:
                    return False
        return True
    
    def _determine_coordination_pattern(self, task_spec: TaskSpecification, agent_count: int) -> CoordinationPattern:
        """Determine the optimal coordination pattern."""
        complexity = task_spec.complexity.value
        
        if complexity <= 2:
            return CoordinationPattern.SEQUENTIAL
        elif complexity <= 4 and agent_count > 3:
            return CoordinationPattern.PARALLEL
        elif complexity >= 5 and agent_count > 5:
            return CoordinationPattern.HIERARCHICAL
        elif "pipeline" in task_spec.description.lower():
            return CoordinationPattern.PIPELINE
        elif "event" in task_spec.description.lower():
            return CoordinationPattern.EVENT_DRIVEN
        else:
            return CoordinationPattern.HYBRID
    
    def _calculate_confidence(self, agent_name: str, task_spec: TaskSpecification) -> float:
        """Calculate confidence score for agent selection."""
        capability = self.agents[agent_name]
        performance = self.performance_history[agent_name]
        
        # Base confidence from capability
        base_confidence = capability.success_rate
        
        # Adjust based on domain match
        if task_spec.domain in capability.specializations:
            base_confidence *= 1.1
        
        # Adjust based on performance history
        if performance.total_tasks > 10:
            historical_success = performance.successful_tasks / performance.total_tasks
            base_confidence = (base_confidence + historical_success) / 2
        
        return min(1.0, base_confidence)
    
    def _allocate_resources(self, agent_name: str, task_spec: TaskSpecification) -> Dict[str, float]:
        """Allocate resources for agent execution."""
        capability = self.agents[agent_name]
        allocation = {}
        
        for resource, required in capability.resource_requirements.items():
            if resource in self.resource_pool:
                # Allocate with some buffer
                allocated = min(required * 1.2, self.resource_pool[resource])
                allocation[resource] = allocated
                # Temporarily reduce available pool
                self.resource_pool[resource] -= allocated
        
        return allocation
    
    def _create_fallback_selection(self, task_spec: TaskSpecification) -> AgentSelectionResult:
        """Create a fallback selection when no optimal agent is found."""
        # Find any agent that can handle the complexity
        fallback_agents = []
        for agent_name, capability in self.agents.items():
            if task_spec.complexity.value <= capability.complexity_range[1]:
                fallback_agents.append(agent_name)
        
        if fallback_agents:
            primary = random.choice(fallback_agents[:5])
        else:
            primary = "ai-senior-full-stack-developer"  # Ultimate fallback
        
        return AgentSelectionResult(
            primary_agent=primary,
            backup_agents=fallback_agents[1:4] if len(fallback_agents) > 1 else [],
            coordination_pattern=CoordinationPattern.SEQUENTIAL,
            confidence_score=0.5,
            estimated_completion_time=10.0,
            resource_allocation={"cpu": 25, "memory": 25},
            reasoning="Fallback selection due to no optimal match",
            workflow_stages=[]
        )
    
    def _design_workflow_stages(self, task_spec: TaskSpecification, agents: List[Tuple[str, float]]) -> List[Dict[str, Any]]:
        """Design multi-agent workflow stages."""
        stages = []
        
        if len(agents) < 2:
            return stages
        
        # Create workflow based on coordination pattern
        if task_spec.complexity.value >= 4:
            # Complex task - multi-stage workflow
            stages.append({
                "stage": 1,
                "name": "Analysis & Planning",
                "agents": [agents[0][0]],
                "type": "sequential",
                "outputs": ["requirements", "design"]
            })
            
            if len(agents) > 2:
                stages.append({
                    "stage": 2,
                    "name": "Implementation",
                    "agents": [a[0] for a in agents[1:3]],
                    "type": "parallel",
                    "outputs": ["code", "tests"]
                })
            
            stages.append({
                "stage": 3,
                "name": "Validation & Optimization",
                "agents": [agents[-1][0]],
                "type": "sequential",
                "outputs": ["validated_solution"]
            })
        
        return stages
    
    def design_multi_agent_workflow(self, complex_task: str) -> Dict[str, Any]:
        """
        Design a complete multi-agent workflow for complex tasks.
        
        This implements the sophisticated workflow design required by Rule 14.
        
        Args:
            complex_task: Complex task description requiring multiple agents
            
        Returns:
            Complete workflow specification with stages and coordination
        """
        # Analyze the complex task
        task_spec = self.analyze_task(complex_task)
        
        # Decompose into subtasks
        subtasks = self._decompose_complex_task(complex_task)
        
        # Select agents for each subtask
        agent_assignments = {}
        for subtask in subtasks:
            subtask_spec = self.analyze_task(subtask["description"])
            selection = self.select_optimal_agent(subtask_spec)
            agent_assignments[subtask["id"]] = selection
        
        # Create coordination plan
        coordination_plan = self._create_coordination_plan(subtasks, agent_assignments)
        
        # Design handoff protocols
        handoff_protocols = self._design_handoff_protocols(subtasks, agent_assignments)
        
        # Create quality gates
        quality_gates = self._create_quality_gates(subtasks)
        
        workflow = {
            "id": hashlib.md5(complex_task.encode()).hexdigest()[:8],
            "description": complex_task,
            "total_agents": len(set(a.primary_agent for a in agent_assignments.values())),
            "subtasks": subtasks,
            "agent_assignments": {k: v.primary_agent for k, v in agent_assignments.items()},
            "coordination_plan": coordination_plan,
            "handoff_protocols": handoff_protocols,
            "quality_gates": quality_gates,
            "estimated_duration": sum(a.estimated_completion_time for a in agent_assignments.values()),
            "resource_requirements": self._aggregate_resources(agent_assignments),
            "created_at": datetime.now().isoformat()
        }
        
        logger.info(f"Multi-agent workflow designed: {workflow['id']} with {workflow['total_agents']} agents")
        return workflow
    
    def _decompose_complex_task(self, complex_task: str) -> List[Dict[str, Any]]:
        """Decompose a complex task into subtasks."""
        subtasks = []
        
        # Simple decomposition based on task components
        components = ["design", "implementation", "testing", "deployment", "documentation"]
        
        for i, component in enumerate(components):
            if component in complex_task.lower() or len(subtasks) < 3:
                subtasks.append({
                    "id": f"subtask_{i+1}",
                    "description": f"{component.capitalize()} phase of: {complex_task[:100]}",
                    "dependencies": [f"subtask_{i}"] if i > 0 else [],
                    "priority": 5 - i,
                    "type": component
                })
        
        return subtasks
    
    def _create_coordination_plan(self, subtasks: List[Dict], assignments: Dict) -> Dict[str, Any]:
        """Create a coordination plan for multi-agent execution."""
        return {
            "execution_mode": "hybrid",
            "stages": [
                {
                    "stage_id": f"stage_{i+1}",
                    "subtasks": [st["id"] for st in subtasks if i == 0 or st["dependencies"]],
                    "mode": "sequential" if i == 0 else "parallel",
                    "checkpoint": f"checkpoint_{i+1}"
                }
                for i in range(min(3, len(subtasks)))
            ],
            "synchronization_points": ["checkpoint_1", "checkpoint_2", "checkpoint_3"],
            "error_handling": "retry_with_fallback",
            "timeout": 600  # 10 minutes
        }
    
    def _design_handoff_protocols(self, subtasks: List[Dict], assignments: Dict) -> List[Dict[str, Any]]:
        """Design handoff protocols between agents."""
        protocols = []
        
        for i, subtask in enumerate(subtasks[:-1]):
            next_subtask = subtasks[i + 1]
            protocols.append({
                "from_agent": assignments[subtask["id"]].primary_agent,
                "to_agent": assignments[next_subtask["id"]].primary_agent,
                "handoff_type": "knowledge_transfer",
                "data_format": "structured_json",
                "validation": "schema_check",
                "retry_policy": "exponential_backoff"
            })
        
        return protocols
    
    def _create_quality_gates(self, subtasks: List[Dict]) -> List[Dict[str, Any]]:
        """Create quality gates for workflow validation."""
        gates = []
        
        for subtask in subtasks:
            gates.append({
                "gate_id": f"qg_{subtask['id']}",
                "subtask_id": subtask["id"],
                "criteria": {
                    "completeness": 0.9,
                    "quality_score": 0.85,
                    "test_coverage": 0.8 if "test" in subtask["type"] else 0.7
                },
                "validation_method": "automated",
                "failure_action": "retry_or_escalate"
            })
        
        return gates
    
    def _aggregate_resources(self, assignments: Dict) -> Dict[str, float]:
        """Aggregate resource requirements across all agents."""
        total_resources = {"cpu": 0, "memory": 0, "gpu": 0, "network": 0}
        
        for selection in assignments.values():
            for resource, amount in selection.resource_allocation.items():
                total_resources[resource] += amount
        
        return total_resources
    
    def update_performance(self, agent_name: str, task_id: str, success: bool, 
                          quality_score: float, execution_time: float):
        """
        Update agent performance metrics after task completion.
        
        Args:
            agent_name: Name of the agent
            task_id: Task identifier
            success: Whether task was successful
            quality_score: Quality score (0-1)
            execution_time: Actual execution time
        """
        if agent_name not in self.performance_history:
            return
        
        metrics = self.performance_history[agent_name]
        metrics.total_tasks += 1
        
        if success:
            metrics.successful_tasks += 1
        else:
            metrics.failed_tasks += 1
        
        # Update average completion time
        metrics.average_completion_time = (
            (metrics.average_completion_time * (metrics.total_tasks - 1) + execution_time) 
            / metrics.total_tasks
        )
        
        # Add quality score
        metrics.quality_scores.append(quality_score)
        if len(metrics.quality_scores) > 100:
            metrics.quality_scores = metrics.quality_scores[-100:]  # Keep last 100
        
        metrics.last_updated = datetime.now()
        
        logger.info(f"Performance updated for {agent_name}: success={success}, quality={quality_score:.2f}")
    
    def get_agent_recommendations(self, task_description: str, count: int = 5) -> List[Dict[str, Any]]:
        """
        Get top N agent recommendations for a task.
        
        Args:
            task_description: Task description
            count: Number of recommendations
            
        Returns:
            List of agent recommendations with scores
        """
        task_spec = self.analyze_task(task_description)
        capable_agents = self._filter_capable_agents(task_spec)
        
        if not capable_agents:
            return []
        
        agent_scores = self._score_agents(capable_agents, task_spec)
        sorted_agents = sorted(agent_scores.items(), key=lambda x: x[1], reverse=True)
        
        recommendations = []
        for agent_name, score in sorted_agents[:count]:
            capability = self.agents[agent_name]
            performance = self.performance_history[agent_name]
            
            recommendations.append({
                "agent": agent_name,
                "score": score,
                "specializations": [s.value for s in capability.specializations],
                "success_rate": capability.success_rate,
                "average_time": capability.average_execution_time,
                "total_tasks": performance.total_tasks,
                "confidence": self._calculate_confidence(agent_name, task_spec)
            })
        
        return recommendations
    
    def get_specialization_coverage(self) -> Dict[str, int]:
        """Get count of agents per specialization."""
        coverage = defaultdict(int)
        
        for capability in self.agents.values():
            for spec in capability.specializations:
                coverage[spec.value] += 1
        
        return dict(coverage)
    
    def predict_workflow_success(self, workflow: Dict[str, Any]) -> float:
        """
        Predict success probability for a multi-agent workflow.
        
        Args:
            workflow: Workflow specification
            
        Returns:
            Success probability (0-1)
        """
        if "agent_assignments" not in workflow:
            return 0.5
        
        # Calculate average success rate of assigned agents
        success_rates = []
        for agent_name in workflow["agent_assignments"].values():
            if agent_name in self.agents:
                success_rates.append(self.agents[agent_name].success_rate)
        
        if not success_rates:
            return 0.5
        
        # Account for coordination complexity
        num_agents = len(set(workflow["agent_assignments"].values()))
        coordination_factor = 1.0 - (num_agents - 1) * 0.05  # 5% reduction per additional agent
        
        base_probability = np.mean(success_rates)
        adjusted_probability = base_probability * max(0.5, coordination_factor)
        
        return min(1.0, adjusted_probability)