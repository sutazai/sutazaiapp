#!/usr/bin/env python3
"""
Advanced Reasoning Engine for SutazAI AGI/ASI System
Implements symbolic reasoning, probabilistic inference, and causal analysis
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import networkx as nx
from pyDatalog import pyDatalog
import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
import aiohttp
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SutazAI Reasoning Engine", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ReasoningType(Enum):
    DEDUCTIVE = "deductive"
    INDUCTIVE = "inductive"
    ABDUCTIVE = "abductive"
    ANALOGICAL = "analogical"
    CAUSAL = "causal"
    PROBABILISTIC = "probabilistic"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    METACOGNITIVE = "metacognitive"

@dataclass
class Proposition:
    """Logical proposition with truth value and confidence"""
    content: str
    truth_value: Optional[bool] = None
    confidence: float = 1.0
    source: Optional[str] = None
    timestamp: Optional[datetime] = None

@dataclass
class Rule:
    """Inference rule for reasoning"""
    name: str
    premises: List[str]
    conclusion: str
    confidence: float = 1.0
    rule_type: str = "modus_ponens"

class KnowledgeBase:
    """Advanced knowledge base with multiple reasoning capabilities"""
    
    def __init__(self):
        self.facts = {}
        self.rules = {}
        self.causal_graph = nx.DiGraph()
        self.bayesian_network = None
        self.temporal_facts = []
        self.spatial_relations = nx.Graph()
        pyDatalog.create_terms('X, Y, Z, parent, ancestor, implies')
        
    def add_fact(self, fact: Proposition):
        """Add a fact to the knowledge base"""
        self.facts[fact.content] = fact
        if fact.timestamp:
            self.temporal_facts.append(fact)
            
    def add_rule(self, rule: Rule):
        """Add an inference rule"""
        self.rules[rule.name] = rule
        
    def add_causal_relation(self, cause: str, effect: str, strength: float = 1.0):
        """Add causal relationship"""
        self.causal_graph.add_edge(cause, effect, weight=strength)
        
    def query(self, query: str, reasoning_type: ReasoningType) -> Dict[str, Any]:
        """Query the knowledge base using specified reasoning type"""
        if reasoning_type == ReasoningType.DEDUCTIVE:
            return self._deductive_reasoning(query)
        elif reasoning_type == ReasoningType.PROBABILISTIC:
            return self._probabilistic_reasoning(query)
        elif reasoning_type == ReasoningType.CAUSAL:
            return self._causal_reasoning(query)
        elif reasoning_type == ReasoningType.TEMPORAL:
            return self._temporal_reasoning(query)
        else:
            return {"error": f"Reasoning type {reasoning_type} not implemented"}

    def _deductive_reasoning(self, query: str) -> Dict[str, Any]:
        """Perform deductive reasoning using logic programming"""
        # Use pyDatalog for logical inference
        results = []
        
        # Check direct facts
        if query in self.facts:
            return {
                "result": True,
                "confidence": self.facts[query].confidence,
                "reasoning": "Direct fact",
                "proof": [query]
            }
        
        # Apply rules
        for rule_name, rule in self.rules.items():
            if rule.conclusion == query:
                premises_satisfied = all(
                    premise in self.facts and self.facts[premise].truth_value
                    for premise in rule.premises
                )
                if premises_satisfied:
                    confidence = min(
                        self.facts[p].confidence for p in rule.premises
                    ) * rule.confidence
                    results.append({
                        "result": True,
                        "confidence": confidence,
                        "reasoning": f"Rule: {rule_name}",
                        "proof": rule.premises + [f"→ {query}"]
                    })
        
        return results[0] if results else {"result": False, "reasoning": "No proof found"}

    def _probabilistic_reasoning(self, query: str) -> Dict[str, Any]:
        """Perform probabilistic reasoning using Bayesian networks"""
        if not self.bayesian_network:
            return {"error": "Bayesian network not initialized"}
        
        inference = VariableElimination(self.bayesian_network)
        
        # Parse query for variable and evidence
        # Example: P(A|B=true,C=false)
        try:
            result = inference.query(variables=[query])
            return {
                "result": result,
                "reasoning": "Bayesian inference",
                "confidence": float(max(result.values))
            }
        except Exception as e:
            return {"error": str(e)}

    def _causal_reasoning(self, query: str) -> Dict[str, Any]:
        """Perform causal reasoning using causal graphs"""
        # Parse query for causal relationship
        # Example: "Does A cause B?"
        parts = query.split(" cause ")
        if len(parts) == 2:
            cause, effect = parts[0].strip(), parts[1].strip("?")
            
            if self.causal_graph.has_edge(cause, effect):
                strength = self.causal_graph[cause][effect]['weight']
                return {
                    "result": True,
                    "causal_strength": strength,
                    "reasoning": "Direct causal link",
                    "path": [cause, effect]
                }
            
            # Check for indirect causation
            try:
                paths = list(nx.all_simple_paths(self.causal_graph, cause, effect))
                if paths:
                    # Calculate path strength as product of edge weights
                    path_strengths = []
                    for path in paths:
                        strength = 1.0
                        for i in range(len(path) - 1):
                            strength *= self.causal_graph[path[i]][path[i+1]]['weight']
                        path_strengths.append((path, strength))
                    
                    best_path = max(path_strengths, key=lambda x: x[1])
                    return {
                        "result": True,
                        "causal_strength": best_path[1],
                        "reasoning": "Indirect causal chain",
                        "path": best_path[0]
                    }
            except nx.NetworkXNoPath:
                pass
        
        return {"result": False, "reasoning": "No causal relationship found"}

    def _temporal_reasoning(self, query: str) -> Dict[str, Any]:
        """Perform temporal reasoning"""
        # Example: "What happened before X?"
        # This would analyze temporal facts and their relationships
        temporal_results = []
        
        for fact in self.temporal_facts:
            # Implement Allen's interval algebra or similar
            temporal_results.append({
                "fact": fact.content,
                "time": fact.timestamp.isoformat() if fact.timestamp else None
            })
        
        return {
            "results": temporal_results,
            "reasoning": "Temporal analysis"
        }

class AdvancedReasoningEngine:
    """Main reasoning engine with multiple reasoning strategies"""
    
    def __init__(self):
        self.knowledge_base = KnowledgeBase()
        self.reasoning_cache = {}
        self.meta_reasoner = MetaReasoner()
        
    async def reason(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Perform reasoning on a query with optional context"""
        
        # Check cache
        cache_key = f"{query}:{json.dumps(context or {}, sort_keys=True)}"
        if cache_key in self.reasoning_cache:
            return self.reasoning_cache[cache_key]
        
        # Determine best reasoning approach
        reasoning_type = self.meta_reasoner.select_reasoning_type(query, context)
        
        # Perform reasoning
        result = await self._perform_reasoning(query, reasoning_type, context)
        
        # Cache result
        self.reasoning_cache[cache_key] = result
        
        return result
    
    async def _perform_reasoning(self, query: str, reasoning_type: ReasoningType, 
                                context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute reasoning based on type"""
        
        # Enrich knowledge base with context
        if context:
            for key, value in context.items():
                self.knowledge_base.add_fact(Proposition(
                    content=f"{key}: {value}",
                    truth_value=True,
                    source="context"
                ))
        
        # Perform reasoning
        base_result = self.knowledge_base.query(query, reasoning_type)
        
        # Enhance with explanation generation
        explanation = await self._generate_explanation(query, base_result, reasoning_type)
        
        # Add confidence calibration
        calibrated_confidence = self._calibrate_confidence(base_result, context)
        
        return {
            **base_result,
            "reasoning_type": reasoning_type.value,
            "explanation": explanation,
            "calibrated_confidence": calibrated_confidence,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _generate_explanation(self, query: str, result: Dict[str, Any], 
                                   reasoning_type: ReasoningType) -> str:
        """Generate human-readable explanation of reasoning"""
        
        explanation_parts = [
            f"Query: {query}",
            f"Reasoning type: {reasoning_type.value}",
            f"Result: {result.get('result', 'Unknown')}"
        ]
        
        if 'reasoning' in result:
            explanation_parts.append(f"Method: {result['reasoning']}")
        
        if 'proof' in result:
            explanation_parts.append("Proof chain:")
            for step in result['proof']:
                explanation_parts.append(f"  - {step}")
        
        if 'confidence' in result:
            explanation_parts.append(f"Confidence: {result['confidence']:.2%}")
        
        return "\n".join(explanation_parts)
    
    def _calibrate_confidence(self, result: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Calibrate confidence based on multiple factors"""
        
        base_confidence = result.get('confidence', 0.5)
        
        # Adjust based on context completeness
        if context:
            context_factor = min(len(context) / 10, 1.0)
            base_confidence *= (0.8 + 0.2 * context_factor)
        
        # Adjust based on reasoning type reliability
        type_factors = {
            ReasoningType.DEDUCTIVE: 1.0,
            ReasoningType.PROBABILISTIC: 0.9,
            ReasoningType.CAUSAL: 0.85,
            ReasoningType.INDUCTIVE: 0.8,
            ReasoningType.ABDUCTIVE: 0.75
        }
        
        reasoning_type = result.get('reasoning_type', ReasoningType.DEDUCTIVE)
        base_confidence *= type_factors.get(reasoning_type, 0.7)
        
        return min(base_confidence, 0.99)

class MetaReasoner:
    """Meta-reasoning component for selecting reasoning strategies"""
    
    def select_reasoning_type(self, query: str, context: Dict[str, Any]) -> ReasoningType:
        """Select appropriate reasoning type based on query and context"""
        
        query_lower = query.lower()
        
        # Pattern matching for reasoning type selection
        if any(word in query_lower for word in ['if', 'then', 'implies', 'therefore']):
            return ReasoningType.DEDUCTIVE
        
        if any(word in query_lower for word in ['probability', 'likely', 'chance']):
            return ReasoningType.PROBABILISTIC
        
        if any(word in query_lower for word in ['cause', 'effect', 'because', 'leads to']):
            return ReasoningType.CAUSAL
        
        if any(word in query_lower for word in ['before', 'after', 'when', 'during']):
            return ReasoningType.TEMPORAL
        
        if any(word in query_lower for word in ['similar', 'like', 'analogous']):
            return ReasoningType.ANALOGICAL
        
        # Default to deductive reasoning
        return ReasoningType.DEDUCTIVE

# Global reasoning engine instance
reasoning_engine = AdvancedReasoningEngine()

# API Models
class ReasoningRequest(BaseModel):
    query: str
    context: Optional[Dict[str, Any]] = None
    reasoning_type: Optional[str] = None

class FactRequest(BaseModel):
    content: str
    truth_value: Optional[bool] = True
    confidence: float = 1.0
    source: Optional[str] = None

class RuleRequest(BaseModel):
    name: str
    premises: List[str]
    conclusion: str
    confidence: float = 1.0

class CausalRelationRequest(BaseModel):
    cause: str
    effect: str
    strength: float = 1.0

# API Endpoints
@app.post("/reason")
async def reason(request: ReasoningRequest):
    """Perform reasoning on a query"""
    try:
        result = await reasoning_engine.reason(
            request.query,
            request.context
        )
        return result
    except Exception as e:
        logger.error(f"Reasoning error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/add_fact")
async def add_fact(request: FactRequest):
    """Add a fact to the knowledge base"""
    try:
        fact = Proposition(
            content=request.content,
            truth_value=request.truth_value,
            confidence=request.confidence,
            source=request.source,
            timestamp=datetime.now()
        )
        reasoning_engine.knowledge_base.add_fact(fact)
        return {"status": "success", "fact": request.content}
    except Exception as e:
        logger.error(f"Error adding fact: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/add_rule")
async def add_rule(request: RuleRequest):
    """Add an inference rule"""
    try:
        rule = Rule(
            name=request.name,
            premises=request.premises,
            conclusion=request.conclusion,
            confidence=request.confidence
        )
        reasoning_engine.knowledge_base.add_rule(rule)
        return {"status": "success", "rule": request.name}
    except Exception as e:
        logger.error(f"Error adding rule: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/add_causal_relation")
async def add_causal_relation(request: CausalRelationRequest):
    """Add a causal relationship"""
    try:
        reasoning_engine.knowledge_base.add_causal_relation(
            request.cause,
            request.effect,
            request.strength
        )
        return {"status": "success", "relation": f"{request.cause} -> {request.effect}"}
    except Exception as e:
        logger.error(f"Error adding causal relation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/knowledge_stats")
async def get_knowledge_stats():
    """Get statistics about the knowledge base"""
    kb = reasoning_engine.knowledge_base
    return {
        "facts": len(kb.facts),
        "rules": len(kb.rules),
        "causal_relations": kb.causal_graph.number_of_edges(),
        "causal_nodes": kb.causal_graph.number_of_nodes(),
        "temporal_facts": len(kb.temporal_facts),
        "cache_size": len(reasoning_engine.reasoning_cache)
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "reasoning-engine",
        "timestamp": datetime.now().isoformat()
    }

# Convenience alias for backwards compatibility
ReasoningEngine = AdvancedReasoningEngine

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)