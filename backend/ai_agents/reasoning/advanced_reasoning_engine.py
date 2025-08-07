"""
Advanced Reasoning Engine for SutazAI automation/advanced automation Platform
=====================================================

Implements sophisticated reasoning capabilities including:
- Multi-step logical reasoning
- Causal inference
- Strategic planning
- Pattern recognition
- Uncertainty handling
- Meta-cognitive monitoring
"""

import asyncio
import json
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import uuid
from collections import defaultdict, deque
import math

logger = logging.getLogger(__name__)


class ReasoningType(Enum):
    """Types of reasoning processes"""
    DEDUCTIVE = "deductive"       # From general to specific
    INDUCTIVE = "inductive"       # From specific to general
    ABDUCTIVE = "abductive"       # Best explanation
    ANALOGICAL = "analogical"     # Reasoning by analogy
    CAUSAL = "causal"            # Cause and effect
    STRATEGIC = "strategic"       # Multi-step planning
    PROBABILISTIC = "probabilistic"  # Under uncertainty


class ConfidenceLevel(Enum):
    """Confidence levels for reasoning outcomes"""
    VERY_LOW = 0.1
    LOW = 0.3
    MEDIUM = 0.5
    HIGH = 0.7
    VERY_HIGH = 0.9


@dataclass
class Fact:
    """Represents a fact in the knowledge base"""
    id: str
    content: str
    confidence: float
    source: str
    timestamp: datetime = field(default_factory=datetime.now)
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Rule:
    """Represents a reasoning rule"""
    id: str
    name: str
    conditions: List[str]  # Condition patterns
    conclusions: List[str]  # Conclusion patterns
    confidence: float
    rule_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReasoningStep:
    """Single step in a reasoning chain"""
    id: str
    step_number: int
    reasoning_type: ReasoningType
    input_facts: List[str]
    applied_rules: List[str]
    output_facts: List[str]
    confidence: float
    explanation: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ReasoningChain:
    """Complete reasoning chain from premise to conclusion"""
    id: str
    query: str
    steps: List[ReasoningStep]
    final_conclusion: str
    overall_confidence: float
    reasoning_path: List[str]
    start_time: datetime
    end_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CausalRelation:
    """Represents a causal relationship"""
    cause: str
    effect: str
    strength: float  # 0-1
    confidence: float
    evidence: List[str]
    mechanism: Optional[str] = None


@dataclass
class Strategy:
    """Strategic plan with steps and goals"""
    id: str
    goal: str
    steps: List[Dict[str, Any]]
    expected_outcomes: List[str]
    risks: List[Dict[str, Any]]
    success_probability: float
    resource_requirements: Dict[str, Any]
    timeline: Dict[str, Any]


class AdvancedReasoningEngine:
    """
    Advanced reasoning engine with multiple reasoning modes
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        
        # Knowledge base
        self.facts: Dict[str, Fact] = {}
        self.rules: Dict[str, Rule] = {}
        self.causal_relations: Dict[str, CausalRelation] = {}
        
        # Reasoning history
        self.reasoning_chains: Dict[str, ReasoningChain] = {}
        self.pattern_library: Dict[str, Dict[str, Any]] = {}
        
        # Performance metrics
        self.reasoning_stats = {
            "total_queries": 0,
            "successful_inferences": 0,
            "average_confidence": 0.0,
            "reasoning_times": deque(maxlen=1000)
        }
        
        # Strategic planning
        self.strategies: Dict[str, Strategy] = {}
        self.goal_hierarchy: Dict[str, List[str]] = {}
        
        logger.info("Advanced Reasoning Engine initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for reasoning engine"""
        return {
            "max_reasoning_depth": 10,
            "min_confidence_threshold": 0.3,
            "max_inference_time": 30.0,  # seconds
            "enable_probabilistic_reasoning": True,
            "enable_causal_inference": True,
            "enable_strategic_planning": True,
            "pattern_recognition_threshold": 0.8,
            "uncertainty_handling": "monte_carlo",
            "metacognitive_monitoring": True
        }
    
    # ==================== Core Reasoning Methods ====================
    
    async def reason(self, 
                    query: str,
                    context: Optional[Dict[str, Any]] = None,
                    reasoning_type: Optional[ReasoningType] = None) -> ReasoningChain:
        """
        Main reasoning entry point
        """
        start_time = datetime.now()
        chain_id = str(uuid.uuid4())
        
        logger.info(f"🧠 Starting reasoning for query: {query[:100]}...")
        
        # Determine reasoning type if not specified
        if not reasoning_type:
            reasoning_type = await self._determine_reasoning_type(query, context)
        
        # Initialize reasoning chain
        chain = ReasoningChain(
            id=chain_id,
            query=query,
            steps=[],
            final_conclusion="",
            overall_confidence=0.0,
            reasoning_path=[],
            start_time=start_time
        )
        
        try:
            # Execute reasoning based on type
            if reasoning_type == ReasoningType.DEDUCTIVE:
                chain = await self._deductive_reasoning(query, context, chain)
            elif reasoning_type == ReasoningType.INDUCTIVE:
                chain = await self._inductive_reasoning(query, context, chain)
            elif reasoning_type == ReasoningType.ABDUCTIVE:
                chain = await self._abductive_reasoning(query, context, chain)
            elif reasoning_type == ReasoningType.CAUSAL:
                chain = await self._causal_reasoning(query, context, chain)
            elif reasoning_type == ReasoningType.STRATEGIC:
                chain = await self._strategic_reasoning(query, context, chain)
            elif reasoning_type == ReasoningType.PROBABILISTIC:
                chain = await self._probabilistic_reasoning(query, context, chain)
            else:
                # Default to mixed reasoning
                chain = await self._mixed_reasoning(query, context, chain)
            
            chain.end_time = datetime.now()
            
            # Store reasoning chain
            self.reasoning_chains[chain_id] = chain
            
            # Update statistics
            self._update_reasoning_stats(chain)
            
            logger.info(f"✅ Reasoning completed with confidence: {chain.overall_confidence:.2f}")
            return chain
        
        except Exception as e:
            logger.error(f"❌ Reasoning failed: {e}")
            chain.final_conclusion = f"Reasoning failed: {str(e)}"
            chain.overall_confidence = 0.0
            chain.end_time = datetime.now()
            return chain
    
    async def _determine_reasoning_type(self, 
                                      query: str,
                                      context: Optional[Dict[str, Any]]) -> ReasoningType:
        """Determine the best reasoning type for a query"""
        query_lower = query.lower()
        
        # Strategic keywords
        if any(word in query_lower for word in ["plan", "strategy", "achieve", "goal", "how to"]):
            return ReasoningType.STRATEGIC
        
        # Causal keywords
        if any(word in query_lower for word in ["why", "because", "cause", "effect", "reason"]):
            return ReasoningType.CAUSAL
        
        # Probabilistic keywords
        if any(word in query_lower for word in ["probability", "likely", "chance", "uncertain"]):
            return ReasoningType.PROBABILISTIC
        
        # Inductive keywords
        if any(word in query_lower for word in ["pattern", "trend", "generally", "usually"]):
            return ReasoningType.INDUCTIVE
        
        # Abductive keywords
        if any(word in query_lower for word in ["explain", "best explanation", "most likely"]):
            return ReasoningType.ABDUCTIVE
        
        # Default to deductive
        return ReasoningType.DEDUCTIVE
    
    # ==================== Deductive Reasoning ====================
    
    async def _deductive_reasoning(self, 
                                 query: str,
                                 context: Optional[Dict[str, Any]],
                                 chain: ReasoningChain) -> ReasoningChain:
        """
        Deductive reasoning: Apply general rules to specific cases
        """
        logger.info("🔍 Performing deductive reasoning")
        
        # Find relevant facts and rules
        relevant_facts = await self._find_relevant_facts(query, context)
        applicable_rules = await self._find_applicable_rules(relevant_facts, query)
        
        step_number = 1
        current_facts = relevant_facts.copy()
        
        # Apply rules iteratively
        for depth in range(self.config["max_reasoning_depth"]):
            new_facts = []
            applied_rules = []
            
            for rule in applicable_rules:
                if await self._can_apply_rule(rule, current_facts):
                    derived_facts = await self._apply_rule(rule, current_facts)
                    new_facts.extend(derived_facts)
                    applied_rules.append(rule.id)
            
            if not new_facts:
                break  # No new facts derived
            
            # Create reasoning step
            step = ReasoningStep(
                id=str(uuid.uuid4()),
                step_number=step_number,
                reasoning_type=ReasoningType.DEDUCTIVE,
                input_facts=[f.id for f in current_facts],
                applied_rules=applied_rules,
                output_facts=[f.id for f in new_facts],
                confidence=self._calculate_step_confidence(new_facts),
                explanation=f"Applied {len(applied_rules)} rules to derive {len(new_facts)} new facts"
            )
            
            chain.steps.append(step)
            current_facts.extend(new_facts)
            step_number += 1
        
        # Find conclusion
        conclusion = await self._find_conclusion(query, current_facts)
        chain.final_conclusion = conclusion["text"]
        chain.overall_confidence = conclusion["confidence"]
        chain.reasoning_path = [step.explanation for step in chain.steps]
        
        return chain
    
    # ==================== Inductive Reasoning ====================
    
    async def _inductive_reasoning(self, 
                                 query: str,
                                 context: Optional[Dict[str, Any]],
                                 chain: ReasoningChain) -> ReasoningChain:
        """
        Inductive reasoning: Find patterns and generalize from specific examples
        """
        logger.info("📊 Performing inductive reasoning")
        
        # Collect relevant examples
        examples = await self._find_relevant_examples(query, context)
        
        if len(examples) < 3:
            chain.final_conclusion = "Insufficient examples for inductive reasoning"
            chain.overall_confidence = 0.1
            return chain
        
        # Find patterns in examples
        patterns = await self._identify_patterns(examples)
        
        step = ReasoningStep(
            id=str(uuid.uuid4()),
            step_number=1,
            reasoning_type=ReasoningType.INDUCTIVE,
            input_facts=[ex["id"] for ex in examples],
            applied_rules=[],
            output_facts=[],
            confidence=max(p["confidence"] for p in patterns) if patterns else 0.0,
            explanation=f"Identified {len(patterns)} patterns from {len(examples)} examples"
        )
        
        chain.steps.append(step)
        
        # Generalize from patterns
        if patterns:
            best_pattern = max(patterns, key=lambda p: p["confidence"])
            generalization = await self._create_generalization(best_pattern, examples)
            
            chain.final_conclusion = generalization["statement"]
            chain.overall_confidence = generalization["confidence"]
        else:
            chain.final_conclusion = "No reliable patterns found"
            chain.overall_confidence = 0.2
        
        return chain
    
    # ==================== Abductive Reasoning ====================
    
    async def _abductive_reasoning(self, 
                                 query: str,
                                 context: Optional[Dict[str, Any]],
                                 chain: ReasoningChain) -> ReasoningChain:
        """
        Abductive reasoning: Find the best explanation for observations
        """
        logger.info("🕵️ Performing abductive reasoning")
        
        # Extract observations from query
        observations = await self._extract_observations(query, context)
        
        # Generate possible explanations
        explanations = await self._generate_explanations(observations)
        
        # Evaluate explanations
        evaluated_explanations = []
        for explanation in explanations:
            evaluation = await self._evaluate_explanation(explanation, observations)
            evaluated_explanations.append((explanation, evaluation))
        
        # Select best explanation
        if evaluated_explanations:
            best_explanation, best_evaluation = max(
                evaluated_explanations, 
                key=lambda x: x[1]["score"]
            )
            
            step = ReasoningStep(
                id=str(uuid.uuid4()),
                step_number=1,
                reasoning_type=ReasoningType.ABDUCTIVE,
                input_facts=[obs["id"] for obs in observations],
                applied_rules=[],
                output_facts=[best_explanation["id"]],
                confidence=best_evaluation["confidence"],
                explanation=f"Selected best explanation from {len(explanations)} candidates"
            )
            
            chain.steps.append(step)
            chain.final_conclusion = best_explanation["text"]
            chain.overall_confidence = best_evaluation["confidence"]
        else:
            chain.final_conclusion = "No plausible explanations found"
            chain.overall_confidence = 0.1
        
        return chain
    
    # ==================== Causal Reasoning ====================
    
    async def _causal_reasoning(self, 
                              query: str,
                              context: Optional[Dict[str, Any]],
                              chain: ReasoningChain) -> ReasoningChain:
        """
        Causal reasoning: Determine cause-effect relationships
        """
        logger.info("🔗 Performing causal reasoning")
        
        # Extract causal elements from query
        causal_elements = await self._extract_causal_elements(query)
        
        if not causal_elements:
            chain.final_conclusion = "No causal elements identified in query"
            chain.overall_confidence = 0.1
            return chain
        
        # Find relevant causal relations
        relevant_relations = await self._find_relevant_causal_relations(causal_elements)
        
        # Build causal chain
        causal_chain = await self._build_causal_chain(causal_elements, relevant_relations)
        
        step = ReasoningStep(
            id=str(uuid.uuid4()),
            step_number=1,
            reasoning_type=ReasoningType.CAUSAL,
            input_facts=[elem["id"] for elem in causal_elements],
            applied_rules=[rel.id for rel in relevant_relations],
            output_facts=[],
            confidence=self._calculate_causal_confidence(causal_chain),
            explanation=f"Built causal chain with {len(causal_chain)} steps"
        )
        
        chain.steps.append(step)
        
        # Generate conclusion
        if causal_chain:
            conclusion = await self._generate_causal_conclusion(causal_chain, query)
            chain.final_conclusion = conclusion["text"]
            chain.overall_confidence = conclusion["confidence"]
        else:
            chain.final_conclusion = "No causal relationship established"
            chain.overall_confidence = 0.2
        
        return chain
    
    # ==================== Strategic Reasoning ====================
    
    async def _strategic_reasoning(self, 
                                 query: str,
                                 context: Optional[Dict[str, Any]],
                                 chain: ReasoningChain) -> ReasoningChain:
        """
        Strategic reasoning: Multi-step planning to achieve goals
        """
        logger.info("🎯 Performing strategic reasoning")
        
        # Extract goal from query
        goal = await self._extract_goal(query, context)
        
        if not goal:
            chain.final_conclusion = "No clear goal identified for strategic planning"
            chain.overall_confidence = 0.1
            return chain
        
        # Analyze current state
        current_state = await self._analyze_current_state(context)
        
        # Generate strategic options
        strategic_options = await self._generate_strategic_options(goal, current_state)
        
        # Evaluate options
        evaluated_options = []
        for option in strategic_options:
            evaluation = await self._evaluate_strategic_option(option, goal, current_state)
            evaluated_options.append((option, evaluation))
        
        # Select best strategy
        if evaluated_options:
            best_strategy, best_evaluation = max(
                evaluated_options,
                key=lambda x: x[1]["score"]
            )
            
            # Create detailed strategic plan
            strategic_plan = await self._create_strategic_plan(best_strategy, goal)
            
            step = ReasoningStep(
                id=str(uuid.uuid4()),
                step_number=1,
                reasoning_type=ReasoningType.STRATEGIC,
                input_facts=[goal["id"]],
                applied_rules=[],
                output_facts=[strategic_plan["id"]],
                confidence=best_evaluation["confidence"],
                explanation=f"Created strategic plan with {len(strategic_plan['steps'])} steps"
            )
            
            chain.steps.append(step)
            chain.final_conclusion = strategic_plan["description"]
            chain.overall_confidence = best_evaluation["confidence"]
            
            # Store strategy
            self.strategies[strategic_plan["id"]] = strategic_plan
        else:
            chain.final_conclusion = "No viable strategic options found"
            chain.overall_confidence = 0.1
        
        return chain
    
    # ==================== Probabilistic Reasoning ====================
    
    async def _probabilistic_reasoning(self, 
                                     query: str,
                                     context: Optional[Dict[str, Any]],
                                     chain: ReasoningChain) -> ReasoningChain:
        """
        Probabilistic reasoning: Handle uncertainty and compute probabilities
        """
        logger.info("🎲 Performing probabilistic reasoning")
        
        # Extract probabilistic elements
        prob_elements = await self._extract_probabilistic_elements(query, context)
        
        # Build probabilistic model
        prob_model = await self._build_probabilistic_model(prob_elements)
        
        # Perform inference
        if self.config["uncertainty_handling"] == "monte_carlo":
            inference_result = await self._monte_carlo_inference(prob_model, query)
        else:
            inference_result = await self._exact_inference(prob_model, query)
        
        step = ReasoningStep(
            id=str(uuid.uuid4()),
            step_number=1,
            reasoning_type=ReasoningType.PROBABILISTIC,
            input_facts=[elem["id"] for elem in prob_elements],
            applied_rules=[],
            output_facts=[],
            confidence=inference_result["confidence"],
            explanation=f"Probabilistic inference with {inference_result['method']}"
        )
        
        chain.steps.append(step)
        chain.final_conclusion = inference_result["conclusion"]
        chain.overall_confidence = inference_result["confidence"]
        
        return chain
    
    # ==================== Pattern Recognition ====================
    
    async def _identify_patterns(self, examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify patterns in a set of examples"""
        patterns = []
        
        # Temporal patterns
        temporal_pattern = await self._find_temporal_patterns(examples)
        if temporal_pattern:
            patterns.append(temporal_pattern)
        
        # Structural patterns
        structural_patterns = await self._find_structural_patterns(examples)
        patterns.extend(structural_patterns)
        
        # Causal patterns
        causal_patterns = await self._find_causal_patterns(examples)
        patterns.extend(causal_patterns)
        
        # Filter by confidence threshold
        return [p for p in patterns if p["confidence"] >= self.config["pattern_recognition_threshold"]]
    
    async def _find_temporal_patterns(self, examples: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Find temporal patterns in examples"""
        if len(examples) < 3:
            return None
        
        # Sort by timestamp if available
        timestamped_examples = [ex for ex in examples if "timestamp" in ex]
        if len(timestamped_examples) < 3:
            return None
        
        timestamped_examples.sort(key=lambda x: x["timestamp"])
        
        # Look for trends
        values = [ex.get("value", 0) for ex in timestamped_examples]
        
        # Simple trend analysis
        if len(values) >= 3:
            # Calculate trend
            x = np.arange(len(values))
            coeffs = np.polyfit(x, values, 1)
            slope = coeffs[0]
            
            trend_strength = abs(slope) / (max(values) - min(values) + 1e-6)
            
            if trend_strength > 0.1:
                direction = "increasing" if slope > 0 else "decreasing"
                return {
                    "type": "temporal_trend",
                    "direction": direction,
                    "strength": trend_strength,
                    "confidence": min(0.9, trend_strength * 2)
                }
        
        return None
    
    # ==================== Knowledge Base Management ====================
    
    async def add_fact(self, content: str, confidence: float = 1.0, source: str = "user") -> str:
        """Add a fact to the knowledge base"""
        fact_id = str(uuid.uuid4())
        
        fact = Fact(
            id=fact_id,
            content=content,
            confidence=confidence,
            source=source
        )
        
        self.facts[fact_id] = fact
        logger.info(f"📝 Added fact: {content[:50]}...")
        
        return fact_id
    
    async def add_rule(self, 
                      name: str,
                      conditions: List[str],
                      conclusions: List[str],
                      confidence: float = 1.0) -> str:
        """Add a reasoning rule"""
        rule_id = str(uuid.uuid4())
        
        rule = Rule(
            id=rule_id,
            name=name,
            conditions=conditions,
            conclusions=conclusions,
            confidence=confidence,
            rule_type="user_defined"
        )
        
        self.rules[rule_id] = rule
        logger.info(f"📐 Added rule: {name}")
        
        return rule_id
    
    async def add_causal_relation(self, 
                                cause: str,
                                effect: str,
                                strength: float,
                                confidence: float = 1.0,
                                evidence: Optional[List[str]] = None) -> str:
        """Add a causal relationship"""
        relation_id = str(uuid.uuid4())
        
        relation = CausalRelation(
            cause=cause,
            effect=effect,
            strength=strength,
            confidence=confidence,
            evidence=evidence or []
        )
        
        self.causal_relations[relation_id] = relation
        logger.info(f"🔗 Added causal relation: {cause} → {effect}")
        
        return relation_id
    
    # ==================== Helper Methods ====================
    
    def _calculate_step_confidence(self, facts: List[Fact]) -> float:
        """Calculate confidence for a reasoning step"""
        if not facts:
            return 0.0
        
        confidences = [f.confidence for f in facts]
        return sum(confidences) / len(confidences)
    
    def _calculate_causal_confidence(self, causal_chain: List[Dict[str, Any]]) -> float:
        """Calculate confidence for a causal chain"""
        if not causal_chain:
            return 0.0
        
        # Multiply confidences (chain is only as strong as weakest link)
        confidence = 1.0
        for link in causal_chain:
            confidence *= link.get("confidence", 0.5)
        
        return confidence
    
    def _update_reasoning_stats(self, chain: ReasoningChain):
        """Update reasoning performance statistics"""
        self.reasoning_stats["total_queries"] += 1
        
        if chain.overall_confidence > self.config["min_confidence_threshold"]:
            self.reasoning_stats["successful_inferences"] += 1
        
        # Update average confidence
        total = self.reasoning_stats["total_queries"]
        current_avg = self.reasoning_stats["average_confidence"]
        new_avg = (current_avg * (total - 1) + chain.overall_confidence) / total
        self.reasoning_stats["average_confidence"] = new_avg
        
        # Record reasoning time
        if chain.end_time:
            reasoning_time = (chain.end_time - chain.start_time).total_seconds()
            self.reasoning_stats["reasoning_times"].append(reasoning_time)
    
    # Additional helper methods would be implemented here...
    # This includes methods for:
    # - Finding relevant facts and rules
    # - Applying rules and generating conclusions
    # - Building probabilistic models
    # - Monte Carlo inference
    # - Strategic planning helpers
    # - Pattern extraction algorithms
    # - And many more supporting functions
    
    async def get_reasoning_statistics(self) -> Dict[str, Any]:
        """Get reasoning engine performance statistics"""
        recent_times = list(self.reasoning_stats["reasoning_times"])[-100:]  # Last 100
        
        return {
            "total_queries": self.reasoning_stats["total_queries"],
            "successful_inferences": self.reasoning_stats["successful_inferences"],
            "success_rate": (self.reasoning_stats["successful_inferences"] / 
                           max(1, self.reasoning_stats["total_queries"])),
            "average_confidence": self.reasoning_stats["average_confidence"],
            "average_reasoning_time": np.mean(recent_times) if recent_times else 0.0,
            "knowledge_base_size": {
                "facts": len(self.facts),
                "rules": len(self.rules),
                "causal_relations": len(self.causal_relations)
            },
            "patterns_learned": len(self.pattern_library),
            "strategies_created": len(self.strategies)
        }


# ==================== Example Usage ====================

async def example_reasoning():
    """Example of using the reasoning engine"""
    
    engine = AdvancedReasoningEngine()
    
    # Add some facts
    await engine.add_fact("The system CPU usage is 85%", confidence=0.9)
    await engine.add_fact("High CPU usage causes slow response times", confidence=0.8)
    await engine.add_fact("The target response time is under 2 seconds", confidence=1.0)
    
    # Add a rule
    await engine.add_rule(
        name="Performance Optimization Rule",
        conditions=["CPU usage > 80%", "Response time > target"],
        conclusions=["System needs optimization"],
        confidence=0.9
    )
    
    # Perform reasoning
    result = await engine.reason(
        query="Why is the system responding slowly?",
        reasoning_type=ReasoningType.CAUSAL
    )
    
    print(f"Conclusion: {result.final_conclusion}")
    print(f"Confidence: {result.overall_confidence}")
    print(f"Steps: {len(result.steps)}")
    
    # Strategic reasoning
    strategic_result = await engine.reason(
        query="How can we improve system performance?",
        context={"current_cpu": 0.85, "target_response_time": 2.0},
        reasoning_type=ReasoningType.STRATEGIC
    )
    
    print(f"Strategic Plan: {strategic_result.final_conclusion}")
    
    # Get statistics
    stats = await engine.get_reasoning_statistics()
    print(f"Reasoning Stats: {stats}")


if __name__ == "__main__":
    asyncio.run(example_reasoning())