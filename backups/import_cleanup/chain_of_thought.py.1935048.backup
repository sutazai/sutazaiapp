"""
Advanced Chain-of-Thought Reasoning System
Implements multi-step reasoning with agent collaboration and verification
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime
import time

logger = logging.getLogger(__name__)

class ReasoningStage(Enum):
    ANALYSIS = "analysis"
    HYPOTHESIS = "hypothesis"
    VERIFICATION = "verification"
    SYNTHESIS = "synthesis"
    CONCLUSION = "conclusion"

@dataclass
class ReasoningStep:
    """Single step in chain of thought"""
    stage: ReasoningStage
    agent_id: str
    thought: str
    confidence: float
    evidence: List[str]
    timestamp: datetime
    parent_step: Optional[str] = None
    children_steps: List[str] = None

    def __post_init__(self):
        if self.children_steps is None:
            self.children_steps = []

@dataclass
class ReasoningChain:
    """Complete chain of reasoning for a problem"""
    problem_id: str
    problem_statement: str
    steps: List[ReasoningStep]
    final_answer: Optional[str] = None
    confidence_score: float = 0.0
    verification_count: int = 0

class AdvancedReasoningEngine:
    """Multi-agent reasoning engine with verification"""
    
    def __init__(self, agent_orchestrator, max_reasoning_time: int = 300):
        self.agent_orchestrator = agent_orchestrator
        self.max_reasoning_time = max_reasoning_time
        self.active_chains: Dict[str, ReasoningChain] = {}
        
    async def reason_about_problem(
        self, 
        problem: str, 
        domain: str = "general",
        min_agents: int = 3,
        require_consensus: bool = True
    ) -> ReasoningChain:
        """
        Implement multi-agent reasoning on complex problems
        Similar to o3's approach but with agent collaboration
        """
        problem_id = f"reasoning_{int(time.time())}"
        chain = ReasoningChain(
            problem_id=problem_id,
            problem_statement=problem,
            steps=[]
        )
        
        self.active_chains[problem_id] = chain
        
        try:
            # Stage 1: Multi-agent analysis
            analysis_steps = await self._multi_agent_analysis(problem, domain, min_agents)
            chain.steps.extend(analysis_steps)
            
            # Stage 2: Hypothesis generation
            hypothesis_steps = await self._generate_hypotheses(problem, analysis_steps)
            chain.steps.extend(hypothesis_steps)
            
            # Stage 3: Cross-verification
            verification_steps = await self._cross_verify_hypotheses(hypothesis_steps)
            chain.steps.extend(verification_steps)
            
            # Stage 4: Synthesis
            synthesis_step = await self._synthesize_reasoning(chain.steps)
            chain.steps.append(synthesis_step)
            
            # Stage 5: Final answer with confidence
            chain.final_answer = synthesis_step.thought
            chain.confidence_score = self._calculate_chain_confidence(chain.steps)
            
            return chain
            
        except Exception as e:
            logger.error(f"Reasoning failed for problem {problem_id}: {e}")
            raise 
            
    async def _multi_agent_analysis(
        self, 
        problem: str, 
        domain: str, 
        min_agents: int
    ) -> List[ReasoningStep]:
        """Get analysis from multiple specialized agents"""
        
        # Select agents based on domain
        agent_types = self._select_agents_for_domain(domain)
        
        analysis_tasks = []
        for agent_type in agent_types[:min_agents]:
            task = {
                "type": "analyze",
                "content": problem,
                "instruction": f"""
                Analyze this problem step by step:
                1. What is the core question being asked?
                2. What information do we have?
                3. What information might we need?
                4. What approaches tested implementation or proof of concept?
                5. What are potential challenges?
                
                Be specific and show your reasoning.
                """
            }
            analysis_tasks.append(self._execute_reasoning_task(agent_type, task))
            
        # Execute in parallel
        results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
        
        steps = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"Agent {agent_types[i]} failed analysis: {result}")
                continue
                
            step = ReasoningStep(
                stage=ReasoningStage.ANALYSIS,
                agent_id=agent_types[i],
                thought=result.get("analysis", ""),
                confidence=result.get("confidence", 0.5),
                evidence=result.get("evidence", []),
                timestamp=datetime.now()
            )
            steps.append(step)
            
        return steps
        
    async def _generate_hypotheses(
        self, 
        problem: str, 
        analysis_steps: List[ReasoningStep]
    ) -> List[ReasoningStep]:
        """Generate hypotheses based on analysis"""
        
        # Combine all analysis insights
        combined_analysis = "\n".join([step.thought for step in analysis_steps])
        
        hypothesis_task = {
            "type": "hypothesize",
            "content": problem,
            "context": combined_analysis,
            "instruction": """
            Based on the analysis provided, generate 3-5 specific hypotheses
            for solving this problem. For each hypothesis:
            1. State the hypothesis clearly
            2. Explain the reasoning behind it
            3. Identify what evidence would support/refute it
            4. Estimate probability of success
            """
        }
        
        # Use the best reasoning agent
        result = await self._execute_reasoning_task("tinyllama", hypothesis_task)
        
        step = ReasoningStep(
            stage=ReasoningStage.HYPOTHESIS,
            agent_id="tinyllama",
            thought=result.get("hypotheses", ""),
            confidence=result.get("confidence", 0.7),
            evidence=result.get("evidence", []),
            timestamp=datetime.now(),
            parent_step=analysis_steps[0].agent_id if analysis_steps else None
        )
        
        return [step]
        
    async def _cross_verify_hypotheses(
        self, 
        hypothesis_steps: List[ReasoningStep]
    ) -> List[ReasoningStep]:
        """Cross-verify hypotheses using different agents"""
        
        verification_steps = []
        
        for hypothesis_step in hypothesis_steps:
            # Get multiple agents to verify each hypothesis
            verification_tasks = []
            
            verification_agents = ["tinyllama"]
            
            for agent_type in verification_agents:
                task = {
                    "type": "verify",
                    "content": hypothesis_step.thought,
                    "instruction": """
                    Critically evaluate these hypotheses:
                    1. Which parts seem most/least plausible?
                    2. What additional evidence would be needed?
                    3. Are there logical flaws or gaps?
                    4. Can you improve or refine any hypothesis?
                    5. Rank the hypotheses by likelihood of success
                    """
                }
                verification_tasks.append(
                    self._execute_reasoning_task(agent_type, task)
                )
                
            results = await asyncio.gather(*verification_tasks, return_exceptions=True)
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    continue
                    
                step = ReasoningStep(
                    stage=ReasoningStage.VERIFICATION,
                    agent_id=verification_agents[i],
                    thought=result.get("verification", ""),
                    confidence=result.get("confidence", 0.6),
                    evidence=result.get("evidence", []),
                    timestamp=datetime.now(),
                    parent_step=hypothesis_step.agent_id
                )
                verification_steps.append(step)
                
        return verification_steps
        
    async def _synthesize_reasoning(
        self, 
        all_steps: List[ReasoningStep]
    ) -> ReasoningStep:
        """Synthesize all reasoning into final answer"""
        
        # Organize steps by stage
        analysis_thoughts = [s.thought for s in all_steps if s.stage == ReasoningStage.ANALYSIS]
        hypothesis_thoughts = [s.thought for s in all_steps if s.stage == ReasoningStage.HYPOTHESIS]
        verification_thoughts = [s.thought for s in all_steps if s.stage == ReasoningStage.VERIFICATION]
        
        synthesis_task = {
            "type": "synthesize",
            "analysis": "\n".join(analysis_thoughts),
            "hypotheses": "\n".join(hypothesis_thoughts),
            "verifications": "\n".join(verification_thoughts),
            "instruction": """
            Synthesize all the reasoning above into a comprehensive answer:
            1. What is the best solution based on all evidence?
            2. Why is this the strongest approach?
            3. What are the key insights from the multi-agent analysis?
            4. What confidence level do you assign to this solution?
            5. What would be the next steps to implement/validate?
            
            Provide a clear, actionable final answer.
            """
        }
        
        result = await self._execute_reasoning_task("tinyllama", synthesis_task)
        
        return ReasoningStep(
            stage=ReasoningStage.SYNTHESIS,
            agent_id="tinyllama",
            thought=result.get("synthesis", ""),
            confidence=result.get("confidence", 0.8),
            evidence=result.get("evidence", []),
            timestamp=datetime.now()
        )
        
    def _select_agents_for_domain(self, domain: str) -> List[str]:
        """Select best agents for specific domain"""
        domain_mapping = {
            "code": ["tinyllama"],
            "math": ["tinyllama"],
            "science": ["tinyllama"],
            "general": ["tinyllama"],
            "analysis": ["tinyllama"]
        }
        
        return domain_mapping.get(domain, domain_mapping["general"])
        
    async def _execute_reasoning_task(self, agent_type: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute reasoning task on specific agent"""
        try:
            # This would interface with your agent orchestrator
            result = await self.agent_orchestrator.execute_on_agent(agent_type, task)
            return result
        except Exception as e:
            logger.error(f"Reasoning task failed on {agent_type}: {e}")
            return {"error": str(e), "confidence": 0.0}
            
    def _calculate_chain_confidence(self, steps: List[ReasoningStep]) -> float:
        """Calculate overall confidence in reasoning chain"""
        if not steps:
            return 0.0
            
        # Weight different stages
        stage_weights = {
            ReasoningStage.ANALYSIS: 0.2,
            ReasoningStage.HYPOTHESIS: 0.3,
            ReasoningStage.VERIFICATION: 0.3,
            ReasoningStage.SYNTHESIS: 0.2
        }
        
        weighted_confidence = 0.0
        total_weight = 0.0
        
        for step in steps:
            weight = stage_weights.get(step.stage, 0.1)
            weighted_confidence += step.confidence * weight
            total_weight += weight
            
        return weighted_confidence / total_weight if total_weight > 0 else 0.0
        
    async def get_reasoning_explanation(self, problem_id: str) -> Dict[str, Any]:
        """Get detailed explanation of reasoning process"""
        if problem_id not in self.active_chains:
            return {"error": "Reasoning chain not found"}
            
        chain = self.active_chains[problem_id]
        
        return {
            "problem": chain.problem_statement,
            "final_answer": chain.final_answer,
            "confidence": chain.confidence_score,
            "reasoning_steps": [
                {
                    "stage": step.stage.value,
                    "agent": step.agent_id,
                    "thought": step.thought,
                    "confidence": step.confidence,
                    "timestamp": step.timestamp.isoformat()
                }
                for step in chain.steps
            ],
            "summary": self._generate_reasoning_summary(chain)
        }
        
    def _generate_reasoning_summary(self, chain: ReasoningChain) -> str:
        """Generate human-readable summary of reasoning process"""
        summary = f"Multi-agent reasoning for: {chain.problem_statement}\n\n"
        
        # Group by stage
        stages = {}
        for step in chain.steps:
            if step.stage not in stages:
                stages[step.stage] = []
            stages[step.stage].append(step)
            
        for stage, steps in stages.items():
            summary += f"{stage.value.upper()}:\n"
            for step in steps:
                summary += f"  - {step.agent_id}: {step.thought[:100]}...\n"
            summary += "\n"
            
        summary += f"FINAL ANSWER: {chain.final_answer}\n"
        summary += f"CONFIDENCE: {chain.confidence_score:.2f}"
        
        return summary 