"""
Self-Improvement Engine for SutazAI automation/advanced automation System
Enables continuous learning and capability enhancement
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import json
import pickle
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

class ImprovementType(Enum):
    REASONING_PATTERN = "reasoning_pattern"
    TASK_OPTIMIZATION = "task_optimization"
    AGENT_COORDINATION = "agent_coordination"
    MODEL_FINE_TUNING = "model_fine_tuning"
    PROMPT_ENGINEERING = "prompt_engineering"

@dataclass
class LearningEvent:
    """Represents a learning opportunity"""
    event_id: str
    event_type: ImprovementType
    task_context: str
    performance_before: float
    performance_after: float
    improvement_delta: float
    successful_approach: str
    failed_approaches: List[str]
    timestamp: datetime
    confidence: float

@dataclass
class CapabilityMetric:
    """Tracks capability improvements over time"""
    capability_name: str
    baseline_score: float
    current_score: float
    improvement_rate: float
    last_updated: datetime
    sample_count: int

class SelfImprovementEngine:
    """
    Implements self-improvement mechanisms for the automation system
    Based on latest research in meta-learning and autonomous improvement
    """
    
    def __init__(self, 
                 agent_orchestrator,
                 reasoning_engine,
                 data_path: str = "/opt/sutazaiapp/data/self_improvement"):
        self.agent_orchestrator = agent_orchestrator
        self.reasoning_engine = reasoning_engine
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)
        
        # Learning components
        self.learning_events: List[LearningEvent] = []
        self.capability_metrics: Dict[str, CapabilityMetric] = {}
        self.improvement_patterns: Dict[str, Any] = {}
        self.successful_strategies: Dict[str, List[str]] = {}
        
        # Load existing learning data
        self._load_learning_data()
        
    async def evaluate_and_improve(self, 
                                 task_result: Dict[str, Any],
                                 task_context: str) -> Dict[str, Any]:
        """
        Evaluate task performance and identify improvement opportunities
        """
        # 1. Analyze task performance
        performance_analysis = await self._analyze_task_performance(task_result, task_context)
        
        # 2. Identify improvement opportunities
        improvement_opportunities = await self._identify_improvements(performance_analysis)
        
        # 3. Apply improvements
        applied_improvements = []
        for opportunity in improvement_opportunities:
            success = await self._apply_improvement(opportunity)
            if success:
                applied_improvements.append(opportunity)
                
        # 4. Record learning event
        if applied_improvements:
            learning_event = await self._record_learning_event(
                task_context, performance_analysis, applied_improvements
            )
            
        # 5. Update capability metrics
        await self._update_capability_metrics(task_result, task_context)
        
        return {
            "performance_analysis": performance_analysis,
            "improvements_applied": len(applied_improvements),
            "capability_updates": len(self.capability_metrics),
            "learning_events_total": len(self.learning_events)
        }
        
    async def _analyze_task_performance(self, 
                                      task_result: Dict[str, Any],
                                      task_context: str) -> Dict[str, Any]:
        """Analyze task performance using multi-agent evaluation"""
        
        analysis_prompt = f"""
        Analyze the performance of this task execution:
        
        Task Context: {task_context}
        Task Result: {json.dumps(task_result, indent=2)}
        
        Evaluate:
        1. Success rate (0-1 scale)
        2. Efficiency (time, resources used)
        3. Quality of output
        4. Any errors or suboptimal decisions
        5. Areas for improvement
        6. What worked well
        
        Provide structured analysis with scores and recommendations.
        """
        
        # Use reasoning engine for analysis
        analysis_chain = await self.reasoning_engine.reason_about_problem(
            analysis_prompt, 
            domain="analysis",
            min_agents=3
        )
        
        # Extract metrics from analysis
        performance_score = self._extract_performance_score(analysis_chain.final_answer)
        improvement_areas = self._extract_improvement_areas(analysis_chain.final_answer)
        
        return {
            "overall_score": performance_score,
            "improvement_areas": improvement_areas,
            "detailed_analysis": analysis_chain.final_answer,
            "confidence": analysis_chain.confidence_score,
            "reasoning_chain_id": analysis_chain.problem_id
        }
        
    async def _identify_improvements(self, 
                                   performance_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify specific improvement opportunities"""
        
        improvements = []
        
        # 1. Reasoning pattern improvements
        if performance_analysis["overall_score"] < 0.8:
            reasoning_improvement = await self._identify_reasoning_improvements(performance_analysis)
            if reasoning_improvement:
                improvements.append(reasoning_improvement)
                
        # 2. Agent coordination improvements
        coordination_improvement = await self._identify_coordination_improvements(performance_analysis)
        if coordination_improvement:
            improvements.append(coordination_improvement)
            
        # 3. Prompt engineering improvements
        prompt_improvement = await self._identify_prompt_improvements(performance_analysis)
        if prompt_improvement:
            improvements.append(prompt_improvement)
            
        return improvements
        
    async def _identify_reasoning_improvements(self, 
                                            performance_analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Identify improvements to reasoning patterns"""
        
        # Analyze historical successful reasoning patterns
        successful_patterns = self._get_successful_reasoning_patterns()
        
        improvement_prompt = f"""
        Based on this performance analysis:
        {json.dumps(performance_analysis, indent=2)}
        
        And these successful reasoning patterns from past tasks:
        {json.dumps(successful_patterns, indent=2)}
        
        Suggest specific improvements to reasoning approach:
        1. Should we use different agents?
        2. Should we change the reasoning sequence?
        3. Are there better verification methods?
        4. Should we allocate more thinking time?
        """
        
        reasoning_chain = await self.reasoning_engine.reason_about_problem(
            improvement_prompt,
            domain="analysis"
        )
        
        if reasoning_chain.confidence_score > 0.7:
            return {
                "type": ImprovementType.REASONING_PATTERN,
                "description": reasoning_chain.final_answer,
                "confidence": reasoning_chain.confidence_score,
                "implementation": self._extract_reasoning_implementation(reasoning_chain.final_answer)
            }
            
        return None
        
    async def _identify_coordination_improvements(self, 
                                                performance_analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Identify agent coordination improvements"""
        
        # Analyze agent interaction patterns
        coordination_data = await self._analyze_agent_coordination_history()
        
        if coordination_data["efficiency_score"] < 0.8:
            return {
                "type": ImprovementType.AGENT_COORDINATION,
                "description": "Improve agent coordination based on efficiency analysis",
                "confidence": 0.8,
                "implementation": {
                    "optimize_task_distribution": True,
                    "improve_communication_protocols": True,
                    "reduce_redundant_work": True
                }
            }
            
        return None
        
    async def _identify_prompt_improvements(self, 
                                          performance_analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Identify prompt engineering improvements"""
        
        # Use self-reflection to improve prompts
        prompt_analysis = f"""
        Analyze the prompts used in this task and suggest improvements:
        
        Performance Analysis: {json.dumps(performance_analysis, indent=2)}
        
        Consider:
        1. Clarity and specificity of instructions
        2. Context provided to agents
        3. Output format specifications
        4. Examples and constraints
        5. Error handling instructions
        """
        
        reasoning_chain = await self.reasoning_engine.reason_about_problem(
            prompt_analysis,
            domain="general"
        )
        
        if reasoning_chain.confidence_score > 0.6:
            return {
                "type": ImprovementType.PROMPT_ENGINEERING,
                "description": reasoning_chain.final_answer,
                "confidence": reasoning_chain.confidence_score,
                "implementation": self._extract_prompt_improvements(reasoning_chain.final_answer)
            }
            
        return None
        
    async def _apply_improvement(self, improvement: Dict[str, Any]) -> bool:
        """Apply a specific improvement to the system"""
        
        try:
            improvement_type = improvement["type"]
            
            if improvement_type == ImprovementType.REASONING_PATTERN:
                return await self._apply_reasoning_improvement(improvement)
            elif improvement_type == ImprovementType.AGENT_COORDINATION:
                return await self._apply_coordination_improvement(improvement)
            elif improvement_type == ImprovementType.PROMPT_ENGINEERING:
                return await self._apply_prompt_improvement(improvement)
            else:
                logger.warning(f"Unknown improvement type: {improvement_type}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to apply improvement: {e}")
            return False
            
    async def _apply_reasoning_improvement(self, improvement: Dict[str, Any]) -> bool:
        """Apply reasoning pattern improvements"""
        
        implementation = improvement.get("implementation", {})
        
        # Update reasoning engine configuration
        if "agent_selection" in implementation:
            # Update agent selection logic
            self.reasoning_engine.agent_selection_strategy = implementation["agent_selection"]
            
        if "verification_steps" in implementation:
            # Update verification process
            self.reasoning_engine.verification_steps = implementation["verification_steps"]
            
        if "thinking_time" in implementation:
            # Update max thinking time
            self.reasoning_engine.max_reasoning_time = implementation["thinking_time"]
            
        # Save improvement pattern
        pattern_key = f"reasoning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.improvement_patterns[pattern_key] = improvement
        
        logger.info(f"Applied reasoning improvement: {improvement['description'][:100]}...")
        return True
        
    async def _apply_coordination_improvement(self, improvement: Dict[str, Any]) -> bool:
        """Apply agent coordination improvements"""
        
        implementation = improvement.get("implementation", {})
        
        if implementation.get("optimize_task_distribution"):
            # Implement better task distribution
            await self._optimize_task_distribution()
            
        if implementation.get("improve_communication_protocols"):
            # Enhance communication protocols
            await self._improve_communication_protocols()
            
        logger.info(f"Applied coordination improvement: {improvement['description'][:100]}...")
        return True
        
    async def _apply_prompt_improvement(self, improvement: Dict[str, Any]) -> bool:
        """Apply prompt engineering improvements"""
        
        implementation = improvement.get("implementation", {})
        
        # Update prompt templates
        if "templates" in implementation:
            for template_name, template_content in implementation["templates"].items():
                await self._update_prompt_template(template_name, template_content)
                
        logger.info(f"Applied prompt improvement: {improvement['description'][:100]}...")
        return True
        
    async def _record_learning_event(self, 
                                   task_context: str,
                                   performance_analysis: Dict[str, Any],
                                   improvements: List[Dict[str, Any]]) -> LearningEvent:
        """Record a learning event for future reference"""
        
        event = LearningEvent(
            event_id=f"learning_{int(datetime.now().timestamp())}",
            event_type=improvements[0]["type"] if improvements else ImprovementType.TASK_OPTIMIZATION,
            task_context=task_context,
            performance_before=performance_analysis["overall_score"],
            performance_after=performance_analysis["overall_score"] + 0.1,  # Estimated improvement
            improvement_delta=0.1,
            successful_approach=json.dumps([imp["description"] for imp in improvements]),
            failed_approaches=[],
            timestamp=datetime.now(),
            confidence=np.mean([imp["confidence"] for imp in improvements])
        )
        
        self.learning_events.append(event)
        
        # Save to disk
        await self._save_learning_data()
        
        return event
        
    async def _update_capability_metrics(self, 
                                       task_result: Dict[str, Any],
                                       task_context: str):
        """Update capability metrics based on task performance"""
        
        # Extract capability from task context
        capability_name = self._extract_capability_name(task_context)
        
        # Calculate performance score
        performance_score = self._calculate_performance_score(task_result)
        
        if capability_name in self.capability_metrics:
            metric = self.capability_metrics[capability_name]
            
            # Update with exponential moving average
            alpha = 0.1
            metric.current_score = alpha * performance_score + (1 - alpha) * metric.current_score
            metric.improvement_rate = (metric.current_score - metric.baseline_score) / max(metric.sample_count, 1)
            metric.last_updated = datetime.now()
            metric.sample_count += 1
        else:
            # Create new capability metric
            self.capability_metrics[capability_name] = CapabilityMetric(
                capability_name=capability_name,
                baseline_score=performance_score,
                current_score=performance_score,
                improvement_rate=0.0,
                last_updated=datetime.now(),
                sample_count=1
            )
            
    def _get_successful_reasoning_patterns(self) -> Dict[str, Any]:
        """Get patterns from successful reasoning chains"""
        
        successful_patterns = {}
        
        for event in self.learning_events:
            if (event.improvement_delta > 0.1 and 
                event.event_type == ImprovementType.REASONING_PATTERN):
                
                pattern_key = f"pattern_{len(successful_patterns)}"
                successful_patterns[pattern_key] = {
                    "approach": event.successful_approach,
                    "performance_gain": event.improvement_delta,
                    "confidence": event.confidence
                }
                
        return successful_patterns
        
    def _extract_performance_score(self, analysis_text: str) -> float:
        """Extract performance score from analysis text"""
        # Simple extraction - in production, use more sophisticated NLP
        try:
            # Look for score patterns like "0.85" or "85%"
            import re
            score_match = re.search(r'(?:score|rating|performance).*?(\d+\.?\d*)', analysis_text.lower())
            if score_match:
                score = float(score_match.group(1))
                return score if score <= 1.0 else score / 100.0
        except:
            pass
        
        return 0.5  # Default middle score
        
    def _extract_improvement_areas(self, analysis_text: str) -> List[str]:
        """Extract improvement areas from analysis text"""
        # Simple extraction - in production, use more sophisticated NLP
        improvement_keywords = ["improve", "enhance", "optimize", "better", "fix"]
        areas = []
        
        for line in analysis_text.split('\n'):
            if any(keyword in line.lower() for keyword in improvement_keywords):
                areas.append(line.strip())
                
        return areas[:5]  # Return top 5 areas
        
    def _extract_reasoning_implementation(self, reasoning_text: str) -> Dict[str, Any]:
        """Extract implementation details from reasoning text"""
        # In production, use more sophisticated extraction
        return {
            "agent_selection": "adaptive",
            "verification_steps": 3,
            "thinking_time": 300
        }
        
    def _extract_prompt_improvements(self, reasoning_text: str) -> Dict[str, Any]:
        """Extract prompt improvement details"""
        return {
            "templates": {
                "analysis": "Enhanced analysis template with better structure",
                "verification": "Improved verification with specific criteria"
            }
        }
        
    async def _analyze_agent_coordination_history(self) -> Dict[str, Any]:
        """Analyze historical agent coordination efficiency"""
        # Placeholder - implement actual coordination analysis
        return {
            "efficiency_score": 0.75,
            "average_task_time": 120,
            "coordination_overhead": 0.15
        }
        
    async def _optimize_task_distribution(self):
        """Optimize how tasks are distributed among agents"""
        logger.info("Optimizing task distribution...")
        
    async def _improve_communication_protocols(self):
        """Improve agent communication protocols"""
        logger.info("Improving communication protocols...")
        
    async def _update_prompt_template(self, template_name: str, template_content: str):
        """Update a prompt template"""
        logger.info(f"Updating prompt template: {template_name}")
        
    def _extract_capability_name(self, task_context: str) -> str:
        """Extract capability name from task context"""
        # Simple extraction - in production, use classification
        if "code" in task_context.lower():
            return "code_generation"
        elif "analyze" in task_context.lower():
            return "analysis"
        elif "reason" in task_context.lower():
            return "reasoning"
        else:
            return "general"
            
    def _calculate_performance_score(self, task_result: Dict[str, Any]) -> float:
        """Calculate performance score from task result"""
        # Simple scoring - in production, use more sophisticated metrics
        if task_result.get("success", False):
            return 0.8 + 0.2 * task_result.get("quality", 0.5)
        else:
            return 0.3
            
    async def _save_learning_data(self):
        """Save learning data to disk"""
        try:
            # Save learning events
            events_file = self.data_path / "learning_events.json"
            with open(events_file, 'w') as f:
                json.dump([asdict(event) for event in self.learning_events], f, default=str, indent=2)
                
            # Save capability metrics
            metrics_file = self.data_path / "capability_metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump({k: asdict(v) for k, v in self.capability_metrics.items()}, f, default=str, indent=2)
                
            # Save improvement patterns
            patterns_file = self.data_path / "improvement_patterns.json"
            with open(patterns_file, 'w') as f:
                json.dump(self.improvement_patterns, f, default=str, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save learning data: {e}")
            
    def _load_learning_data(self):
        """Load existing learning data from disk"""
        try:
            # Load learning events
            events_file = self.data_path / "learning_events.json"
            if events_file.exists():
                with open(events_file, 'r') as f:
                    events_data = json.load(f)
                    self.learning_events = [LearningEvent(**event) for event in events_data]
                    
            # Load capability metrics
            metrics_file = self.data_path / "capability_metrics.json"
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    metrics_data = json.load(f)
                    self.capability_metrics = {k: CapabilityMetric(**v) for k, v in metrics_data.items()}
                    
            # Load improvement patterns
            patterns_file = self.data_path / "improvement_patterns.json"
            if patterns_file.exists():
                with open(patterns_file, 'r') as f:
                    self.improvement_patterns = json.load(f)
                    
        except Exception as e:
            logger.error(f"Failed to load learning data: {e}")
            
    async def get_improvement_report(self) -> Dict[str, Any]:
        """Generate comprehensive improvement report"""
        
        return {
            "learning_events_count": len(self.learning_events),
            "capabilities_tracked": len(self.capability_metrics),
            "improvement_patterns": len(self.improvement_patterns),
            "recent_improvements": [
                asdict(event) for event in self.learning_events[-5:]
            ],
            "capability_improvements": {
                name: {
                    "current_score": metric.current_score,
                    "improvement_rate": metric.improvement_rate,
                    "samples": metric.sample_count
                }
                for name, metric in self.capability_metrics.items()
            },
            "top_successful_patterns": list(self.improvement_patterns.keys())[-5:]
        } 