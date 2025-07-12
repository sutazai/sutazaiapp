import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio
import json

from .base_agent import BaseAgent
from models import model_manager
from memory import vector_memory

logger = logging.getLogger(__name__)

class BigAGIAgent(BaseAgent):
    """BigAGI - Advanced AI system with multiple intelligence domains."""
    
    def __init__(self, agent_id: str = "big_agi_agent"):
        super().__init__(agent_id, "big_agi")
        self.capabilities = [
            "general_intelligence",
            "domain_expertise",
            "creative_thinking",
            "logical_reasoning",
            "memory_management",
            "learning_adaptation",
            "multi_modal_processing",
            "strategic_planning",
            "problem_decomposition",
            "solution_synthesis"
        ]
        self.intelligence_domains = {
            "linguistic": {"weight": 0.9, "active": True},
            "logical_mathematical": {"weight": 0.85, "active": True},
            "spatial": {"weight": 0.7, "active": True},
            "musical": {"weight": 0.6, "active": False},
            "bodily_kinesthetic": {"weight": 0.5, "active": False},
            "interpersonal": {"weight": 0.8, "active": True},
            "intrapersonal": {"weight": 0.75, "active": True},
            "naturalistic": {"weight": 0.65, "active": True}
        }
        self.knowledge_base = {}
        self.learning_history = []
    
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute BigAGI task with advanced intelligence processing."""
        try:
            task_type = task.get("type", "")
            
            if task_type == "general_intelligence":
                return await self._general_intelligence_task(task)
            elif task_type == "domain_expertise":
                return await self._domain_expertise_task(task)
            elif task_type == "creative_thinking":
                return await self._creative_thinking_task(task)
            elif task_type == "strategic_planning":
                return await self._strategic_planning_task(task)
            elif task_type == "problem_decomposition":
                return await self._problem_decomposition_task(task)
            elif task_type == "learning":
                return await self._learning_task(task)
            else:
                return await self._adaptive_intelligence_task(task)
                
        except Exception as e:
            logger.error(f"Error executing BigAGI task: {e}")
            return {
                "success": False,
                "error": str(e),
                "agent_id": self.agent_id,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def _general_intelligence_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Apply general intelligence to complex problems."""
        problem = task.get("problem", "")
        context = task.get("context", {})
        intelligence_level = task.get("intelligence_level", "high")
        
        if not problem:
            return {"success": False, "error": "No problem specified"}
        
        # Analyze problem complexity
        complexity_analysis = await self._analyze_problem_complexity(problem)
        
        # Select appropriate intelligence domains
        relevant_domains = await self._select_intelligence_domains(problem, complexity_analysis)
        
        # Apply multi-domain intelligence
        domain_responses = {}
        for domain in relevant_domains:
            domain_response = await self._apply_domain_intelligence(domain, problem, context)
            domain_responses[domain] = domain_response
        
        # Synthesize intelligent response
        synthesis = await self._synthesize_intelligence(problem, domain_responses, complexity_analysis)
        
        # Learn from this interaction
        await self._record_learning_experience({
            "problem": problem,
            "complexity": complexity_analysis,
            "domains_used": relevant_domains,
            "solution_quality": synthesis.get("quality_score", 0.7)
        })
        
        return {
            "success": True,
            "problem": problem,
            "complexity_analysis": complexity_analysis,
            "intelligence_domains_used": relevant_domains,
            "domain_responses": domain_responses,
            "synthesized_solution": synthesis,
            "intelligence_level": intelligence_level,
            "capabilities_used": ["general_intelligence", "multi_modal_processing"],
            "agent_id": self.agent_id,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _domain_expertise_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Apply specialized domain expertise."""
        domain = task.get("domain", "general")
        query = task.get("query", "")
        depth = task.get("depth", "intermediate")
        
        if not query:
            return {"success": False, "error": "No query specified"}
        
        # Build domain-specific prompt
        expertise_prompt = f"""
        As a domain expert in {domain}, provide {depth}-level analysis for:
        
        Query: {query}
        
        Consider:
        1. Domain-specific principles and theories
        2. Best practices and methodologies
        3. Current trends and developments
        4. Potential challenges and solutions
        5. Real-world applications and examples
        
        Provide comprehensive expertise-based response.
        """
        
        expert_response = await model_manager.general_ai_response(expertise_prompt)
        
        # Enhance with knowledge base
        if domain in self.knowledge_base:
            domain_knowledge = self.knowledge_base[domain]
            enhanced_response = await self._enhance_with_knowledge(expert_response, domain_knowledge)
        else:
            enhanced_response = expert_response
        
        # Store expertise for future use
        if domain not in self.knowledge_base:
            self.knowledge_base[domain] = []
        
        self.knowledge_base[domain].append({
            "query": query,
            "response": enhanced_response,
            "depth": depth,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return {
            "success": True,
            "domain": domain,
            "query": query,
            "depth": depth,
            "expert_response": enhanced_response,
            "knowledge_enhanced": domain in self.knowledge_base,
            "capabilities_used": ["domain_expertise", "memory_management"],
            "agent_id": self.agent_id,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _creative_thinking_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Apply creative thinking and innovation."""
        challenge = task.get("challenge", "")
        creativity_type = task.get("type", "divergent")  # convergent, divergent, lateral
        constraints = task.get("constraints", [])
        
        if not challenge:
            return {"success": False, "error": "No challenge specified"}
        
        creative_techniques = {
            "divergent": [
                "brainstorming", "mind_mapping", "random_word_association",
                "analogical_thinking", "reverse_thinking"
            ],
            "convergent": [
                "critical_analysis", "evaluation_matrix", "pros_cons_analysis",
                "feasibility_assessment", "optimization"
            ],
            "lateral": [
                "provocative_questions", "assumption_challenging", "metaphorical_thinking",
                "perspective_shifting", "boundary_relaxation"
            ]
        }
        
        techniques = creative_techniques.get(creativity_type, creative_techniques["divergent"])
        creative_solutions = []
        
        for technique in techniques:
            technique_prompt = f"""
            Apply {technique} to this creative challenge:
            
            Challenge: {challenge}
            Creativity type: {creativity_type}
            Constraints: {', '.join(constraints) if constraints else 'None'}
            
            Using {technique}, generate innovative solutions or ideas.
            Be creative, unconventional, and think outside the box.
            """
            
            technique_response = await model_manager.general_ai_response(technique_prompt)
            
            creative_solutions.append({
                "technique": technique,
                "ideas": technique_response,
                "creativity_score": await self._assess_creativity(technique_response)
            })
        
        # Synthesize best creative solutions
        best_solutions = sorted(creative_solutions, 
                               key=lambda x: x["creativity_score"], 
                               reverse=True)[:3]
        
        synthesis_prompt = f"""
        Challenge: {challenge}
        
        Top creative solutions:
        {chr(10).join([f"{sol['technique']}: {sol['ideas'][:150]}..." for sol in best_solutions])}
        
        Synthesize these into the most innovative, practical, and creative final solution.
        """
        
        final_creative_solution = await model_manager.general_ai_response(synthesis_prompt)
        
        return {
            "success": True,
            "challenge": challenge,
            "creativity_type": creativity_type,
            "constraints": constraints,
            "techniques_used": techniques,
            "creative_solutions": creative_solutions,
            "best_solutions": best_solutions,
            "final_solution": final_creative_solution,
            "capabilities_used": ["creative_thinking", "solution_synthesis"],
            "agent_id": self.agent_id,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _strategic_planning_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Develop strategic plans and long-term thinking."""
        objective = task.get("objective", "")
        timeframe = task.get("timeframe", "medium-term")
        resources = task.get("resources", {})
        constraints = task.get("constraints", [])
        
        if not objective:
            return {"success": False, "error": "No objective specified"}
        
        # Strategic analysis components
        strategy_components = {
            "situation_analysis": f"Analyze current situation regarding: {objective}",
            "goal_setting": f"Define SMART goals for: {objective}",
            "strategy_formulation": f"Develop strategies to achieve: {objective}",
            "resource_planning": f"Plan resource allocation for: {objective}",
            "risk_assessment": f"Identify and assess risks for: {objective}",
            "implementation_plan": f"Create implementation roadmap for: {objective}",
            "monitoring_metrics": f"Define success metrics for: {objective}"
        }
        
        strategic_analysis = {}
        for component, prompt_base in strategy_components.items():
            full_prompt = f"""
            {prompt_base}
            
            Timeframe: {timeframe}
            Available resources: {json.dumps(resources, indent=2) if resources else 'To be determined'}
            Constraints: {', '.join(constraints) if constraints else 'None specified'}
            
            Provide strategic insights for this component.
            """
            
            component_analysis = await model_manager.general_ai_response(full_prompt)
            strategic_analysis[component] = component_analysis
        
        # Create comprehensive strategic plan
        plan_synthesis_prompt = f"""
        Objective: {objective}
        Timeframe: {timeframe}
        
        Strategic analysis components:
        {chr(10).join([f"{comp}: {analysis[:100]}..." for comp, analysis in strategic_analysis.items()])}
        
        Synthesize into a comprehensive, actionable strategic plan.
        Include priorities, milestones, and success criteria.
        """
        
        strategic_plan = await model_manager.general_ai_response(plan_synthesis_prompt)
        
        # Store strategic plan in memory
        await vector_memory.store(
            content=strategic_plan,
            metadata={
                "type": "strategic_plan",
                "objective": objective,
                "timeframe": timeframe,
                "agent": self.agent_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        return {
            "success": True,
            "objective": objective,
            "timeframe": timeframe,
            "resources": resources,
            "constraints": constraints,
            "strategic_analysis": strategic_analysis,
            "comprehensive_plan": strategic_plan,
            "capabilities_used": ["strategic_planning", "logical_reasoning"],
            "agent_id": self.agent_id,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _problem_decomposition_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Break down complex problems into manageable components."""
        problem = task.get("problem", "")
        decomposition_level = task.get("level", "detailed")
        
        if not problem:
            return {"success": False, "error": "No problem specified"}
        
        # Multi-level decomposition
        decomposition_prompt = f"""
        Problem: {problem}
        
        Decompose this problem using multiple approaches:
        
        1. Functional decomposition (what needs to be done)
        2. Structural decomposition (system components)
        3. Temporal decomposition (sequence and timing)
        4. Causal decomposition (cause and effect relationships)
        5. Stakeholder decomposition (who is involved)
        
        Level of detail: {decomposition_level}
        
        Provide systematic breakdown for each approach.
        """
        
        decomposition_result = await model_manager.general_ai_response(decomposition_prompt)
        
        # Create action plan from decomposition
        action_plan_prompt = f"""
        Problem: {problem}
        
        Decomposition result:
        {decomposition_result}
        
        Create an actionable plan with:
        1. Prioritized sub-problems
        2. Dependencies between components
        3. Recommended solution approach for each component
        4. Resource requirements
        5. Success criteria
        """
        
        action_plan = await model_manager.general_ai_response(action_plan_prompt)
        
        return {
            "success": True,
            "original_problem": problem,
            "decomposition_level": decomposition_level,
            "decomposition_result": decomposition_result,
            "action_plan": action_plan,
            "capabilities_used": ["problem_decomposition", "logical_reasoning"],
            "agent_id": self.agent_id,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _learning_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Learn and adapt from new information."""
        learning_content = task.get("content", "")
        learning_type = task.get("type", "experiential")  # experiential, theoretical, practical
        domain = task.get("domain", "general")
        
        if not learning_content:
            return {"success": False, "error": "No learning content provided"}
        
        # Process learning content
        learning_analysis = await self._analyze_learning_content(learning_content, learning_type, domain)
        
        # Extract key insights
        insights = await self._extract_insights(learning_content, domain)
        
        # Update knowledge base
        await self._update_knowledge_base(domain, learning_content, insights)
        
        # Record learning experience
        learning_record = {
            "content": learning_content[:500],  # Store summary
            "type": learning_type,
            "domain": domain,
            "insights": insights,
            "timestamp": datetime.utcnow().isoformat(),
            "confidence": learning_analysis.get("confidence", 0.7)
        }
        
        self.learning_history.append(learning_record)
        
        return {
            "success": True,
            "learning_type": learning_type,
            "domain": domain,
            "analysis": learning_analysis,
            "insights_extracted": insights,
            "knowledge_updated": True,
            "learning_confidence": learning_analysis.get("confidence", 0.7),
            "capabilities_used": ["learning_adaptation", "memory_management"],
            "agent_id": self.agent_id,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _adaptive_intelligence_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Apply adaptive intelligence to any task."""
        content = task.get("content", "")
        adaptation_context = task.get("context", {})
        
        if not content:
            return {"success": False, "error": "No content provided"}
        
        # Adaptive intelligence processing
        intelligence_response = await self._apply_adaptive_intelligence(content, adaptation_context)
        
        return {
            "success": True,
            "content": content,
            "adaptation_context": adaptation_context,
            "intelligence_response": intelligence_response,
            "capabilities_used": ["general_intelligence", "learning_adaptation"],
            "agent_id": self.agent_id,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _analyze_problem_complexity(self, problem: str) -> Dict[str, Any]:
        """Analyze the complexity of a given problem."""
        complexity_prompt = f"""
        Analyze the complexity of this problem:
        
        Problem: {problem}
        
        Assess:
        1. Cognitive complexity (mental effort required)
        2. Information complexity (amount of information to process)
        3. Structural complexity (number of components and relationships)
        4. Dynamic complexity (changes over time)
        5. Solution complexity (difficulty of finding solutions)
        
        Rate each dimension from 1-10 and provide overall complexity assessment.
        """
        
        complexity_analysis = await model_manager.general_ai_response(complexity_prompt)
        
        return {
            "analysis": complexity_analysis,
            "estimated_difficulty": "high",  # Could be extracted from analysis
            "recommended_approach": "systematic"
        }
    
    async def _select_intelligence_domains(self, problem: str, complexity: Dict[str, Any]) -> List[str]:
        """Select appropriate intelligence domains for a problem."""
        # Simple heuristic for domain selection
        selected_domains = []
        
        # Always include linguistic for text processing
        selected_domains.append("linguistic")
        
        # Include logical-mathematical for analysis
        selected_domains.append("logical_mathematical")
        
        # Include interpersonal for social problems
        if any(word in problem.lower() for word in ["people", "team", "social", "group", "collaboration"]):
            selected_domains.append("interpersonal")
        
        # Include spatial for visual/design problems
        if any(word in problem.lower() for word in ["design", "visual", "layout", "structure", "spatial"]):
            selected_domains.append("spatial")
        
        return selected_domains
    
    async def _apply_domain_intelligence(self, domain: str, problem: str, context: Dict[str, Any]) -> str:
        """Apply specific intelligence domain to problem."""
        domain_prompts = {
            "linguistic": f"Apply linguistic intelligence to analyze and solve: {problem}",
            "logical_mathematical": f"Apply logical and mathematical reasoning to: {problem}",
            "spatial": f"Apply spatial and visual thinking to: {problem}",
            "interpersonal": f"Apply interpersonal and social intelligence to: {problem}",
            "intrapersonal": f"Apply self-awareness and reflection to: {problem}",
            "naturalistic": f"Apply pattern recognition and natural understanding to: {problem}"
        }
        
        prompt = domain_prompts.get(domain, f"Apply {domain} intelligence to: {problem}")
        return await model_manager.general_ai_response(prompt)
    
    async def _synthesize_intelligence(self, problem: str, domain_responses: Dict[str, str], complexity: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize responses from multiple intelligence domains."""
        synthesis_prompt = f"""
        Problem: {problem}
        Complexity: {complexity.get('estimated_difficulty', 'medium')}
        
        Intelligence domain responses:
        {chr(10).join([f"{domain}: {response[:150]}..." for domain, response in domain_responses.items()])}
        
        Synthesize these perspectives into a comprehensive, intelligent solution.
        Consider the strengths of each domain and create an integrated response.
        """
        
        synthesized_response = await model_manager.general_ai_response(synthesis_prompt)
        
        return {
            "synthesized_solution": synthesized_response,
            "domains_integrated": len(domain_responses),
            "quality_score": 0.8,  # Could be calculated based on various factors
            "confidence": 0.85
        }
    
    async def _record_learning_experience(self, experience: Dict[str, Any]):
        """Record learning experience for future adaptation."""
        self.learning_history.append({
            **experience,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Keep only recent learning experiences (last 100)
        if len(self.learning_history) > 100:
            self.learning_history = self.learning_history[-100:]
    
    async def _enhance_with_knowledge(self, response: str, domain_knowledge: List[Dict]) -> str:
        """Enhance response with existing domain knowledge."""
        if not domain_knowledge:
            return response
        
        # Simple enhancement - could be more sophisticated
        recent_knowledge = domain_knowledge[-3:]  # Use last 3 entries
        knowledge_text = "\n".join([entry["response"][:100] for entry in recent_knowledge])
        
        enhancement_prompt = f"""
        Original response: {response}
        
        Related knowledge:
        {knowledge_text}
        
        Enhance the original response by incorporating relevant insights from the knowledge base.
        """
        
        return await model_manager.general_ai_response(enhancement_prompt)
    
    async def _assess_creativity(self, content: str) -> float:
        """Assess creativity score of content."""
        # Simple creativity assessment - could use more sophisticated metrics
        creativity_indicators = [
            "innovative", "unique", "original", "creative", "novel",
            "unconventional", "imaginative", "breakthrough", "revolutionary"
        ]
        
        score = sum(1 for indicator in creativity_indicators if indicator in content.lower())
        return min(score / len(creativity_indicators), 1.0)
    
    async def _analyze_learning_content(self, content: str, learning_type: str, domain: str) -> Dict[str, Any]:
        """Analyze learning content for insights."""
        analysis_prompt = f"""
        Analyze this learning content:
        
        Content: {content}
        Learning type: {learning_type}
        Domain: {domain}
        
        Provide:
        1. Key concepts and principles
        2. Practical applications
        3. Learning confidence assessment
        4. Knowledge gaps identified
        5. Integration opportunities
        """
        
        analysis = await model_manager.general_ai_response(analysis_prompt)
        
        return {
            "analysis": analysis,
            "confidence": 0.75,
            "complexity": "medium"
        }
    
    async def _extract_insights(self, content: str, domain: str) -> List[str]:
        """Extract key insights from content."""
        insights_prompt = f"""
        Extract key insights from this content in the {domain} domain:
        
        {content}
        
        Provide 3-5 actionable insights that can be applied in future situations.
        """
        
        insights_text = await model_manager.general_ai_response(insights_prompt)
        
        # Simple extraction - could be more sophisticated
        insights = [insight.strip() for insight in insights_text.split('\n') if insight.strip()]
        return insights[:5]
    
    async def _update_knowledge_base(self, domain: str, content: str, insights: List[str]):
        """Update knowledge base with new learning."""
        if domain not in self.knowledge_base:
            self.knowledge_base[domain] = []
        
        self.knowledge_base[domain].append({
            "content": content[:200],  # Store summary
            "insights": insights,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Keep knowledge base manageable
        if len(self.knowledge_base[domain]) > 50:
            self.knowledge_base[domain] = self.knowledge_base[domain][-50:]
    
    async def _apply_adaptive_intelligence(self, content: str, context: Dict[str, Any]) -> str:
        """Apply adaptive intelligence processing."""
        adaptation_prompt = f"""
        Apply advanced artificial general intelligence to this content:
        
        Content: {content}
        Context: {json.dumps(context, indent=2) if context else 'No specific context'}
        
        Use adaptive intelligence to:
        1. Understand the deep meaning and implications
        2. Identify patterns and connections
        3. Generate insights and recommendations
        4. Consider multiple perspectives
        5. Provide actionable conclusions
        
        Apply the full spectrum of intelligence capabilities.
        """
        
        return await model_manager.general_ai_response(adaptation_prompt)
    
    def get_intelligence_status(self) -> Dict[str, Any]:
        """Get current intelligence system status."""
        return {
            "intelligence_domains": self.intelligence_domains,
            "knowledge_domains": list(self.knowledge_base.keys()),
            "learning_experiences": len(self.learning_history),
            "total_knowledge_entries": sum(len(entries) for entries in self.knowledge_base.values()),
            "active_capabilities": [cap for cap in self.capabilities if cap in ["general_intelligence", "learning_adaptation"]],
            "last_learning": self.learning_history[-1]["timestamp"] if self.learning_history else None
        }

# Global instance
big_agi_agent = BigAGIAgent()