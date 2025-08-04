#!/usr/bin/env python3
"""
Task Planner - Creates execution plans for voice commands and tasks
Integrates with Ollama/TinyLlama for intelligent planning
"""

import asyncio
import json
import logging
import httpx
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class TaskPlanner:
    """Intelligent task planner using local LLM"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.ollama_url = config.get('ollama_url', 'http://ollama:11434')
        self.planning_model = config.get('planning_model', 'tinyllama')
        self.max_steps = config.get('max_steps', 10)
        self.enable_reflection = config.get('enable_reflection', True)
        self.planning_templates = self._load_planning_templates()
        
    async def initialize(self):
        """Initialize task planner"""
        try:
            # Test connection to Ollama
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.ollama_url}/api/tags")
                if response.status_code == 200:
                    logger.info("Task planner connected to Ollama successfully")
                else:
                    logger.warning(f"Ollama connection test failed: {response.status_code}")
                    
        except Exception as e:
            logger.error(f"Failed to initialize task planner: {e}")
            
    async def shutdown(self):
        """Shutdown task planner"""
        logger.info("Task planner shutdown complete")
        
    def _load_planning_templates(self) -> Dict[str, str]:
        """Load planning templates for different task types"""
        return {
            'development': """
            You are a senior software architect. Break down this development task: "{task}"
            
            Consider:
            - Code architecture and design patterns
            - Testing requirements
            - Security considerations
            - Performance optimization
            - Documentation needs
            
            Create a step-by-step plan with clear deliverables.
            """,
            
            'deployment': """
            You are a DevOps expert. Plan this deployment task: "{task}"
            
            Consider:
            - Infrastructure requirements
            - Security measures
            - Monitoring and logging
            - Rollback strategies
            - Performance testing
            
            Create a detailed deployment plan.
            """,
            
            'analysis': """
            You are a data analyst. Plan this analysis task: "{task}"
            
            Consider:
            - Data sources and quality
            - Analytical methods
            - Visualization requirements
            - Statistical validation
            - Reporting format
            
            Create an analysis plan with clear methodology.
            """,
            
            'security': """
            You are a cybersecurity expert. Plan this security task: "{task}"
            
            Consider:
            - Threat assessment
            - Vulnerability analysis
            - Risk mitigation
            - Compliance requirements
            - Incident response
            
            Create a comprehensive security plan.
            """,
            
            'general': """
            You are an intelligent assistant. Break down this task: "{task}"
            
            Consider:
            - Task complexity and dependencies
            - Required resources and skills
            - Potential risks and mitigation
            - Success criteria
            - Timeline considerations
            
            Create a logical step-by-step plan.
            """
        }
        
    async def create_plan(self, task: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create an execution plan for a task"""
        try:
            # Determine task type
            task_type = self._classify_task(task)
            
            # Generate plan using LLM
            llm_plan = await self._generate_llm_plan(task, task_type, context)
            
            # Structure the plan
            structured_plan = self._structure_plan(llm_plan, task, task_type)
            
            # Add reflection if enabled
            if self.enable_reflection:
                structured_plan = await self._reflect_on_plan(structured_plan)
                
            return structured_plan
            
        except Exception as e:
            logger.error(f"Failed to create plan: {e}")
            return self._create_fallback_plan(task)
            
    def _classify_task(self, task: str) -> str:
        """Classify task type for appropriate planning template"""
        task_lower = task.lower()
        
        # Development keywords
        if any(word in task_lower for word in ['code', 'develop', 'build', 'implement', 'program', 'software']):
            return 'development'
            
        # Deployment keywords
        elif any(word in task_lower for word in ['deploy', 'release', 'infrastructure', 'server', 'container']):
            return 'deployment'
            
        # Analysis keywords
        elif any(word in task_lower for word in ['analyze', 'data', 'report', 'metrics', 'statistics']):
            return 'analysis'
            
        # Security keywords
        elif any(word in task_lower for word in ['security', 'secure', 'vulnerability', 'penetration', 'audit']):
            return 'security'
            
        else:
            return 'general'
            
    async def _generate_llm_plan(self, task: str, task_type: str, context: Dict[str, Any] = None) -> str:
        """Generate plan using local LLM"""
        try:
            # Get appropriate template
            template = self.planning_templates.get(task_type, self.planning_templates['general'])
            prompt = template.format(task=task)
            
            # Add context if available
            if context:
                context_str = "\n\nAdditional context:\n"
                for key, value in context.items():
                    context_str += f"- {key}: {value}\n"
                prompt += context_str
                
            # Add planning constraints
            prompt += f"""
            
            Requirements:
            - Maximum {self.max_steps} steps
            - Each step should be actionable and specific
            - Include success criteria for each step
            - Consider dependencies between steps
            - Provide estimated effort/time if possible
            
            Format the response as a structured plan with clear steps.
            """
            
            # Call Ollama
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.planning_model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.3,  # Lower temperature for more focused planning
                            "top_p": 0.9,
                            "num_predict": 1024
                        }
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result.get('response', '')
                else:
                    logger.error(f"Ollama request failed: {response.status_code}")
                    return ""
                    
        except Exception as e:
            logger.error(f"LLM plan generation failed: {e}")
            return ""
            
    def _structure_plan(self, llm_response: str, original_task: str, task_type: str) -> Dict[str, Any]:
        """Structure the LLM response into a formal plan"""
        try:
            # Parse the LLM response to extract steps
            steps = self._parse_steps_from_response(llm_response)
            
            # If parsing failed, create basic steps
            if not steps:
                steps = self._create_basic_steps(original_task, task_type)
                
            # Ensure we don't exceed max steps
            if len(steps) > self.max_steps:
                steps = steps[:self.max_steps]
                
            plan = {
                'id': f"plan_{datetime.now().timestamp()}",
                'goal': original_task,
                'type': task_type,
                'steps': steps,
                'estimated_duration': self._estimate_duration(steps),
                'created_at': datetime.now().isoformat(),
                'llm_response': llm_response[:500] + "..." if len(llm_response) > 500 else llm_response
            }
            
            return plan
            
        except Exception as e:
            logger.error(f"Failed to structure plan: {e}")
            return self._create_fallback_plan(original_task)
            
    def _parse_steps_from_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse steps from LLM response"""
        steps = []
        lines = response.split('\n')
        
        current_step = None
        step_counter = 1
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Look for step indicators
            if (line.lower().startswith(('step', 'phase', 'stage')) or 
                any(line.startswith(f"{i}.") for i in range(1, 20))):
                
                # Save previous step
                if current_step:
                    steps.append(current_step)
                    
                # Start new step
                current_step = {
                    'id': f"step_{step_counter}",
                    'type': 'execution',
                    'description': line,
                    'input': {},
                    'dependencies': [],
                    'estimated_effort': 'medium'
                }
                step_counter += 1
                
            elif current_step and line:
                # Add to current step description
                current_step['description'] += f" {line}"
                
        # Add final step
        if current_step:
            steps.append(current_step)
            
        return steps
        
    def _create_basic_steps(self, task: str, task_type: str) -> List[Dict[str, Any]]:
        """Create basic steps when LLM parsing fails"""
        return [
            {
                'id': 'step_1',
                'type': task_type,
                'description': f'Analyze and understand the task: {task}',
                'input': {'task': task},
                'dependencies': [],
                'estimated_effort': 'low'
            },
            {
                'id': 'step_2',
                'type': 'execution',
                'description': f'Execute the main task: {task}',
                'input': {'task': task},
                'dependencies': ['step_1'],
                'estimated_effort': 'high'
            },
            {
                'id': 'step_3',
                'type': 'validation',
                'description': 'Validate and verify the results',
                'input': {},
                'dependencies': ['step_2'],
                'estimated_effort': 'medium'
            }
        ]
        
    def _estimate_duration(self, steps: List[Dict[str, Any]]) -> int:
        """Estimate total duration in minutes"""
        effort_mapping = {
            'low': 5,
            'medium': 15,
            'high': 30,
            'very_high': 60
        }
        
        total_minutes = 0
        for step in steps:
            effort = step.get('estimated_effort', 'medium')
            total_minutes += effort_mapping.get(effort, 15)
            
        return total_minutes
        
    async def _reflect_on_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Use LLM to reflect on and improve the plan"""
        try:
            reflection_prompt = f"""
            Review this execution plan for the task: "{plan['goal']}"
            
            Current plan:
            {json.dumps(plan['steps'], indent=2)}
            
            Please analyze:
            1. Are all necessary steps included?
            2. Are the dependencies correct?
            3. Are there any risks or potential issues?
            4. Could the plan be optimized?
            
            Provide specific suggestions for improvement.
            """
            
            async with httpx.AsyncClient(timeout=20.0) as client:
                response = await client.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.planning_model,
                        "prompt": reflection_prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.4,
                            "num_predict": 512
                        }
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    reflection = result.get('response', '')
                    plan['reflection'] = reflection
                    plan['reflected_at'] = datetime.now().isoformat()
                    
        except Exception as e:
            logger.error(f"Plan reflection failed: {e}")
            
        return plan
        
    def _create_fallback_plan(self, task: str) -> Dict[str, Any]:
        """Create a simple fallback plan when all else fails"""
        return {
            'id': f"fallback_plan_{datetime.now().timestamp()}",
            'goal': task,
            'type': 'general',
            'steps': [
                {
                    'id': 'fallback_step',
                    'type': 'general',
                    'description': f'Execute task: {task}',
                    'input': {'task': task},
                    'dependencies': [],
                    'estimated_effort': 'medium'
                }
            ],
            'estimated_duration': 15,
            'created_at': datetime.now().isoformat(),
            'fallback': True
        }
        
    async def validate_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a plan for correctness and feasibility"""
        validation_result = {
            'valid': True,
            'issues': [],
            'warnings': []
        }
        
        # Check basic structure
        if not plan.get('steps'):
            validation_result['valid'] = False
            validation_result['issues'].append('Plan has no steps')
            
        # Check step dependencies
        step_ids = {step['id'] for step in plan.get('steps', [])}
        for step in plan.get('steps', []):
            for dep in step.get('dependencies', []):
                if dep not in step_ids:
                    validation_result['warnings'].append(f"Step {step['id']} depends on non-existent step {dep}")
                    
        # Check for circular dependencies
        if self._has_circular_dependencies(plan.get('steps', [])):
            validation_result['valid'] = False
            validation_result['issues'].append('Plan has circular dependencies')
            
        return validation_result
        
    def _has_circular_dependencies(self, steps: List[Dict[str, Any]]) -> bool:
        """Check for circular dependencies in the plan"""
        try:
            # Build dependency graph
            graph = {}
            for step in steps:
                graph[step['id']] = step.get('dependencies', [])
                
            # DFS to detect cycles
            visited = set()
            rec_stack = set()
            
            def has_cycle(node):
                if node in rec_stack:
                    return True
                if node in visited:
                    return False
                    
                visited.add(node)
                rec_stack.add(node)
                
                for neighbor in graph.get(node, []):
                    if has_cycle(neighbor):
                        return True
                        
                rec_stack.remove(node)
                return False
                
            for node in graph:
                if node not in visited:
                    if has_cycle(node):
                        return True
                        
            return False
            
        except Exception as e:
            logger.error(f"Error checking circular dependencies: {e}")
            return False