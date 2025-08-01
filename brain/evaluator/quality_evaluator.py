#!/usr/bin/env python3
"""
Quality Evaluator for the Brain
Uses LLMs to evaluate the quality of agent outputs
"""

import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import asyncio

from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QualityScore(BaseModel):
    """Structure for quality evaluation results"""
    accuracy: float = Field(description="How accurate is the output (0-1)")
    completeness: float = Field(description="How complete is the output (0-1)")
    relevance: float = Field(description="How relevant is the output to the task (0-1)")
    coherence: float = Field(description="How coherent and well-structured is the output (0-1)")
    usefulness: float = Field(description="How useful is the output for the user (0-1)")
    overall_score: float = Field(description="Overall quality score (0-1)")
    explanation: str = Field(description="Brief explanation of the scores")
    improvements: List[str] = Field(description="Specific improvements needed")


class QualityEvaluator:
    """Evaluates the quality of agent outputs using LLMs"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize evaluation LLM (using tinyllama for reasoning)
        self.evaluator_llm = Ollama(
            model=config.get('evaluation_model', 'tinyllama'),
            base_url=config.get('ollama_host', 'http://sutazai-ollama:11434'),
            temperature=0.1  # Low temperature for consistent evaluation
        )
        
        # Initialize comparison LLM for multi-agent outputs
        self.comparison_llm = Ollama(
            model=config.get('comparison_model', 'qwen2.5:7b'),
            base_url=config.get('ollama_host', 'http://sutazai-ollama:11434'),
            temperature=0.1
        )
        
        # Output parser
        self.parser = PydanticOutputParser(pydantic_object=QualityScore)
        
        # Evaluation prompt
        self.evaluation_prompt = PromptTemplate(
            input_variables=["task", "output", "expected_criteria"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()},
            template="""You are an expert quality evaluator for AI agent outputs.

Task: {task}

Agent Output:
{output}

Expected Criteria:
{expected_criteria}

Please evaluate this output on the following dimensions:
1. Accuracy - Is the information correct and factual?
2. Completeness - Does it fully address the task?
3. Relevance - Is it focused on what was asked?
4. Coherence - Is it well-structured and logical?
5. Usefulness - Will it help the user achieve their goal?

Provide specific improvement suggestions if the output is lacking.

{format_instructions}
"""
        )
        
        # Comparison prompt for multiple outputs
        self.comparison_prompt = PromptTemplate(
            input_variables=["task", "outputs"],
            template="""Compare these agent outputs for the same task and identify the best one.

Task: {task}

Agent Outputs:
{outputs}

Analyze each output and determine:
1. Which output best fulfills the task requirements?
2. What are the strengths of each output?
3. What are the weaknesses of each output?
4. Which output would be most useful to the user?

Provide a ranking from best to worst with justification.
"""
        )
        
        # Track evaluation metrics
        self.evaluation_history: List[Dict[str, Any]] = []
        
    async def evaluate_result(
        self,
        result: Dict[str, Any],
        original_input: str,
        expected_output: Optional[Dict[str, Any]] = None
    ) -> float:
        """Evaluate a single agent result"""
        try:
            # Skip evaluation for failed results
            if not result.get('success', False) or result.get('output') is None:
                return 0.0
            
            # Prepare evaluation criteria
            criteria = self._prepare_evaluation_criteria(original_input, expected_output)
            
            # Format the evaluation prompt
            prompt = self.evaluation_prompt.format(
                task=original_input,
                output=self._format_output(result['output']),
                expected_criteria=criteria
            )
            
            # Get evaluation from LLM
            evaluation_response = await self.evaluator_llm.ainvoke(prompt)
            
            # Parse the response
            try:
                quality_score = self.parser.parse(evaluation_response)
            except Exception as e:
                logger.warning(f"Failed to parse evaluation response: {e}")
                # Fallback to simple scoring
                quality_score = self._fallback_evaluation(result)
            
            # Record evaluation
            self._record_evaluation(result['agent'], quality_score)
            
            # Add improvements to result if needed
            if quality_score.overall_score < 0.8:
                result['quality_improvements'] = quality_score.improvements
            
            return quality_score.overall_score
            
        except Exception as e:
            logger.error(f"Error evaluating result: {e}")
            return self._calculate_basic_score(result)
    
    async def compare_results(
        self,
        results: List[Dict[str, Any]],
        original_input: str
    ) -> Dict[str, Any]:
        """Compare multiple agent results and rank them"""
        # Filter successful results
        successful_results = [r for r in results if r.get('success', False) and r.get('output')]
        
        if not successful_results:
            return {'best_agent': None, 'ranking': []}
        
        if len(successful_results) == 1:
            return {
                'best_agent': successful_results[0]['agent'],
                'ranking': [successful_results[0]['agent']]
            }
        
        # Format outputs for comparison
        outputs_text = "\n\n".join([
            f"Agent: {r['agent']}\nOutput: {self._format_output(r['output'])}"
            for r in successful_results
        ])
        
        # Get comparison from LLM
        comparison_prompt = self.comparison_prompt.format(
            task=original_input,
            outputs=outputs_text
        )
        
        comparison_response = await self.comparison_llm.ainvoke(comparison_prompt)
        
        # Extract ranking from response
        ranking = self._extract_ranking(comparison_response, successful_results)
        
        return {
            'best_agent': ranking[0] if ranking else None,
            'ranking': ranking,
            'comparison_analysis': comparison_response
        }
    
    def _prepare_evaluation_criteria(
        self,
        task: str,
        expected_output: Optional[Dict[str, Any]]
    ) -> str:
        """Prepare evaluation criteria based on task and expectations"""
        criteria = []
        
        # Analyze task for implicit criteria
        task_lower = task.lower()
        
        if any(word in task_lower for word in ['code', 'implement', 'function', 'class']):
            criteria.append("Code should be syntactically correct and follow best practices")
            criteria.append("Implementation should be complete and runnable")
        
        if any(word in task_lower for word in ['analyze', 'research', 'investigate']):
            criteria.append("Analysis should be thorough and well-researched")
            criteria.append("Include relevant data and sources")
        
        if any(word in task_lower for word in ['explain', 'describe', 'summarize']):
            criteria.append("Explanation should be clear and easy to understand")
            criteria.append("Include relevant examples if applicable")
        
        if any(word in task_lower for word in ['create', 'generate', 'write']):
            criteria.append("Output should be original and creative")
            criteria.append("Follow any specified format or structure")
        
        # Add expected output criteria if provided
        if expected_output:
            if 'format' in expected_output:
                criteria.append(f"Output should be in {expected_output['format']} format")
            if 'length' in expected_output:
                criteria.append(f"Output length should be approximately {expected_output['length']}")
            if 'requirements' in expected_output:
                criteria.extend(expected_output['requirements'])
        
        return "\n".join(f"- {c}" for c in criteria) if criteria else "General quality and usefulness"
    
    def _format_output(self, output: Any) -> str:
        """Format output for evaluation"""
        if isinstance(output, str):
            return output[:2000]  # Limit length for evaluation
        elif isinstance(output, dict):
            return json.dumps(output, indent=2)[:2000]
        elif isinstance(output, list):
            return str(output)[:2000]
        else:
            return str(output)[:2000]
    
    def _fallback_evaluation(self, result: Dict[str, Any]) -> QualityScore:
        """Fallback evaluation when LLM parsing fails"""
        # Simple heuristic-based scoring
        output = str(result.get('output', ''))
        
        # Calculate basic scores
        has_content = len(output) > 50
        execution_time = result.get('execution_time', 999)
        
        base_score = 0.5
        if has_content:
            base_score += 0.2
        if execution_time < 10:
            base_score += 0.1
        if execution_time < 5:
            base_score += 0.1
        if result.get('agent') in ['gpt-engineer', 'autogen', 'crewai']:
            base_score += 0.1  # Bonus for advanced agents
        
        return QualityScore(
            accuracy=base_score,
            completeness=base_score,
            relevance=base_score,
            coherence=base_score,
            usefulness=base_score,
            overall_score=min(base_score, 1.0),
            explanation="Fallback evaluation based on heuristics",
            improvements=["Could not perform detailed evaluation"]
        )
    
    def _calculate_basic_score(self, result: Dict[str, Any]) -> float:
        """Calculate basic score when evaluation fails"""
        if not result.get('success', False):
            return 0.0
        
        # Base score for successful execution
        score = 0.5
        
        # Bonus for fast execution
        if result.get('execution_time', 999) < 5:
            score += 0.2
        elif result.get('execution_time', 999) < 10:
            score += 0.1
        
        # Bonus for having output
        if result.get('output'):
            score += 0.2
        
        # Bonus for known good agents
        if result.get('agent') in ['langchain', 'autogen', 'gpt-engineer']:
            score += 0.1
        
        return min(score, 1.0)
    
    def _extract_ranking(
        self,
        comparison_response: str,
        results: List[Dict[str, Any]]
    ) -> List[str]:
        """Extract agent ranking from comparison response"""
        agent_names = [r['agent'] for r in results]
        
        # Simple extraction based on order mentioned
        ranking = []
        response_lower = comparison_response.lower()
        
        for agent in agent_names:
            if agent.lower() in response_lower:
                if agent not in ranking:
                    ranking.append(agent)
        
        # Add any missing agents at the end
        for agent in agent_names:
            if agent not in ranking:
                ranking.append(agent)
        
        return ranking
    
    def _record_evaluation(self, agent: str, score: QualityScore):
        """Record evaluation for analysis"""
        self.evaluation_history.append({
            'timestamp': datetime.now().isoformat(),
            'agent': agent,
            'scores': {
                'accuracy': score.accuracy,
                'completeness': score.completeness,
                'relevance': score.relevance,
                'coherence': score.coherence,
                'usefulness': score.usefulness,
                'overall': score.overall_score
            },
            'improvements': score.improvements
        })
        
        # Keep only recent history
        if len(self.evaluation_history) > 1000:
            self.evaluation_history = self.evaluation_history[-1000:]
    
    def get_agent_performance_stats(self) -> Dict[str, Dict[str, float]]:
        """Get performance statistics for each agent"""
        agent_stats = {}
        
        for eval_record in self.evaluation_history:
            agent = eval_record['agent']
            if agent not in agent_stats:
                agent_stats[agent] = {
                    'total_evaluations': 0,
                    'average_overall': 0,
                    'average_accuracy': 0,
                    'average_completeness': 0,
                    'average_relevance': 0,
                    'average_coherence': 0,
                    'average_usefulness': 0
                }
            
            stats = agent_stats[agent]
            stats['total_evaluations'] += 1
            
            # Update averages
            n = stats['total_evaluations']
            for metric in ['overall', 'accuracy', 'completeness', 'relevance', 'coherence', 'usefulness']:
                old_avg = stats[f'average_{metric}']
                new_val = eval_record['scores'][metric]
                stats[f'average_{metric}'] = (old_avg * (n - 1) + new_val) / n
        
        return agent_stats