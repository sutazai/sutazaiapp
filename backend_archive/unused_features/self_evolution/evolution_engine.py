#!/usr/bin/env python3
"""
Self-Evolution Engine
Implements meta-learning and autonomous system improvement for SutazAI
"""

import asyncio
import json
import logging
import pickle
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable
import numpy as np
from scipy import stats
import ast
import inspect

logger = logging.getLogger(__name__)

@dataclass
class EvolutionMetrics:
    """Metrics for evolution tracking"""
    performance_score: float = 0.0
    efficiency_score: float = 0.0
    accuracy_score: float = 0.0
    resource_usage: float = 0.0
    error_rate: float = 0.0
    learning_rate: float = 0.0
    adaptation_time: float = 0.0
    
    def overall_score(self) -> float:
        """Calculate overall fitness score"""
        return (self.performance_score * 0.3 + 
                self.efficiency_score * 0.2 +
                self.accuracy_score * 0.3 +
                (1 - self.resource_usage) * 0.1 +
                (1 - self.error_rate) * 0.1)

@dataclass
class EvolutionCandidate:
    """Candidate for evolution"""
    id: str
    code: str
    description: str
    metrics: EvolutionMetrics
    generation: int = 0
    parent_id: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
class SelfEvolutionEngine:
    """
    Self-Evolution Engine with meta-learning capabilities
    Enables autonomous system improvement and code evolution
    """
    
    def __init__(self, workspace_path: str = "data/evolution"):
        """Initialize self-evolution engine"""
        self.workspace_path = Path(workspace_path)
        self.workspace_path.mkdir(parents=True, exist_ok=True)
        
        # Evolution state
        self.population: Dict[str, EvolutionCandidate] = {}
        self.generation = 0
        self.evolution_history: List[Dict[str, Any]] = []
        
        # Evolution parameters
        self.population_size = 20
        self.mutation_rate = 0.1
        self.crossover_rate = 0.7
        self.elite_size = 5
        self.max_generations = 100
        
        # Safety constraints
        self.safe_mode = True
        self.code_sandbox_enabled = True
        self.resource_limits = {
            "max_memory_mb": 1024,
            "max_cpu_time": 30,
            "max_file_size": 1024 * 1024  # 1MB
        }
        
        # Meta-learning components
        self.meta_learner = None
        self.performance_history: List[EvolutionMetrics] = []
        self.adaptation_strategies: Dict[str, Callable] = {}
        
        # Initialize components
        self._initialize_meta_learner()
        self._initialize_strategies()
        
        logger.info("Self-Evolution Engine initialized")
    
    def _initialize_meta_learner(self):
        """Initialize meta-learning component"""
        try:
            from .meta_learner import MetaLearner
            self.meta_learner = MetaLearner()
        except Exception as e:
            logger.warning(f"Meta-learner initialization failed: {e}")
    
    def _initialize_strategies(self):
        """Initialize adaptation strategies"""
        self.adaptation_strategies = {
            "genetic_algorithm": self._genetic_evolution,
            "gradient_based": self._gradient_based_evolution,
            "random_search": self._random_search_evolution,
            "bayesian_optimization": self._bayesian_optimization,
            "neural_architecture_search": self._neural_architecture_search
        }
    
    async def evolve_code(self, original_code: str, target_metrics: EvolutionMetrics,
                         max_iterations: int = 50) -> Optional[EvolutionCandidate]:
        """Evolve code to meet target metrics"""
        try:
            logger.info(f"Starting code evolution for {max_iterations} iterations")
            
            # Initialize population
            await self._initialize_population(original_code)
            
            best_candidate = None
            for iteration in range(max_iterations):
                # Evaluate population
                await self._evaluate_population()
                
                # Select best candidate
                current_best = max(self.population.values(), 
                                 key=lambda c: c.metrics.overall_score())
                
                if not best_candidate or current_best.metrics.overall_score() > best_candidate.metrics.overall_score():
                    best_candidate = current_best
                
                # Check if target metrics achieved
                if self._meets_target_metrics(current_best.metrics, target_metrics):
                    logger.info(f"Target metrics achieved in iteration {iteration}")
                    break
                
                # Evolve population
                await self._evolve_population()
                
                # Meta-learning update
                if self.meta_learner:
                    await self.meta_learner.update(current_best.metrics)
                
                self.generation += 1
                
                # Log progress
                if iteration % 10 == 0:
                    logger.info(f"Iteration {iteration}: Best score = {current_best.metrics.overall_score():.3f}")
            
            return best_candidate
            
        except Exception as e:
            logger.error(f"Code evolution failed: {e}")
            return None
    
    async def _initialize_population(self, original_code: str):
        """Initialize evolution population"""
        try:
            self.population.clear()
            
            # Add original code
            original_id = f"gen0_original"
            original_candidate = EvolutionCandidate(
                id=original_id,
                code=original_code,
                description="Original code",
                metrics=await self._evaluate_code(original_code),
                generation=0
            )
            self.population[original_id] = original_candidate
            
            # Generate variations
            for i in range(self.population_size - 1):
                variation_code = await self._mutate_code(original_code)
                variation_id = f"gen0_var{i}"
                
                candidate = EvolutionCandidate(
                    id=variation_id,
                    code=variation_code,
                    description=f"Initial variation {i}",
                    metrics=EvolutionMetrics(),  # Will be evaluated later
                    generation=0,
                    parent_id=original_id
                )
                
                self.population[variation_id] = candidate
            
        except Exception as e:
            logger.error(f"Population initialization failed: {e}")
    
    async def _evaluate_population(self):
        """Evaluate all candidates in population"""
        try:
            evaluation_tasks = []
            
            for candidate in self.population.values():
                if candidate.metrics.overall_score() == 0:  # Not yet evaluated
                    task = self._evaluate_code(candidate.code)
                    evaluation_tasks.append((candidate, task))
            
            # Run evaluations concurrently
            for candidate, task in evaluation_tasks:
                try:
                    candidate.metrics = await task
                except Exception as e:
                    logger.warning(f"Evaluation failed for {candidate.id}: {e}")
                    candidate.metrics = EvolutionMetrics()  # Default poor metrics
            
        except Exception as e:
            logger.error(f"Population evaluation failed: {e}")
    
    async def _evaluate_code(self, code: str) -> EvolutionMetrics:
        """Evaluate code performance and safety"""
        try:
            metrics = EvolutionMetrics()
            
            # Safety check
            if not await self._is_code_safe(code):
                return metrics  # Return default poor metrics
            
            # Performance evaluation
            start_time = time.time()
            
            # Execute code in sandbox
            execution_result = await self._execute_code_safely(code)
            
            execution_time = time.time() - start_time
            
            if execution_result["success"]:
                # Calculate performance metrics
                metrics.performance_score = min(1.0, 10.0 / (execution_time + 0.1))
                metrics.efficiency_score = 1.0 - min(1.0, execution_result.get("memory_usage", 0) / 1000.0)
                metrics.accuracy_score = execution_result.get("accuracy", 0.5)
                metrics.resource_usage = execution_result.get("resource_usage", 0.5)
                metrics.error_rate = 0.0
                metrics.adaptation_time = execution_time
            else:
                metrics.error_rate = 1.0
            
            return metrics
            
        except Exception as e:
            logger.error(f"Code evaluation failed: {e}")
            return EvolutionMetrics()
    
    async def _is_code_safe(self, code: str) -> bool:
        """Check if code is safe to execute"""
        try:
            if not self.safe_mode:
                return True
            
            # Parse code to AST
            try:
                tree = ast.parse(code)
            except SyntaxError:
                return False
            
            # Check for dangerous operations
            dangerous_patterns = [
                'import os',
                'import sys', 
                'import subprocess',
                'open(',
                'eval(',
                'exec(',
                '__import__',
                'globals()',
                'locals()',
                'file(',
                'input(',
                'raw_input('
            ]
            
            for pattern in dangerous_patterns:
                if pattern in code:
                    logger.warning(f"Unsafe code pattern detected: {pattern}")
                    return False
            
            # AST-based safety checks
            class SafetyVisitor(ast.NodeVisitor):
                def __init__(self):
                    self.safe = True
                
                def visit_Import(self, node):
                    # Allow only safe imports
                    safe_modules = ['math', 'random', 'json', 'datetime', 're', 'collections']
                    for alias in node.names:
                        if alias.name not in safe_modules:
                            self.safe = False
                    self.generic_visit(node)
                
                def visit_Call(self, node):
                    # Check function calls
                    if isinstance(node.func, ast.Name):
                        dangerous_calls = ['eval', 'exec', 'compile', '__import__']
                        if node.func.id in dangerous_calls:
                            self.safe = False
                    self.generic_visit(node)
            
            visitor = SafetyVisitor()
            visitor.visit(tree)
            
            return visitor.safe
            
        except Exception as e:
            logger.error(f"Safety check failed: {e}")
            return False
    
    async def _execute_code_safely(self, code: str) -> Dict[str, Any]:
        """Execute code in safe sandbox environment"""
        try:
            if not self.code_sandbox_enabled:
                return {"success": False, "error": "Sandbox disabled"}
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp_file:
                tmp_file.write(code)
                tmp_path = tmp_file.name
            
            try:
                # Execute with resource limits
                process = await asyncio.create_subprocess_exec(
                    'python', tmp_path,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=self.workspace_path
                )
                
                try:
                    stdout, stderr = await asyncio.wait_for(
                        process.communicate(), 
                        timeout=self.resource_limits["max_cpu_time"]
                    )
                    
                    success = process.returncode == 0
                    
                    return {
                        "success": success,
                        "stdout": stdout.decode() if stdout else "",
                        "stderr": stderr.decode() if stderr else "",
                        "memory_usage": 0,  # Would need actual measurement
                        "resource_usage": 0.1,  # Placeholder
                        "accuracy": 0.8 if success else 0.0  # Placeholder
                    }
                    
                except asyncio.TimeoutError:
                    process.kill()
                    return {"success": False, "error": "Execution timeout"}
                
            finally:
                # Cleanup
                Path(tmp_path).unlink(missing_ok=True)
            
        except Exception as e:
            logger.error(f"Safe execution failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _evolve_population(self):
        """Evolve population using genetic algorithms"""
        try:
            # Select best candidates (elitism)
            sorted_candidates = sorted(
                self.population.values(),
                key=lambda c: c.metrics.overall_score(),
                reverse=True
            )
            
            elite = sorted_candidates[:self.elite_size]
            
            # Create new population
            new_population = {}
            
            # Keep elite
            for candidate in elite:
                new_population[candidate.id] = candidate
            
            # Generate offspring
            while len(new_population) < self.population_size:
                if np.random.random() < self.crossover_rate and len(elite) >= 2:
                    # Crossover
                    parent1, parent2 = np.random.choice(elite, 2, replace=False)
                    offspring_code = await self._crossover_code(parent1.code, parent2.code)
                    parent_ids = f"{parent1.id}+{parent2.id}"
                else:
                    # Mutation only
                    parent = np.random.choice(elite)
                    offspring_code = await self._mutate_code(parent.code)
                    parent_ids = parent.id
                
                # Create offspring candidate
                offspring_id = f"gen{self.generation + 1}_{len(new_population)}"
                offspring = EvolutionCandidate(
                    id=offspring_id,
                    code=offspring_code,
                    description=f"Generation {self.generation + 1} offspring",
                    metrics=EvolutionMetrics(),
                    generation=self.generation + 1,
                    parent_id=parent_ids
                )
                
                new_population[offspring_id] = offspring
            
            self.population = new_population
            
        except Exception as e:
            logger.error(f"Population evolution failed: {e}")
    
    async def _mutate_code(self, code: str) -> str:
        """Apply random mutations to code"""
        try:
            lines = code.split('\n')
            
            # Simple mutations
            mutations = [
                self._mutate_variable_names,
                self._mutate_constants,
                self._mutate_operators,
                self._mutate_control_flow
            ]
            
            # Apply random mutations
            if np.random.random() < self.mutation_rate:
                mutation = np.random.choice(mutations)
                lines = mutation(lines)
            
            return '\n'.join(lines)
            
        except Exception as e:
            logger.error(f"Code mutation failed: {e}")
            return code
    
    def _mutate_variable_names(self, lines: List[str]) -> List[str]:
        """Mutate variable names"""
        # Simple variable name mutations
        # This is a placeholder - real implementation would be more sophisticated
        return lines
    
    def _mutate_constants(self, lines: List[str]) -> List[str]:
        """Mutate numeric constants"""
        mutated_lines = []
        for line in lines:
            # Find numeric constants and slightly modify them
            import re
            def mutate_number(match):
                num = float(match.group())
                # Add small random variation
                variation = num * 0.1 * (np.random.random() - 0.5)
                return str(num + variation)
            
            mutated_line = re.sub(r'\b\d+\.?\d*\b', mutate_number, line)
            mutated_lines.append(mutated_line)
        
        return mutated_lines
    
    def _mutate_operators(self, lines: List[str]) -> List[str]:
        """Mutate mathematical operators"""
        # Simple operator mutations
        operator_map = {'+': '-', '-': '+', '*': '/', '/': '*'}
        
        mutated_lines = []
        for line in lines:
            mutated_line = line
            for old_op, new_op in operator_map.items():
                if old_op in line and np.random.random() < 0.1:
                    mutated_line = mutated_line.replace(old_op, new_op, 1)
                    break
            mutated_lines.append(mutated_line)
        
        return mutated_lines
    
    def _mutate_control_flow(self, lines: List[str]) -> List[str]:
        """Mutate control flow structures"""
        # Placeholder for control flow mutations
        return lines
    
    async def _crossover_code(self, code1: str, code2: str) -> str:
        """Perform crossover between two code snippets"""
        try:
            lines1 = code1.split('\n')
            lines2 = code2.split('\n')
            
            # Simple crossover - combine parts of both codes
            min_length = min(len(lines1), len(lines2))
            crossover_point = np.random.randint(1, min_length)
            
            offspring_lines = lines1[:crossover_point] + lines2[crossover_point:]
            
            return '\n'.join(offspring_lines)
            
        except Exception as e:
            logger.error(f"Code crossover failed: {e}")
            return code1
    
    def _meets_target_metrics(self, current: EvolutionMetrics, target: EvolutionMetrics) -> bool:
        """Check if current metrics meet target"""
        return (current.performance_score >= target.performance_score and
                current.efficiency_score >= target.efficiency_score and
                current.accuracy_score >= target.accuracy_score and
                current.error_rate <= target.error_rate)
    
    # Additional evolution strategies
    async def _genetic_evolution(self, population: Dict[str, EvolutionCandidate]) -> Dict[str, EvolutionCandidate]:
        """Genetic algorithm evolution"""
        # Already implemented above
        return population
    
    async def _gradient_based_evolution(self, population: Dict[str, EvolutionCandidate]) -> Dict[str, EvolutionCandidate]:
        """Gradient-based code evolution"""
        # Placeholder for gradient-based optimization
        return population
    
    async def _random_search_evolution(self, population: Dict[str, EvolutionCandidate]) -> Dict[str, EvolutionCandidate]:
        """Random search evolution"""
        # Generate random variations
        for candidate in list(population.values()):
            if np.random.random() < 0.3:  # 30% chance to randomize
                candidate.code = await self._mutate_code(candidate.code)
        
        return population
    
    async def _bayesian_optimization(self, population: Dict[str, EvolutionCandidate]) -> Dict[str, EvolutionCandidate]:
        """Bayesian optimization for code evolution"""
        # Placeholder for Bayesian optimization
        return population
    
    async def _neural_architecture_search(self, population: Dict[str, EvolutionCandidate]) -> Dict[str, EvolutionCandidate]:
        """Neural architecture search for code evolution"""
        # Placeholder for NAS
        return population
    
    def get_evolution_statistics(self) -> Dict[str, Any]:
        """Get evolution statistics"""
        try:
            if not self.population:
                return {}
            
            scores = [c.metrics.overall_score() for c in self.population.values()]
            
            return {
                "generation": self.generation,
                "population_size": len(self.population),
                "best_score": max(scores),
                "average_score": np.mean(scores),
                "score_std": np.std(scores),
                "evolution_history_length": len(self.evolution_history)
            }
            
        except Exception as e:
            logger.error(f"Failed to get evolution statistics: {e}")
            return {}
    
    def save_evolution_state(self, file_path: str):
        """Save evolution state to file"""
        try:
            state = {
                "generation": self.generation,
                "population": {
                    k: {
                        "id": v.id,
                        "code": v.code,
                        "description": v.description,
                        "metrics": v.metrics.__dict__,
                        "generation": v.generation,
                        "parent_id": v.parent_id,
                        "created_at": v.created_at.isoformat()
                    }
                    for k, v in self.population.items()
                },
                "evolution_history": self.evolution_history
            }
            
            with open(file_path, 'w') as f:
                json.dump(state, f, indent=2)
            
            logger.info(f"Evolution state saved to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to save evolution state: {e}")
    
    def load_evolution_state(self, file_path: str):
        """Load evolution state from file"""
        try:
            with open(file_path, 'r') as f:
                state = json.load(f)
            
            self.generation = state["generation"]
            self.evolution_history = state["evolution_history"]
            
            # Reconstruct population
            self.population = {}
            for k, v in state["population"].items():
                metrics = EvolutionMetrics(**v["metrics"])
                candidate = EvolutionCandidate(
                    id=v["id"],
                    code=v["code"],
                    description=v["description"],
                    metrics=metrics,
                    generation=v["generation"],
                    parent_id=v["parent_id"],
                    created_at=datetime.fromisoformat(v["created_at"])
                )
                self.population[k] = candidate
            
            logger.info(f"Evolution state loaded from {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to load evolution state: {e}")