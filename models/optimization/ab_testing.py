"""
A/B Testing Framework for SutazAI Model Comparison
Implements statistical testing and experimental design for model evaluation
"""
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import logging
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import json
import os
from pathlib import Path
import asyncio
import aiohttp
import sqlite3
import time
import hashlib
from collections import defaultdict, deque
import statistics
import random
from scipy import stats
import uuid

logger = logging.getLogger(__name__)

class ExperimentType(Enum):
    """Types of A/B testing experiments"""
    MODEL_COMPARISON = "model_comparison"
    PARAMETER_OPTIMIZATION = "parameter_optimization"
    PROMPT_OPTIMIZATION = "prompt_optimization"
    ENSEMBLE_STRATEGY = "ensemble_strategy"
    PERFORMANCE_BENCHMARK = "performance_benchmark"

class SplitStrategy(Enum):
    """Strategies for splitting traffic between variants"""
    RANDOM = "random"
    ROUND_ROBIN = "round_robin"
    WEIGHTED = "weighted"
    HASH_BASED = "hash_based"
    CONTEXTUAL = "contextual"

class StatisticalTest(Enum):
    """Statistical tests for significance"""
    T_TEST = "t_test"
    MANN_WHITNEY = "mann_whitney"
    CHI_SQUARE = "chi_square"
    BOOTSTRAP = "bootstrap"
    BAYESIAN = "bayesian"

@dataclass
class ExperimentConfig:
    """Configuration for A/B testing experiments"""
    name: str
    experiment_type: ExperimentType
    
    # Experiment design
    variants: List[str] = field(default_factory=list)
    traffic_split: Dict[str, float] = field(default_factory=dict)
    split_strategy: SplitStrategy = SplitStrategy.RANDOM
    
    # Sample size and duration
    min_sample_size: int = 100
    max_sample_size: int = 10000
    duration_hours: float = 24.0
    confidence_level: float = 0.95
    
    # Statistical settings
    statistical_test: StatisticalTest = StatisticalTest.T_TEST
    significance_threshold: float = 0.05
    effect_size_threshold: float = 0.1
    
    # Metrics to track
    primary_metric: str = "response_quality"
    secondary_metrics: List[str] = field(default_factory=lambda: [
        "response_time", "confidence_score", "user_satisfaction"
    ])
    
    # Quality control
    enable_early_stopping: bool = True
    early_stopping_threshold: float = 0.01
    outlier_detection: bool = True
    outlier_threshold: float = 2.0
    
    # Contextual factors
    contextual_features: List[str] = field(default_factory=list)
    stratification_features: List[str] = field(default_factory=list)

@dataclass
class ExperimentResult:
    """Results from an A/B test experiment"""
    experiment_id: str
    variant: str
    timestamp: float
    
    # Core metrics
    response_quality: float
    response_time: float
    confidence_score: float
    user_satisfaction: Optional[float] = None
    
    # Context information
    context: Dict[str, Any] = field(default_factory=dict)
    
    # Additional metrics
    additional_metrics: Dict[str, float] = field(default_factory=dict)

class TrafficSplitter:
    """Handles traffic splitting for A/B tests"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.assignment_history = {}
        self.current_assignments = defaultdict(int)
    
    def assign_variant(self, user_id: str = None, context: Dict[str, Any] = None) -> str:
        """Assign a variant to a user/request"""
        if self.config.split_strategy == SplitStrategy.RANDOM:
            return self._random_assignment()
        elif self.config.split_strategy == SplitStrategy.ROUND_ROBIN:
            return self._round_robin_assignment()
        elif self.config.split_strategy == SplitStrategy.WEIGHTED:
            return self._weighted_assignment()
        elif self.config.split_strategy == SplitStrategy.HASH_BASED:
            return self._hash_based_assignment(user_id or str(time.time()))
        elif self.config.split_strategy == SplitStrategy.CONTEXTUAL:
            return self._contextual_assignment(context or {})
        else:
            return self._random_assignment()
    
    def _random_assignment(self) -> str:
        """Random assignment to variants"""
        if not self.config.variants:
            return "control"
        
        if self.config.traffic_split:
            # Weighted random based on traffic split
            variants = list(self.config.traffic_split.keys())
            weights = list(self.config.traffic_split.values())
            return np.random.choice(variants, p=weights)
        else:
            # Equal probability
            return random.choice(self.config.variants)
    
    def _round_robin_assignment(self) -> str:
        """Round-robin assignment"""
        if not self.config.variants:
            return "control"
        
        # Find variant with least assignments
        min_assignments = min(self.current_assignments[v] for v in self.config.variants)
        candidates = [v for v in self.config.variants 
                     if self.current_assignments[v] == min_assignments]
        
        selected = random.choice(candidates)
        self.current_assignments[selected] += 1
        return selected
    
    def _weighted_assignment(self) -> str:
        """Weighted assignment based on traffic split"""
        if not self.config.traffic_split:
            return self._random_assignment()
        
        # Calculate current distribution
        total_assignments = sum(self.current_assignments.values())
        
        if total_assignments == 0:
            return self._random_assignment()
        
        # Find variant that's most under-represented
        target_ratios = self.config.traffic_split
        current_ratios = {
            v: self.current_assignments[v] / total_assignments 
            for v in self.config.variants
        }
        
        # Calculate how far each variant is from target
        deviations = {
            v: target_ratios.get(v, 1.0 / len(self.config.variants)) - current_ratios.get(v, 0)
            for v in self.config.variants
        }
        
        # Select variant with highest positive deviation
        selected = max(deviations, key=deviations.get)
        self.current_assignments[selected] += 1
        return selected
    
    def _hash_based_assignment(self, identifier: str) -> str:
        """Hash-based assignment for consistent user experience"""
        if not self.config.variants:
            return "control"
        
        # Hash identifier to get consistent assignment
        hash_value = hashlib.md5(identifier.encode()).hexdigest()
        hash_int = int(hash_value[:8], 16)
        
        if self.config.traffic_split:
            # Use cumulative distribution
            cumulative = 0
            normalized_hash = (hash_int % 10000) / 10000.0
            
            for variant, weight in self.config.traffic_split.items():
                cumulative += weight
                if normalized_hash <= cumulative:
                    return variant
            
            return list(self.config.traffic_split.keys())[-1]
        else:
            # Equal distribution
            variant_index = hash_int % len(self.config.variants)
            return self.config.variants[variant_index]
    
    def _contextual_assignment(self, context: Dict[str, Any]) -> str:
        """Context-aware assignment"""
        # Simple contextual assignment based on features
        if not self.config.contextual_features or not context:
            return self._random_assignment()
        
        # Score variants based on context
        variant_scores = {}
        
        for variant in self.config.variants:
            score = 0.0
            
            # Simple scoring based on context features
            for feature in self.config.contextual_features:
                if feature in context:
                    feature_value = context[feature]
                    
                    # Simplified context scoring
                    if isinstance(feature_value, (int, float)):
                        score += feature_value * 0.1
                    elif isinstance(feature_value, str):
                        score += len(feature_value) * 0.01
            
            variant_scores[variant] = score
        
        # Select variant with highest score (with some randomness)
        if variant_scores:
            scores = list(variant_scores.values())
            max_score = max(scores)
            
            # Add randomness to prevent deterministic assignment
            adjusted_scores = [s + random.random() * 0.1 for s in scores]
            best_idx = np.argmax(adjusted_scores)
            return list(variant_scores.keys())[best_idx]
        
        return self._random_assignment()

class StatisticalAnalyzer:
    """Performs statistical analysis of A/B test results"""
    
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
    
    def analyze_experiment(self, results: Dict[str, List[ExperimentResult]], 
                          primary_metric: str = "response_quality") -> Dict[str, Any]:
        """Analyze A/B test results"""
        if len(results) < 2:
            return {"error": "Need at least 2 variants to compare"}
        
        analysis = {
            "summary": self._calculate_summary_stats(results, primary_metric),
            "statistical_tests": {},
            "effect_sizes": {},
            "recommendations": []
        }
        
        # Pairwise comparisons
        variants = list(results.keys())
        
        for i in range(len(variants)):
            for j in range(i + 1, len(variants)):
                variant_a = variants[i]
                variant_b = variants[j]
                
                comparison_key = f"{variant_a}_vs_{variant_b}"
                
                # Extract metric values
                values_a = [getattr(r, primary_metric) for r in results[variant_a]]
                values_b = [getattr(r, primary_metric) for r in results[variant_b]]
                
                # Perform statistical tests
                test_results = self._perform_statistical_tests(values_a, values_b)
                analysis["statistical_tests"][comparison_key] = test_results
                
                # Calculate effect size
                effect_size = self._calculate_effect_size(values_a, values_b)
                analysis["effect_sizes"][comparison_key] = effect_size
        
        # Generate recommendations
        analysis["recommendations"] = self._generate_recommendations(analysis)
        
        return analysis
    
    def _calculate_summary_stats(self, results: Dict[str, List[ExperimentResult]], 
                                metric: str) -> Dict[str, Dict[str, float]]:
        """Calculate summary statistics for each variant"""
        summary = {}
        
        for variant, variant_results in results.items():
            values = [getattr(r, metric) for r in variant_results]
            
            if values:
                summary[variant] = {
                    "count": len(values),
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "median": np.median(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "q25": np.percentile(values, 25),
                    "q75": np.percentile(values, 75)
                }
            else:
                summary[variant] = {
                    "count": 0,
                    "mean": 0,
                    "std": 0,
                    "median": 0,
                    "min": 0,
                    "max": 0,
                    "q25": 0,
                    "q75": 0
                }
        
        return summary
    
    def _perform_statistical_tests(self, values_a: List[float], 
                                 values_b: List[float]) -> Dict[str, Any]:
        """Perform various statistical tests"""
        tests = {}
        
        if len(values_a) == 0 or len(values_b) == 0:
            return {"error": "Empty value lists"}
        
        # T-test
        try:
            t_stat, t_p_value = stats.ttest_ind(values_a, values_b)
            tests["t_test"] = {
                "statistic": float(t_stat),
                "p_value": float(t_p_value),
                "significant": t_p_value < self.alpha
            }
        except Exception as e:
            tests["t_test"] = {"error": str(e)}
        
        # Mann-Whitney U test (non-parametric)
        try:
            u_stat, u_p_value = stats.mannwhitneyu(values_a, values_b, alternative='two-sided')
            tests["mann_whitney"] = {
                "statistic": float(u_stat),
                "p_value": float(u_p_value),
                "significant": u_p_value < self.alpha
            }
        except Exception as e:
            tests["mann_whitney"] = {"error": str(e)}
        
        # Bootstrap confidence interval
        try:
            bootstrap_result = self._bootstrap_comparison(values_a, values_b)
            tests["bootstrap"] = bootstrap_result
        except Exception as e:
            tests["bootstrap"] = {"error": str(e)}
        
        return tests
    
    def _bootstrap_comparison(self, values_a: List[float], values_b: List[float], 
                            n_bootstrap: int = 1000) -> Dict[str, Any]:
        """Bootstrap comparison of means"""
        differences = []
        
        for _ in range(n_bootstrap):
            # Bootstrap samples
            sample_a = np.random.choice(values_a, len(values_a), replace=True)
            sample_b = np.random.choice(values_b, len(values_b), replace=True)
            
            # Calculate difference in means
            diff = np.mean(sample_a) - np.mean(sample_b)
            differences.append(diff)
        
        # Calculate confidence interval
        lower = np.percentile(differences, (self.alpha / 2) * 100)
        upper = np.percentile(differences, (1 - self.alpha / 2) * 100)
        
        return {
            "mean_difference": float(np.mean(differences)),
            "confidence_interval": [float(lower), float(upper)],
            "significant": not (lower <= 0 <= upper)
        }
    
    def _calculate_effect_size(self, values_a: List[float], values_b: List[float]) -> Dict[str, float]:
        """Calculate effect size measures"""
        if len(values_a) == 0 or len(values_b) == 0:
            return {"error": "Empty value lists"}
        
        mean_a = np.mean(values_a)
        mean_b = np.mean(values_b)
        std_a = np.std(values_a)
        std_b = np.std(values_b)
        
        # Cohen's d
        pooled_std = np.sqrt(((len(values_a) - 1) * std_a ** 2 + 
                             (len(values_b) - 1) * std_b ** 2) / 
                            (len(values_a) + len(values_b) - 2))
        
        cohens_d = (mean_a - mean_b) / pooled_std if pooled_std > 0 else 0
        
        # Relative difference
        relative_diff = (mean_a - mean_b) / mean_b if mean_b != 0 else 0
        
        return {
            "cohens_d": float(cohens_d),
            "relative_difference": float(relative_diff),
            "absolute_difference": float(mean_a - mean_b)
        }
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []
        
        # Check statistical significance
        significant_results = []
        for comparison, tests in analysis["statistical_tests"].items():
            if "t_test" in tests and tests["t_test"].get("significant", False):
                significant_results.append(comparison)
        
        if significant_results:
            recommendations.append(f"Statistically significant differences found in: {', '.join(significant_results)}")
        else:
            recommendations.append("No statistically significant differences detected")
        
        # Check effect sizes
        large_effects = []
        for comparison, effect in analysis["effect_sizes"].items():
            if abs(effect.get("cohens_d", 0)) > 0.8:  # Large effect size
                large_effects.append(comparison)
        
        if large_effects:
            recommendations.append(f"Large effect sizes detected in: {', '.join(large_effects)}")
        
        # Sample size recommendations
        summary = analysis.get("summary", {})
        small_samples = [v for v, stats in summary.items() if stats["count"] < 100]
        
        if small_samples:
            recommendations.append(f"Consider increasing sample size for: {', '.join(small_samples)}")
        
        return recommendations

class ABTestManager:
    """Main A/B testing manager"""
    
    def __init__(self, db_path: str = "ab_testing.db"):
        self.db_path = db_path
        self.active_experiments = {}
        self.traffic_splitters = {}
        self.analyzers = {}
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for storing experiment data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Experiments table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS experiments (
                id TEXT PRIMARY KEY,
                name TEXT,
                config TEXT,
                status TEXT,
                created_at REAL,
                started_at REAL,
                ended_at REAL
            )
        ''')
        
        # Results table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS experiment_results (
                id TEXT PRIMARY KEY,
                experiment_id TEXT,
                variant TEXT,
                timestamp REAL,
                response_quality REAL,
                response_time REAL,
                confidence_score REAL,
                user_satisfaction REAL,
                context TEXT,
                additional_metrics TEXT,
                FOREIGN KEY (experiment_id) REFERENCES experiments (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def create_experiment(self, config: ExperimentConfig) -> str:
        """Create a new A/B test experiment"""
        experiment_id = str(uuid.uuid4())
        
        # Store experiment configuration
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO experiments (id, name, config, status, created_at)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            experiment_id, 
            config.name, 
            json.dumps(config.__dict__, default=str),
            'created',
            time.time()
        ))
        
        conn.commit()
        conn.close()
        
        # Set up experiment components
        self.active_experiments[experiment_id] = config
        self.traffic_splitters[experiment_id] = TrafficSplitter(config)
        self.analyzers[experiment_id] = StatisticalAnalyzer(config.confidence_level)
        
        logger.info(f"Created experiment '{config.name}' with ID: {experiment_id}")
        return experiment_id
    
    def start_experiment(self, experiment_id: str):
        """Start an experiment"""
        if experiment_id not in self.active_experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        # Update database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE experiments 
            SET status = ?, started_at = ?
            WHERE id = ?
        ''', ('running', time.time(), experiment_id))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Started experiment: {experiment_id}")
    
    def stop_experiment(self, experiment_id: str):
        """Stop an experiment"""
        if experiment_id not in self.active_experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        # Update database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE experiments 
            SET status = ?, ended_at = ?
            WHERE id = ?
        ''', ('stopped', time.time(), experiment_id))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Stopped experiment: {experiment_id}")
    
    def assign_variant(self, experiment_id: str, user_id: str = None, 
                      context: Dict[str, Any] = None) -> str:
        """Assign a variant for a user/request"""
        if experiment_id not in self.active_experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        if experiment_id not in self.traffic_splitters:
            raise ValueError(f"Traffic splitter not found for experiment {experiment_id}")
        
        splitter = self.traffic_splitters[experiment_id]
        variant = splitter.assign_variant(user_id, context)
        
        return variant
    
    def record_result(self, experiment_id: str, result: ExperimentResult):
        """Record an experiment result"""
        if experiment_id not in self.active_experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        result_id = str(uuid.uuid4())
        
        cursor.execute('''
            INSERT INTO experiment_results (
                id, experiment_id, variant, timestamp,
                response_quality, response_time, confidence_score, 
                user_satisfaction, context, additional_metrics
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            result_id,
            experiment_id,
            result.variant,
            result.timestamp,
            result.response_quality,
            result.response_time,
            result.confidence_score,
            result.user_satisfaction,
            json.dumps(result.context),
            json.dumps(result.additional_metrics)
        ))
        
        conn.commit()
        conn.close()
        
        logger.debug(f"Recorded result for experiment {experiment_id}, variant {result.variant}")
    
    def analyze_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """Analyze experiment results"""
        if experiment_id not in self.active_experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        # Load results from database
        results = self._load_experiment_results(experiment_id)
        
        if not results:
            return {"error": "No results found for experiment"}
        
        # Group results by variant
        variant_results = defaultdict(list)
        for result in results:
            variant_results[result.variant].append(result)
        
        # Get analyzer and perform analysis
        analyzer = self.analyzers[experiment_id]
        config = self.active_experiments[experiment_id]
        
        analysis = analyzer.analyze_experiment(
            dict(variant_results), 
            config.primary_metric
        )
        
        # Add experiment metadata
        analysis["experiment_info"] = {
            "id": experiment_id,
            "name": config.name,
            "variants": config.variants,
            "primary_metric": config.primary_metric,
            "total_samples": len(results)
        }
        
        return analysis
    
    def _load_experiment_results(self, experiment_id: str) -> List[ExperimentResult]:
        """Load experiment results from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT variant, timestamp, response_quality, response_time,
                   confidence_score, user_satisfaction, context, additional_metrics
            FROM experiment_results
            WHERE experiment_id = ?
            ORDER BY timestamp
        ''', (experiment_id,))
        
        rows = cursor.fetchall()
        conn.close()
        
        results = []
        for row in rows:
            try:
                context = json.loads(row[6]) if row[6] else {}
                additional_metrics = json.loads(row[7]) if row[7] else {}
                
                result = ExperimentResult(
                    experiment_id=experiment_id,
                    variant=row[0],
                    timestamp=row[1],
                    response_quality=row[2],
                    response_time=row[3],
                    confidence_score=row[4],
                    user_satisfaction=row[5],
                    context=context,
                    additional_metrics=additional_metrics
                )
                results.append(result)
                
            except Exception as e:
                logger.warning(f"Error loading result: {e}")
                continue
        
        return results
    
    def get_experiment_status(self, experiment_id: str) -> Dict[str, Any]:
        """Get current status of an experiment"""
        if experiment_id not in self.active_experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        # Load from database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT name, status, created_at, started_at, ended_at
            FROM experiments
            WHERE id = ?
        ''', (experiment_id,))
        
        row = cursor.fetchone()
        
        # Count results per variant
        cursor.execute('''
            SELECT variant, COUNT(*) as count
            FROM experiment_results
            WHERE experiment_id = ?
            GROUP BY variant
        ''', (experiment_id,))
        
        variant_counts = dict(cursor.fetchall())
        conn.close()
        
        if not row:
            raise ValueError(f"Experiment {experiment_id} not found in database")
        
        config = self.active_experiments[experiment_id]
        
        status = {
            "experiment_id": experiment_id,
            "name": row[0],
            "status": row[1],
            "created_at": row[2],
            "started_at": row[3],
            "ended_at": row[4],
            "config": {
                "variants": config.variants,
                "traffic_split": config.traffic_split,
                "min_sample_size": config.min_sample_size,
                "primary_metric": config.primary_metric
            },
            "variant_counts": variant_counts,
            "total_samples": sum(variant_counts.values())
        }
        
        return status
    
    def list_experiments(self) -> List[Dict[str, Any]]:
        """List all experiments"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, name, status, created_at, started_at, ended_at
            FROM experiments
            ORDER BY created_at DESC
        ''')
        
        rows = cursor.fetchall()
        conn.close()
        
        experiments = []
        for row in rows:
            experiments.append({
                "id": row[0],
                "name": row[1],
                "status": row[2],
                "created_at": row[3],
                "started_at": row[4],
                "ended_at": row[5]
            })
        
        return experiments
    
    def generate_experiment_report(self, experiment_id: str) -> str:
        """Generate a comprehensive experiment report"""
        try:
            analysis = self.analyze_experiment(experiment_id)
            
            if "error" in analysis:
                return f"Error generating report: {analysis['error']}"
            
            report = []
            report.append(f"# A/B Test Report: {analysis['experiment_info']['name']}")
            report.append("=" * 60)
            
            # Experiment info
            info = analysis['experiment_info']
            report.append(f"\n## Experiment Details")
            report.append(f"ID: {info['id']}")
            report.append(f"Primary Metric: {info['primary_metric']}")
            report.append(f"Variants: {', '.join(info['variants'])}")
            report.append(f"Total Samples: {info['total_samples']}")
            
            # Summary statistics
            if 'summary' in analysis:
                report.append(f"\n## Summary Statistics")
                for variant, stats in analysis['summary'].items():
                    report.append(f"\n### {variant}")
                    report.append(f"  Count: {stats['count']}")
                    report.append(f"  Mean: {stats['mean']:.4f}")
                    report.append(f"  Std: {stats['std']:.4f}")
                    report.append(f"  Median: {stats['median']:.4f}")
            
            # Statistical tests
            if 'statistical_tests' in analysis:
                report.append(f"\n## Statistical Tests")
                for comparison, tests in analysis['statistical_tests'].items():
                    report.append(f"\n### {comparison}")
                    
                    if 't_test' in tests and 'error' not in tests['t_test']:
                        t_test = tests['t_test']
                        report.append(f"  T-test p-value: {t_test['p_value']:.6f}")
                        report.append(f"  Significant: {t_test['significant']}")
                    
                    if 'bootstrap' in tests and 'error' not in tests['bootstrap']:
                        bootstrap = tests['bootstrap']
                        ci = bootstrap['confidence_interval']
                        report.append(f"  Bootstrap 95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
            
            # Effect sizes
            if 'effect_sizes' in analysis:
                report.append(f"\n## Effect Sizes")
                for comparison, effect in analysis['effect_sizes'].items():
                    report.append(f"\n### {comparison}")
                    report.append(f"  Cohen's d: {effect.get('cohens_d', 0):.4f}")
                    report.append(f"  Relative difference: {effect.get('relative_difference', 0):.2%}")
            
            # Recommendations
            if 'recommendations' in analysis:
                report.append(f"\n## Recommendations")
                for i, recommendation in enumerate(analysis['recommendations'], 1):
                    report.append(f"{i}. {recommendation}")
            
            return "\n".join(report)
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return f"Error generating report: {str(e)}"

# Factory functions and utilities
def create_model_comparison_experiment(model_a: str, model_b: str, 
                                     experiment_name: str = None) -> ExperimentConfig:
    """Create a model comparison experiment configuration"""
    name = experiment_name or f"Compare {model_a} vs {model_b}"
    
    return ExperimentConfig(
        name=name,
        experiment_type=ExperimentType.MODEL_COMPARISON,
        variants=[model_a, model_b],
        traffic_split={model_a: 0.5, model_b: 0.5},
        min_sample_size=100,
        primary_metric="response_quality"
    )

def create_parameter_optimization_experiment(base_model: str, 
                                           parameter_configs: Dict[str, Dict],
                                           experiment_name: str = None) -> ExperimentConfig:
    """Create a parameter optimization experiment"""
    name = experiment_name or f"Optimize parameters for {base_model}"
    
    variants = list(parameter_configs.keys())
    traffic_split = {v: 1.0 / len(variants) for v in variants}
    
    return ExperimentConfig(
        name=name,
        experiment_type=ExperimentType.PARAMETER_OPTIMIZATION,
        variants=variants,
        traffic_split=traffic_split,
        min_sample_size=50,
        primary_metric="response_quality",
        secondary_metrics=["response_time", "confidence_score"]
    )

# Example usage
async def example_ab_testing():
    """Example A/B testing usage"""
    # Create A/B test manager
    manager = ABTestManager()
    
    # Create experiment configuration
    config = create_model_comparison_experiment(
        "gpt-oss2.5-coder:7b", 
        "gpt-oss",
        "Model Performance Comparison"
    )
    
    # Create and start experiment
    experiment_id = manager.create_experiment(config)
    manager.start_experiment(experiment_id)
    
    # Simulate experiment data
    for i in range(200):
        # Assign variant
        variant = manager.assign_variant(experiment_id, f"user_{i}")
        
        # Simulate different performance for each variant
        if variant == "gpt-oss2.5-coder:7b":
            quality = np.random.normal(0.8, 0.1)
            response_time = np.random.normal(3.0, 0.5)
            confidence = np.random.normal(0.85, 0.1)
        else:  # gpt-oss
            quality = np.random.normal(0.7, 0.15)
            response_time = np.random.normal(1.5, 0.3)
            confidence = np.random.normal(0.75, 0.15)
        
        # Create and record result
        result = ExperimentResult(
            experiment_id=experiment_id,
            variant=variant,
            timestamp=time.time(),
            response_quality=max(0, min(1, quality)),
            response_time=max(0.1, response_time),
            confidence_score=max(0, min(1, confidence))
        )
        
        manager.record_result(experiment_id, result)
    
    # Analyze results
    analysis = manager.analyze_experiment(experiment_id)
    
    # Generate report
    report = manager.generate_experiment_report(experiment_id)
    
    print("A/B Test Analysis:")
    print(json.dumps(analysis, indent=2, default=str))
    print("\nA/B Test Report:")
    print(report)
    
    return analysis

if __name__ == "__main__":
    # Run example
    import asyncio
    
    async def main():
        analysis = await example_ab_testing()
        return analysis
    
    # asyncio.run(main())