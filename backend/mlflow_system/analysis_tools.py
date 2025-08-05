"""
ML Experiment Analysis and Comparison Tools
Advanced analytics for experiment results and model performance
"""

import asyncio
import logging
import json
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import statistics

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType, Run, Experiment
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from .config import mlflow_config
from .metrics import mlflow_metrics
from .database import mlflow_database


logger = logging.getLogger(__name__)


@dataclass
class ExperimentComparison:
    """Results of experiment comparison"""
    
    experiment_ids: List[str]
    experiment_names: List[str]
    total_runs: int
    comparison_metrics: Dict[str, Any]
    statistical_tests: Dict[str, Any]
    recommendations: List[str]
    charts: Dict[str, str] = field(default_factory=dict)  # Chart file paths
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_ids": self.experiment_ids,
            "experiment_names": self.experiment_names,
            "total_runs": self.total_runs,
            "comparison_metrics": self.comparison_metrics,
            "statistical_tests": self.statistical_tests,
            "recommendations": self.recommendations,
            "charts": self.charts
        }


@dataclass 
class ModelPerformanceAnalysis:
    """Model performance analysis results"""
    
    model_name: str
    model_version: str
    performance_metrics: Dict[str, float]
    performance_trends: Dict[str, List[float]]
    anomalies: List[Dict[str, Any]]
    drift_detected: bool
    drift_score: float
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "performance_metrics": self.performance_metrics,
            "performance_trends": self.performance_trends,
            "anomalies": self.anomalies,
            "drift_detected": self.drift_detected,
            "drift_score": self.drift_score,
            "recommendations": self.recommendations
        }


class ExperimentAnalyzer:
    """Analyzes and compares ML experiments"""
    
    def __init__(self):
        self.client = MlflowClient(tracking_uri=mlflow_config.tracking_uri)
        self.charts_dir = Path("/opt/sutazaiapp/backend/mlflow_system/analysis_charts")
        self.charts_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    async def compare_experiments(self, experiment_ids: List[str], 
                                 metrics_to_compare: List[str] = None) -> ExperimentComparison:
        """Compare multiple experiments comprehensively"""
        try:
            logger.info(f"Comparing {len(experiment_ids)} experiments")
            
            # Get experiment information
            experiments = []
            experiment_names = []
            all_runs = []
            
            for exp_id in experiment_ids:
                try:
                    experiment = self.client.get_experiment(exp_id)
                    experiments.append(experiment)
                    experiment_names.append(experiment.name)
                    
                    # Get all runs for this experiment
                    runs = self.client.search_runs(
                        experiment_ids=[exp_id],
                        run_view_type=ViewType.ACTIVE_ONLY,
                        max_results=1000
                    )
                    all_runs.extend([(run, experiment.name) for run in runs])
                    
                except Exception as e:
                    logger.warning(f"Failed to get experiment {exp_id}: {e}")
            
            if not all_runs:
                raise ValueError("No runs found for the specified experiments")
            
            # Extract run data
            run_data = await self._extract_run_data(all_runs, metrics_to_compare)
            
            # Perform statistical analysis
            statistical_tests = await self._perform_statistical_tests(run_data)
            
            # Generate comparison metrics
            comparison_metrics = await self._generate_comparison_metrics(run_data)
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(
                run_data, statistical_tests, comparison_metrics
            )
            
            # Generate visualizations
            charts = await self._generate_comparison_charts(run_data, experiment_names)
            
            # Create comparison result
            comparison = ExperimentComparison(
                experiment_ids=experiment_ids,
                experiment_names=experiment_names,
                total_runs=len(all_runs),
                comparison_metrics=comparison_metrics,
                statistical_tests=statistical_tests,
                recommendations=recommendations,
                charts=charts
            )
            
            logger.info("Experiment comparison completed")
            return comparison
            
        except Exception as e:
            logger.error(f"Experiment comparison failed: {e}")
            raise
    
    async def _extract_run_data(self, runs_with_experiments: List[Tuple[Run, str]], 
                               metrics_to_compare: List[str] = None) -> pd.DataFrame:
        """Extract data from runs into a structured format"""
        data = []
        
        for run, experiment_name in runs_with_experiments:
            run_info = {
                'run_id': run.info.run_id,
                'experiment_name': experiment_name,
                'status': run.info.status,
                'start_time': run.info.start_time,
                'end_time': run.info.end_time,
                'duration': (run.info.end_time - run.info.start_time) / 1000 if run.info.end_time else None
            }
            
            # Add parameters
            for key, value in run.data.params.items():
                try:
                    # Try to convert to numeric
                    run_info[f'param_{key}'] = pd.to_numeric(value, errors='ignore')
                except:
                    run_info[f'param_{key}'] = value
            
            # Add metrics
            for key, value in run.data.metrics.items():
                if metrics_to_compare is None or key in metrics_to_compare:
                    run_info[f'metric_{key}'] = value
            
            # Add tags
            for key, value in run.data.tags.items():
                run_info[f'tag_{key}'] = value
            
            data.append(run_info)
        
        return pd.DataFrame(data)
    
    async def _perform_statistical_tests(self, run_data: pd.DataFrame) -> Dict[str, Any]:
        """Perform statistical tests on experiment results"""
        tests = {}
        
        try:
            # Get metric columns
            metric_cols = [col for col in run_data.columns if col.startswith('metric_')]
            
            if len(metric_cols) == 0:
                return tests
            
            # Group by experiment
            experiments = run_data['experiment_name'].unique()
            
            if len(experiments) < 2:
                return tests
            
            for metric_col in metric_cols:
                metric_name = metric_col.replace('metric_', '')
                tests[metric_name] = {}
                
                # Get data for each experiment
                experiment_data = []
                for exp_name in experiments:
                    exp_data = run_data[run_data['experiment_name'] == exp_name][metric_col].dropna()
                    if len(exp_data) > 0:
                        experiment_data.append(exp_data.values)
                
                if len(experiment_data) >= 2:
                    # Perform ANOVA if more than 2 groups
                    if len(experiment_data) > 2:
                        try:
                            f_stat, p_value = stats.f_oneway(*experiment_data)
                            tests[metric_name]['anova'] = {
                                'f_statistic': float(f_stat),
                                'p_value': float(p_value),
                                'significant': p_value < 0.05
                            }
                        except Exception as e:
                            logger.warning(f"ANOVA failed for {metric_name}: {e}")
                    
                    # Perform pairwise t-tests
                    pairwise_tests = {}
                    for i in range(len(experiments)):
                        for j in range(i + 1, len(experiments)):
                            if i < len(experiment_data) and j < len(experiment_data):
                                try:
                                    t_stat, p_value = stats.ttest_ind(
                                        experiment_data[i], experiment_data[j]
                                    )
                                    
                                    pair_key = f"{experiments[i]}_vs_{experiments[j]}"
                                    pairwise_tests[pair_key] = {
                                        't_statistic': float(t_stat),
                                        'p_value': float(p_value),
                                        'significant': p_value < 0.05,
                                        'mean_diff': float(np.mean(experiment_data[i]) - np.mean(experiment_data[j]))
                                    }
                                except Exception as e:
                                    logger.warning(f"T-test failed for {metric_name} pair {i},{j}: {e}")
                    
                    tests[metric_name]['pairwise_tests'] = pairwise_tests
                    
                    # Effect size (Cohen's d for two groups)
                    if len(experiment_data) == 2:
                        try:
                            pooled_std = np.sqrt(
                                ((len(experiment_data[0]) - 1) * np.var(experiment_data[0], ddof=1) +
                                 (len(experiment_data[1]) - 1) * np.var(experiment_data[1], ddof=1)) /
                                (len(experiment_data[0]) + len(experiment_data[1]) - 2)
                            )
                            
                            cohens_d = (np.mean(experiment_data[0]) - np.mean(experiment_data[1])) / pooled_std
                            tests[metric_name]['effect_size'] = {
                                'cohens_d': float(cohens_d),
                                'interpretation': self._interpret_effect_size(abs(cohens_d))
                            }
                        except Exception as e:
                            logger.warning(f"Effect size calculation failed for {metric_name}: {e}")
        
        except Exception as e:
            logger.error(f"Statistical tests failed: {e}")
        
        return tests
    
    def _interpret_effect_size(self, d: float) -> str:
        """Interpret Cohen's d effect size"""
        if d < 0.2:
            return "negligible"
        elif d < 0.5:
            return "small"
        elif d < 0.8:
            return "medium"
        else:
            return "large"
    
    async def _generate_comparison_metrics(self, run_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive comparison metrics"""
        metrics = {}
        
        try:
            # Basic statistics by experiment
            experiments = run_data['experiment_name'].unique()
            metrics['experiments'] = {}
            
            for exp_name in experiments:
                exp_data = run_data[run_data['experiment_name'] == exp_name]
                exp_metrics = {
                    'total_runs': len(exp_data),
                    'successful_runs': len(exp_data[exp_data['status'] == 'FINISHED']),
                    'failed_runs': len(exp_data[exp_data['status'] == 'FAILED']),
                    'success_rate': len(exp_data[exp_data['status'] == 'FINISHED']) / len(exp_data) if len(exp_data) > 0 else 0
                }
                
                # Duration statistics
                durations = exp_data['duration'].dropna()
                if len(durations) > 0:
                    exp_metrics['avg_duration_seconds'] = float(durations.mean())
                    exp_metrics['median_duration_seconds'] = float(durations.median())
                    exp_metrics['std_duration_seconds'] = float(durations.std())
                
                # Metric statistics
                metric_cols = [col for col in exp_data.columns if col.startswith('metric_')]
                exp_metrics['metrics'] = {}
                
                for metric_col in metric_cols:
                    metric_name = metric_col.replace('metric_', '')
                    metric_values = exp_data[metric_col].dropna()
                    
                    if len(metric_values) > 0:
                        exp_metrics['metrics'][metric_name] = {
                            'mean': float(metric_values.mean()),
                            'median': float(metric_values.median()),
                            'std': float(metric_values.std()),
                            'min': float(metric_values.min()),
                            'max': float(metric_values.max()),
                            'count': len(metric_values)
                        }
                
                metrics['experiments'][exp_name] = exp_metrics
            
            # Overall comparison
            metrics['overall'] = {
                'total_experiments': len(experiments),
                'total_runs': len(run_data),
                'overall_success_rate': len(run_data[run_data['status'] == 'FINISHED']) / len(run_data) if len(run_data) > 0 else 0
            }
            
            # Best performing runs
            metric_cols = [col for col in run_data.columns if col.startswith('metric_')]
            if metric_cols:
                # Assume the first metric is the primary one for ranking
                primary_metric = metric_cols[0]
                best_runs = run_data.nlargest(5, primary_metric)[['run_id', 'experiment_name', primary_metric]]
                metrics['best_runs'] = best_runs.to_dict('records')
            
        except Exception as e:
            logger.error(f"Comparison metrics generation failed: {e}")
        
        return metrics
    
    async def _generate_recommendations(self, run_data: pd.DataFrame, 
                                      statistical_tests: Dict[str, Any],
                                      comparison_metrics: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        try:
            # Check success rates
            for exp_name, exp_metrics in comparison_metrics.get('experiments', {}).items():
                success_rate = exp_metrics.get('success_rate', 0)
                if success_rate < 0.8:
                    recommendations.append(
                        f"Low success rate ({success_rate:.1%}) in experiment '{exp_name}' - "
                        "investigate failed runs and improve robustness"
                    )
            
            # Check statistical significance
            for metric_name, tests in statistical_tests.items():
                if 'anova' in tests and tests['anova'].get('significant', False):
                    recommendations.append(
                        f"Significant differences found in metric '{metric_name}' across experiments - "
                        "consider the best performing approach for future work"
                    )
                
                # Check effect sizes
                if 'effect_size' in tests:
                    effect_size = tests['effect_size']
                    if effect_size['interpretation'] in ['medium', 'large']:
                        recommendations.append(
                            f"Medium to large effect size ({effect_size['interpretation']}) detected for "
                            f"metric '{metric_name}' - this represents a meaningful practical difference"
                        )
            
            # Check consistency
            for exp_name, exp_metrics in comparison_metrics.get('experiments', {}).items():
                for metric_name, metric_stats in exp_metrics.get('metrics', {}).items():
                    if metric_stats.get('std', 0) > metric_stats.get('mean', 0) * 0.3:
                        recommendations.append(
                            f"High variability in metric '{metric_name}' for experiment '{exp_name}' - "
                            "consider improving experimental setup for more consistent results"
                        )
            
            # Duration recommendations
            duration_stats = []
            for exp_name, exp_metrics in comparison_metrics.get('experiments', {}).items():
                if 'avg_duration_seconds' in exp_metrics:
                    duration_stats.append((exp_name, exp_metrics['avg_duration_seconds']))
            
            if len(duration_stats) > 1:
                duration_stats.sort(key=lambda x: x[1])
                fastest_exp = duration_stats[0]
                slowest_exp = duration_stats[-1]
                
                if slowest_exp[1] > fastest_exp[1] * 2:
                    recommendations.append(
                        f"Experiment '{fastest_exp[0]}' is significantly faster than '{slowest_exp[0]}' - "
                        "consider optimizing slower experiments or adopting efficient approaches"
                    )
            
            # General recommendations
            if len(recommendations) == 0:
                recommendations.append(
                    "No major issues detected. Continue monitoring and consider expanding "
                    "the analysis with more experiments or longer time periods."
                )
            
        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
            recommendations.append("Analysis completed but recommendation generation encountered issues.")
        
        return recommendations
    
    async def _generate_comparison_charts(self, run_data: pd.DataFrame, 
                                        experiment_names: List[str]) -> Dict[str, str]:
        """Generate visualization charts for comparison"""
        charts = {}
        
        try:
            timestamp = int(time.time())
            
            # 1. Metric comparison box plots
            metric_cols = [col for col in run_data.columns if col.startswith('metric_')]
            
            if metric_cols:
                for metric_col in metric_cols[:5]:  # Limit to first 5 metrics
                    metric_name = metric_col.replace('metric_', '')
                    
                    plt.figure(figsize=(10, 6))
                    
                    # Prepare data for boxplot
                    plot_data = []
                    plot_labels = []
                    
                    for exp_name in experiment_names:
                        exp_data = run_data[run_data['experiment_name'] == exp_name][metric_col].dropna()
                        if len(exp_data) > 0:
                            plot_data.append(exp_data.values)
                            plot_labels.append(exp_name)
                    
                    if plot_data:
                        plt.boxplot(plot_data, labels=plot_labels)
                        plt.title(f'Distribution of {metric_name} Across Experiments')
                        plt.ylabel(metric_name)
                        plt.xticks(rotation=45, ha='right')
                        plt.tight_layout()
                        
                        chart_file = self.charts_dir / f"metric_comparison_{metric_name}_{timestamp}.png"
                        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
                        plt.close()
                        
                        charts[f"metric_{metric_name}_boxplot"] = str(chart_file)
            
            # 2. Success rate comparison
            success_rates = []
            exp_names = []
            
            for exp_name in experiment_names:
                exp_data = run_data[run_data['experiment_name'] == exp_name]
                if len(exp_data) > 0:
                    success_rate = len(exp_data[exp_data['status'] == 'FINISHED']) / len(exp_data)
                    success_rates.append(success_rate * 100)
                    exp_names.append(exp_name)
            
            if success_rates:
                plt.figure(figsize=(10, 6))
                bars = plt.bar(exp_names, success_rates)
                plt.title('Success Rate by Experiment')
                plt.ylabel('Success Rate (%)')
                plt.ylim(0, 100)
                
                # Add value labels on bars
                for bar, rate in zip(bars, success_rates):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                           f'{rate:.1f}%', ha='center', va='bottom')
                
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                
                chart_file = self.charts_dir / f"success_rates_{timestamp}.png"
                plt.savefig(chart_file, dpi=300, bbox_inches='tight')
                plt.close()
                
                charts["success_rates"] = str(chart_file)
            
            # 3. Duration comparison
            durations_by_exp = {}
            for exp_name in experiment_names:
                exp_data = run_data[run_data['experiment_name'] == exp_name]
                durations = exp_data['duration'].dropna()
                if len(durations) > 0:
                    durations_by_exp[exp_name] = durations.values / 60  # Convert to minutes
            
            if durations_by_exp:
                plt.figure(figsize=(10, 6))
                
                plot_data = list(durations_by_exp.values())
                plot_labels = list(durations_by_exp.keys())
                
                plt.boxplot(plot_data, labels=plot_labels)
                plt.title('Run Duration Distribution by Experiment')
                plt.ylabel('Duration (minutes)')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                
                chart_file = self.charts_dir / f"duration_comparison_{timestamp}.png"
                plt.savefig(chart_file, dpi=300, bbox_inches='tight')
                plt.close()
                
                charts["duration_comparison"] = str(chart_file)
            
            # 4. Correlation matrix for metrics
            if len(metric_cols) > 1:
                metric_data = run_data[metric_cols].select_dtypes(include=[np.number])
                
                if not metric_data.empty and len(metric_data.columns) > 1:
                    correlation_matrix = metric_data.corr()
                    
                    plt.figure(figsize=(10, 8))
                    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                              square=True, fmt='.2f')
                    plt.title('Metric Correlation Matrix')
                    plt.tight_layout()
                    
                    chart_file = self.charts_dir / f"correlation_matrix_{timestamp}.png"
                    plt.savefig(chart_file, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    charts["correlation_matrix"] = str(chart_file)
            
            logger.info(f"Generated {len(charts)} comparison charts")
            
        except Exception as e:
            logger.error(f"Chart generation failed: {e}")
        
        return charts
    
    async def analyze_model_performance(self, model_name: str, 
                                      time_window_days: int = 30) -> ModelPerformanceAnalysis:
        """Analyze model performance over time"""
        try:
            logger.info(f"Analyzing performance for model {model_name}")
            
            # Get model versions and runs
            model_versions = await self._get_model_versions(model_name)
            
            if not model_versions:
                raise ValueError(f"No versions found for model {model_name}")
            
            # Get recent runs for the model
            end_date = datetime.now()
            start_date = end_date - timedelta(days=time_window_days)
            
            runs_data = await self._get_model_runs_data(model_name, start_date, end_date)
            
            if runs_data.empty:
                raise ValueError(f"No recent runs found for model {model_name}")
            
            # Analyze performance metrics
            performance_metrics = await self._analyze_performance_metrics(runs_data)
            
            # Detect performance trends
            performance_trends = await self._detect_performance_trends(runs_data)
            
            # Detect anomalies
            anomalies = await self._detect_performance_anomalies(runs_data)
            
            # Detect model drift
            drift_detected, drift_score = await self._detect_model_drift(runs_data)
            
            # Generate recommendations
            recommendations = await self._generate_model_recommendations(
                performance_metrics, performance_trends, anomalies, drift_detected
            )
            
            # Get latest version
            latest_version = max(model_versions, key=lambda v: v.get('creation_timestamp', 0))
            
            analysis = ModelPerformanceAnalysis(
                model_name=model_name,
                model_version=latest_version.get('version', 'unknown'),
                performance_metrics=performance_metrics,
                performance_trends=performance_trends,
                anomalies=anomalies,
                drift_detected=drift_detected,
                drift_score=drift_score,
                recommendations=recommendations
            )
            
            logger.info(f"Model performance analysis completed for {model_name}")
            return analysis
            
        except Exception as e:
            logger.error(f"Model performance analysis failed: {e}")
            raise
    
    async def _get_model_versions(self, model_name: str) -> List[Dict[str, Any]]:
        """Get all versions of a model"""
        try:
            # This would integrate with MLflow Model Registry
            # For now, return mock data
            return [
                {
                    "version": "1",
                    "creation_timestamp": int(time.time()) - 86400,
                    "stage": "Production"
                }
            ]
        except Exception as e:
            logger.error(f"Failed to get model versions: {e}")
            return []
    
    async def _get_model_runs_data(self, model_name: str, 
                                  start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get runs data for a specific model within time window"""
        try:
            # Search for runs with model name tag
            runs = self.client.search_runs(
                experiment_ids=[],  # Search across all experiments
                filter_string=f"tags.model_name = '{model_name}'",
                run_view_type=ViewType.ACTIVE_ONLY,
                max_results=1000
            )
            
            # Filter by time window and convert to DataFrame
            filtered_runs = []
            
            for run in runs:
                run_start = datetime.fromtimestamp(run.info.start_time / 1000)
                if start_date <= run_start <= end_date:
                    run_data = {
                        'run_id': run.info.run_id,
                        'start_time': run_start,
                        'status': run.info.status
                    }
                    
                    # Add metrics
                    for key, value in run.data.metrics.items():
                        run_data[key] = value
                    
                    # Add parameters
                    for key, value in run.data.params.items():
                        run_data[f'param_{key}'] = value
                    
                    filtered_runs.append(run_data)
            
            return pd.DataFrame(filtered_runs)
            
        except Exception as e:
            logger.error(f"Failed to get model runs data: {e}")
            return pd.DataFrame()
    
    async def _analyze_performance_metrics(self, runs_data: pd.DataFrame) -> Dict[str, float]:
        """Analyze current performance metrics"""
        metrics = {}
        
        try:
            # Get numeric columns (metrics)
            numeric_cols = runs_data.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                if col not in ['start_time']:  # Exclude timestamp columns
                    values = runs_data[col].dropna()
                    if len(values) > 0:
                        metrics[f"{col}_mean"] = float(values.mean())
                        metrics[f"{col}_std"] = float(values.std())
                        metrics[f"{col}_latest"] = float(values.iloc[-1]) if len(values) > 0 else 0.0
            
            # Success rate
            if 'status' in runs_data.columns:
                success_rate = len(runs_data[runs_data['status'] == 'FINISHED']) / len(runs_data)
                metrics['success_rate'] = success_rate
            
        except Exception as e:
            logger.error(f"Performance metrics analysis failed: {e}")
        
        return metrics
    
    async def _detect_performance_trends(self, runs_data: pd.DataFrame) -> Dict[str, List[float]]:
        """Detect trends in performance metrics over time"""
        trends = {}
        
        try:
            if runs_data.empty:
                return trends
            
            # Sort by time
            runs_data_sorted = runs_data.sort_values('start_time')
            
            # Get numeric columns
            numeric_cols = runs_data_sorted.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                if col not in ['start_time']:
                    values = runs_data_sorted[col].dropna()
                    if len(values) > 1:
                        trends[col] = values.tolist()
            
        except Exception as e:
            logger.error(f"Trend detection failed: {e}")
        
        return trends
    
    async def _detect_performance_anomalies(self, runs_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect performance anomalies using statistical methods"""
        anomalies = []
        
        try:
            if runs_data.empty:
                return anomalies
            
            # Get numeric columns
            numeric_cols = runs_data.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                if col not in ['start_time']:
                    values = runs_data[col].dropna()
                    
                    if len(values) > 3:  # Need minimum data points
                        # Use IQR method for anomaly detection
                        Q1 = values.quantile(0.25)
                        Q3 = values.quantile(0.75)
                        IQR = Q3 - Q1
                        
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        
                        # Find anomalies
                        outliers = values[(values < lower_bound) | (values > upper_bound)]
                        
                        for idx, value in outliers.items():
                            run_id = runs_data.loc[idx, 'run_id'] if 'run_id' in runs_data.columns else idx
                            
                            anomaly = {
                                'run_id': run_id,
                                'metric': col,
                                'value': float(value),
                                'expected_range': [float(lower_bound), float(upper_bound)],
                                'severity': 'high' if abs(value - values.median()) > 2 * values.std() else 'medium'
                            }
                            anomalies.append(anomaly)
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
        
        return anomalies
    
    async def _detect_model_drift(self, runs_data: pd.DataFrame) -> Tuple[bool, float]:
        """Detect model drift using statistical tests"""
        try:
            if runs_data.empty or len(runs_data) < 10:
                return False, 0.0
            
            # Split data into two halves (early vs recent)
            split_point = len(runs_data) // 2
            early_data = runs_data.iloc[:split_point]
            recent_data = runs_data.iloc[split_point:]
            
            # Look for drift in primary metrics (accuracy, loss, etc.)
            primary_metrics = ['accuracy', 'loss', 'f1_score', 'precision', 'recall']
            drift_scores = []
            
            for metric in primary_metrics:
                if metric in runs_data.columns:
                    early_values = early_data[metric].dropna()
                    recent_values = recent_data[metric].dropna()
                    
                    if len(early_values) > 2 and len(recent_values) > 2:
                        # Perform Kolmogorov-Smirnov test
                        ks_stat, p_value = stats.ks_2samp(early_values, recent_values)
                        
                        # Consider drift if p < 0.05
                        if p_value < 0.05:
                            drift_scores.append(ks_stat)
            
            if drift_scores:
                avg_drift_score = np.mean(drift_scores)
                drift_detected = avg_drift_score > 0.2  # Threshold for significant drift
                return drift_detected, float(avg_drift_score)
            
            return False, 0.0
            
        except Exception as e:
            logger.error(f"Drift detection failed: {e}")
            return False, 0.0
    
    async def _generate_model_recommendations(self, performance_metrics: Dict[str, float],
                                            performance_trends: Dict[str, List[float]],
                                            anomalies: List[Dict[str, Any]],
                                            drift_detected: bool) -> List[str]:
        """Generate recommendations for model performance"""
        recommendations = []
        
        try:
            # Check success rate
            success_rate = performance_metrics.get('success_rate', 1.0)
            if success_rate < 0.9:
                recommendations.append(
                    f"Low success rate ({success_rate:.1%}) detected. "
                    "Investigate failed runs and improve model robustness."
                )
            
            # Check for drift
            if drift_detected:
                recommendations.append(
                    "Model drift detected. Consider retraining the model with recent data "
                    "or implementing online learning techniques."
                )
            
            # Check for anomalies
            high_severity_anomalies = [a for a in anomalies if a.get('severity') == 'high']
            if high_severity_anomalies:
                recommendations.append(
                    f"Found {len(high_severity_anomalies)} high-severity performance anomalies. "
                    "Review the affected runs and investigate potential causes."
                )
            
            # Check trends
            for metric, trend in performance_trends.items():
                if len(trend) > 5:
                    # Simple trend analysis - check if last 5 values are consistently decreasing
                    recent_trend = trend[-5:]
                    if all(recent_trend[i] >= recent_trend[i+1] for i in range(len(recent_trend)-1)):
                        recommendations.append(
                            f"Declining trend detected in {metric}. "
                            "Monitor closely and consider model updates."
                        )
            
            # General recommendations
            if not recommendations:
                recommendations.append(
                    "Model performance appears stable. Continue regular monitoring "
                    "and consider periodic retraining to maintain performance."
                )
            
        except Exception as e:
            logger.error(f"Model recommendation generation failed: {e}")
            recommendations.append(
                "Performance analysis completed but recommendation generation encountered issues."
            )
        
        return recommendations


# Global analyzer instance
experiment_analyzer = ExperimentAnalyzer()