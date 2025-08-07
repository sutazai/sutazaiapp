"""
Knowledge Graph Reasoning Engine
===============================

Provides advanced reasoning capabilities for the SutazAI knowledge graph.
Implements inference rules, pattern matching, capability reasoning, and
intelligent recommendations for system optimization and decision support.

Features:
- Rule-based inference engine
- Graph pattern matching
- Capability-based reasoning
- Performance optimization recommendations
- System health insights
- Agent orchestration suggestions
- Anomaly detection
- Predictive analytics
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import statistics

from .query_engine import QueryEngine, QueryResult
from .schema import NodeType, RelationshipType


@dataclass
class ReasoningRule:
    """Represents an inference rule"""
    rule_id: str
    name: str
    description: str
    condition_cypher: str
    action_cypher: str
    priority: int = 1
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Inference:
    """Represents an inference result"""
    inference_id: str
    rule_id: str
    confidence: float
    description: str
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    applied: bool = False


@dataclass
class Recommendation:
    """Represents a system recommendation"""
    recommendation_id: str
    type: str  # performance, security, reliability, optimization
    title: str
    description: str
    priority: str  # high, medium, low
    confidence: float
    actions: List[Dict[str, Any]]
    evidence: List[Dict[str, Any]]
    impact_score: float
    timestamp: datetime = field(default_factory=datetime.utcnow)


class RuleEngine:
    """Core rule-based inference engine"""
    
    def __init__(self, query_engine: QueryEngine):
        self.query_engine = query_engine
        self.rules = {}
        self.logger = logging.getLogger("rule_engine")
        
        # Initialize built-in rules
        self._initialize_builtin_rules()
    
    def _initialize_builtin_rules(self):
        """Initialize built-in reasoning rules"""
        
        # Agent capability inference
        self.add_rule(ReasoningRule(
            rule_id="infer_composite_capabilities",
            name="Infer Composite Capabilities",
            description="Infer agents that have combinations of capabilities",
            condition_cypher="""
            MATCH (a:Agent)-[:HAS_CAPABILITY]->(c:Capability)
            WITH a, collect(c.name) as capabilities
            WHERE size(capabilities) >= 2
            RETURN a, capabilities
            """,
            action_cypher="""
            // This would create virtual composite capability nodes
            MATCH (a:Agent) WHERE a.id = $agent_id
            SET a.composite_capabilities = $capabilities
            """,
            priority=2
        ))
        
        # Service dependency chain inference
        self.add_rule(ReasoningRule(
            rule_id="infer_dependency_chains",
            name="Infer Dependency Chains",
            description="Identify critical service dependency chains",
            condition_cypher="""
            MATCH path = (s:Service)-[:DEPENDS_ON*2..5]->(critical:Service)
            WHERE critical.service_type = 'database'
            RETURN s, critical, length(path) as chain_length
            ORDER BY chain_length DESC
            """,
            action_cypher="""
            MATCH (s:Service {id: $service_id})
            SET s.dependency_chain_length = $chain_length,
                s.critical_dependency = true
            """,
            priority=3
        ))
        
        # Health status propagation
        self.add_rule(ReasoningRule(
            rule_id="propagate_health_issues",
            name="Propagate Health Issues",
            description="Propagate health issues through dependency chains",
            condition_cypher="""
            MATCH (unhealthy:Agent {health_status: 'critical'})-[:ORCHESTRATES]->(s:Service)
            MATCH (s)-[:DEPENDS_ON]->(dependent:Service)
            WHERE NOT dependent.health_status = 'warning'
            RETURN unhealthy, s, dependent
            """,
            action_cypher="""
            MATCH (s:Service {id: $service_id})
            SET s.health_status = 'warning',
                s.health_reason = 'upstream_dependency_critical'
            """,
            priority=5
        ))
        
        # Capability gap detection
        self.add_rule(ReasoningRule(
            rule_id="detect_capability_gaps",
            name="Detect Capability Gaps", 
            description="Detect workflows without adequate agent coverage",
            condition_cypher="""
            MATCH (w:Workflow)-[:REQUIRES]->(c:Capability)
            WHERE NOT EXISTS {
                MATCH (a:Agent)-[:HAS_CAPABILITY]->(c)
                WHERE a.health_status = 'healthy'
            }
            RETURN w, c
            """,
            action_cypher="""
            MATCH (w:Workflow {id: $workflow_id})
            SET w.capability_gap = true,
                w.missing_capabilities = coalesce(w.missing_capabilities, []) + [$capability]
            """,
            priority=4
        ))
    
    def add_rule(self, rule: ReasoningRule):
        """Add a reasoning rule"""
        self.rules[rule.rule_id] = rule
        self.logger.debug(f"Added rule: {rule.name}")
    
    def remove_rule(self, rule_id: str):
        """Remove a reasoning rule"""
        if rule_id in self.rules:
            del self.rules[rule_id]
            self.logger.debug(f"Removed rule: {rule_id}")
    
    def enable_rule(self, rule_id: str):
        """Enable a rule"""
        if rule_id in self.rules:
            self.rules[rule_id].enabled = True
    
    def disable_rule(self, rule_id: str):
        """Disable a rule"""
        if rule_id in self.rules:
            self.rules[rule_id].enabled = False
    
    async def execute_rules(self) -> List[Inference]:
        """Execute all enabled rules and generate inferences"""
        inferences = []
        
        # Sort rules by priority (higher priority first)
        sorted_rules = sorted(
            [rule for rule in self.rules.values() if rule.enabled],
            key=lambda r: r.priority,
            reverse=True
        )
        
        for rule in sorted_rules:
            try:
                rule_inferences = await self._execute_rule(rule)
                inferences.extend(rule_inferences)
                
            except Exception as e:
                self.logger.error(f"Error executing rule {rule.rule_id}: {e}")
        
        return inferences
    
    async def _execute_rule(self, rule: ReasoningRule) -> List[Inference]:
        """Execute a single rule"""
        inferences = []
        
        # Execute condition query
        results = await self.query_engine.neo4j_manager.execute_cypher(
            rule.condition_cypher
        )
        
        for record in results:
            try:
                # Create inference
                inference = Inference(
                    inference_id=f"{rule.rule_id}_{len(inferences)}_{int(datetime.now().timestamp())}",
                    rule_id=rule.rule_id,
                    confidence=self._calculate_confidence(rule, record),
                    description=f"Rule '{rule.name}' triggered",
                    data=dict(record)
                )
                
                inferences.append(inference)
                
                # Optionally apply the action (if configured to do so)
                if rule.metadata.get("auto_apply", False):
                    await self._apply_inference(rule, inference)
                
            except Exception as e:
                self.logger.error(f"Error processing rule result: {e}")
        
        return inferences
    
    def _calculate_confidence(self, rule: ReasoningRule, record: Dict[str, Any]) -> float:
        """Calculate confidence score for an inference"""
        # Simple confidence calculation - could be more sophisticated
        base_confidence = 0.8
        
        # Adjust based on rule priority
        priority_factor = min(rule.priority / 5.0, 1.0)
        
        # Adjust based on data quality indicators
        quality_factor = 1.0
        
        return min(base_confidence * priority_factor * quality_factor, 1.0)
    
    async def _apply_inference(self, rule: ReasoningRule, inference: Inference):
        """Apply an inference by executing the rule's action"""
        try:
            # Extract parameters from inference data
            parameters = {}
            for key, value in inference.data.items():
                if isinstance(value, dict) and 'id' in value:
                    parameters[f"{key}_id"] = value['id']
                else:
                    parameters[key] = value
            
            # Execute action query
            await self.query_engine.neo4j_manager.execute_cypher(
                rule.action_cypher,
                parameters
            )
            
            inference.applied = True
            self.logger.debug(f"Applied inference: {inference.inference_id}")
            
        except Exception as e:
            self.logger.error(f"Error applying inference: {e}")


class CapabilityReasoner:
    """Specialized reasoning for agent capabilities"""
    
    def __init__(self, query_engine: QueryEngine):
        self.query_engine = query_engine
        self.logger = logging.getLogger("capability_reasoner")
    
    async def find_optimal_agent_combinations(self, required_capabilities: List[str]) -> List[Dict[str, Any]]:
        """Find optimal combinations of agents for required capabilities"""
        
        # Find all possible agent combinations that cover the required capabilities
        cypher = """
        MATCH (a:Agent)-[:HAS_CAPABILITY]->(c:Capability)
        WHERE c.name IN $capabilities
        AND a.health_status IN ['healthy', 'warning']
        WITH a, collect(c.name) as agent_caps
        RETURN a, agent_caps, 
               size([cap IN $capabilities WHERE cap IN agent_caps]) as coverage_count
        ORDER BY coverage_count DESC
        """
        
        results = await self.query_engine.neo4j_manager.execute_cypher(
            cypher, {"capabilities": required_capabilities}
        )
        
        # Generate combinations
        combinations = self._generate_agent_combinations(results, required_capabilities)
        
        # Score and rank combinations
        scored_combinations = []
        for combo in combinations:
            score = self._score_agent_combination(combo, required_capabilities)
            scored_combinations.append({
                "agents": combo,
                "score": score,
                "total_capabilities": len(set().union(*[a["agent_caps"] for a in combo])),
                "redundancy": self._calculate_redundancy(combo),
                "estimated_performance": self._estimate_performance(combo)
            })
        
        return sorted(scored_combinations, key=lambda x: x["score"], reverse=True)[:5]
    
    def _generate_agent_combinations(self, agent_data: List[Dict], required_capabilities: List[str]) -> List[List[Dict]]:
        """Generate valid agent combinations"""
        combinations = []
        
        # Simple greedy approach - could be more sophisticated
        remaining_caps = set(required_capabilities)
        current_combo = []
        
        for agent_record in agent_data:
            agent_caps = set(agent_record.get("agent_caps", []))
            
            if remaining_caps.intersection(agent_caps):
                current_combo.append(agent_record)
                remaining_caps -= agent_caps
                
                if not remaining_caps:
                    combinations.append(current_combo.copy())
                    break
        
        return combinations
    
    def _score_agent_combination(self, combination: List[Dict], required_capabilities: List[str]) -> float:
        """Score an agent combination"""
        if not combination:
            return 0.0
        
        # Coverage score
        all_caps = set().union(*[set(a.get("agent_caps", [])) for a in combination])
        coverage_score = len(all_caps.intersection(required_capabilities)) / len(required_capabilities)
        
        # Efficiency score (fewer agents is better)
        efficiency_score = 1.0 / len(combination)
        
        # Health score
        health_scores = []
        for agent in combination:
            agent_data = agent.get("a", {})
            health = agent_data.get("health_status", "unknown")
            health_score = {"healthy": 1.0, "warning": 0.7, "critical": 0.3}.get(health, 0.1)
            health_scores.append(health_score)
        
        avg_health_score = statistics.mean(health_scores) if health_scores else 0.0
        
        # Combined score
        return (coverage_score * 0.5) + (efficiency_score * 0.3) + (avg_health_score * 0.2)
    
    def _calculate_redundancy(self, combination: List[Dict]) -> float:
        """Calculate capability redundancy in the combination"""
        if len(combination) <= 1:
            return 0.0
        
        all_caps = []
        for agent in combination:
            all_caps.extend(agent.get("agent_caps", []))
        
        cap_counts = Counter(all_caps)
        redundant_caps = sum(count - 1 for count in cap_counts.values() if count > 1)
        total_caps = len(all_caps)
        
        return redundant_caps / total_caps if total_caps > 0 else 0.0
    
    def _estimate_performance(self, combination: List[Dict]) -> float:
        """Estimate performance of the agent combination"""
        performance_scores = []
        
        for agent in combination:
            agent_data = agent.get("a", {})
            performance_metrics = agent_data.get("performance_metrics", {})
            
            if isinstance(performance_metrics, str):
                # Parse JSON string if needed
                try:
                    import json
                    performance_metrics = json.loads(performance_metrics)
                except:
                    performance_metrics = {}
            
            success_rate = performance_metrics.get("success_rate", 80.0)
            performance_scores.append(success_rate / 100.0)
        
        return statistics.mean(performance_scores) if performance_scores else 0.5
    
    async def identify_capability_bottlenecks(self) -> List[Dict[str, Any]]:
        """Identify capability bottlenecks in the system"""
        
        cypher = """
        MATCH (c:Capability)<-[:HAS_CAPABILITY]-(a:Agent)
        WHERE a.health_status IN ['healthy', 'warning']
        WITH c, count(a) as agent_count, 
             collect(a.health_status) as health_statuses
        WITH c, agent_count,
             size([h IN health_statuses WHERE h = 'healthy']) as healthy_count
        RETURN c.name as capability,
               c.capability_type as category,
               agent_count,
               healthy_count,
               CASE 
                 WHEN healthy_count = 0 THEN 'critical'
                 WHEN healthy_count = 1 THEN 'high_risk'
                 WHEN healthy_count <= 2 THEN 'medium_risk'
                 ELSE 'low_risk'
               END as risk_level
        ORDER BY healthy_count ASC, agent_count ASC
        """
        
        results = await self.query_engine.neo4j_manager.execute_cypher(cypher)
        
        bottlenecks = []
        for record in results:
            if record.get("risk_level") in ["critical", "high_risk"]:
                bottlenecks.append({
                    "capability": record["capability"],
                    "category": record["category"],
                    "total_agents": record["agent_count"],
                    "healthy_agents": record["healthy_count"],
                    "risk_level": record["risk_level"],
                    "recommendations": self._generate_bottleneck_recommendations(record)
                })
        
        return bottlenecks
    
    def _generate_bottleneck_recommendations(self, bottleneck_data: Dict) -> List[str]:
        """Generate recommendations for capability bottlenecks"""
        recommendations = []
        risk_level = bottleneck_data["risk_level"]
        capability = bottleneck_data["capability"]
        
        if risk_level == "critical":
            recommendations.extend([
                f"URGENT: Deploy new agents with {capability} capability",
                f"Consider cross-training existing agents for {capability}",
                "Implement failover procedures for this capability"
            ])
        elif risk_level == "high_risk":
            recommendations.extend([
                f"Scale up agents with {capability} capability",
                "Monitor health of existing capable agents closely",
                "Prepare backup agents with this capability"
            ])
        
        return recommendations


class SystemOptimizer:
    """Provides system optimization recommendations"""
    
    def __init__(self, query_engine: QueryEngine):
        self.query_engine = query_engine
        self.logger = logging.getLogger("system_optimizer")
    
    async def analyze_system_performance(self) -> List[Recommendation]:
        """Analyze system performance and generate recommendations"""
        recommendations = []
        
        # Agent performance analysis
        agent_recommendations = await self._analyze_agent_performance()
        recommendations.extend(agent_recommendations)
        
        # Service dependency analysis
        dependency_recommendations = await self._analyze_service_dependencies()
        recommendations.extend(dependency_recommendations)
        
        # Resource utilization analysis
        resource_recommendations = await self._analyze_resource_utilization()
        recommendations.extend(resource_recommendations)
        
        # Sort by priority and impact
        recommendations.sort(key=lambda r: (
            {"high": 3, "medium": 2, "low": 1}[r.priority],
            r.impact_score
        ), reverse=True)
        
        return recommendations
    
    async def _analyze_agent_performance(self) -> List[Recommendation]:
        """Analyze agent performance patterns"""
        recommendations = []
        
        # Find underperforming agents
        cypher = """
        MATCH (a:Agent)
        WHERE a.performance_metrics IS NOT NULL
        WITH a, 
             CASE WHEN a.performance_metrics.success_rate < 70 THEN 'underperforming'
                  WHEN a.performance_metrics.success_rate < 85 THEN 'needs_attention'
                  ELSE 'good' END as performance_category
        WHERE performance_category IN ['underperforming', 'needs_attention']
        RETURN a, performance_category
        """
        
        results = await self.query_engine.neo4j_manager.execute_cypher(cypher)
        
        for record in results:
            agent = record["a"]
            category = record["performance_category"]
            
            if category == "underperforming":
                recommendations.append(Recommendation(
                    recommendation_id=f"agent_performance_{agent['id']}",
                    type="performance",
                    title=f"Agent {agent['name']} Performance Issue",
                    description=f"Agent {agent['name']} has low success rate and needs attention",
                    priority="high",
                    confidence=0.8,
                    actions=[
                        {"type": "investigate", "target": agent["id"], "description": "Investigate performance issues"},
                        {"type": "retrain", "target": agent["id"], "description": "Consider retraining or reconfiguration"},
                        {"type": "replace", "target": agent["id"], "description": "Consider replacing with better performing agent"}
                    ],
                    evidence=[{"metric": "success_rate", "value": agent.get("performance_metrics", {}).get("success_rate", 0)}],
                    impact_score=0.7
                ))
        
        return recommendations
    
    async def _analyze_service_dependencies(self) -> List[Recommendation]:
        """Analyze service dependency patterns"""
        recommendations = []
        
        # Find services with too many dependencies
        cypher = """
        MATCH (s:Service)-[:DEPENDS_ON]->(dep:Service)
        WITH s, count(dep) as dependency_count
        WHERE dependency_count > 5
        RETURN s, dependency_count
        ORDER BY dependency_count DESC
        """
        
        results = await self.query_engine.neo4j_manager.execute_cypher(cypher)
        
        for record in results:
            service = record["s"]
            dep_count = record["dependency_count"]
            
            recommendations.append(Recommendation(
                recommendation_id=f"service_deps_{service['id']}",
                type="reliability",
                title=f"High Dependency Count for {service['name']}",
                description=f"Service {service['name']} has {dep_count} dependencies, creating potential reliability issues",
                priority="medium",
                confidence=0.7,
                actions=[
                    {"type": "refactor", "target": service["id"], "description": "Consider breaking down service into smaller components"},
                    {"type": "cache", "target": service["id"], "description": "Implement caching to reduce dependency calls"},
                    {"type": "circuit_breaker", "target": service["id"], "description": "Implement circuit breakers for dependencies"}
                ],
                evidence=[{"metric": "dependency_count", "value": dep_count}],
                impact_score=min(dep_count / 10.0, 1.0)
            ))
        
        return recommendations
    
    async def _analyze_resource_utilization(self) -> List[Recommendation]:
        """Analyze resource utilization patterns"""
        recommendations = []
        
        # Find agents with high resource usage
        cypher = """
        MATCH (a:Agent)
        WHERE a.performance_metrics IS NOT NULL
        AND (a.performance_metrics.cpu_utilization > 80 OR a.performance_metrics.memory_usage > 80)
        RETURN a
        """
        
        results = await self.query_engine.neo4j_manager.execute_cypher(cypher)
        
        for record in results:
            agent = record["a"]
            metrics = agent.get("performance_metrics", {})
            cpu_usage = metrics.get("cpu_utilization", 0)
            memory_usage = metrics.get("memory_usage", 0)
            
            recommendations.append(Recommendation(
                recommendation_id=f"resource_usage_{agent['id']}",
                type="performance",
                title=f"High Resource Usage: {agent['name']}",
                description=f"Agent {agent['name']} is using high CPU ({cpu_usage}%) or memory ({memory_usage}%)",
                priority="medium" if max(cpu_usage, memory_usage) < 90 else "high",
                confidence=0.8,
                actions=[
                    {"type": "scale", "target": agent["id"], "description": "Scale agent resources"},
                    {"type": "optimize", "target": agent["id"], "description": "Optimize agent algorithms"},
                    {"type": "load_balance", "target": agent["id"], "description": "Distribute load across multiple agents"}
                ],
                evidence=[
                    {"metric": "cpu_utilization", "value": cpu_usage},
                    {"metric": "memory_usage", "value": memory_usage}
                ],
                impact_score=max(cpu_usage, memory_usage) / 100.0
            ))
        
        return recommendations


class ReasoningEngine:
    """
    Main reasoning engine that coordinates all reasoning components
    """
    
    def __init__(self, query_engine: QueryEngine):
        self.query_engine = query_engine
        
        # Initialize reasoning components
        self.rule_engine = RuleEngine(query_engine)
        self.capability_reasoner = CapabilityReasoner(query_engine)
        self.system_optimizer = SystemOptimizer(query_engine)
        
        self.logger = logging.getLogger("reasoning_engine")
        
        # Reasoning history
        self.inference_history = []
        self.recommendation_history = []
    
    async def perform_reasoning_cycle(self) -> Dict[str, Any]:
        """Perform a complete reasoning cycle"""
        start_time = datetime.utcnow()
        results = {
            "timestamp": start_time.isoformat(),
            "inferences": [],
            "capability_analysis": {},
            "recommendations": [],
            "performance": {}
        }
        
        try:
            # 1. Execute inference rules
            self.logger.info("Executing inference rules")
            inferences = await self.rule_engine.execute_rules()
            results["inferences"] = [
                {
                    "inference_id": inf.inference_id,
                    "rule_id": inf.rule_id,
                    "confidence": inf.confidence,
                    "description": inf.description,
                    "applied": inf.applied
                }
                for inf in inferences
            ]
            self.inference_history.extend(inferences)
            
            # 2. Analyze capabilities
            self.logger.info("Analyzing system capabilities")
            bottlenecks = await self.capability_reasoner.identify_capability_bottlenecks()
            results["capability_analysis"] = {
                "bottlenecks": bottlenecks,
                "bottleneck_count": len(bottlenecks)
            }
            
            # 3. Generate system recommendations
            self.logger.info("Generating system recommendations")
            recommendations = await self.system_optimizer.analyze_system_performance()
            results["recommendations"] = [
                {
                    "recommendation_id": rec.recommendation_id,
                    "type": rec.type,
                    "title": rec.title,
                    "priority": rec.priority,
                    "confidence": rec.confidence,
                    "impact_score": rec.impact_score,
                    "actions": rec.actions
                }
                for rec in recommendations
            ]
            self.recommendation_history.extend(recommendations)
            
            # 4. Performance metrics
            end_time = datetime.utcnow()
            processing_time = (end_time - start_time).total_seconds()
            results["performance"] = {
                "processing_time_seconds": processing_time,
                "inferences_generated": len(inferences),
                "recommendations_generated": len(recommendations),
                "rules_executed": len([r for r in self.rule_engine.rules.values() if r.enabled])
            }
            
            self.logger.info(f"Reasoning cycle completed in {processing_time:.2f}s")
            
        except Exception as e:
            self.logger.error(f"Error in reasoning cycle: {e}")
            results["error"] = str(e)
        
        return results
    
    async def find_optimal_agent_for_task(self, task_description: str, 
                                        required_capabilities: List[str]) -> Dict[str, Any]:
        """Find optimal agent assignment for a task"""
        
        # Find optimal combinations
        combinations = await self.capability_reasoner.find_optimal_agent_combinations(
            required_capabilities
        )
        
        # Generate reasoning explanation
        explanation = self._generate_task_assignment_explanation(
            task_description,
            required_capabilities,
            combinations
        )
        
        return {
            "task_description": task_description,
            "required_capabilities": required_capabilities,
            "agent_combinations": combinations,
            "recommendation": combinations[0] if combinations else None,
            "explanation": explanation,
            "confidence": combinations[0]["score"] if combinations else 0.0
        }
    
    def _generate_task_assignment_explanation(self, task_description: str,
                                           required_capabilities: List[str],
                                           combinations: List[Dict]) -> str:
        """Generate explanation for task assignment recommendation"""
        if not combinations:
            return f"No suitable agent combinations found for capabilities: {', '.join(required_capabilities)}"
        
        best_combo = combinations[0]
        agent_names = [agent["a"]["name"] for agent in best_combo["agents"]]
        
        explanation = f"For task '{task_description}', the optimal agent combination is: {', '.join(agent_names)}. "
        explanation += f"This combination provides {best_combo['total_capabilities']} capabilities "
        explanation += f"with a confidence score of {best_combo['score']:.2f}. "
        
        if best_combo["redundancy"] > 0:
            explanation += f"The combination has {best_combo['redundancy']:.1%} capability redundancy for reliability. "
        
        explanation += f"Estimated performance: {best_combo['estimated_performance']:.1%}."
        
        return explanation
    
    def get_reasoning_statistics(self) -> Dict[str, Any]:
        """Get reasoning engine statistics"""
        return {
            "total_inferences": len(self.inference_history),
            "total_recommendations": len(self.recommendation_history),
            "active_rules": len([r for r in self.rule_engine.rules.values() if r.enabled]),
            "recent_inferences": len([
                inf for inf in self.inference_history 
                if inf.timestamp > datetime.utcnow() - timedelta(hours=24)
            ]),
            "recent_recommendations": len([
                rec for rec in self.recommendation_history
                if rec.timestamp > datetime.utcnow() - timedelta(hours=24)
            ]),
            "rule_breakdown": {
                rule_id: {"name": rule.name, "priority": rule.priority, "enabled": rule.enabled}
                for rule_id, rule in self.rule_engine.rules.items()
            }
        }
    
    def clear_history(self):
        """Clear reasoning history"""
        self.inference_history.clear()
        self.recommendation_history.clear()
        self.logger.info("Reasoning history cleared")


# Global reasoning engine instance
_reasoning_engine: Optional[ReasoningEngine] = None


def get_reasoning_engine() -> Optional[ReasoningEngine]:
    """Get the global reasoning engine instance"""
    return _reasoning_engine


def set_reasoning_engine(engine: ReasoningEngine):
    """Set the global reasoning engine instance"""
    global _reasoning_engine
    _reasoning_engine = engine