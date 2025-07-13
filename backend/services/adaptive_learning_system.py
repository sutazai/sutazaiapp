"""
Adaptive Learning System
Advanced system for learning from user interactions and improving responses
"""

import asyncio
import logging
import json
import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import pickle
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class InteractionType(str, Enum):
    CHAT = "chat"
    CODE_GENERATION = "code_generation"
    ANALYSIS = "analysis"
    SEARCH = "search"
    TASK_EXECUTION = "task_execution"
    FEEDBACK = "feedback"

class LearningMode(str, Enum):
    PASSIVE = "passive"  # Learn from observations
    ACTIVE = "active"    # Ask for feedback
    REINFORCEMENT = "reinforcement"  # Learn from rewards

@dataclass
class Interaction:
    """User interaction data"""
    id: str
    timestamp: float
    user_id: str
    interaction_type: InteractionType
    input_data: Dict[str, Any]
    system_response: Dict[str, Any]
    user_feedback: Optional[Dict[str, Any]] = None
    success_score: float = 0.5  # 0.0 to 1.0
    context: Dict[str, Any] = None
    learned_from: bool = False
    
    def __post_init__(self):
        if self.context is None:
            self.context = {}

@dataclass
class LearningPattern:
    """Learned pattern from interactions"""
    id: str
    pattern_type: str
    input_pattern: Dict[str, Any]
    successful_response_pattern: Dict[str, Any]
    confidence: float
    usage_count: int
    success_rate: float
    created_at: float
    last_used: float
    
@dataclass
class UserProfile:
    """User behavior profile"""
    user_id: str
    preferences: Dict[str, Any]
    interaction_history: List[str]  # Interaction IDs
    success_patterns: List[str]  # Pattern IDs
    average_satisfaction: float
    interaction_count: int
    created_at: float
    last_active: float

class AdaptiveLearningSystem:
    """
    Advanced Adaptive Learning System
    Learns from user interactions to improve system responses
    """
    
    def __init__(self, data_dir: str = "/opt/sutazaiapp/data/learning"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Core learning components
        self.interactions = {}
        self.patterns = {}
        self.user_profiles = {}
        
        # Learning metrics
        self.learning_metrics = {
            "total_interactions": 0,
            "patterns_learned": 0,
            "improvement_rate": 0.0,
            "user_satisfaction_trend": [],
            "response_quality_trend": []
        }
        
        # Memory systems
        self.short_term_memory = deque(maxlen=1000)  # Recent interactions
        self.long_term_memory = {}  # Persistent patterns
        self.episodic_memory = {}   # Specific memorable interactions
        
        # Learning parameters
        self.learning_rate = 0.1
        self.confidence_threshold = 0.7
        self.pattern_usage_threshold = 5
        
        # Initialize
        self._load_existing_data()
        self._initialize_learning_models()
    
    def _initialize_learning_models(self):
        """Initialize machine learning models"""
        try:
            # Simple pattern matching and clustering models
            self.pattern_classifier = self._create_pattern_classifier()
            self.satisfaction_predictor = self._create_satisfaction_predictor()
            self.response_optimizer = self._create_response_optimizer()
            
            logger.info("✅ Learning models initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize learning models: {e}")
    
    def _create_pattern_classifier(self):
        """Create pattern classification model"""
        # Simplified pattern classifier using basic similarity matching
        return {
            "model_type": "similarity_matcher",
            "feature_weights": {
                "input_similarity": 0.4,
                "context_similarity": 0.3,
                "user_similarity": 0.2,
                "temporal_similarity": 0.1
            }
        }
    
    def _create_satisfaction_predictor(self):
        """Create user satisfaction prediction model"""
        return {
            "model_type": "regression",
            "features": [
                "response_time",
                "response_length",
                "relevance_score",
                "user_history",
                "context_match"
            ]
        }
    
    def _create_response_optimizer(self):
        """Create response optimization model"""
        return {
            "model_type": "reinforcement_learning",
            "optimization_targets": [
                "user_satisfaction",
                "task_completion",
                "response_quality",
                "efficiency"
            ]
        }
    
    async def record_interaction(self, interaction_data: Dict[str, Any]) -> str:
        """Record a new user interaction"""
        try:
            interaction = Interaction(
                id=str(uuid.uuid4()),
                timestamp=time.time(),
                user_id=interaction_data.get("user_id", "anonymous"),
                interaction_type=InteractionType(interaction_data.get("type", "chat")),
                input_data=interaction_data.get("input", {}),
                system_response=interaction_data.get("response", {}),
                user_feedback=interaction_data.get("feedback"),
                success_score=interaction_data.get("success_score", 0.5),
                context=interaction_data.get("context", {})
            )
            
            # Store interaction
            self.interactions[interaction.id] = interaction
            self.short_term_memory.append(interaction.id)
            
            # Update user profile
            await self._update_user_profile(interaction)
            
            # Trigger learning if feedback is available
            if interaction.user_feedback:
                await self._learn_from_interaction(interaction)
            
            # Update metrics
            self.learning_metrics["total_interactions"] += 1
            
            logger.info(f"📝 Recorded interaction: {interaction.id}")
            return interaction.id
            
        except Exception as e:
            logger.error(f"Failed to record interaction: {e}")
            return ""
    
    async def _update_user_profile(self, interaction: Interaction):
        """Update or create user profile"""
        try:
            user_id = interaction.user_id
            
            if user_id not in self.user_profiles:
                self.user_profiles[user_id] = UserProfile(
                    user_id=user_id,
                    preferences={},
                    interaction_history=[],
                    success_patterns=[],
                    average_satisfaction=0.5,
                    interaction_count=0,
                    created_at=time.time(),
                    last_active=time.time()
                )
            
            profile = self.user_profiles[user_id]
            
            # Update profile
            profile.interaction_history.append(interaction.id)
            profile.interaction_count += 1
            profile.last_active = interaction.timestamp
            
            # Update satisfaction average
            if interaction.user_feedback and "satisfaction" in interaction.user_feedback:
                satisfaction = interaction.user_feedback["satisfaction"]
                profile.average_satisfaction = (
                    (profile.average_satisfaction * (profile.interaction_count - 1) + satisfaction) /
                    profile.interaction_count
                )
            
            # Keep only last 100 interactions per user
            if len(profile.interaction_history) > 100:
                profile.interaction_history = profile.interaction_history[-100:]
            
            # Learn user preferences
            await self._learn_user_preferences(profile, interaction)
            
        except Exception as e:
            logger.error(f"Failed to update user profile: {e}")
    
    async def _learn_user_preferences(self, profile: UserProfile, interaction: Interaction):
        """Learn and update user preferences"""
        try:
            # Learn from interaction patterns
            if interaction.success_score > 0.7:  # Successful interaction
                
                # Learn response format preferences
                response = interaction.system_response
                if "format" in response:
                    format_pref = profile.preferences.get("response_format", {})
                    format_type = response["format"]
                    format_pref[format_type] = format_pref.get(format_type, 0) + 1
                    profile.preferences["response_format"] = format_pref
                
                # Learn content preferences
                if "topics" in interaction.context:
                    topics = interaction.context["topics"]
                    topic_prefs = profile.preferences.get("topics", {})
                    for topic in topics:
                        topic_prefs[topic] = topic_prefs.get(topic, 0) + 1
                    profile.preferences["topics"] = topic_prefs
                
                # Learn communication style
                if "style" in interaction.user_feedback:
                    style = interaction.user_feedback["style"]
                    profile.preferences["communication_style"] = style
                    
        except Exception as e:
            logger.error(f"Failed to learn user preferences: {e}")
    
    async def _learn_from_interaction(self, interaction: Interaction):
        """Learn patterns from successful interactions"""
        try:
            if interaction.success_score < 0.6:  # Don't learn from poor interactions
                return
            
            # Extract patterns
            input_pattern = await self._extract_input_pattern(interaction)
            response_pattern = await self._extract_response_pattern(interaction)
            
            # Check if pattern already exists
            similar_pattern = await self._find_similar_pattern(input_pattern)
            
            if similar_pattern:
                # Update existing pattern
                pattern = self.patterns[similar_pattern]
                pattern.usage_count += 1
                pattern.success_rate = (
                    (pattern.success_rate * (pattern.usage_count - 1) + interaction.success_score) /
                    pattern.usage_count
                )
                pattern.last_used = interaction.timestamp
                pattern.confidence = min(1.0, pattern.confidence + 0.1)
                
            else:
                # Create new pattern
                pattern = LearningPattern(
                    id=str(uuid.uuid4()),
                    pattern_type=interaction.interaction_type.value,
                    input_pattern=input_pattern,
                    successful_response_pattern=response_pattern,
                    confidence=0.6,
                    usage_count=1,
                    success_rate=interaction.success_score,
                    created_at=interaction.timestamp,
                    last_used=interaction.timestamp
                )
                
                self.patterns[pattern.id] = pattern
                self.learning_metrics["patterns_learned"] += 1
            
            interaction.learned_from = True
            
            # Update user's successful patterns
            user_profile = self.user_profiles.get(interaction.user_id)
            if user_profile and pattern.id not in user_profile.success_patterns:
                user_profile.success_patterns.append(pattern.id)
            
            logger.info(f"🧠 Learned from interaction: {interaction.id}")
            
        except Exception as e:
            logger.error(f"Failed to learn from interaction: {e}")
    
    async def _extract_input_pattern(self, interaction: Interaction) -> Dict[str, Any]:
        """Extract pattern from user input"""
        input_data = interaction.input_data
        
        pattern = {
            "type": interaction.interaction_type.value,
            "keywords": self._extract_keywords(str(input_data)),
            "length": len(str(input_data)),
            "complexity": self._calculate_input_complexity(input_data),
            "intent": self._classify_intent(input_data),
            "context_features": self._extract_context_features(interaction.context)
        }
        
        return pattern
    
    async def _extract_response_pattern(self, interaction: Interaction) -> Dict[str, Any]:
        """Extract pattern from successful response"""
        response = interaction.system_response
        
        pattern = {
            "response_type": response.get("type", "text"),
            "structure": self._analyze_response_structure(response),
            "length": len(str(response)),
            "components": list(response.keys()),
            "style": response.get("style", "standard"),
            "effectiveness_score": interaction.success_score
        }
        
        return pattern
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text"""
        # Simple keyword extraction
        words = text.lower().split()
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        keywords = [word for word in words if len(word) > 3 and word not in stopwords]
        return list(set(keywords))[:10]  # Top 10 unique keywords
    
    def _calculate_input_complexity(self, input_data: Dict[str, Any]) -> float:
        """Calculate complexity score of input"""
        text = str(input_data)
        
        # Simple complexity metrics
        word_count = len(text.split())
        char_count = len(text)
        unique_words = len(set(text.lower().split()))
        
        # Normalize to 0-1 scale
        complexity = min(1.0, (word_count * 0.1 + char_count * 0.001 + unique_words * 0.05))
        return complexity
    
    def _classify_intent(self, input_data: Dict[str, Any]) -> str:
        """Classify user intent"""
        text = str(input_data).lower()
        
        # Simple intent classification
        if any(word in text for word in ['create', 'generate', 'make', 'build']):
            return 'creation'
        elif any(word in text for word in ['analyze', 'examine', 'review', 'check']):
            return 'analysis'
        elif any(word in text for word in ['help', 'how', 'what', 'explain']):
            return 'assistance'
        elif any(word in text for word in ['search', 'find', 'look', 'locate']):
            return 'search'
        else:
            return 'general'
    
    def _extract_context_features(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features from context"""
        features = {
            "time_of_day": time.strftime("%H", time.localtime()),
            "session_length": context.get("session_length", 0),
            "previous_interactions": context.get("previous_interactions", 0),
            "user_expertise": context.get("user_expertise", "beginner"),
            "task_urgency": context.get("urgency", "normal")
        }
        return features
    
    def _analyze_response_structure(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze structure of response"""
        structure = {
            "has_explanation": "explanation" in response,
            "has_examples": "examples" in response,
            "has_code": "code" in response,
            "has_links": "links" in response,
            "sections": len(response.keys()),
            "format": response.get("format", "text")
        }
        return structure
    
    async def _find_similar_pattern(self, input_pattern: Dict[str, Any]) -> Optional[str]:
        """Find similar existing pattern"""
        try:
            for pattern_id, pattern in self.patterns.items():
                similarity = await self._calculate_pattern_similarity(
                    input_pattern, 
                    pattern.input_pattern
                )
                
                if similarity > 0.8:  # High similarity threshold
                    return pattern_id
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to find similar pattern: {e}")
            return None
    
    async def _calculate_pattern_similarity(self, pattern1: Dict[str, Any], pattern2: Dict[str, Any]) -> float:
        """Calculate similarity between two patterns"""
        try:
            similarity_scores = []
            
            # Compare keywords
            keywords1 = set(pattern1.get("keywords", []))
            keywords2 = set(pattern2.get("keywords", []))
            if keywords1 or keywords2:
                keyword_similarity = len(keywords1 & keywords2) / len(keywords1 | keywords2)
                similarity_scores.append(keyword_similarity)
            
            # Compare intent
            if pattern1.get("intent") == pattern2.get("intent"):
                similarity_scores.append(1.0)
            else:
                similarity_scores.append(0.0)
            
            # Compare complexity (normalized difference)
            complexity1 = pattern1.get("complexity", 0)
            complexity2 = pattern2.get("complexity", 0)
            complexity_similarity = 1.0 - abs(complexity1 - complexity2)
            similarity_scores.append(complexity_similarity)
            
            # Return average similarity
            return sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0.0
            
        except Exception as e:
            logger.error(f"Failed to calculate pattern similarity: {e}")
            return 0.0
    
    async def suggest_response_improvements(self, interaction_id: str) -> Dict[str, Any]:
        """Suggest improvements for a response based on learned patterns"""
        try:
            interaction = self.interactions.get(interaction_id)
            if not interaction:
                return {"error": "Interaction not found"}
            
            # Find relevant patterns
            input_pattern = await self._extract_input_pattern(interaction)
            relevant_patterns = await self._find_relevant_patterns(input_pattern)
            
            if not relevant_patterns:
                return {"message": "No relevant patterns found for improvement"}
            
            suggestions = []
            
            for pattern_id in relevant_patterns[:3]:  # Top 3 patterns
                pattern = self.patterns[pattern_id]
                
                suggestion = {
                    "pattern_id": pattern_id,
                    "confidence": pattern.confidence,
                    "success_rate": pattern.success_rate,
                    "suggested_improvements": await self._generate_improvement_suggestions(
                        interaction.system_response,
                        pattern.successful_response_pattern
                    )
                }
                suggestions.append(suggestion)
            
            return {
                "interaction_id": interaction_id,
                "suggestions": suggestions,
                "generated_at": time.time()
            }
            
        except Exception as e:
            logger.error(f"Failed to suggest improvements: {e}")
            return {"error": str(e)}
    
    async def _find_relevant_patterns(self, input_pattern: Dict[str, Any]) -> List[str]:
        """Find patterns relevant to the input"""
        relevant_patterns = []
        
        for pattern_id, pattern in self.patterns.items():
            if pattern.confidence > self.confidence_threshold:
                similarity = await self._calculate_pattern_similarity(
                    input_pattern,
                    pattern.input_pattern
                )
                
                if similarity > 0.6:  # Relevance threshold
                    relevant_patterns.append((pattern_id, similarity))
        
        # Sort by similarity and return pattern IDs
        relevant_patterns.sort(key=lambda x: x[1], reverse=True)
        return [pattern_id for pattern_id, _ in relevant_patterns]
    
    async def _generate_improvement_suggestions(self, current_response: Dict[str, Any], successful_pattern: Dict[str, Any]) -> List[str]:
        """Generate specific improvement suggestions"""
        suggestions = []
        
        # Compare response structures
        current_structure = self._analyze_response_structure(current_response)
        pattern_structure = successful_pattern.get("structure", {})
        
        if not current_structure.get("has_examples") and pattern_structure.get("has_examples"):
            suggestions.append("Consider adding examples to illustrate your points")
        
        if not current_structure.get("has_explanation") and pattern_structure.get("has_explanation"):
            suggestions.append("Provide more detailed explanations")
        
        if current_structure.get("sections", 0) < pattern_structure.get("sections", 0):
            suggestions.append("Consider organizing response into more sections")
        
        # Compare response length
        current_length = len(str(current_response))
        pattern_length = successful_pattern.get("length", 0)
        
        if current_length < pattern_length * 0.7:
            suggestions.append("Consider providing a more comprehensive response")
        elif current_length > pattern_length * 1.5:
            suggestions.append("Consider making the response more concise")
        
        return suggestions
    
    async def get_learning_insights(self) -> Dict[str, Any]:
        """Get insights about the learning progress"""
        try:
            # Calculate trends
            recent_interactions = [
                interaction for interaction in self.interactions.values()
                if time.time() - interaction.timestamp < 86400  # Last 24 hours
            ]
            
            satisfaction_trend = [
                interaction.success_score for interaction in recent_interactions
                if interaction.user_feedback
            ]
            
            # Calculate improvement rate
            if len(satisfaction_trend) > 10:
                early_avg = sum(satisfaction_trend[:5]) / 5
                recent_avg = sum(satisfaction_trend[-5:]) / 5
                improvement_rate = (recent_avg - early_avg) / early_avg * 100
            else:
                improvement_rate = 0.0
            
            # Pattern effectiveness
            effective_patterns = [
                pattern for pattern in self.patterns.values()
                if pattern.success_rate > 0.8 and pattern.usage_count >= self.pattern_usage_threshold
            ]
            
            insights = {
                "learning_metrics": self.learning_metrics,
                "total_patterns": len(self.patterns),
                "effective_patterns": len(effective_patterns),
                "recent_satisfaction_avg": sum(satisfaction_trend) / len(satisfaction_trend) if satisfaction_trend else 0.0,
                "improvement_rate": improvement_rate,
                "user_profiles": len(self.user_profiles),
                "most_successful_pattern_types": self._get_most_successful_pattern_types(),
                "learning_recommendations": await self._generate_learning_recommendations()
            }
            
            return insights
            
        except Exception as e:
            logger.error(f"Failed to get learning insights: {e}")
            return {"error": str(e)}
    
    def _get_most_successful_pattern_types(self) -> Dict[str, float]:
        """Get most successful pattern types"""
        type_success = defaultdict(list)
        
        for pattern in self.patterns.values():
            if pattern.usage_count >= self.pattern_usage_threshold:
                type_success[pattern.pattern_type].append(pattern.success_rate)
        
        # Calculate average success rate per type
        type_averages = {
            pattern_type: sum(scores) / len(scores)
            for pattern_type, scores in type_success.items()
        }
        
        return dict(sorted(type_averages.items(), key=lambda x: x[1], reverse=True))
    
    async def _generate_learning_recommendations(self) -> List[str]:
        """Generate recommendations for improving learning"""
        recommendations = []
        
        # Check data sufficiency
        if len(self.interactions) < 100:
            recommendations.append("Collect more interaction data to improve learning accuracy")
        
        # Check pattern diversity
        pattern_types = set(pattern.pattern_type for pattern in self.patterns.values())
        if len(pattern_types) < 3:
            recommendations.append("Encourage diverse interaction types to learn more patterns")
        
        # Check feedback availability
        feedback_ratio = len([i for i in self.interactions.values() if i.user_feedback]) / max(len(self.interactions), 1)
        if feedback_ratio < 0.3:
            recommendations.append("Collect more user feedback to improve learning quality")
        
        # Check pattern effectiveness
        effective_patterns = len([p for p in self.patterns.values() if p.success_rate > 0.8])
        if effective_patterns / max(len(self.patterns), 1) < 0.5:
            recommendations.append("Focus on improving response quality based on successful patterns")
        
        return recommendations
    
    def _load_existing_data(self):
        """Load existing learning data"""
        try:
            # Load interactions
            interactions_file = self.data_dir / "interactions.json"
            if interactions_file.exists():
                with open(interactions_file, 'r') as f:
                    data = json.load(f)
                    for interaction_data in data.get("interactions", []):
                        interaction = Interaction(**interaction_data)
                        self.interactions[interaction.id] = interaction
            
            # Load patterns
            patterns_file = self.data_dir / "patterns.json"
            if patterns_file.exists():
                with open(patterns_file, 'r') as f:
                    data = json.load(f)
                    for pattern_data in data.get("patterns", []):
                        pattern = LearningPattern(**pattern_data)
                        self.patterns[pattern.id] = pattern
            
            # Load user profiles
            profiles_file = self.data_dir / "user_profiles.json"
            if profiles_file.exists():
                with open(profiles_file, 'r') as f:
                    data = json.load(f)
                    for profile_data in data.get("user_profiles", []):
                        profile = UserProfile(**profile_data)
                        self.user_profiles[profile.user_id] = profile
            
            # Load metrics
            metrics_file = self.data_dir / "learning_metrics.json"
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    self.learning_metrics.update(json.load(f))
            
            logger.info("✅ Learning data loaded from previous sessions")
            
        except Exception as e:
            logger.error(f"Failed to load learning data: {e}")
    
    async def save_learning_data(self):
        """Save learning data"""
        try:
            # Save interactions
            interactions_data = {
                "interactions": [asdict(interaction) for interaction in self.interactions.values()]
            }
            with open(self.data_dir / "interactions.json", 'w') as f:
                json.dump(interactions_data, f, indent=2, default=str)
            
            # Save patterns
            patterns_data = {
                "patterns": [asdict(pattern) for pattern in self.patterns.values()]
            }
            with open(self.data_dir / "patterns.json", 'w') as f:
                json.dump(patterns_data, f, indent=2, default=str)
            
            # Save user profiles
            profiles_data = {
                "user_profiles": [asdict(profile) for profile in self.user_profiles.values()]
            }
            with open(self.data_dir / "user_profiles.json", 'w') as f:
                json.dump(profiles_data, f, indent=2, default=str)
            
            # Save metrics
            with open(self.data_dir / "learning_metrics.json", 'w') as f:
                json.dump(self.learning_metrics, f, indent=2, default=str)
            
            logger.info("✅ Learning data saved")
            
        except Exception as e:
            logger.error(f"Failed to save learning data: {e}")

# Global instance
adaptive_learning_system = AdaptiveLearningSystem()

# Convenience functions
async def record_interaction(interaction_data: Dict[str, Any]) -> str:
    """Record interaction"""
    return await adaptive_learning_system.record_interaction(interaction_data)

async def suggest_response_improvements(interaction_id: str) -> Dict[str, Any]:
    """Suggest improvements"""
    return await adaptive_learning_system.suggest_response_improvements(interaction_id)

async def get_learning_insights() -> Dict[str, Any]:
    """Get learning insights"""
    return await adaptive_learning_system.get_learning_insights()