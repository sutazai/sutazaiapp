"""Learning System"""
import logging
import json
import time
from typing import Dict, List, Any
from pathlib import Path

logger = logging.getLogger(__name__)

class LearningSystem:
    def __init__(self):
        self.learning_data = []
        self.performance_metrics = {}
        self.adaptations = []
    
    async def initialize(self):
        """Initialize learning system"""
        logger.info("🧠 Initializing Learning System")
        
        # Setup learning parameters
        self.learning_rate = 0.01
        self.adaptation_threshold = 0.1
        
        logger.info("✅ Learning system initialized")
    
    async def add_learning_example(self, input_data: Dict[str, Any], output_data: Dict[str, Any], feedback: float):
        """Add learning example"""
        example = {
            "input": input_data,
            "output": output_data,
            "feedback": feedback,
            "timestamp": time.time()
        }
        
        self.learning_data.append(example)
        
        # Trigger adaptation if needed
        if len(self.learning_data) % 10 == 0:
            await self._adapt_system()
    
    async def _adapt_system(self):
        """Adapt system based on learning"""
        # Analyze recent performance
        recent_feedback = [ex["feedback"] for ex in self.learning_data[-10:]]
        avg_feedback = sum(recent_feedback) / len(recent_feedback)
        
        if avg_feedback < 0.7:  # Below threshold
            adaptation = {
                "timestamp": time.time(),
                "reason": "Low feedback score",
                "avg_feedback": avg_feedback,
                "action": "Adjust learning parameters"
            }
            self.adaptations.append(adaptation)
            logger.info(f"System adaptation triggered: {adaptation['reason']}")
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get learning statistics"""
        if not self.learning_data:
            return {"message": "No learning data available"}
        
        recent_feedback = [ex["feedback"] for ex in self.learning_data[-20:]]
        
        return {
            "total_examples": len(self.learning_data),
            "recent_performance": sum(recent_feedback) / len(recent_feedback),
            "adaptations_made": len(self.adaptations),
            "learning_rate": self.learning_rate
        }

# Global instance
learning_system = LearningSystem()
