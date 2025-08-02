#!/usr/bin/env python3
"""
Brain Integration Service for SutazAI
Connects the brain system to the main application backend
"""

import asyncio
import logging
import json
import requests
from typing import Dict, List, Any, Optional
from datetime import datetime
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BrainIntegration:
    """
    Integration layer between SutazAI backend and brain system
    """
    
    def __init__(self):
        self.brain_api_url = "http://localhost:8888"
        self.backend_url = "http://localhost:8000"  # Main backend
        self.is_connected = False
        self.last_health_check = None
        
    async def initialize(self):
        """Initialize brain integration"""
        logger.info("🔗 Initializing Brain Integration...")
        
        # Check brain availability
        if await self.check_brain_health():
            self.is_connected = True
            logger.info("✅ Brain system connected")
            
            # Start integration loops
            asyncio.create_task(self.health_monitoring_loop())
            asyncio.create_task(self.experience_sharing_loop())
            
            return True
        else:
            logger.error("❌ Brain system not available")
            return False
    
    async def check_brain_health(self) -> bool:
        """Check if brain system is healthy"""
        try:
            response = requests.get(f"{self.brain_api_url}/health", timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                self.last_health_check = datetime.now()
                logger.info(f"📊 Brain health: {health_data['status']} (Intelligence: {health_data['intelligence_level']:.3f})")
                return True
        except Exception as e:
            logger.warning(f"Brain health check failed: {e}")
        return False
    
    async def process_with_brain(self, user_input: str, context: Optional[Dict] = None, require_learning: bool = False) -> Dict[str, Any]:
        """Process user input through the brain system"""
        if not self.is_connected:
            return {
                'response': 'Brain system not available',
                'confidence': 0.0,
                'source': 'fallback'
            }
        
        try:
            payload = {
                'query': user_input,
                'context': context,
                'require_learning': require_learning
            }
            
            response = requests.post(
                f"{self.brain_api_url}/process",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                result['source'] = 'brain'
                return result
            else:
                logger.error(f"Brain processing failed: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error processing with brain: {e}")
        
        # Fallback response
        return {
            'response': f'I encountered an issue processing your request: {user_input}',
            'confidence': 0.1,
            'source': 'fallback'
        }
    
    async def store_memory(self, content: str, importance: float = 0.5, memory_type: str = "user_interaction") -> bool:
        """Store a memory in the brain system"""
        if not self.is_connected:
            return False
        
        try:
            payload = {
                'content': content,
                'importance': importance,
                'memory_type': memory_type
            }
            
            response = requests.post(
                f"{self.brain_api_url}/memory/store",
                json=payload,
                timeout=10
            )
            
            return response.status_code == 200
            
        except Exception as e:
            logger.error(f"Error storing memory: {e}")
            return False
    
    async def search_memories(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search brain memories"""
        if not self.is_connected:
            return []
        
        try:
            response = requests.get(
                f"{self.brain_api_url}/memory/search",
                params={'query': query, 'top_k': top_k},
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json().get('memories', [])
                
        except Exception as e:
            logger.error(f"Error searching memories: {e}")
        
        return []
    
    async def get_brain_status(self) -> Dict[str, Any]:
        """Get current brain status"""
        if not self.is_connected:
            return {'status': 'disconnected'}
        
        try:
            response = requests.get(f"{self.brain_api_url}/status", timeout=5)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"Error getting brain status: {e}")
        
        return {'status': 'error'}
    
    async def trigger_learning(self) -> bool:
        """Manually trigger brain learning cycle"""
        if not self.is_connected:
            return False
        
        try:
            response = requests.post(f"{self.brain_api_url}/learning/trigger", timeout=15)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Error triggering learning: {e}")
            return False
    
    async def health_monitoring_loop(self):
        """Monitor brain health continuously"""
        while True:
            try:
                was_connected = self.is_connected
                self.is_connected = await self.check_brain_health()
                
                if was_connected and not self.is_connected:
                    logger.warning("⚠️ Brain system disconnected")
                elif not was_connected and self.is_connected:
                    logger.info("✨ Brain system reconnected")
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in health monitoring: {e}")
                await asyncio.sleep(60)
    
    async def experience_sharing_loop(self):
        """Share experiences between backend and brain"""
        while True:
            try:
                if self.is_connected:
                    # This would collect experiences from the backend
                    # and share them with the brain for learning
                    
                    # For now, just log the connection status
                    brain_status = await self.get_brain_status()
                    if brain_status.get('status') != 'error':
                        logger.info(
                            f"🧠 Brain Update - Intelligence: {brain_status.get('intelligence_level', 0):.3f} | "
                            f"Requests: {brain_status.get('total_requests', 0)} | "
                            f"Memories: {brain_status.get('memory_entries', 0)} | "
                            f"Learning Cycles: {brain_status.get('learning_cycles', 0)}"
                        )
                
                await asyncio.sleep(300)  # Update every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in experience sharing: {e}")
                await asyncio.sleep(300)


class BrainEnhancedAgent:
    """
    Enhanced agent that uses the brain system for processing
    """
    
    def __init__(self, brain_integration: BrainIntegration):
        self.brain = brain_integration
        self.conversation_history = []
        
    async def process_message(self, message: str, user_id: str = None) -> Dict[str, Any]:
        """Process a message through the brain-enhanced system"""
        # Add to conversation history
        self.conversation_history.append({
            'type': 'user',
            'message': message,
            'timestamp': datetime.now(),
            'user_id': user_id
        })
        
        # Create context from recent conversation (JSON serializable)
        recent_messages = []
        for msg in self.conversation_history[-5:]:
            serializable_msg = {
                'type': msg['type'],
                'message': msg['message'],
                'timestamp': msg['timestamp'].isoformat(),
                'user_id': msg.get('user_id'),
                'confidence': msg.get('confidence')
            }
            recent_messages.append(serializable_msg)
        
        context = {
            'recent_messages': recent_messages,
            'user_id': user_id
        }
        
        # Process through brain
        result = await self.brain.process_with_brain(
            user_input=message,
            context=context,
            require_learning=True  # Enable learning from user interactions
        )
        
        # Add response to history
        self.conversation_history.append({
            'type': 'assistant',
            'message': result['response'],
            'confidence': result['confidence'],
            'timestamp': datetime.now()
        })
        
        # Store important interactions as memories
        if result['confidence'] > 0.8:
            await self.brain.store_memory(
                content=f"User: {message}\nAssistant: {result['response']}",
                importance=result['confidence'],
                memory_type="user_interaction"
            )
        
        return result
    
    async def get_conversation_summary(self) -> str:
        """Get a summary of the conversation"""
        if not self.conversation_history:
            return "No conversation history"
        
        # Create summary from recent messages
        recent_messages = self.conversation_history[-10:]  # Last 10 messages
        summary_text = "\n".join([
            f"{msg['type'].title()}: {msg['message'][:100]}..."
            for msg in recent_messages
        ])
        
        # Use brain to generate summary
        result = await self.brain.process_with_brain(
            f"Please summarize this conversation: {summary_text}",
            require_learning=False
        )
        
        return result['response']


async def main():
    """Main integration test"""
    logger.info("🎆 Starting SutazAI Brain Integration...")
    
    # Initialize brain integration
    integration = BrainIntegration()
    if await integration.initialize():
        logger.info("✅ Brain integration initialized")
        
        # Create enhanced agent
        agent = BrainEnhancedAgent(integration)
        
        # Test conversation
        test_messages = [
            "Hello! What can you tell me about yourself?",
            "How do you learn and improve over time?",
            "What's the current state of artificial intelligence?",
            "Can you remember what we've discussed so far?"
        ]
        
        logger.info("\n💬 Testing brain-enhanced conversation...")
        
        for i, message in enumerate(test_messages, 1):
            logger.info(f"\n--- Test {i} ---")
            logger.info(f"User: {message}")
            
            result = await agent.process_message(message, user_id="test_user")
            
            logger.info(f"Brain: {result['response'][:200]}...")
            logger.info(f"Confidence: {result['confidence']:.2f}, Source: {result.get('source', 'unknown')}")
            
            # Wait a bit between messages
            await asyncio.sleep(1)
        
        # Get conversation summary
        logger.info("\n📝 Getting conversation summary...")
        summary = await agent.get_conversation_summary()
        logger.info(f"Summary: {summary[:200]}...")
        
        # Show final brain statistics
        brain_status = await integration.get_brain_status()
        logger.info("\n📊 Final Brain Statistics:")
        logger.info(f"Intelligence Level: {brain_status.get('intelligence_level', 0):.3f}")
        logger.info(f"Total Requests: {brain_status.get('total_requests', 0)}")
        logger.info(f"Memory Entries: {brain_status.get('memory_entries', 0)}")
        logger.info(f"Learning Cycles: {brain_status.get('learning_cycles', 0)}")
        
        logger.info("\n✨ Brain integration test completed successfully!")
        
        # Keep integration running
        logger.info("🔄 Brain integration will continue running...")
        while True:
            await asyncio.sleep(60)
    
    else:
        logger.error("❌ Failed to initialize brain integration")


if __name__ == "__main__":
    asyncio.run(main())