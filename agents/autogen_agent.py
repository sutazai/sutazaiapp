import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio

from .base_agent import BaseAgent
from models import model_manager
from memory import vector_memory

logger = logging.getLogger(__name__)

class AutoGenAgent(BaseAgent):
    """AutoGen multi-agent conversation framework integration."""
    
    def __init__(self, agent_id: str = "autogen_agent"):
        super().__init__(agent_id, "autogen")
        self.capabilities = [
            "multi_agent_conversation",
            "role_based_interaction",
            "collaborative_problem_solving",
            "group_decision_making",
            "specialized_expertise",
            "conversation_management"
        ]
        self.agents_registry = {}
        self.active_conversations = {}
    
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute AutoGen task with multi-agent conversation."""
        try:
            task_type = task.get("type", "")
            
            if task_type == "create_conversation":
                return await self._create_conversation_task(task)
            elif task_type == "add_agent":
                return await self._add_agent_task(task)
            elif task_type == "start_discussion":
                return await self._start_discussion_task(task)
            elif task_type == "collaborate":
                return await self._collaborate_task(task)
            else:
                return await self._general_autogen_task(task)
                
        except Exception as e:
            logger.error(f"Error executing AutoGen task: {e}")
            return {
                "success": False,
                "error": str(e),
                "agent_id": self.agent_id,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def _create_conversation_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new multi-agent conversation."""
        conversation_id = task.get("conversation_id", f"conv_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}")
        participants = task.get("participants", [])
        topic = task.get("topic", "General Discussion")
        
        conversation = {
            "id": conversation_id,
            "topic": topic,
            "participants": participants,
            "messages": [],
            "status": "active",
            "created_at": datetime.utcnow(),
            "moderator": "autogen_agent"
        }
        
        self.active_conversations[conversation_id] = conversation
        
        return {
            "success": True,
            "conversation_id": conversation_id,
            "conversation": conversation,
            "capabilities_used": ["multi_agent_conversation", "conversation_management"],
            "agent_id": self.agent_id,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _add_agent_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Add a specialized agent to the registry."""
        agent_config = task.get("agent_config", {})
        agent_name = agent_config.get("name", f"agent_{len(self.agents_registry) + 1}")
        
        agent_spec = {
            "name": agent_name,
            "role": agent_config.get("role", "assistant"),
            "expertise": agent_config.get("expertise", []),
            "system_message": agent_config.get("system_message", "You are a helpful assistant."),
            "capabilities": agent_config.get("capabilities", []),
            "created_at": datetime.utcnow()
        }
        
        self.agents_registry[agent_name] = agent_spec
        
        return {
            "success": True,
            "agent_name": agent_name,
            "agent_spec": agent_spec,
            "total_agents": len(self.agents_registry),
            "capabilities_used": ["role_based_interaction", "specialized_expertise"],
            "agent_id": self.agent_id,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _start_discussion_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Start a discussion among multiple agents."""
        conversation_id = task.get("conversation_id", "")
        initial_message = task.get("message", "")
        max_rounds = task.get("max_rounds", 5)
        
        if conversation_id not in self.active_conversations:
            return {"success": False, "error": "Conversation not found"}
        
        conversation = self.active_conversations[conversation_id]
        discussion_log = []
        
        # Simulate multi-agent discussion
        for round_num in range(max_rounds):
            for participant in conversation["participants"]:
                if participant in self.agents_registry:
                    agent_spec = self.agents_registry[participant]
                    
                    # Generate response from this agent's perspective
                    prompt = f"""
                    You are {agent_spec['name']} with the role: {agent_spec['role']}
                    Your expertise: {', '.join(agent_spec['expertise'])}
                    System message: {agent_spec['system_message']}
                    
                    Discussion topic: {conversation['topic']}
                    Current discussion: {discussion_log[-3:] if discussion_log else 'Starting discussion'}
                    Initial message: {initial_message}
                    
                    Provide your perspective on this topic as {agent_spec['name']}.
                    """
                    
                    # Use model manager to generate response
                    response = await model_manager.general_ai_response(prompt)
                    
                    message = {
                        "round": round_num + 1,
                        "agent": participant,
                        "role": agent_spec["role"],
                        "content": response,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    
                    discussion_log.append(message)
                    conversation["messages"].append(message)
        
        # Store conversation in vector memory
        conversation_text = "\n".join([f"{msg['agent']}: {msg['content']}" for msg in discussion_log])
        await vector_memory.store(
            content=conversation_text[:1000],  # Limit content size
            metadata={
                "type": "autogen_conversation",
                "conversation_id": conversation_id,
                "topic": conversation["topic"],
                "participants": conversation["participants"],
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        return {
            "success": True,
            "conversation_id": conversation_id,
            "discussion_log": discussion_log,
            "rounds_completed": max_rounds,
            "total_messages": len(discussion_log),
            "capabilities_used": ["multi_agent_conversation", "collaborative_problem_solving"],
            "agent_id": self.agent_id,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _collaborate_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Collaborate on a specific problem or task."""
        problem = task.get("problem", "")
        required_expertise = task.get("required_expertise", [])
        collaboration_type = task.get("type", "problem_solving")
        
        if not problem:
            return {"success": False, "error": "No problem specified"}
        
        # Find suitable agents for this problem
        suitable_agents = []
        for agent_name, agent_spec in self.agents_registry.items():
            agent_expertise = set(agent_spec.get("expertise", []))
            required_set = set(required_expertise)
            
            if agent_expertise.intersection(required_set) or not required_expertise:
                suitable_agents.append(agent_name)
        
        if not suitable_agents:
            suitable_agents = list(self.agents_registry.keys())[:3]  # Use first 3 as fallback
        
        # Create collaborative solution
        solutions = []
        for agent_name in suitable_agents[:5]:  # Limit to 5 agents
            agent_spec = self.agents_registry[agent_name]
            
            prompt = f"""
            You are {agent_spec['name']} with expertise in: {', '.join(agent_spec['expertise'])}
            Role: {agent_spec['role']}
            
            Problem to solve: {problem}
            Required expertise: {', '.join(required_expertise)}
            Collaboration type: {collaboration_type}
            
            Provide your solution approach and recommendations.
            """
            
            solution = await model_manager.general_ai_response(prompt)
            
            solutions.append({
                "agent": agent_name,
                "expertise": agent_spec["expertise"],
                "solution": solution,
                "confidence": 0.8  # Simulated confidence score
            })
        
        # Synthesize collaborative result
        synthesis_prompt = f"""
        Problem: {problem}
        
        Multiple expert solutions:
        {chr(10).join([f"{sol['agent']} ({', '.join(sol['expertise'])}): {sol['solution'][:200]}..." for sol in solutions])}
        
        Synthesize these solutions into a comprehensive, collaborative recommendation.
        """
        
        final_solution = await model_manager.general_ai_response(synthesis_prompt)
        
        return {
            "success": True,
            "problem": problem,
            "collaboration_type": collaboration_type,
            "participating_agents": suitable_agents,
            "individual_solutions": solutions,
            "synthesized_solution": final_solution,
            "capabilities_used": ["collaborative_problem_solving", "group_decision_making"],
            "agent_id": self.agent_id,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _general_autogen_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle general AutoGen tasks."""
        content = task.get("content", "")
        agents_needed = task.get("agents_needed", 2)
        
        if not content:
            return {"success": False, "error": "No content provided"}
        
        # Create a quick multi-agent discussion
        available_agents = list(self.agents_registry.keys())[:agents_needed]
        
        if not available_agents:
            # Create default agents if none exist
            await self._create_default_agents()
            available_agents = list(self.agents_registry.keys())[:agents_needed]
        
        responses = []
        for agent_name in available_agents:
            agent_spec = self.agents_registry[agent_name]
            
            prompt = f"""
            You are {agent_spec['name']} with role: {agent_spec['role']}
            System message: {agent_spec['system_message']}
            
            Task: {content}
            
            Provide your response as {agent_spec['name']}.
            """
            
            response = await model_manager.general_ai_response(prompt)
            responses.append({
                "agent": agent_name,
                "role": agent_spec["role"],
                "response": response
            })
        
        return {
            "success": True,
            "task_content": content,
            "agents_responded": len(responses),
            "responses": responses,
            "capabilities_used": ["multi_agent_conversation"],
            "agent_id": self.agent_id,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _create_default_agents(self):
        """Create default agent configurations."""
        default_agents = [
            {
                "name": "analyst",
                "role": "data_analyst",
                "expertise": ["data_analysis", "statistics", "reporting"],
                "system_message": "You are a data analyst focused on extracting insights from information."
            },
            {
                "name": "developer",
                "role": "software_developer",
                "expertise": ["programming", "software_design", "debugging"],
                "system_message": "You are a software developer focused on creating and improving code."
            },
            {
                "name": "researcher",
                "role": "researcher",
                "expertise": ["research", "fact_checking", "knowledge_discovery"],
                "system_message": "You are a researcher focused on finding and verifying information."
            },
            {
                "name": "coordinator",
                "role": "project_manager",
                "expertise": ["project_management", "coordination", "planning"],
                "system_message": "You are a project manager focused on organizing and coordinating tasks."
            }
        ]
        
        for agent_config in default_agents:
            agent_config["created_at"] = datetime.utcnow()
            agent_config["capabilities"] = agent_config["expertise"]
            self.agents_registry[agent_config["name"]] = agent_config
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get current status of AutoGen system."""
        return {
            "registered_agents": len(self.agents_registry),
            "active_conversations": len(self.active_conversations),
            "agent_types": list(set([agent["role"] for agent in self.agents_registry.values()])),
            "expertise_areas": list(set([exp for agent in self.agents_registry.values() for exp in agent.get("expertise", [])])),
            "last_activity": datetime.utcnow().isoformat()
        }

# Global instance
autogen_agent = AutoGenAgent()