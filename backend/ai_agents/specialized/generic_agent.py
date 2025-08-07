"""
Generic Agent - Universal Fallback Agent
========================================

This agent serves as a universal fallback for any tasks that don't match
specialized agents. It can handle general reasoning, communication, and
basic task execution using local Ollama models.
"""

import asyncio
import json
from typing import Dict, Any, List
from datetime import datetime

from ..core.base_agent import BaseAgent, AgentMessage, AgentStatus, AgentCapability


class GenericAgent(BaseAgent):
    """
    Generic fallback agent that can handle any type of task
    
    Capabilities:
    - General reasoning and problem solving
    - Text processing and analysis
    - Basic communication tasks
    - Flexible task execution
    """
    
    async def on_initialize(self):
        """Initialize generic agent"""
        self.logger.info("Initializing Generic Agent")
        
        # Register message handlers
        self.register_message_handler("general_task", self._handle_general_task)
        self.register_message_handler("analyze_text", self._handle_analyze_text)
        self.register_message_handler("generate_response", self._handle_generate_response)
        self.register_message_handler("process_request", self._handle_process_request)
        
        # Add basic capabilities
        self.add_capability(AgentCapability.REASONING)
        self.add_capability(AgentCapability.COMMUNICATION)
        
        self.logger.info("Generic Agent initialized successfully")
    
    async def on_task_execute(self, task_id: str, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute any type of task"""
        task_type = task_data.get("task_type", "general")
        
        try:
            if task_type == "general":
                return await self._execute_general_task(task_id, task_data)
            elif task_type == "analyze":
                return await self._execute_analysis_task(task_id, task_data)
            elif task_type == "generate":
                return await self._execute_generation_task(task_id, task_data)
            elif task_type == "process":
                return await self._execute_processing_task(task_id, task_data)
            elif task_type == "communicate":
                return await self._execute_communication_task(task_id, task_data)
            else:
                # Handle unknown task types with general approach
                return await self._execute_unknown_task(task_id, task_data)
                
        except Exception as e:
            self.logger.error(f"Generic task {task_id} failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "task_id": task_id
            }
    
    async def _execute_general_task(self, task_id: str, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a general task using AI reasoning"""
        task_description = task_data.get("description", "")
        task_input = task_data.get("input", "")
        context = task_data.get("context", "")
        
        if not task_description and not task_input:
            return {
                "success": False,
                "error": "No task description or input provided",
                "task_id": task_id
            }
        
        system_prompt = """You are a versatile AI assistant capable of handling any type of task. Analyze the given task and provide appropriate results.

Guidelines:
- Understand the task requirements clearly
- Provide comprehensive and accurate results
- Use logical reasoning and problem-solving skills
- Be helpful and informative in your responses
- Consider context and provide relevant insights"""

        user_prompt = f"""Execute the following task:

Task Description: {task_description}
Input: {task_input}
Context: {context}

Please analyze the task requirements and provide appropriate results. Be thorough and helpful in your response."""

        result = await self.query_model(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.5,
            max_tokens=2000
        )
        
        if not result:
            return {
                "success": False,
                "error": "Failed to execute general task",
                "task_id": task_id
            }
        
        return {
            "success": True,
            "result": {
                "task_description": task_description,
                "input": task_input,
                "output": result,
                "context": context,
                "executed_at": datetime.utcnow().isoformat()
            },
            "task_id": task_id
        }
    
    async def _execute_analysis_task(self, task_id: str, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an analysis task"""
        content_to_analyze = task_data.get("content", "")
        analysis_type = task_data.get("analysis_type", "general")
        focus_areas = task_data.get("focus_areas", [])
        
        if not content_to_analyze:
            return {
                "success": False,
                "error": "No content provided for analysis",
                "task_id": task_id
            }
        
        system_prompt = f"""You are an expert analyst. Perform {analysis_type} analysis on the given content.

Guidelines:
- Provide thorough and objective analysis
- Structure your analysis clearly
- Include insights and observations
- Highlight key findings and patterns
- Be specific and actionable in your recommendations"""

        focus_instruction = ""
        if focus_areas:
            focus_instruction = f"\nFocus Areas: {', '.join(focus_areas)}"

        user_prompt = f"""Analyze the following content:

Analysis Type: {analysis_type}{focus_instruction}

Content to Analyze:
{content_to_analyze}

Please provide a comprehensive analysis including:
1. Summary and overview
2. Key findings and insights
3. Patterns and trends identified
4. Strengths and weaknesses
5. Recommendations and next steps"""

        analysis_result = await self.query_model(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.3,
            max_tokens=2500
        )
        
        if not analysis_result:
            return {
                "success": False,
                "error": "Failed to perform analysis",
                "task_id": task_id
            }
        
        return {
            "success": True,
            "result": {
                "content": content_to_analyze,
                "analysis_type": analysis_type,
                "focus_areas": focus_areas,
                "analysis": analysis_result,
                "analyzed_at": datetime.utcnow().isoformat()
            },
            "task_id": task_id
        }
    
    async def _execute_generation_task(self, task_id: str, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a content generation task"""
        generation_type = task_data.get("generation_type", "text")
        prompt = task_data.get("prompt", "")
        parameters = task_data.get("parameters", {})
        
        if not prompt:
            return {
                "success": False,
                "error": "No prompt provided for generation",
                "task_id": task_id
            }
        
        system_prompt = f"""You are a creative content generator. Generate high-quality {generation_type} content based on the given prompt.

Guidelines:
- Create original and engaging content
- Follow the specified parameters and requirements
- Ensure the content is appropriate and useful
- Be creative while maintaining quality standards
- Structure the content clearly and logically"""

        user_prompt = f"""Generate {generation_type} content based on the following:

Prompt: {prompt}

Parameters: {json.dumps(parameters, indent=2)}

Please create high-quality content that meets the requirements and follows the specified parameters."""

        temperature = parameters.get("temperature", 0.7)
        max_tokens = parameters.get("max_tokens", 2000)
        
        generated_content = await self.query_model(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        if not generated_content:
            return {
                "success": False,
                "error": "Failed to generate content",
                "task_id": task_id
            }
        
        return {
            "success": True,
            "result": {
                "generation_type": generation_type,
                "prompt": prompt,
                "parameters": parameters,
                "generated_content": generated_content,
                "generated_at": datetime.utcnow().isoformat()
            },
            "task_id": task_id
        }
    
    async def _execute_processing_task(self, task_id: str, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a data processing task"""
        data_to_process = task_data.get("data", "")
        processing_type = task_data.get("processing_type", "transform")
        processing_rules = task_data.get("processing_rules", [])
        
        if not data_to_process:
            return {
                "success": False,
                "error": "No data provided for processing",
                "task_id": task_id
            }
        
        system_prompt = f"""You are a data processing expert. Process the given data according to the specified type and rules.

Guidelines:
- Follow the processing rules precisely
- Maintain data integrity and accuracy
- Provide clear processing results
- Handle edge cases appropriately
- Document any transformations applied"""

        rules_text = "\n".join(f"- {rule}" for rule in processing_rules) if processing_rules else "Apply standard processing"

        user_prompt = f"""Process the following data:

Processing Type: {processing_type}

Processing Rules:
{rules_text}

Data to Process:
{data_to_process}

Please process the data according to the specified type and rules, and provide the processed results along with a summary of the transformations applied."""

        processed_result = await self.query_model(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.2,
            max_tokens=2000
        )
        
        if not processed_result:
            return {
                "success": False,
                "error": "Failed to process data",
                "task_id": task_id
            }
        
        return {
            "success": True,
            "result": {
                "original_data": data_to_process,
                "processing_type": processing_type,
                "processing_rules": processing_rules,
                "processed_data": processed_result,
                "processed_at": datetime.utcnow().isoformat()
            },
            "task_id": task_id
        }
    
    async def _execute_communication_task(self, task_id: str, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a communication task"""
        message_type = task_data.get("message_type", "response")
        recipient_context = task_data.get("recipient_context", "")
        message_content = task_data.get("message_content", "")
        communication_style = task_data.get("style", "professional")
        
        if not message_content:
            return {
                "success": False,
                "error": "No message content provided",
                "task_id": task_id
            }
        
        system_prompt = f"""You are a communication expert. Craft appropriate {message_type} messages with a {communication_style} style.

Guidelines:
- Adapt the tone and style to the context
- Ensure clear and effective communication
- Be respectful and professional
- Consider the recipient's perspective
- Provide complete and useful information"""

        user_prompt = f"""Create a {message_type} message with the following details:

Message Type: {message_type}
Communication Style: {communication_style}
Recipient Context: {recipient_context}

Message Content:
{message_content}

Please craft an appropriate message that effectively communicates the content while considering the recipient context and maintaining the specified style."""

        communication_result = await self.query_model(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.4,
            max_tokens=1500
        )
        
        if not communication_result:
            return {
                "success": False,
                "error": "Failed to create communication",
                "task_id": task_id
            }
        
        return {
            "success": True,
            "result": {
                "message_type": message_type,
                "style": communication_style,
                "recipient_context": recipient_context,
                "original_content": message_content,
                "crafted_message": communication_result,
                "created_at": datetime.utcnow().isoformat()
            },
            "task_id": task_id
        }
    
    async def _execute_unknown_task(self, task_id: str, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle unknown task types with flexible approach"""
        task_type = task_data.get("task_type", "unknown")
        
        # Try to understand what the task is asking for
        system_prompt = """You are a versatile AI assistant. You've been given a task that doesn't fit standard categories. Analyze the task and do your best to provide helpful results.

Guidelines:
- Interpret the task requirements as best as possible
- Provide useful and relevant output
- Be creative in your approach
- Ask for clarification if something is unclear
- Focus on being helpful and informative"""

        user_prompt = f"""I need help with a task of type: {task_type}

Task Data: {json.dumps(task_data, indent=2)}

Please analyze this task and provide appropriate results. If the task requirements are unclear, please provide your best interpretation and suggest how to clarify the requirements."""

        result = await self.query_model(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.6,
            max_tokens=2000
        )
        
        if not result:
            return {
                "success": False,
                "error": f"Failed to handle unknown task type: {task_type}",
                "task_id": task_id
            }
        
        return {
            "success": True,
            "result": {
                "task_type": task_type,
                "task_data": task_data,
                "interpretation": result,
                "handled_at": datetime.utcnow().isoformat(),
                "note": "This was an unknown task type handled with flexible approach"
            },
            "task_id": task_id
        }
    
    # Message handlers
    async def _handle_general_task(self, message: AgentMessage):
        """Handle general task request"""
        content = message.content
        
        result = await self._execute_general_task(
            message.id,
            {
                "description": content.get("description", ""),
                "input": content.get("input", ""),
                "context": content.get("context", "")
            }
        )
        
        await self.send_message(
            message.sender_id,
            "general_task_result",
            result
        )
    
    async def _handle_analyze_text(self, message: AgentMessage):
        """Handle text analysis request"""
        content = message.content
        
        result = await self._execute_analysis_task(
            message.id,
            {
                "content": content.get("text", ""),
                "analysis_type": content.get("analysis_type", "general"),
                "focus_areas": content.get("focus_areas", [])
            }
        )
        
        await self.send_message(
            message.sender_id,
            "text_analysis_result",
            result
        )
    
    async def _handle_generate_response(self, message: AgentMessage):
        """Handle response generation request"""
        content = message.content
        
        result = await self._execute_generation_task(
            message.id,
            {
                "generation_type": content.get("type", "response"),
                "prompt": content.get("prompt", ""),
                "parameters": content.get("parameters", {})
            }
        )
        
        await self.send_message(
            message.sender_id,
            "generated_response",
            result
        )
    
    async def _handle_process_request(self, message: AgentMessage):
        """Handle general processing request"""
        content = message.content
        
        # Determine the best task type based on content
        if "analyze" in content.get("action", "").lower():
            result = await self._execute_analysis_task(message.id, content)
        elif "generate" in content.get("action", "").lower():
            result = await self._execute_generation_task(message.id, content)
        elif "process" in content.get("action", "").lower():
            result = await self._execute_processing_task(message.id, content)
        else:
            result = await self._execute_general_task(message.id, content)
        
        await self.send_message(
            message.sender_id,
            "process_request_result",
            result
        )
    
    async def on_message_received(self, message: AgentMessage):
        """Handle any unknown message types"""
        self.logger.info(f"Handling unknown message type: {message.message_type}")
        
        # Try to handle the message generically
        result = await self._execute_unknown_task(
            message.id,
            {
                "task_type": message.message_type,
                "message_content": message.content,
                "sender_id": message.sender_id
            }
        )
        
        await self.send_message(
            message.sender_id,
            "generic_response",
            result
        )
    
    async def on_shutdown(self):
        """Cleanup when shutting down"""
        self.logger.info("Generic Agent shutting down")