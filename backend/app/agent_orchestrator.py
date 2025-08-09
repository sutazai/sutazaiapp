"""
Compatibility shim for Agent Orchestrator.
Source of truth lives at app.services.agent_orchestrator.
This module re-exports the canonical implementation to preserve imports.
"""
from app.services.agent_orchestrator import AgentOrchestrator  # noqa: F401
                url="http://privategpt",
                port=8001,
                capabilities=["private_llm", "document_qa"]
            ),
            AgentType.LLAMAINDEX: Agent(
                name="LlamaIndex",
                type=AgentType.LLAMAINDEX,
                url="http://llamaindex",
                port=8080,
                capabilities=["data_indexing", "rag", "retrieval"]
            ),
            AgentType.FLOWISE: Agent(
                name="FlowiseAI",
                type=AgentType.FLOWISE,
                url="http://flowise",
                port=3000,
                capabilities=["chatflow_builder", "visual_ai_flows"]
            ),
            AgentType.SHELLGPT: Agent(
                name="ShellGPT",
                type=AgentType.SHELLGPT,
                url="http://shellgpt",
                port=8080,
                capabilities=["cli_assistance", "command_generation"]
            ),
            AgentType.PENTESTGPT: Agent(
                name="PentestGPT",
                type=AgentType.PENTESTGPT,
                url="http://pentestgpt",
                port=8080,
                capabilities=["security_testing", "penetration_testing"]
            )
        }
        
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task using the appropriate agent(s)"""
        task_id = f"task_{datetime.now().timestamp()}"
        task["id"] = task_id
        
        # Determine best agent for the task
        agent_type = self._select_agent(task)
        
        if agent_type:
            agent = self.agents[agent_type]
            
            # Check if agent is healthy
            if agent.status != "healthy":
                await self._check_agent_health(agent)
                
            if agent.status == "healthy":
                result = await self._execute_on_agent(agent, task)
                return {
                    "task_id": task_id,
                    "agent": agent.name,
                    "result": result,
                    "status": "completed"
                }
            else:
                return {
                    "task_id": task_id,
                    "error": f"Agent {agent.name} is not available",
                    "status": "failed"
                }
        else:
            # Use multi-agent collaboration
            return await self._multi_agent_execution(task)
            
    def _select_agent(self, task: Dict[str, Any]) -> Optional[AgentType]:
        """Select the best agent for a task"""
        task_type = task.get("type", "").lower()
        task_desc = task.get("description", "").lower()
        
        # Match task to agent capabilities
        if "code" in task_type or "code" in task_desc:
            if "generate" in task_desc:
                return AgentType.GPT_ENGINEER
            elif "complete" in task_desc:
                return AgentType.TABBYML
            elif "edit" in task_desc or "fix" in task_desc:
                return AgentType.AIDER
        elif "security" in task_type:
            if "scan" in task_desc:
                return AgentType.SEMGREP
            elif "pentest" in task_desc:
                return AgentType.PENTESTGPT
        elif "document" in task_type:
            return AgentType.DOCUMIND
        elif "financial" in task_type:
            return AgentType.FINROBOT
        elif "web" in task_type:
            if "automate" in task_desc:
                return AgentType.BROWSER_USE
            elif "scrape" in task_desc:
                return AgentType.SKYVERN
        elif "workflow" in task_type:
            return AgentType.LANGFLOW
        elif "chat" in task_type:
            return AgentType.BIGAGI
            
        # Default to AutoGPT for general tasks
        return AgentType.AUTOGPT
        
    async def _execute_on_agent(self, agent: Agent, task: Dict[str, Any]) -> Any:
        """Execute a task on a specific agent"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{agent.url}:{agent.port}/execute",
                    json=task,
                    timeout=60.0
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    return {"error": f"Agent returned status {response.status_code}"}
                    
        except Exception as e:
            logger.error(f"Error executing on {agent.name}: {e}")
            return {"error": str(e)}
            
    async def _multi_agent_execution(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute task using multiple agents collaboratively"""
        # Use CrewAI for multi-agent coordination
        crew_agent = self.agents[AgentType.CREWAI]
        
        if crew_agent.status == "healthy":
            return await self._execute_on_agent(crew_agent, task)
        else:
            # Fallback to sequential execution
            results = []
            for agent_type, agent in self.agents.items():
                if agent.status == "healthy":
                    result = await self._execute_on_agent(agent, task)
                    results.append({
                        "agent": agent.name,
                        "result": result
                    })
                    
            return {
                "task_id": task["id"],
                "results": results,
                "status": "completed"
            }
            
    async def _health_monitor(self):
        """Monitor health of all agents"""
        while True:
            try:
                for agent in self.agents.values():
                    await self._check_agent_health(agent)
                    
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                
    async def _check_agent_health(self, agent: Agent):
        """Check health of a specific agent"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{agent.url}:{agent.port}/health",
                    timeout=5.0
                )
                
                if response.status_code == 200:
                    agent.status = "healthy"
                else:
                    agent.status = "unhealthy"
                    
        except Exception:
            agent.status = "unreachable"
            
        agent.last_health_check = datetime.now()
        
    async def _task_processor(self):
        """Process tasks from the queue"""
        while True:
            try:
                task = await self.task_queue.get()
                result = await self.execute_task(task)
                
                # Store result
                self.active_tasks[task["id"]] = result
                
            except Exception as e:
                logger.error(f"Task processor error: {e}")
                
    async def list_agents(self) -> List[Dict[str, Any]]:
        """List all available agents and their status"""
        return [
            {
                "name": agent.name,
                "type": agent.type.value,
                "url": f"{agent.url}:{agent.port}",
                "capabilities": agent.capabilities,
                "status": agent.status,
                "last_health_check": agent.last_health_check.isoformat() if agent.last_health_check else None
            }
            for agent in self.agents.values()
        ]
        
    async def health_check(self) -> Dict[str, Any]:
        """Check orchestrator health"""
        healthy_agents = sum(1 for a in self.agents.values() if a.status == "healthy")
        total_agents = len(self.agents)
        
        return {
            "status": "healthy" if self.initialized else "initializing",
            "healthy_agents": healthy_agents,
            "total_agents": total_agents,
            "task_queue_size": self.task_queue.qsize(),
            "active_tasks": len(self.active_tasks)
        }
        
    async def shutdown(self):
        """Gracefully shutdown the orchestrator"""
        logger.info("Shutting down Agent Orchestrator...")
        # Cancel running tasks
        # Clean up resources
        self.initialized = False 
