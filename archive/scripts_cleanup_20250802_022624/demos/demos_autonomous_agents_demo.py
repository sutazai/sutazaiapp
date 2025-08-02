#!/usr/bin/env python3
"""
SutazAI Autonomous Multi-Agent Demonstration System
====================================================

This demo showcases the power of LocalAGI orchestration with autonomous AI agents
working together through Redis communication, using local Ollama models.

Features:
- Multi-agent task delegation
- Redis-based agent communication
- LocalAGI orchestration
- Ollama model integration
- Code analysis and improvement workflow
- Autonomous decision making
- Real-time monitoring

Usage: python autonomous_agents_demo.py [--task TASK] [--agents N]
"""

import asyncio
import json
import logging
import random
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
import argparse
import sys
import os

# Add project root to path
sys.path.append('/opt/sutazaiapp')

try:
    import redis
    import httpx
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.live import Live
    from rich.layout import Layout
    from rich.text import Text
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Please install: pip install redis httpx rich")
    sys.exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

console = Console()

class AgentMessage:
    """Message structure for agent communication"""
    def __init__(self, agent_id: str, task_id: str, message_type: str, 
                 content: Any, target_agent: Optional[str] = None):
        self.id = str(uuid.uuid4())
        self.agent_id = agent_id
        self.task_id = task_id
        self.message_type = message_type
        self.content = content
        self.target_agent = target_agent
        self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'agent_id': self.agent_id,
            'task_id': self.task_id,
            'message_type': self.message_type,
            'content': self.content,
            'target_agent': self.target_agent,
            'timestamp': self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'AgentMessage':
        msg = cls(
            agent_id=data['agent_id'],
            task_id=data['task_id'],
            message_type=data['message_type'],
            content=data['content'],
            target_agent=data.get('target_agent')
        )
        msg.id = data['id']
        msg.timestamp = data['timestamp']
        return msg

class BaseAgent:
    """Base class for all AI agents in the system"""
    
    def __init__(self, agent_id: str, redis_client: redis.Redis, 
                 ollama_url: str = "http://localhost:11434"):
        self.agent_id = agent_id
        self.redis_client = redis_client
        self.ollama_url = ollama_url
        self.active_tasks = {}
        self.capabilities = []
        self.status = "idle"
        self.message_queue = f"agent_queue:{agent_id}"
        
        # Register agent in Redis
        self.register_agent()
    
    def register_agent(self):
        """Register this agent in the system"""
        agent_data = {
            'id': self.agent_id,
            'type': self.__class__.__name__,
            'capabilities': self.capabilities,
            'status': self.status,
            'last_seen': datetime.now().isoformat()
        }
        self.redis_client.hset(f"agent:{self.agent_id}", mapping=agent_data)
        self.redis_client.sadd("active_agents", self.agent_id)
    
    async def send_message(self, message: AgentMessage):
        """Send message to another agent or broadcast"""
        message_data = json.dumps(message.to_dict())
        
        if message.target_agent:
            # Direct message to specific agent
            target_queue = f"agent_queue:{message.target_agent}"
            self.redis_client.lpush(target_queue, message_data)
        else:
            # Broadcast to all agents
            self.redis_client.publish("agent_broadcast", message_data)
    
    async def receive_messages(self) -> List[AgentMessage]:
        """Receive messages from Redis queue"""
        messages = []
        while True:
            message_data = self.redis_client.rpop(self.message_queue)
            if not message_data:
                break
            try:
                message_dict = json.loads(message_data)
                messages.append(AgentMessage.from_dict(message_dict))
            except json.JSONDecodeError:
                logger.error(f"Failed to decode message: {message_data}")
        return messages
    
    async def call_ollama(self, model: str, prompt: str, temperature: float = 0.7) -> str:
        """Call Ollama API for AI inference"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": temperature}
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result.get("response", "")
                else:
                    logger.error(f"Ollama API error: {response.status_code}")
                    return f"Error: Failed to get response from {model}"
                    
        except Exception as e:
            logger.error(f"Error calling Ollama: {e}")
            return f"Error: {str(e)}"
    
    async def process_task(self, task: Dict) -> Dict:
        """Process a task - to be implemented by subclasses"""
        raise NotImplementedError
    
    async def run(self):
        """Main agent loop"""
        logger.info(f"Agent {self.agent_id} starting...")
        
        while True:
            try:
                # Update status
                self.redis_client.hset(f"agent:{self.agent_id}", 
                                     "last_seen", datetime.now().isoformat())
                
                # Process incoming messages
                messages = await self.receive_messages()
                for message in messages:
                    await self.handle_message(message)
                
                # Perform agent-specific work
                await self.work_cycle()
                
                await asyncio.sleep(1)  # Prevent busy waiting
                
            except Exception as e:
                logger.error(f"Error in agent {self.agent_id}: {e}")
                await asyncio.sleep(5)
    
    async def handle_message(self, message: AgentMessage):
        """Handle incoming message"""
        logger.info(f"Agent {self.agent_id} received {message.message_type} from {message.agent_id}")
        
        if message.message_type == "task_request":
            await self.handle_task_request(message)
        elif message.message_type == "task_result":
            await self.handle_task_result(message)
        elif message.message_type == "coordination":
            await self.handle_coordination(message)
    
    async def handle_task_request(self, message: AgentMessage):
        """Handle task request"""
        task_data = message.content
        self.status = "working"
        
        try:
            result = await self.process_task(task_data)
            
            # Send result back
            response = AgentMessage(
                agent_id=self.agent_id,
                task_id=message.task_id,
                message_type="task_result",
                content=result,
                target_agent=message.agent_id
            )
            await self.send_message(response)
            
        except Exception as e:
            error_result = {"error": str(e), "status": "failed"}
            response = AgentMessage(
                agent_id=self.agent_id,
                task_id=message.task_id,
                message_type="task_result",
                content=error_result,
                target_agent=message.agent_id
            )
            await self.send_message(response)
        
        self.status = "idle"
    
    async def handle_task_result(self, message: AgentMessage):
        """Handle task result"""
        pass
    
    async def handle_coordination(self, message: AgentMessage):
        """Handle coordination message"""
        pass
    
    async def work_cycle(self):
        """Agent-specific work cycle"""
        pass

class CodeAnalyzerAgent(BaseAgent):
    """Agent specialized in code analysis"""
    
    def __init__(self, agent_id: str, redis_client: redis.Redis, ollama_url: str = "http://localhost:11434"):
        super().__init__(agent_id, redis_client, ollama_url)
        self.capabilities = ["code_analysis", "security_scanning", "complexity_analysis"]
    
    async def process_task(self, task: Dict) -> Dict:
        """Analyze code for issues and improvements"""
        code = task.get("code", "")
        analysis_type = task.get("analysis_type", "general")
        
        if analysis_type == "security":
            prompt = f"""Analyze this code for security vulnerabilities:

{code}

Provide a detailed security analysis including:
1. Potential vulnerabilities
2. Security best practices violations
3. Recommended fixes
4. Risk assessment (Low/Medium/High)

Format your response as JSON with fields: vulnerabilities, recommendations, risk_level, summary."""

        elif analysis_type == "performance":
            prompt = f"""Analyze this code for performance issues:

{code}

Provide a performance analysis including:
1. Performance bottlenecks
2. Memory usage concerns
3. Algorithm complexity
4. Optimization suggestions

Format your response as JSON with fields: bottlenecks, optimizations, complexity, summary."""

        else:
            prompt = f"""Analyze this code for general quality and improvements:

{code}

Provide a comprehensive code analysis including:
1. Code quality assessment
2. Design patterns usage
3. Maintainability score (1-10)
4. Specific improvement suggestions
5. Best practices compliance

Format your response as JSON with fields: quality_score, patterns, maintainability, improvements, summary."""

        # Use Ollama for analysis
        analysis_result = await self.call_ollama("tinyllama", prompt, temperature=0.3)
        
        return {
            "agent": self.agent_id,
            "task_type": "code_analysis",
            "analysis_type": analysis_type,
            "result": analysis_result,
            "timestamp": datetime.now().isoformat(),
            "status": "completed"
        }

class CodeImproverAgent(BaseAgent):
    """Agent specialized in code improvement and refactoring"""
    
    def __init__(self, agent_id: str, redis_client: redis.Redis, ollama_url: str = "http://localhost:11434"):
        super().__init__(agent_id, redis_client, ollama_url)
        self.capabilities = ["code_improvement", "refactoring", "optimization"]
    
    async def process_task(self, task: Dict) -> Dict:
        """Improve code based on analysis results"""
        code = task.get("code", "")
        analysis_results = task.get("analysis_results", {})
        improvement_type = task.get("improvement_type", "general")
        
        prompt = f"""Based on the following code analysis, provide improved code:

Original Code:
{code}

Analysis Results:
{json.dumps(analysis_results, indent=2)}

Improvement Type: {improvement_type}

Please provide:
1. Improved/refactored code
2. Explanation of changes made
3. Expected benefits
4. Any additional recommendations

Format your response as JSON with fields: improved_code, changes_made, benefits, recommendations."""

        # Use Ollama for code improvement
        improvement_result = await self.call_ollama("tinyllama", prompt, temperature=0.5)
        
        return {
            "agent": self.agent_id,
            "task_type": "code_improvement",
            "improvement_type": improvement_type,
            "result": improvement_result,
            "timestamp": datetime.now().isoformat(),
            "status": "completed"
        }

class LocalAGIOrchestratorAgent(BaseAgent):
    """LocalAGI orchestrator for managing multi-agent workflows"""
    
    def __init__(self, agent_id: str, redis_client: redis.Redis, ollama_url: str = "http://localhost:11434"):
        super().__init__(agent_id, redis_client, ollama_url)
        self.capabilities = ["orchestration", "task_delegation", "workflow_management"]
        self.pending_tasks = {}
        self.agent_capabilities = {}
    
    async def orchestrate_code_analysis_workflow(self, code: str, task_id: str):
        """Orchestrate a complete code analysis and improvement workflow"""
        logger.info(f"Starting orchestration for task {task_id}")
        
        # Step 1: Find available agents
        available_agents = await self.discover_agents()
        
        # Step 2: Delegate code analysis
        analyzer_agents = [agent for agent in available_agents 
                          if "code_analysis" in self.agent_capabilities.get(agent, [])]
        
        if not analyzer_agents:
            logger.error("No code analyzer agents available")
            return {"error": "No analyzer agents available"}
        
        # Delegate analysis tasks
        analysis_tasks = [
            {"code": code, "analysis_type": "security"},
            {"code": code, "analysis_type": "performance"},
            {"code": code, "analysis_type": "general"}
        ]
        
        analysis_results = []
        for i, task in enumerate(analysis_tasks):
            agent_id = analyzer_agents[i % len(analyzer_agents)]
            
            message = AgentMessage(
                agent_id=self.agent_id,
                task_id=f"{task_id}_analysis_{i}",
                message_type="task_request",
                content=task,
                target_agent=agent_id
            )
            
            await self.send_message(message)
            self.pending_tasks[f"{task_id}_analysis_{i}"] = {
                "type": "analysis",
                "started": time.time(),
                "agent": agent_id
            }
        
        # Wait for analysis results
        logger.info("Waiting for analysis results...")
        analysis_results = await self.wait_for_results([f"{task_id}_analysis_{i}" for i in range(len(analysis_tasks))])
        
        # Step 3: Delegate code improvement
        improver_agents = [agent for agent in available_agents 
                          if "code_improvement" in self.agent_capabilities.get(agent, [])]
        
        if improver_agents:
            improvement_task = {
                "code": code,
                "analysis_results": analysis_results,
                "improvement_type": "comprehensive"
            }
            
            agent_id = improver_agents[0]
            message = AgentMessage(
                agent_id=self.agent_id,
                task_id=f"{task_id}_improvement",
                message_type="task_request",
                content=improvement_task,
                target_agent=agent_id
            )
            
            await self.send_message(message)
            self.pending_tasks[f"{task_id}_improvement"] = {
                "type": "improvement",
                "started": time.time(),
                "agent": agent_id
            }
            
            improvement_results = await self.wait_for_results([f"{task_id}_improvement"])
        else:
            improvement_results = []
        
        # Step 4: Compile final report
        final_report = await self.compile_workflow_report(analysis_results, improvement_results, task_id)
        
        return final_report
    
    async def discover_agents(self) -> List[str]:
        """Discover available agents and their capabilities"""
        agent_ids = self.redis_client.smembers("active_agents")
        available_agents = []
        
        for agent_id in agent_ids:
            agent_data = self.redis_client.hgetall(f"agent:{agent_id.decode()}")
            if agent_data:
                agent_id_str = agent_id.decode()
                if agent_id_str != self.agent_id:  # Don't include self
                    available_agents.append(agent_id_str)
                    capabilities = agent_data.get(b'capabilities', b'[]').decode()
                    try:
                        self.agent_capabilities[agent_id_str] = json.loads(capabilities)
                    except json.JSONDecodeError:
                        self.agent_capabilities[agent_id_str] = []
        
        return available_agents
    
    async def wait_for_results(self, task_ids: List[str], timeout: int = 60) -> List[Dict]:
        """Wait for task results with timeout"""
        results = []
        start_time = time.time()
        completed_tasks = set()
        
        while len(completed_tasks) < len(task_ids) and (time.time() - start_time) < timeout:
            messages = await self.receive_messages()
            
            for message in messages:
                if (message.message_type == "task_result" and 
                    message.task_id in task_ids and 
                    message.task_id not in completed_tasks):
                    
                    results.append({
                        "task_id": message.task_id,
                        "agent_id": message.agent_id,
                        "result": message.content
                    })
                    completed_tasks.add(message.task_id)
                    
                    if message.task_id in self.pending_tasks:
                        del self.pending_tasks[message.task_id]
            
            await asyncio.sleep(0.5)
        
        # Handle timeouts
        for task_id in task_ids:
            if task_id not in completed_tasks:
                logger.warning(f"Task {task_id} timed out")
                results.append({
                    "task_id": task_id,
                    "agent_id": "timeout",
                    "result": {"error": "Task timed out", "status": "timeout"}
                })
        
        return results
    
    async def compile_workflow_report(self, analysis_results: List[Dict], 
                                    improvement_results: List[Dict], task_id: str) -> Dict:
        """Compile final workflow report using LocalAGI reasoning"""
        
        prompt = f"""As LocalAGI, compile a comprehensive workflow report from the following agent results:

Analysis Results:
{json.dumps(analysis_results, indent=2)}

Improvement Results:
{json.dumps(improvement_results, indent=2)}

Task ID: {task_id}

Provide a comprehensive report including:
1. Executive summary of findings
2. Critical issues identified
3. Recommended actions (prioritized)
4. Implementation roadmap
5. Success metrics
6. Risk assessment

Format as JSON with fields: executive_summary, critical_issues, recommendations, roadmap, metrics, risks."""

        report_content = await self.call_ollama("tinyllama", prompt, temperature=0.4)
        
        return {
            "workflow_id": task_id,
            "orchestrator": self.agent_id,
            "completion_time": datetime.now().isoformat(),
            "agents_involved": len(set([r.get("agent_id") for r in analysis_results + improvement_results])),
            "analysis_results": analysis_results,
            "improvement_results": improvement_results,
            "final_report": report_content,
            "status": "completed"
        }
    
    async def process_task(self, task: Dict) -> Dict:
        """Process orchestration tasks"""
        task_type = task.get("type", "unknown")
        
        if task_type == "code_workflow":
            code = task.get("code", "")
            task_id = task.get("task_id", str(uuid.uuid4()))
            return await self.orchestrate_code_analysis_workflow(code, task_id)
        
        return {"error": f"Unknown task type: {task_type}"}

class DemoManager:
    """Manages the autonomous agents demonstration"""
    
    def __init__(self, num_analyzer_agents: int = 2, num_improver_agents: int = 1):
        self.redis_client = redis.Redis(host='localhost', port=6379, 
                                      password='redis_password', decode_responses=False)
        self.agents = []
        self.console = Console()
        self.demo_active = False
        
        # Create agents
        self.create_agents(num_analyzer_agents, num_improver_agents)
    
    def create_agents(self, num_analyzers: int, num_improvers: int):
        """Create agent instances"""
        
        # Create LocalAGI orchestrator
        orchestrator = LocalAGIOrchestratorAgent("localagi_orchestrator", self.redis_client)
        self.agents.append(orchestrator)
        
        # Create analyzer agents
        for i in range(num_analyzers):
            agent = CodeAnalyzerAgent(f"analyzer_{i+1}", self.redis_client)
            self.agents.append(agent)
        
        # Create improver agents
        for i in range(num_improvers):
            agent = CodeImproverAgent(f"improver_{i+1}", self.redis_client)
            self.agents.append(agent)
    
    async def start_agents(self):
        """Start all agents"""
        self.console.print("[bold green]Starting SutazAI Autonomous Agent Demonstration[/bold green]")
        self.console.print(f"Initializing {len(self.agents)} agents...")
        
        # Start all agents
        agent_tasks = [asyncio.create_task(agent.run()) for agent in self.agents]
        
        # Wait a moment for agents to initialize
        await asyncio.sleep(3)
        
        self.console.print("[green]✓ All agents initialized and ready[/green]")
        
        return agent_tasks
    
    async def run_demo_workflow(self, code_sample: str):
        """Run the demonstration workflow"""
        self.console.print("\n[bold cyan]Starting Autonomous Code Analysis Workflow[/bold cyan]")
        
        # Get orchestrator
        orchestrator = self.agents[0]  # First agent is orchestrator
        
        # Create workflow task
        task = {
            "type": "code_workflow",
            "code": code_sample,
            "task_id": f"demo_{int(time.time())}"
        }
        
        # Start workflow
        with self.console.status("LocalAGI orchestrating multi-agent workflow..."):
            result = await orchestrator.orchestrate_code_analysis_workflow(
                code_sample, task["task_id"]
            )
        
        return result
    
    def display_results(self, result: Dict):
        """Display workflow results"""
        
        # Create main layout
        layout = Layout()
        
        # Workflow summary
        summary_table = Table(title="Workflow Summary")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")
        
        summary_table.add_row("Workflow ID", result.get("workflow_id", "N/A"))
        summary_table.add_row("Orchestrator", result.get("orchestrator", "N/A"))
        summary_table.add_row("Agents Involved", str(result.get("agents_involved", 0)))
        summary_table.add_row("Completion Time", result.get("completion_time", "N/A"))
        summary_table.add_row("Status", result.get("status", "N/A"))
        
        self.console.print(summary_table)
        
        # Analysis results
        if "analysis_results" in result:
            analysis_table = Table(title="Analysis Results")
            analysis_table.add_column("Task ID", style="cyan")
            analysis_table.add_column("Agent", style="yellow")
            analysis_table.add_column("Status", style="green")
            
            for analysis in result["analysis_results"]:
                task_id = analysis.get("task_id", "N/A")
                agent_id = analysis.get("agent_id", "N/A")
                status = analysis.get("result", {}).get("status", "N/A")
                analysis_table.add_row(task_id, agent_id, status)
            
            self.console.print(analysis_table)
        
        # Final report
        if "final_report" in result:
            report_panel = Panel(
                result["final_report"][:500] + "..." if len(result["final_report"]) > 500 else result["final_report"],
                title="LocalAGI Final Report",
                border_style="green"
            )
            self.console.print(report_panel)
    
    async def monitor_agents(self, duration: int = 30):
        """Monitor agent activity"""
        self.console.print(f"\n[yellow]Monitoring agent activity for {duration} seconds...[/yellow]")
        
        end_time = time.time() + duration
        
        while time.time() < end_time:
            # Get agent status
            agent_table = Table(title="Agent Status")
            agent_table.add_column("Agent ID", style="cyan")
            agent_table.add_column("Type", style="yellow")
            agent_table.add_column("Status", style="green")
            agent_table.add_column("Last Seen", style="blue")
            
            active_agents = self.redis_client.smembers("active_agents")
            for agent_id in active_agents:
                agent_data = self.redis_client.hgetall(f"agent:{agent_id.decode()}")
                if agent_data:
                    agent_table.add_row(
                        agent_id.decode(),
                        agent_data.get(b'type', b'Unknown').decode(),
                        agent_data.get(b'status', b'Unknown').decode(),
                        agent_data.get(b'last_seen', b'Unknown').decode()
                    )
            
            self.console.clear()
            self.console.print(agent_table)
            await asyncio.sleep(2)

async def main():
    """Main demonstration function"""
    parser = argparse.ArgumentParser(description="SutazAI Autonomous Agents Demo")
    parser.add_argument("--task", default="analyze", help="Demo task to run")
    parser.add_argument("--agents", type=int, default=3, help="Number of agents to create")
    parser.add_argument("--code-file", help="Path to code file to analyze")
    
    args = parser.parse_args()
    
    # Sample code for analysis
    if args.code_file and os.path.exists(args.code_file):
        with open(args.code_file, 'r') as f:
            code_sample = f.read()
    else:
        code_sample = '''
def vulnerable_function(user_input):
    """Example function with multiple issues"""
    import os
    import subprocess
    
    # Security issue: SQL injection vulnerability
    query = f"SELECT * FROM users WHERE name = '{user_input}'"
    
    # Security issue: Command injection
    result = subprocess.run(f"echo {user_input}", shell=True, capture_output=True)
    
    # Performance issue: Inefficient algorithm
    data = []
    for i in range(1000):
        for j in range(1000):
            if i * j % 2 == 0:
                data.append(i * j)
    
    # Code quality issue: No error handling
    with open("/tmp/data.txt", "w") as f:
        f.write(str(data))
    
    return query, result.stdout, len(data)

class BadClass:
    def __init__(self):
        self.data = None
    
    def process_data(self, data):
        # No input validation
        self.data = data
        return self.data.upper()  # Will fail if data is not string
'''
    
    # Create demo manager
    num_agents = max(2, args.agents)
    demo = DemoManager(num_analyzer_agents=2, num_improver_agents=1)
    
    try:
        # Start agents
        agent_tasks = await demo.start_agents()
        
        # Run demonstration workflow
        if args.task == "analyze":
            result = await demo.run_demo_workflow(code_sample)
            demo.display_results(result)
        
        elif args.task == "monitor":
            await demo.monitor_agents(60)
        
        else:
            console.print(f"[red]Unknown task: {args.task}[/red]")
    
    except KeyboardInterrupt:
        console.print("\n[yellow]Demo interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"[red]Demo error: {e}[/red]")
        logger.error(f"Demo error: {e}", exc_info=True)
    finally:
        # Cleanup
        console.print("[yellow]Cleaning up agents...[/yellow]")
        for task in agent_tasks:
            task.cancel()

if __name__ == "__main__":
    asyncio.run(main())