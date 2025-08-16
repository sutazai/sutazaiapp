SutazAI Agent Services – Architecture and Operations Manual
Introduction

SutazAI’s agent system is composed of multiple specialized AI agents coordinated by a central Agent Orchestrator service. This document provides a comprehensive guide to understanding, extending, operating, testing, and deploying all agent services in the SutazAI platform. We detail the runtime architecture, orchestration workflows, inter-service protocols, agent endpoints, messaging queues, and agent-specific behaviors. It also covers shared schemas and message contracts, required HTTP APIs, metrics and tracing strategies, development lifecycle (from creation to deployment and rollback), agent capabilities and configurations (including integration with MCP, Ollama, and vector databases), security controls and secret management, failure recovery mechanisms, CI/testing standards, performance targets (SLOs), resource management policies, and CI/CD and monitoring integration. By following this guide, an engineer can manage the entire agent ecosystem end-to-end with zero ambiguity.

System Architecture and Orchestration
Microservices Overview

The SutazAI agent system follows a microservices model. The core components include:

AI Agent Orchestrator – a FastAPI-based coordination service (running in its own container, e.g. sutazai-agent-orchestrator) that registers agents, routes tasks to the appropriate agent, monitors agent health, and resolves conflicts
GitHub
GitHub
. It listens on a dedicated port (e.g. 8589 or as per config) and provides a REST API for orchestrating agent interactions. It is considered the “brain” of the multi-agent network.

Agent Services (Agents) – multiple specialized agent microservices, each typically running a FastAPI server (with Uvicorn) on port 8080 by default
GitHub
GitHub
. These agents encapsulate specific capabilities (e.g. coding assistance, data analysis, task automation, etc.) and expose a standard interface (HTTP endpoints like health and task execution) to receive tasks from the orchestrator and return results. Agents may be custom internal services or wrappers around third-party AI tools. For example, SutazAI integrates 23+ agent types including well-known systems like AutoGPT, AgentGPT, GPT-Engineer, LangChain-based tools, etc., each deployed as its own container
GitHub
GitHub
.

Supporting Services – shared infrastructure used by the agents, including a Redis cache (for transient state and optional discovery), a RabbitMQ message broker (for asynchronous task queueing and broadcast messaging), and Consul for service discovery in some configurations. For instance, the orchestrator can use Redis to discover agents by keys or persist state, and RabbitMQ to dispatch tasks and receive agent heartbeats on topics
GitHub
GitHub
. Additionally, an MCP (Model Context Protocol) Server is integrated to bridge the agent system with external interfaces like Claude Desktop, providing model management via Ollama, vector database queries, and multi-agent orchestration tools
GitHub
GitHub
. (The MCP server runs separately, coordinating with the orchestrator and agents to fulfill requests from the UI.)

Architecture Diagram (Conceptual): The Agent Orchestrator sits at the center, maintaining a registry of all agent services. Agents register themselves or are statically configured at startup. When a client or external system submits a task to the orchestrator, the orchestrator selects the appropriate agent to handle it (based on capabilities, load, and task type), then forwards the task to that agent. Agents execute tasks (often leveraging local AI models via Ollama or performing specialized computations) and return results back to the orchestrator, which aggregates responses and sends them to the requester. All agent services expose HTTP endpoints for health and task execution so that the orchestrator can communicate with them uniformly
GitHub
GitHub
. In addition, a message queue (RabbitMQ) may be used under the hood for decoupled communication: e.g. agents can send heartbeat and status messages to the orchestrator’s exchange, and the orchestrator can broadcast tasks or notifications via messaging
GitHub
. For service discovery, agents can be auto-registered when they start up by publishing a registration message or writing to a known store (e.g. a Redis key or Consul service entry) that the orchestrator monitors
GitHub
.

Technology Stack: Agents and orchestrator are implemented in Python (FastAPI for web framework, asyncio for concurrency) and run in Docker containers
GitHub
. Each agent container uses a standardized Python 3.11 slim base with dependencies like FastAPI, Uvicorn, Pydantic, httpx, etc. installed
GitHub
GitHub
. Agents that integrate AI models rely on Ollama (a local LLM runtime) via its HTTP API for generating completions or embeddings using local models – ensuring the solution is 100% local with no external API calls
GitHub
GitHub
. Other infrastructure components include RabbitMQ (for messaging), Postgres (for any persistent state if needed by MCP or others), and Prometheus/Grafana for observability. The system is containerized and orchestrated via Docker Compose or Kubernetes, following a declarative port registry for consistent port assignments
GitHub
GitHub
.

Agent Orchestrator and Workflow Coordination

The Agent Orchestrator is responsible for high-level coordination of tasks across agents. It maintains an in-memory registry of available agents (including their name, type, base URL, port, capabilities, and current status)
GitHub
GitHub
. In the current implementation, the orchestrator populates this registry at startup with all known agent types. For example, it registers agents like:

AutoGPT (autogpt service) on port 8080, capabilities: task_automation, planning, execution
GitHub

LocalAGI (localagi) on 8080, capabilities: agi_orchestration, workflow_management
GitHub

TabbyML (tabbyml) on 8080, capabilities: code_completion, code_suggestions
GitHub

Semgrep (semgrep) on 8080, capabilities: security_scanning, vulnerability_detection
GitHub

CrewAI (crewai) on 8080, capabilities: multi_agent_collaboration, team_coordination
GitHub

GPT-Engineer (gpt-engineer) on 8080, capabilities: code_generation, project_scaffolding
GitHub
, and so on through many others (Aider, LangFlow, Dify, AgentGPT, etc., up to specialized ones like PentestGPT for security testing
GitHub
). In total, the orchestrator knows about dozens of agents by type. (The Port Registry file also documents each agent service with a unique container name and a standardized port in the range 11000-11148 mapped to internal 8080
GitHub
GitHub
, ensuring no conflicts.)

Task Submission: When a task request comes in (either via the orchestrator’s API or an internal schedule), the orchestrator determines which agent should handle it. Each incoming task is characterized by a task_type and a description/payload. The orchestrator uses a selection algorithm to match the task to an agent’s capabilities
GitHub
GitHub
. For example, if the task is code-related (task type or description contains "code"), it might choose GPT-Engineer for code generation or TabbyML for code completion depending on context
GitHub
. If it’s a security scan task, Semgrep or PentestGPT would be selected
GitHub
. If it’s a web automation task, BrowserUse or Skyvern would be appropriate
GitHub
. These matching rules can be configured and extended; a centralized agents.yaml configuration defines capability-taxonomy and preferred agents for each task type
GitHub
GitHub
. If a single best-fit agent is identified, the orchestrator proceeds with that agent.

Multi-Agent Orchestration: If no single agent is clearly suited or the task is complex, the orchestrator can engage multiple agents. For instance, if the task requires a collaborative workflow (e.g., building a full-stack application involving front-end and back-end), the orchestrator might delegate parts of the task to different specialized agents and coordinate the sequence. There is a special CrewAI agent for multi-agent collaboration – the orchestrator will offload the task to CrewAI which in turn orchestrates a team of agents
GitHub
. If CrewAI is unavailable, the orchestrator can fall back to a sequential execution: iterating through available agents and having each perform what they can on the task
GitHub
GitHub
. The system also supports defined workflow templates for common multi-phase projects (e.g. a Code Development Pipeline, Security Audit pipeline, etc.), where the orchestrator (or a higher-level Orchestrator Agent in the “Jarvis” core) breaks down a complex goal into phases and tasks for various agents
GitHub
GitHub
. In such cases, the orchestrator ensures dependencies between sub-tasks are respected (e.g., code must be generated before tests can run) and oversees the entire workflow through completion
GitHub
GitHub
.

Task Queue and Processing: The orchestrator runs an internal async task queue to manage incoming tasks and not overload agents
GitHub
GitHub
. Tasks are enqueued (with unique IDs and timestamps) and a background task processor coroutine pulls from the queue and dispatches tasks one by one
GitHub
. This design allows throttling and prioritization. Each task carries a priority level (1-10) – by default normal priority 5 unless specified, allowing critical tasks to be prioritized. The orchestrator (and the optional Task Assignment Coordinator agent, described later) can use a priority queue to always run highest priority tasks first
GitHub
GitHub
. The AgentTask schema tracks the task’s status (pending/running/completed/etc.), creation time, optional timeout and retry count
GitHub
GitHub
. The orchestrator marks tasks with statuses and stores results in an active_tasks dictionary for retrieval
GitHub
GitHub
. If a task fails or times out, the orchestrator can decide to retry it (up to a configured max retries) or mark it failed; the AgentTask model provides a can_retry() helper to decide if a retry is allowed
GitHub
.

Inter-Service Communication: The primary mode of communication between orchestrator and agents is via HTTP calls to the agents’ endpoints (especially for immediate task execution). For example, to dispatch a task, the orchestrator will send an HTTP POST to the target agent’s /execute (or /task) endpoint with a JSON payload of the task
GitHub
. All agents implement a Task execution API that accepts a JSON task (often containing fields like id, type, description or data) and returns a JSON result. In our standard agent container template, this is implemented as a POST /task endpoint that returns a TaskResponse containing status, agent name, result dict, and timestamp
GitHub
GitHub
. The orchestrator uses either that or an /execute alias, ensuring a consistent contract. Additionally, asynchronous messaging is used for certain control flows: the orchestrator and agents can communicate via RabbitMQ topics for broadcasts (e.g., an agent can send a heartbeat message to a common exchange). The system defines structured message schemas for such events, including AgentRegistrationMessage, AgentHeartbeatMessage, AgentStatusMessage, etc., all carrying standardized fields
GitHub
GitHub
. For instance, a heartbeat message includes the agent’s ID, status (e.g. active/idle), current load (0.0–1.0), active task count, available capacity, CPU and memory usage percentages, uptime, and error count
GitHub
. These keep the orchestrator informed of agent health in real-time and allow dynamic adjustments (auto-deregistering an agent if heartbeats stop, etc.).

Background Monitors: The orchestrator launches background tasks to continuously monitor the system. A Health Monitor loop pings each agent’s /health endpoint periodically (e.g. every 30 seconds) to update their status
GitHub
GitHub
. A healthy agent responds with HTTP 200 and perhaps a simple JSON like {"status":"healthy","agent":"X", ...}; the orchestrator then marks it as "healthy" in the registry
GitHub
. If an agent’s health check fails or times out, the orchestrator marks it "unreachable" or "unhealthy"
GitHub
 and will avoid routing tasks to it until it recovers. (Agents themselves implement a GET /health endpoint that returns their basic status and possibly version info
GitHub
. The orchestrator relies on this for liveness checks.) The orchestrator may also run a background interaction monitor (for multi-step interactions), conflict detector, and performance optimizer, though in the current code these are placeholders with no substantial logic
GitHub
GitHub
. Those components are envisioned to handle more complex scenarios like coordinating multi-agent dialogues or optimizing task assignments over time.

Task Assignment Coordinator (Optional): In advanced deployments, a dedicated agent service called Task Assignment Coordinator (running on a separate port, e.g. 8551) takes over the responsibility of queuing and assigning tasks to agents in an optimal way
GitHub
. The Orchestrator can forward tasks to this coordinator which maintains a priority queue of up to 10,000 tasks and uses various scheduling strategies
GitHub
. Strategies include round-robin, least-loaded agent first, capability matching, and strict priority-based assignment
GitHub
. By default a capability-based matching is used (ensuring tasks go to agents that declare the required capability)
GitHub
GitHub
. The coordinator will pop tasks from the queue in order and dispatch them to agents, honoring concurrency limits and timeouts. It tracks metrics on queue length and processing times, accessible via its /queue and /metrics endpoints
GitHub
. In essence, this Coordinator agent is an extension of orchestrator logic to handle high throughput scenarios or complex scheduling policies. If enabled, the Orchestrator’s /submit_task endpoint will hand off tasks to the coordinator’s queue instead of calling the agent directly
GitHub
GitHub
, and the coordinator then calls the target agent’s /execute. The use of a separate coordinator is optional; in many cases the Orchestrator’s internal async queue suffices, but the architecture allows plugging one in for enterprise scenarios.

Resource Arbitration Agent (Optional): Another specialized internal service is the Resource Arbitration Agent (port ~8588)
GitHub
. This agent manages system resources (CPU, GPU, memory, etc.) across all agents to prevent resource contention and enforce fairness or priority policies. When an agent is about to run a heavy task, it can request resources from the Arbitrator (via a POST /allocate with requested resource amounts); the Arbitrator will check current usage and either grant an allocation or ask the agent to wait
GitHub
GitHub
. The Arbitrator keeps track of active allocations and can preempt lower priority tasks if a higher priority request comes in and resources are scarce. By default, resource allocation policies might limit each agent to a certain share (e.g. max 30% CPU per agent) and ensure some headroom: e.g. total CPU usage across agents capped at 80%, memory at 85%, etc
GitHub
. It periodically monitors actual system metrics using psutil and adjusts statuses (if an agent exceeds its allotment, it could be flagged or throttled)
GitHub
. The orchestrator or coordinator can consult the Arbitrator before dispatching a task to ensure the target agent has capacity. The Arbitrator provides APIs like /resources for current capacity, /allocations for current allocated slices, and /policies to view or update resource management policies
GitHub
. In practice, not all deployments will run this service, but it’s a crucial part of an end-to-end autonomous AI system where many agents might otherwise compete for finite resources. When enabled, agents should integrate calls to the Arbitrator; if disabled, agents rely on static Docker resource limits and OS scheduling.

Data/Memory Management: Some agents require sharing state or accessing common knowledge (vector database for embeddings, etc.). The platform includes a Vector DB Manager and adapters (for FAISS, Qdrant, Chroma, etc.) that agents can use to store or query vectorized knowledge. For example, an agent may embed incoming data via Ollama’s embedding API
GitHub
 and send it to the vector store for later retrieval. The MCP server provides a tool query_knowledge_base to allow queries against these vector stores
GitHub
GitHub
. Additionally, an Agent Memory subsystem exists for ephemeral memory: the MemoryManager and SharedMemoryManager allow agents to save context or share it across the team in memory or through Redis
GitHub
GitHub
. Each agent may have its own memory space, and a global shared memory for coordination is also possible (e.g. agents can publish key facts for others to use). These memory managers abstract the underlying storage (could be in-memory Python dict, Redis, or a vector DB for semantic memory). The orchestrator or agent manager ensures that when a new agent is created, a memory space is initialized for it
GitHub
. This is particularly useful for iterative workflows where intermediate results from one agent need to be accessible by another.

In summary, the architecture is a modular mesh of agent services orchestrated in a hub-and-spoke fashion by the Agent Orchestrator. The orchestrator handles registration, health monitoring, task routing (optionally delegating to a coordinator), and aggregates results. Agents perform the heavy lifting of AI tasks, each in isolation, but cooperating via defined protocols. The design emphasizes extensibility – new agents can be added easily without modifying the orchestrator (especially when dynamic discovery is fully enabled), and responsibilities can be split among specialized coordinator or arbitrator services for scalability and clarity.

Agent Types and Capabilities

Each agent in the system has a defined set of capabilities and an expected domain of tasks it can handle. Below is a breakdown of the various agents and their roles:

Agent Orchestrator: Capabilities: Orchestration, task routing, conflict resolution, agent coordination
GitHub
. This is the central service (not an AI model per se, but a coordinating agent). It does not execute user-domain tasks itself; instead, it orchestrates other agents. It tracks all agents and their capabilities and chooses which agent(s) should perform a given task. It also resolves conflicts (e.g. if two agents might perform overlapping actions or if an agent becomes unresponsive mid-task). The orchestrator emits metrics about overall system status (number of active agents, tasks completed/failed, etc.)
GitHub
. It has configuration for maximum concurrent tasks (e.g. 10) and monitors heartbeats to auto-deregister dead agents
GitHub
. The orchestrator should always be running as it’s the entry point for any task requests.

Task Assignment Coordinator: Capabilities: Task assignment, load balancing, priority management, queue management
GitHub
. This agent (if running) is essentially a scheduler. It manages a global priority queue of tasks. It implements multiple scheduling strategies – round-robin, least-loaded, capability_match, priority_based – all of which can be toggled or configured, with one set as default (in config, capability_match is default and enabled
GitHub
GitHub
). The coordinator ensures that high-priority tasks (e.g. urgent fixes) jump ahead of low-priority ones, and that tasks go to agents that are both capable and free. It collects metrics like queue length, average wait time, etc., which are exposed via its API
GitHub
. In operation, the orchestrator hands off tasks to the coordinator; the coordinator picks an agent and invokes it, then updates the orchestrator or a shared task status when done. This separation of concerns improves scalability in large systems.

Resource Arbitration Agent: Capabilities: Resource allocation, conflict detection, capacity management, resource monitoring
GitHub
. This agent is responsible for system-level resource arbitration as described. It knows the current capacity of CPU, memory, GPU, etc., and monitors usage (via psutil) across all agent containers. Each agent has resource usage metrics (CPU%, memory%) tracked in an AgentMetrics object
GitHub
. The Arbitrator collects these and can mark an agent as over-utilizing or relieve loads. If two agents demand the same limited resource (e.g. GPU), the arbitrator’s policies decide who gets it, possibly queueing the other’s request or even stopping a running task if preemption is allowed. The policies are configurable, e.g. maximum 80% CPU allocation to ensure OS overhead
GitHub
, per-agent limits (30% each) to avoid one agent starving others, etc. Agents that perform resource-intensive operations (like training models or running large queries) should request resources from this agent to get a time-bounded allocation ID, which they include in their workload. The arbitrator will release allocations on completion or timeout, and has an endpoint to list all current allocations and another to adjust policy thresholds on the fly
GitHub
GitHub
. This agent enhances system stability and fairness under heavy load.

“Meta” Agents (Internal Tools): There are several specialized internal agents that assist in maintaining the agent system itself or providing meta-functions:

Hardware Resource Optimizer: Optimizes local hardware usage – e.g. performs memory cleanup, disk cleanup (removing temp files, etc.), Docker optimization (pruning unused images/containers), CPU pinning strategies. Capabilities: memory_optimization, disk_cleanup, docker_optimization, resource_monitoring, cpu_management
GitHub
. This agent might run periodic tasks to keep the environment healthy or respond to orchestrator triggers (like if memory usage high, orchestrator may task this agent to free memory). It can also produce reports or recommendations (for example, if logs are consuming too much disk). Typically lower priority (priority 3 in config) and runs less frequently (health checks every 60s, tasks can have longer timeouts up to 600s)
GitHub
.

Multi-Agent Coordinator: Focuses on coordinating groups of agents working in parallel on a single complex goal. Capabilities: multi_agent_orchestration, workflow_management, parallel_execution, dependency_resolution
GitHub
. This agent could be conceptually similar to CrewAI (and indeed might be implemented by it or vice versa). It might take a high-level instruction like “build this project using multiple agents” and then allocate subtasks, synchronize results, and ensure dependencies are resolved (e.g., Agent A must finish design before Agent B starts coding). Orchestrator can delegate an coordinate_agents or orchestrate_workflow task to this agent
GitHub
GitHub
. It’s priority 2 (important, but orchestrator and coordinator are priority 1, meaning those critical tasks run first)
GitHub
.

Ollama Integration Specialist: Manages interactions with the Ollama LLM backend. Capabilities: model_management, inference_optimization, prompt_engineering, model_deployment
GitHub
. This agent ensures that models are present and optimized on the local Ollama server. For example, if a task requires a specific model, this agent can pre-fetch or “pull” it via Ollama’s API (the base agent class has a pull_model_if_needed() method to assist with this
GitHub
). It might also tune prompts or manage multiple smaller models for efficiency. In the toolset exposed to external interfaces, there’s a command manage_model which likely routes to this agent for performing actions like downloading or removing models
GitHub
. This agent operates at priority 3, given that model management tasks (like downloading a new model) can be time-consuming (timeouts up to 900s) and should not block critical orchestration
GitHub
.

Domain/Skill-Specific Agents: These are the majority of the 20+ agents – each encapsulating a particular domain expertise or AI tool:

Coding/Development Agents: e.g. GPT-Engineer (project scaffolding and code generation)
GitHub
, Aider (AI pair programmer for code editing and debugging)
GitHub
, AgentZero (autonomous self-directed agent for complex coding tasks, perhaps an AutoGPT variant)
GitHub
. LangFlow (visual flow builder for chains/LLMs) is available for orchestrating flows and might be integrated to let users design custom workflows visually
GitHub
. Dify (AI application dev framework) is integrated to allow building mini-apps or chatbots easily
GitHub
. ShellGPT is available to help with CLI commands or automation via natural language
GitHub
. These agents each have capabilities aligned with their function (as set in orchestrator registry and config). For example, GPT-Engineer’s capabilities include code_generation and project_scaffolding
GitHub
, Aider’s include code_editing and pair_programming
GitHub
, etc. They likely interface with code repositories or accept prompts describing a coding task and produce code or actions.

Autonomous Task Agents: e.g. AutoGPT and AgentGPT – goal-driven autonomous agents that decompose objectives into sub-tasks and iterate (AutoGPT is known for web search + reasoning loops; AgentGPT is a similar concept for goal-oriented tasks). These are integrated as services that the orchestrator can call to handle open-ended problems. For instance, if a user request is broad (“research topic X and produce a report”), orchestrator might pass it to AutoGPT which will then internally spawn multiple steps (using the tools it has like internet access, etc.) to fulfill the request
GitHub
. AutoGen (by Microsoft) is another multi-agent framework for collaborative problem solving, also integrated. BabyAGI (not explicitly listed but possibly included given the context) would similarly handle long-horizon planning tasks. The orchestrator might default to AutoGPT for any “general” tasks that don’t clearly fall under another agent’s domain
GitHub
. These autonomous agents are powerful but should be used carefully (monitored for loops, etc.). They typically respond with a plan or final output after executing their cycles.

Knowledge and Data Agents: LlamaIndex (integration for retrieval-augmented generation) allows indexing documents and querying them
GitHub
. The agent might be responsible for building indexes from data and answering questions using those indexes (private GPT-style Q&A). PrivateGPT is explicitly integrated to handle local document question-answering (no internet, using local models)
GitHub
. FlowiseAI provides a UI for designing chat flows, but also a runtime to execute them – this agent could host Flowise flows for multi-step Q&A or data processing
GitHub
. There’s also DocuMind for document processing/PDF analysis (likely an agent that can extract data from PDFs or long documents). FinRobot for financial analysis and market data tasks (it might connect to financial data APIs or use trained models for finance). These agents enable domain-specific analyses. They often integrate with vector databases to store embeddings of documents (DocuMind might vectorize PDFs for semantic search, FinRobot might store historical data vectors). The MCP server provides the sutazai://knowledge/embeddings resource and query_knowledge_base tool to tap into these agents’ data
GitHub
.

Security and Testing Agents: Semgrep (static code security scanning)
GitHub
, PentestGPT (AI-guided penetration testing)
GitHub
, possibly Vulnerability Scanner agents (the CI config mentions one for compliance monitoring and vulnerability scanning). These agents focus on code and system security. The orchestrator can route a “security_audit” task to Semgrep or PentestGPT based on whether it’s static analysis or dynamic testing
GitHub
. They produce reports on vulnerabilities or exploit attempts. Similarly, QA/testing agents exist: e.g. QA Tester or Testing QA Validator in the port registry
GitHub
GitHub
. These would generate or run tests on code, or verify outputs. There is also mention of an AI System Validator (which might perform end-to-end validation of the AI system’s outputs, ensuring quality or compliance). These agents ensure that output from other agents meets certain standards before final delivery.

Web/Automation Agents: BrowserUse agent can control a headless browser to perform web automation tasks (e.g. scrape data, click links)
GitHub
. Skyvern is likely an agent for web scraping and data extraction from websites
GitHub
. If the orchestrator gets a task like “gather information from a list of URLs”, it could delegate to BrowserUse or Skyvern. They might also be used as tools by AutoGPT (which often needs a browser). Their capabilities (web_automation, browser_control for BrowserUse; web_scraping, data_extraction for Skyvern) reflect that
GitHub
GitHub
.

DevOps and Infrastructure Agents: Some agent names indicate roles like Infrastructure DevOps Manager, Deployment Automation Master, Docker Specialist, Container Orchestrator (k3s), etc. These would handle tasks such as deploying applications, managing CI/CD pipelines, optimizing infrastructure, and automating deployment tasks. For example, Deployment-Automation-Master is listed with priority critical
GitHub
 – it likely ensures smooth automated deployments (maybe interacts with the CI pipeline or Kubernetes). Docker-Specialist might build Docker images or manage container registry tasks. The orchestrator or external user could invoke these to deploy new versions of services or to reconfigure infrastructure via code.

Project Management Agents: There are agents acting as AI versions of team roles: AI Product Manager, AI Scrum Master, AI QA Team Lead, AI Senior Engineer/Frontend/Backend/Full-Stack, etc., as seen in the port registry
GitHub
GitHub
. These agents encapsulate best practices and checks related to their roles – for instance, AI Scrum Master might generate project plans or retrospectives, QA Team Lead could double-check testing coverage or produce QA reports, Product Manager agent might prioritize features or assess requirements. In practice, these could be realized as specialized prompt templates (they might not all have heavy code behind them, but rather a persona and some rules). They are considered critical agents since they ensure the overall output of the system aligns with organizational needs and quality
GitHub
GitHub
. Developers can consult these agents for advice or use them to evaluate the outputs from coding agents (for example, the AI QA Lead agent might review code changes suggested by GPT-Engineer for potential quality issues).

Miscellaneous/Other: The list of agents is extensive (the CI pipeline suggests around 69 agents in total) covering niches like Adversarial Attack Detector
GitHub
 (which might monitor for malicious inputs or model attacks), Bias and Fairness Auditor, Experiment Tracker, Causal Inference Expert, Federated Learning Coordinator, Edge Inference Proxy, Energy Consumption Optimizer, Explainability Specialist, etc. Each of these agents has a narrow focus. For example, a Bias Auditor agent could analyze outputs or models for bias; an Explainability agent could produce explanations for model decisions (perhaps using tools like LIME or SHAP); a Federated Learning Coordinator could simulate federated training across nodes. While not every agent is fully implemented yet, the architecture is designed to accommodate them. Each agent’s capabilities are declared so the orchestrator or coordinator can route tasks appropriately. For instance, a task type “explain_model” might require the capability explainability, which the Explainable-AI agent provides, so it would be chosen
GitHub
GitHub
. These agents often integrate with external libraries or systems (e.g. a Federated Learning agent might interface with a federated server or other peers).

Agent Configuration and Behavior: All agents share some common behavior and configuration patterns:

Standard HTTP API: By convention, agents implement at minimum:

GET /health – returns a simple health status JSON. If this returns HTTP 200 and contains "status": "healthy", the orchestrator regards the agent as online
GitHub
. This may also include the agent’s name and a timestamp or version.

POST /execute or /task – receives a task request (often a JSON with type and data) and returns the result. The exact schema can vary by agent, but typically it includes a status (success/failure), the agent name, a result payload (which could be the completed answer or subtasks results), and a timestamp
GitHub
. Agents are expected to handle this call asynchronously (not blocking the entire server if possible) and respond within a reasonable time or stream intermediate output if long-running.

Optionally, GET /metrics – many agents include the Prometheus client, so if configured, they can expose a /metrics endpoint with Prometheus-format metrics on their internal performance (requests handled, latency, memory usage, etc.). This isn’t explicitly shown in code, but prometheus-client is installed for agents
GitHub
 and the orchestrator’s environment expects Prom/Grafana integration. We strongly recommend enabling prometheus-client to start an HTTP metrics endpoint or having a sidecar exporter for each agent container.

Agents may also provide domain-specific endpoints (e.g., an agent managing resources might have POST /allocate as the Arbitrator does
GitHub
, or an agent that has a UI might serve that on / or /docs). But these are secondary to the core contract of /health and /task.

Agent State and Concurrency: Each agent can handle multiple tasks concurrently up to a limit. The max_concurrent_tasks is configured per agent (for example, orchestrator 10, coordinator 20, others between 3-15 depending on their function)
GitHub
GitHub
. This prevents an agent from being overloaded. Agents can maintain internal state for each task if needed (e.g., a context or chain-of-thought for autonomous agents), but generally tasks are treated independently. If an agent cannot accept a task (because it’s at capacity or offline), the orchestrator will either queue the task or route to a backup agent.

Agent Implementation: Some agents are implemented as wrappers around existing tools. For instance, the AutoGPT agent might internally launch an AutoGPT process or call its API. Agents like TabbyML, LangFlow, Dify, Flowise likely correspond to services with their own implementation (possibly in other languages like Node or provided via Docker images). SutazAI’s deployment includes these by reference. Meanwhile, many of the “AI role” agents (Product Manager, QA Lead, etc.) are essentially prompt templates. They may not have complex logic beyond loading a markdown definition of their persona and then using the base LLM to respond. In fact, the system uses markdown files for agent definitions: for each agent persona, there is an .md file (often created in the .claude/agents/ directory during development) containing a YAML frontmatter (with name, description, tool permissions) and a detailed prompt describing the agent’s expertise and behavior
GitHub
GitHub
. At runtime, the agent’s container includes these definition files (placed in /agents/<agent_name>.md), and the base agent class loads the content to build the agent’s system prompt
GitHub
GitHub
. For example, a tag-agent.md defines a “Tag Standardization” agent with responsibilities and workflow for maintaining a tag taxonomy
GitHub
GitHub
, while a moc-agent.md defines a “Map Of Content” generator agent for knowledge management tasks
GitHub
. The OllamaLocalAgent base class then wraps this persona description with the current task to form the prompt it sends to the LLM
GitHub
GitHub
. This mechanism allows easy creation of new specialized agents by writing a markdown spec for their persona and capabilities, without changing code. The Agent Expert (an internal design agent whose definition is in agent-expert.md) provides guidance on writing such agent markdown templates, ensuring they follow a standard format and best practices
GitHub
GitHub
.

Integration with MCP Server: The Model Context Protocol server acts as an external orchestrator and integration layer – effectively bridging user commands (from a UI or chat interface) to the SutazAI agent backend. It exposes “tools” to the UI (Claude Desktop) such as deploy_agent, execute_agent_task, orchestrate_multi_agent, etc.
GitHub
GitHub
. When a user invokes these, the MCP server will call the appropriate orchestrator or agent API. For example, deploy_agent could call the orchestrator’s /register_agent to dynamically add a new agent (if supported), or run a script to start a new agent container. execute_agent_task calls the orchestrator’s task submission endpoint (or directly the agent if targeting a specific one)
GitHub
. The MCP also provides resource URIs like sutazai://agents/list which likely maps to the orchestrator’s /agents endpoint to list all registered agents and their statuses
GitHub
. Thus, the MCP server is a client-facing facade that uses the underlying agent network. It is important that the agent orchestrator and MCP server are in sync in terms of agent registry and health; typically, the MCP might query the orchestrator for available agents to display them. Agents, orchestrator, and MCP together implement a multi-agent orchestration platform that an end-user can interact with through a friendly interface.

Communication Protocols and Message Contracts
Task and Interaction Protocols

Communication between the orchestrator and agents follows defined protocols to ensure consistency across diverse agents. There are two primary communication mechanisms: synchronous HTTP requests for direct commands and results, and asynchronous messaging for status updates, heartbeats, and event broadcasts.

Synchronous HTTP (REST API): The orchestrator’s REST API endpoints allow external clients (or the MCP server) to interact with the system, and the agents themselves expose REST endpoints for orchestrator-to-agent calls:

Submit Task: The orchestrator offers POST /submit_task (or an equivalent, such as /orchestrate_interaction or /agents/interaction/execute in the unified API) to accept a new task from a client
GitHub
GitHub
. The request typically includes details like agent_name (if targeting a specific agent or null/general if let orchestrator decide), task_type, parameters (task payload), priority, and optional timeout
GitHub
. The orchestrator responds immediately with a task_id and status “pending”
GitHub
, and then processes the task asynchronously. A client can poll GET /task/{task_id} to get the status or result when ready
GitHub
.

Direct Agent Execution: Alternatively, if a specific agent is targeted or known, the orchestrator (or the client via orchestrator’s proxy) can call POST /agents/interaction/execute with an AgentTaskRequest specifying the agent and task details
GitHub
. This triggers the orchestrator to immediately dispatch the task to that agent in a background task (so the HTTP call returns quickly)
GitHub
. The orchestrator enqueues the task and returns a response with task_id, agent_name, status “pending”, timestamp
GitHub
. The actual result will be filled asynchronously, and presumably, the client can retrieve it via another endpoint or get a callback.

Multi-Agent Collaboration: There is an endpoint POST /agents/interaction/collaborate to initiate a collaboration among multiple agents for a complex task
GitHub
. The request includes a task description, a list of required agent names, a coordination strategy (e.g. sequential vs parallel), and any shared context data
GitHub
GitHub
. The orchestrator will then engage those agents (likely via the Multi-Agent Coordinator agent or orchestrator’s internal logic). A collaboration_id is returned along with a list of participating agents and status “in_progress”
GitHub
GitHub
. The orchestrator might then continuously manage the interaction (e.g. passing messages between agents, or orchestrating a round-robin conversation if that’s the strategy). Results of the collaboration (could be intermediate results per agent or a combined output) will be collected and eventually returned to the client, either via polling or a final callback. The system ensures that even if one agent fails in the collaboration, it logs the failure event and continues or aborts gracefully
GitHub
. This endpoint exemplifies orchestrating a scenario like “agent A generate a plan, agent B critique it, agent C improve it” style workflows.

Agent Query/Status: GET /agents returns the list of all registered agents with their status. In the orchestrator, this is implemented by iterating over the registry and returning each agent’s name, type, URL, capabilities, status, and last health check timestamp
GitHub
. A similar unified endpoint might exist at /agents/interaction/list or just /agents. There’s also GET /agents/interaction/status/{agent_name} which returns detailed status and metrics for one agent
GitHub
. This calls into the Agent Manager’s get_agent_status which checks the agent’s health and heartbeat and returns structured info including last heartbeat timestamp, current known status, and possibly current task if any
GitHub
GitHub
. This is useful for monitoring and for external systems (like a dashboard or MCP Inspector) to display agent health.

Agent Management: The orchestrator supports registering new agents via POST /register_agent
GitHub
. The client provides the agent’s ID, human-readable name, capabilities list, endpoint URL, and max concurrency in the JSON
GitHub
. The orchestrator will add it to its registry (in-memory) and reply with confirmation
GitHub
. This currently is not persistent (no DB write)
GitHub
, so a restart would lose it, but it suffices for dynamic addition in a running system. There might not be an explicit /deregister_agent endpoint; instead, if an agent stops responding (health checks failing beyond a threshold), the orchestrator will mark it offline. In a dynamic scenario, an agent can also “register” itself by sending a message or calling this endpoint on startup.

Health and Status Checks: GET /health on the orchestrator returns overall system health – including orchestrator’s own status and some summary metrics
GitHub
. For example, it reports whether orchestrator is fully initialized, how many healthy agents out of total, the current task queue length, and number of active tasks being processed
GitHub
. A similar GET /status might provide more detailed orchestrator stats (like total interactions orchestrated, successes/failures, conflict counts, etc.)
GitHub
GitHub
. Each agent also has GET /health as discussed, used both by orchestrator and external monitors (like Kubernetes liveness probes or Docker healthcheck, which is configured to curl the agent’s /health every 30s in the Dockerfile template
GitHub
).

Custom Endpoints: Some specialized agents expose additional endpoints. For instance, the Resource Arbitrator has POST /allocate and DELETE /allocations/{id} for allocation requests and releases
GitHub
. The Task Coordinator might have GET /queue and POST /strategy to inspect the queue or change scheduling strategy at runtime
GitHub
. These allow runtime control and introspection of the agent’s internal state. They typically have protective access (perhaps only accessible to an admin or orchestrator service account).

All HTTP endpoints use JSON for request and response bodies, and Pydantic models are used to define these schemas in code (for automatic docs and validation). For example, AgentTaskRequest and AgentTaskResponse models define exactly what fields are expected
GitHub
GitHub
. Preserving these contracts is critical – any change in an endpoint or model should be communicated to all developers and the MCP integration updated accordingly. We maintain versioning in the API if needed (the repository shows some endpoints under backend/app/api/v1/, indicating version 1 of the API).

Asynchronous Messaging: For intra-system communication that doesn’t require immediate response, RabbitMQ (or another message bus) is used. The system defines a topic exchange named sutazai.agents with routing keys for agent-specific or broadcast messages
GitHub
. Some message types and their typical flows:

Agent Registration: When an agent starts, instead of or in addition to calling REST /register_agent, it can publish an AgentRegistrationMessage on the bus with its details
GitHub
. This message includes the agent’s unique ID, type (e.g. “worker” or specific category), capabilities list, version, host, port, max concurrent tasks, and supported message types
GitHub
. The orchestrator (or Agent Manager) listens on the bus for agent.* topics and when it sees a registration message, it adds the agent to its registry and possibly acknowledges it. Using messaging allows auto-discovery without a direct HTTP call.

Heartbeats: Agents periodically (say every 30s) publish AgentHeartbeatMessage to convey liveness and load
GitHub
. This contains fields like current status (could be an enum such as ONLINE/BUSY/OFFLINE or here AgentStatus with values like ACTIVE, etc.), current load (a fractional representation of how busy the agent is, 0 meaning idle, 1 meaning fully busy), number of active tasks, available capacity (e.g. slots free), CPU and memory usage percentages, timestamp of last completed task, uptime, and error count
GitHub
. The orchestrator or a monitoring service consumes these heartbeats. If a heartbeat hasn’t been received within a heartbeat timeout (say 90 seconds as per config)
GitHub
, the orchestrator will mark the agent as stale and possibly unregister it. Heartbeat messages enable more robust detection than simple health ping, because they carry load info, which orchestrator can use for better scheduling (e.g., the least_loaded strategy uses these to find the agent with lowest load).

Task Status Updates: When an agent finishes a task (especially if it’s long-running or asynchronous), it can send an AgentStatusMessage or a specialized TaskResultMessage to report the outcome
GitHub
. The AgentStatusMessage includes the agent’s current status and metrics (like performance metrics, active/queued task IDs, etc.)
GitHub
. If integrated with RabbitMQ, the orchestrator might prefer to get task results via a message on task.<agent_id>.completed routing key rather than polling the agent. In the current implementation, the orchestrator’s HTTP call waits for the response, but an alternative design is to have the orchestrator send a task message and not block, then later receive the result message. The config’s supported_message_types for each agent (in AgentRegistration) can indicate whether that agent uses direct HTTP or messaging or both
GitHub
. Fully utilizing RabbitMQ can increase throughput and decouple services.

Capability Broadcast: Agents can advertise their capabilities or any changes via AgentCapabilityMessage
GitHub
, which lists detailed capability descriptions, resource requirements, performance metrics, supported task types, etc. This could be used when an agent’s abilities change (for example, if a new tool is installed in an agent, it could broadcast that it now supports a new task type). Orchestrator can update its registry accordingly. In practice, capabilities are mostly static, so this is not frequently used.

Error Notifications: If an agent encounters a serious error (like it had to shut down or a task crashed), it might emit an ErrorMessage on the bus that the orchestrator or a monitoring service can catch to log or trigger recovery. The Implementation Guide implies error and status messages are part of the planned messaging module
GitHub
.

The message routing keys follow a convention
GitHub
:

agent.{agent_id}.# – messages intended for a specific agent (could be commands sent by orchestrator to the agent’s queue).

agent.all.# – broadcast to all agents (e.g. a system-wide pause or shutdown signal, or a prompt to all to send heartbeat).

task.{agent_id}.# – task-specific messages for an agent (like dispatching a task or returning a result).

Agents likely create a queue named after themselves (agent.<id>) binding to relevant topics. The orchestrator might publish a task message to task.agent123.execute which goes to queue agent.agent123. That agent, listening on that queue, executes the task and then publishes task.agent123.completed with the result.

Using RabbitMQ in this way allows flexible orchestration beyond simple request-response. For example, orchestrator can broadcast a shutdown command to all agents by sending a message to agent.all.shutdown – each agent’s communication client (if implemented) would pick that up and initiate graceful stop. Or orchestrator could query all agents for a certain info by sending a broadcast message and collecting replies on a reply queue.

Currently, the codebase shows the scaffold for this advanced messaging (the AgentCommunication class, message protocols, etc.), but much logic is still being implemented. Nonetheless, the design is set for enterprise scenarios requiring robust decoupling.

Agent Manager and Internal API: Within the backend, there is an AgentManager class that provides a programmatic API for controlling agents (used by the FastAPI routers). This is used when agents are running in-process or to manage their lifecycles in a more monolithic deployment. The manager’s methods like create_agent, start_agent, pause_agent, resume_agent, execute_task etc. allow fine-grained control
GitHub
GitHub
. For example, create_agent('coding') will instantiate a new coding-focused agent via the factory, initialize it, and mark it READY
GitHub
GitHub
. start_agent(id) marks it RUNNING and it begins accepting tasks
GitHub
. execute_task(id, task) will directly call the agent’s execute() method in-process and return the result synchronously
GitHub
. This internal API is used by the router endpoints when orchestrator is compiled into the main application (for example, the /agents/interaction/execute route uses background_tasks.add_task(orchestrator.execute_task, agent_name=..., task_type=..., ...) for an asynchronous execution
GitHub
, and likely inside that it may use AgentManager if running locally). Essentially, the system supports both external microservice mode (agents in separate containers, orchestrator calls them via HTTP) and a single-process mode (all agents as Python classes managed by AgentManager, e.g. for testing or a lightweight deployment). The behaviors are similar, but in single-process mode, the AgentManager handles locking (to avoid race conditions with threads) and directly invokes agent methods, capturing exceptions.

For day-to-day operations, engineers will primarily interact with the system via the orchestrator’s HTTP API or via the MCP server’s high-level interface. However, understanding the messaging and internal APIs is useful for debugging complex issues (like an agent not responding might be debugged by checking both its HTTP and message consumption).

Shared Schemas and Contracts

Consistency across agents is enforced by shared Pydantic models defining the structure of messages and API payloads:

Task Schema: While not explicitly shown as a single model, the structure can be inferred. The JSON passed to an agent’s /task likely has keys like type (a short code for task category), data (a dict of task parameters or content), and priority. The template’s TaskRequest expects at least type and data
GitHub
. The orchestrator when calling an agent constructs a JSON with an id (task_id), type, description and maybe parameters
GitHub
. Agents should handle gracefully if extra fields are present. On response, the TaskResponse includes status ("success" or "error"), agent (the agent name), result (a dict, could be anything – often containing the actual output or a message), and timestamp
GitHub
GitHub
. All agents should follow this convention in their output for consistency. The orchestrator wraps this into its own result format when returning to clients (adding the task_id and status).

Agent Status Schema: The AgentStatusResponse returned by the GET status endpoint provides a snapshot of an agent
GitHub
. It includes the agent_name, status (like "active"/"inactive"/"busy" – perhaps from an enum), current_task (if any), list of capabilities, and a dict of performance_metrics
GitHub
. This is constructed from the AgentManager’s data: status from agent_status dict, metrics from agent_metrics dict (which tracks cpu_percent, memory_percent, etc.)
GitHub
. It gives operators a quick view of each agent’s health and load.

Inter-agent Messages: Already discussed, the AgentRegistrationMessage, AgentHeartbeatMessage, etc., have exact fields defined in schemas/agent_messages.py
GitHub
GitHub
 and they are used to serialize/deserialize messages on the queue.

Error Handling Contract: If an agent fails to execute a task (throws exception, etc.), how is it reported? By convention, the HTTP response would be a non-200 status with a JSON error message. The orchestrator’s _execute_on_agent method catches exceptions and wraps them into a JSON {"error": "...error message..."} for the result
GitHub
. Similarly, if an agent returns a non-200 (like 500 on error), orchestrator interprets that as an error and encapsulates it. Thus, the result field in a TaskResponse might contain either the result or an error key. At a higher level, the AgentTaskResponse model has an optional error field to capture an error message
GitHub
. The orchestrator always tries to return a well-formed JSON indicating success or failure rather than dropping the connection or returning plain text. This standardization is important for the MCP server or any client which will check for an error field or the status field in the JSON to decide if the task succeeded.

Engineers extending the system should adhere to these schemas. When adding a new endpoint or message type, define a Pydantic model for it so that documentation and validation remain consistent.

Configuration and Deployment
Configuration Management

The behavior of the agent system is governed by configuration files and environment variables:

Central Agent Config (config/agents.yaml): This YAML file defines the core settings for each agent type and the orchestration rules
GitHub
. It lists each critical agent by a key (e.g. ai_agent_orchestrator, task_assignment_coordinator, etc.) with its queue name, capabilities, max tasks, priority, health check interval, and timeout
GitHub
GitHub
. It also contains sections for task routing rules (mapping task identifiers to required capabilities and which agents are preferred)
GitHub
GitHub
, assignment strategies flags
GitHub
, and global settings (like max_retry_attempts, default timeouts, heartbeat thresholds, max queue size)
GitHub
. This file is essentially the “source of truth” for how tasks should flow in the system. The orchestrator/AgentManager reads this on startup to configure itself (the AgentFactory may also use it for agent creation defaults). When adding a new agent service, one should update this file: add the agent under agents: with its properties and include it in any relevant task routing rules if it handles specific tasks. This ensures the orchestrator is aware of its capabilities and can assign tasks to it properly.

Environment Variables: Key configurations are provided via environment:

REDIS_URL – connection string for Redis (e.g. redis://redis:6379/0)
GitHub
. The orchestrator and some agents use this if caching or pub/sub via Redis is enabled (for example, the orchestrator’s agent discovery could scan Redis keys agent:*).

RABBITMQ_URL – AMQP connection for RabbitMQ (e.g. amqp://guest:guest@rabbitmq:5672/)
GitHub
. If the advanced messaging is in use, agents and orchestrator will use this to connect to the broker. RabbitMQ details (exchanges, queues) are usually configured in code or via environment as well.

OLLAMA_BASE_URL – URL where the Ollama server is running (default http://ollama:10104) 
GitHub
. All agents using LLM functions will send requests to this base URL for generating text, chat completions, or embeddings. MODEL_NAME is also configurable (the default model each agent uses; environment variable MODEL_NAME with default “tinyllama:latest” as seen in base class)
GitHub
.

AGENT_CONTEXT_WINDOW – context length for LLM generation (default 2048 tokens)
GitHub
.

AGENT_TIMEOUT_SECONDS – default timeout for LLM requests (default 30 seconds)
GitHub
. Agents should set timeouts for their calls to avoid hanging if the model doesn’t respond.

PORT – each agent container uses this to know what port to bind to (the Dockerfiles typically default to 8080, but PORT can override, e.g. orchestrator might set PORT=8589)
GitHub
GitHub
. In Docker Compose or K8s, these are set according to the port registry.

Authentication and service URLs: For security integration, environment variables for SERVICE_ACCOUNT_MANAGER_URL, JWT_SERVICE_URL, KONG_PROXY_URL, and potentially Keycloak details (realm, etc.) are provided to the update-auth script and to agents that implement security
GitHub
. Agents that need to validate tokens might use these to contact the JWT service or introspect tokens. Usually, these are injected via the container runtime or a Kubernetes secret/configmap.

Logging and monitoring: e.g. LOG_LEVEL, ENABLE_TRACING flags can be used. In code, get_settings() likely loads from env (maybe via pydantic BaseSettings) to configure such options
GitHub
. If we have OpenTelemetry, we might have env vars for OTLP endpoint.

Secret Management: Sensitive values like API keys, service account credentials, or encryption keys are not stored in code or static config. Instead, the platform uses HashiCorp Vault (noted in port registry at port 8200 for sutazai-vault)
GitHub
 to store secrets. For example, each agent that needs to call internal APIs might have a client_id and client_secret stored in Vault. The update-agent-auth.py script pulls service account info (client IDs, scopes) presumably from the Service Account Manager and expects to fill in the client secret from Vault (the script placeholder shows "[STORED_IN_VAULT]" for client_secret)
GitHub
. During deployment, Vault injector or environment variables would provide the actual secret to the agent. This ensures that if someone opens the agent’s code or config, they won’t see raw secrets. The script can insert the necessary code in the agent’s startup to fetch or use those secrets. For instance, an agent’s app.py may include code to retrieve a token from the JWT service using its client credentials from env. Overall, security-critical configuration is handled via Vault and environment, not hard-coded.

Docker Configuration: The agents and orchestrator run in Docker containers defined by Dockerfiles. The base image for agents is standardized – docker/Dockerfile.agent-base or similar (and other base images for NodeJS-based or GPU-enabled agents) are built first
GitHub
. Each agent then has its own Dockerfile (often generated by the prepare script), which typically:

Inherits from the Python base image.

Copies the common code and agent-specific code.

Installs any agent-specific Python requirements (the prepare script creates a requirements.txt even if empty for that purpose)
GitHub
GitHub
.

Sets the AGENT_NAME environment variable and a healthcheck to hit /health every 30s
GitHub
GitHub
.

Runs the FastAPI app with Uvicorn on startup
GitHub
.

Possibly includes a user switch to run as non-root (the template adds a user “agent” and uses it)
GitHub
.

The orchestrator’s container is similarly defined (possibly in agents/ai_agent_orchestrator/Dockerfile or in backend/Dockerfile). The port registry assigns each container’s external port mapping. For example, orchestrator is mapped host 11000 -> container 8080
GitHub
, the coordinator might be 11001 -> 8080, etc. This means when running via Docker Compose, one could reach orchestrator at localhost:11000 (or via the internal network by container name). In Kubernetes, these port numbers serve as Service NodePort or other fixed references.

Service Registration: If using Consul for service discovery, each agent could register itself with Consul on startup (Consul agent running sidecar or via HTTP API on sutazai-consul-discovery:8500
GitHub
). The code includes python-consul library
GitHub
, indicating an intention to use Consul. This might allow orchestrator or other system components to query Consul for available agents instead of relying on static config. For example, the orchestrator’s background discovery task (currently just scanning Redis keys) could instead query Consul for any service with tag "ai-agent". If implemented, each agent container would register under a service name (like "agent-<id>") with its IP/port and health check. This aspect isn’t fully fleshed out in the code we saw, but it’s part of the architecture.

Updating Configuration: Changes to agent capabilities or adding new tasks should be done in the config files (agents.yaml, etc.) and corresponding code if needed. The system was designed to minimize code changes for such updates – ideally one can add a new agent’s entry in YAML, deploy the agent container, and orchestrator will pick it up (especially if dynamic reg is working). However, if static, one might need to update the orchestrator’s _register_agents list in code for now
GitHub
. A future improvement could be to auto-generate that from agents.yaml. The port registry YAML (config/port-registry.yaml) should also be updated with the new agent’s port and container name to maintain consistency in documentation and deployment scripts.

Deployment Lifecycle

SutazAI’s agent platform is deployed via a CI/CD pipeline that emphasizes safety, observability, and the ability to roll back quickly:

CI Build and Test: The GitLab CI pipeline (or similar CI) runs through stages for validation, building, testing, security scans, performance tests, staging deployment, and production deployment
GitHub
. In the validation stage, code style and quality are enforced: the pipeline runs Black for formatting check, Flake8 for linting (with some style rules configured, e.g. max line length 88, ignoring specific warnings)
GitHub
, mypy for static typing, and pylint
GitHub
. Any issues here will fail the pipeline, ensuring that all code merged respects coding standards and type safety. This keeps the agent code maintainable and consistent.

Automated Testing: There are extensive unit and integration tests for agent services. The Implementation Guide indicates high test coverage: ~95% for orchestrator, ~93-94% for coordinator and arbitrator
GitHub
. Test files (e.g. test_ai_agent_orchestrator.py, etc.) exist and cover typical scenarios
GitHub
. CI runs these tests in the “test” stage. Additionally, a “performance” stage likely runs load tests or benchmark scripts (ensuring that e.g. orchestrator can handle 10 concurrent requests in under a certain time
GitHub
, and that success rate is 100%). The pipeline may also run a “security” stage using SAST (static analysis for security) or dependency scans – the mention of a security-scan-results directory and the requirements-security-summary.md suggests dependencies are scanned for known vulnerabilities.

Container Image Build: In the build stage, Docker images for all components are built. This includes base images (Python, Node, etc.) and then the core services and agent services. The pipeline shows parallelization: it defines groups of agents (core, specialized, research, security, monitoring) and builds each group in parallel to speed up the process
GitHub
GitHub
. For example, in the build:agents step, it loops through agent directories and does docker build for each agent’s Dockerfile, tags it with a version (timestamp + commit SHA) and pushes to the registry
GitHub
GitHub
. This ensures each agent is packaged as an independent microservice image. The IMAGE_PREFIX and registry variables mean images are named like registry.gitlab.com/<project_path>/agents/<agent_name>:<version>
GitHub
. Versioning is automatic per commit, but for production, they might also tag with latest or a semantic version when releasing.

Deployment (Staging/Prod): The pipeline likely uses Helm or kubectl in the staging and production stages. The presence of KUBERNETES_VERSION and HELM_VERSION variables
GitHub
 implies they use Kubernetes to deploy. A blue-green or canary deployment strategy is indicated by DEPLOYMENT_STRATEGY: "blue-green" and rollback provisions
GitHub
. In a blue-green deployment, a new set of pods (green) is brought up alongside the existing (blue) ones, traffic is switched gradually or instantaneously once the green is confirmed healthy. The ROLLBACK_TIMEOUT: 300 suggests if the new deployment isn’t healthy within 5 minutes, it will automatically roll back to the previous known-good state. The pipeline also has a rollback stage as a safety net. Each agent service likely is defined in a Helm chart template (the TOTAL_AGENTS: "69" might be used to template out all agent deployments, or they use a dynamic approach). More likely, they have a chart where each agent is a sub-chart or they rely on the port registry to generate service definitions.

Observability Integration: As part of deployment, monitoring is configured. Each container has a healthcheck (so Kubernetes liveness/readiness probes are set to call /health endpoints). Prometheus is likely set up to scrape metrics from each agent and orchestrator pod (either directly via a ServiceMonitor or via an aggregated metrics gateway). Grafana dashboards would be configured to visualize key metrics: e.g. number of tasks queued, agent CPU usage (from AgentMetrics), average execution time per agent, errors per hour, etc. Logging from each service is aggregated – possibly using a centralized logging (ELK or Loki) given the mention of “Loki” in monitoring range
GitHub
. Traces, if using Jaeger, would capture cross-service calls: since orchestrator calls agents, distributed tracing could correlate a task across orchestrator and agent spans. If an OpenTelemetry middleware is used in FastAPI, then each request between orchestrator and agent could carry a trace context header; Jaeger would then show a span from orchestrator -> agent call.

Canary Releases: In addition to blue-green, canaries might be used for agents where applicable. For example, if updating just one agent’s logic (say a new version of GPT-Engineer container), one could deploy it as a canary (only a fraction of tasks go to the new version while the rest still go to old version). The orchestrator can support this if configured to treat them as two instances of the same agent type: e.g., temporarily register GPT-Engineer-v2 along with GPT-Engineer-v1, but have orchestrator prefer v2 for a small subset of requests. Alternatively, if using Kubernetes, one might use weighted services or manual routing. This is not explicitly described, but standard practice in CI/CD with such microservices.

Scaling: Each agent can be scaled horizontally if needed. Orchestrator’s design (with health checks and the possibility of service discovery) allows multiple instances of the same agent type. For example, if agentzero-coordinator is a bottleneck, one could run 2 replicas of it. They would both register as type AgentZero; orchestrator could either treat them as one (via a load balancer) or even register both with slightly different IDs. Consul or Kubernetes service would load-balance calls to “agentzero” across pods. The agent’s own concurrency limit plus scaling defines overall throughput for that kind of task. Orchestrator itself can also be scaled for high availability (though then tasks assignment needs coordination – using an external task queue or a leader election for orchestrator instances might be necessary; likely only one orchestrator is active at a time for simplicity).

Deployment Example (Local): For a developer, a simplified flow to run the system might be:

Use Docker Compose with the provided docker-compose.yml to spin up infrastructure: Postgres, Redis, RabbitMQ, Vault, Consul, and core services (backend, orchestrator, maybe MCP server, and a couple of key agents).

The orchestrator (ai-agent-orchestrator service) starts and logs that it has initialized and how many agents it registered
GitHub
GitHub
.

Each agent service container starts, registers with orchestrator (via HTTP or Rabbit), and begins heartbeating.

The system is now ready to accept tasks. If using MCP, start Claude Desktop and verify that the SutazAI MCP server is connected (in Claude’s settings, the sutazai-mcp-server should show as running)
GitHub
.

Run a test: e.g. use Claude to issue “Deploy a new AutoGPT agent called 'research-bot' and then use it to analyze trends”
GitHub
. Behind the scenes, Claude calls the MCP tool deploy_agent (which might invoke an orchestrator API or script to launch an AutoGPT agent container dynamically) and then execute_agent_task for that agent. Or orchestrator might reuse the existing autogpt agent instance if it’s general enough. The orchestrator coordinates the actual analysis task (which AutoGPT agent performs), and results flow back.

Monitor Grafana or logs to see the interactions. If any agent fails (say AutoGPT runs out of context or crashes), orchestrator will catch an error and return it to the client with a clear message.

If deploying a new version of an agent, push the change to Git, CI builds the image, and then (in staging first) the new container is deployed. Health checks will verify it is up (each agent container will only be marked ready when /health returns healthy). Then traffic or task assignment can be switched.

The system ensures that even if one agent is failing, others continue to operate. The orchestrator’s design avoids a single failing agent from bringing down the system: it will mark it unhealthy and skip it, possibly logging the issue or notifying via metrics.

Security and Access Control

Security is critical in an AI agent system that may have capabilities to execute code, access files, or call external APIs. SutazAI employs multiple layers of security controls:

Authentication & Authorization: All inter-service communications and external API calls are protected by a token-based authentication mechanism. The platform uses JWT tokens and service accounts to ensure that only authorized services or users can trigger agent actions:

Each agent service is associated with a service account (managed by the Service Account Manager service) that has a client ID and secret
GitHub
GitHub
. The orchestrator and MCP server also have their own service identities.

A Keycloak identity provider is used for issuing and validating tokens (the config refers to Keycloak realm “sutazai”)
GitHub
. The Service Account Manager likely provisions client credentials in Keycloak for each agent.

The JWT Service (running at port 10054)
GitHub
GitHub
 is an internal component that agents use to obtain JWTs or verify them. For example, an agent can call the JWT service’s endpoint to exchange its client credentials for a signed JWT, which it will then include in requests to other services.

Kong API Gateway (running at ports 10051, 8000)
GitHub
GitHub
 is deployed as a reverse proxy in front of services. Kong is configured with plugins to verify JWT tokens on incoming requests to agent and orchestrator APIs. Essentially, any external request hitting an agent’s /execute or orchestrator’s /submit_task must carry a valid Authorization header (Bearer <token>). Kong uses the JWT service or Keycloak’s public keys to validate this before allowing the request through. This prevents unauthorized access – for instance, you couldn’t directly call an agent’s endpoint from outside the cluster without a proper token.

The update-agent-auth.py script automates the injection of auth logic into agents
GitHub
. It opens each agent’s app code and adds a “SutazAI Authentication Setup” section if not present
GitHub
. This inserted code likely sets up an async routine to fetch tokens and attach them to outbound requests or require them on inbound requests. For example, it defines a SutazAIAuth class with config including Keycloak URL, client_id, and references to JWT and Kong URLs
GitHub
GitHub
. Agents that call other agents (some might directly call each other’s APIs) will use this to get a JWT and include it in the Authorization header (the config shows headers: authorization: Bearer {ACCESS_TOKEN} for outgoing calls)
GitHub
GitHub
. On the inbound side, FastAPI might integrate with a JWT verification dependency – possibly using python-jose and decoding incoming tokens (the presence of python-jose and passlib[bcrypt] suggests token signature verification and maybe hashing secrets)
GitHub
. If a token is missing or invalid, the agent’s API would return 401.

Each agent’s service account scopes are defined such that they only have access to what’s needed. The config snippet shows each agent’s config to be updated with scopes from the service account
GitHub
GitHub
. For example, an agent might have scope only to call the orchestrator and nothing else, or maybe some have more privileges. This fine-grained scope approach means if one agent is compromised, it cannot impersonate others freely.

User Roles: The system possibly distinguishes between internal service calls (which use service accounts) and end-user or UI calls. The MCP server, for instance, when it receives a user action, might use its own token to call orchestrator. If human end-users have accounts, they would authenticate (maybe via the Claude Desktop app’s OAuth to this system) and that token would propagate. Or simpler, all UI calls funnel through the MCP which is trusted. In any case, all points of entry validate identity and roles.

Network Segmentation: All agent containers run on a secure network (Docker network or Kubernetes namespace) where only the API gateway and orchestrator are exposed externally. Agents themselves likely are not exposed to the public network – external access is through the front door (API Gateway at controlled ports). This prevents malicious actors from bypassing auth by directly hitting an agent. Additionally, the orchestrator’s management endpoints might be protected or not exposed publicly. For example, /register_agent could be limited to internal network or require an admin token.

Secret Handling: As mentioned, secrets (like client secrets, API keys for external services, database passwords) are stored in Vault and not hardcoded. Agents fetch what they need at startup via environment or Vault injection. The Vault ensures audit logging of secret access and can rotate secrets if needed. Within code, secrets are referenced by names, not values, and the actual injection may happen out-of-band.

Filesystem and Execution Security: Agents like those capable of running code or shell commands (e.g. a “Bash” tool, or the Browser agent controlling a browser) need sandboxing:

Containerization inherently provides a layer of isolation. Each agent runs in its container with a limited file system (only its code and necessary tools). The Dockerfiles create a non-root user agent to run the service
GitHub
, preventing root-level changes inside the container. For example, even if an agent is exploited, it can’t modify the host or other containers easily.

Resource limits can be set on containers (CPU, memory) to prevent a runaway agent from consuming all system resources (these can be configured in Kubernetes or Docker Compose).

If an agent uses external tools, those should be carefully chosen. For instance, if an agent uses subprocess to run shell, it should only run predefined safe scripts. Ideally, an agent performing system operations (like the hardware optimizer) is given only the privileges it needs (maybe some docker socket access but nothing else).

Logging of agent actions: Agents that have wide capabilities (like AutoGPT that could potentially write files or execute code) should log every such action to a secure log. This allows auditing what the AI is doing. The system may include a “safety monitor” agent (the port registry has hygiene-* services and a hygiene-agent-orchestrator in a script)
GitHub
, which likely monitors outputs or code changes for unsafe content. This is another layer – e.g., scanning if an agent is trying to access forbidden files.

There is mention of a Secure Agent Communication module in the repo (e.g. security/agent-communication/secure_agent_comm.py) which possibly implements encryption or signing of messages. While internal network communications might be plaintext, using service tokens, one could also encrypt payloads especially if going over an untrusted network. For now, we rely on network-level encryption (if needed, e.g. ensure that if going over internet, use TLS – Kong can terminate TLS at the gateway, and internal communication can remain on a private network).

Permissioning of Actions: Not all tasks should be allowed to all agents. The YAML frontmatter in agent definition sometimes lists allowed tools: (like Read, Write, Bash, etc.)
GitHub
, indicating a whitelist of operations that agent can perform. This is akin to how AutoGPT plugins are controlled. The orchestrator or agent manager can enforce this by not executing requests outside the scope. For instance, a Tag-Standardization agent might have permission to read and write markdown files in a certain content vault, but not to make network calls. Ensuring each agent is constrained to its role reduces the potential damage from mistakes or misuse.

Security Testing: The CI pipeline likely includes security tests. Possibly a penetration testing stage using PentestGPT or other scanning of the running dev deployment, and a dependency vulnerability scan. The presence of an agent like Compliance Monitor suggests continuous checks for security compliance (maybe verifying that all components have valid tokens, or scanning logs for secrets). Additionally, issues like ensuring no PII is inadvertently leaked by an agent might be addressed by design (the knowledge base likely stays on-prem, etc.).

Upgrades and Patches: Keeping the system secure means promptly updating agent dependencies and base images for any security patches (the requirements-security-summary.md likely tracks outdated packages). The pipeline and monitoring would flag any CVEs in the Python packages listed. Because all agents share a base image, updating that base (e.g. Python version, OS packages) and rebuilding will propagate fixes to all agent containers.

Finally, physical security and access control to the production environment: Only authorized engineers should have access to the infrastructure running these agents. The integration with Claude Desktop presumably runs on the user’s machine connecting to their local MCP server, so that is user-level. But the central agent cluster should be locked behind firewalls and VPN if necessary. Consul UI, RabbitMQ UI, etc., are likely not exposed publicly (or are secured with creds).

In summary, the agent ecosystem employs comprehensive security: mutual authentication for service calls, least privilege for agents, container isolation, secret vaulting, and ongoing security audits. Engineers adding new agents must also follow these practices: e.g. register a new service account for the agent, ensure the agent’s Dockerfile runs as non-root, and verify that it only does what it’s intended to.

Observability and Performance Monitoring

Having many moving parts demands strong observability. SutazAI’s platform is instrumented to allow monitoring, logging, and tracing of the agents and their interactions.

Logging: Every agent and orchestrator uses structured logging. The Python logging library is configured (often to INFO level by default) and in many places logging.getLogger(__name__) is used for module-specific logs
GitHub
GitHub
. Additionally, Structlog is included for structured JSON logs
GitHub
, which suggests logs are output in a machine-parseable format with key fields (like event_type, agent name, task_id, etc.). Key events logged include:

Agent lifecycle events: creation, start, pause, resume, stop (the AgentManager logs e.g. "Created agent: {agent_id}"
GitHub
, "Started agent: X"
GitHub
, and on stop, it logs and also logs if an agent was cleaned up
GitHub
).

Task events: The agent_interaction router logs each task request via the MonitoringService (e.g. logs an event "agent_task_requested" with details of agent and task_id)
GitHub
. Agents themselves log when they start processing a task and when they finish or error (the template code logs “Processing task of type: X” and logs errors if exceptions occur)
GitHub
GitHub
. The orchestrator logs error messages if an agent call fails
GitHub
 or if health monitor catches an error
GitHub
. These logs help trace what happened if something fails.

System events: Orchestrator initialization, background thread exceptions, any conflict detection, etc., would be logged. If conflict resolution were implemented, it would log details of the conflict and resolution strategy used.

All logs from containers are aggregated by a centralized system. Likely using Loki (as it was reserved in monitoring ports) or ELK stack. Logging is configured to not be overly verbose in normal operations, but with DEBUG mode available for deeper troubleshooting.

Metrics: The Prometheus client is installed in agents and orchestrator
GitHub
. The orchestrator and possibly the coordinator and arbitrator have internal counters/stats. For example:

The orchestrator can maintain counters for total_tasks, successful_tasks, failed_tasks, etc., which might be exposed via Prom metrics or the /status endpoint
GitHub
.

The Task Coordinator tracks queue length, which is likely in a metric or at least the /queue API. Also metrics like tasks_assigned_count, average_wait_time, tasks_per_agent (which agent got how many tasks).

The Resource Arbitrator tracks resource usage – it could expose metrics like current_cpu_usage_per_agent, current_memory_usage, allocation_requests, allocations_denied, etc.

Each agent could expose domain-specific metrics (e.g., a code generation agent might count “code_suggestions_generated_total”).

The presence of psutil means CPU and memory usage can be measured in-code; indeed the AgentMetrics dataclass stores cpu_percent and memory_percent per agent
GitHub
. The orchestrator (or agent manager) updates these periodically (the _update_agent_metrics function updates execution_time and likely could update CPU% by sampling psutil)
GitHub
GitHub
. These metrics can be exported.

Prometheus & Grafana: A Prometheus instance is configured to scrape metrics from all running services. Given the port ranges, it might scrape orchestrator on 11000/metrics, etc., or via service discovery. Grafana dashboards are presumably set up to visualize:

Overall System Health: Number of active agents vs total, number of tasks in queue, task success/failure rate, average task latency.

Agent-level Performance: For each agent, graphs of CPU%, memory%, tasks executed per minute, error count, last heartbeat time (perhaps displayed as time since last heartbeat).

Throughput and Latency: Charts for orchestrator showing how many tasks per second it’s handling, distribution of task durations, etc.

Resource usage: From the arbitrator or directly from node metrics (if Kubernetes, node exporter would give cluster CPU/mem usage).

Alerts: Likely Prometheus alerts are configured for conditions like: an agent’s last heartbeat > 2 minutes ago (meaning agent down), orchestrator task queue length growing too large (could indicate backlog), high error rate from any agent, or resource saturation (e.g., >90% CPU usage sustained).

Tracing: If enabled, distributed tracing (via Jaeger or OpenTelemetry) can track a task across services. For instance, when orchestrator receives a task (span start), then calls an agent (child span), and agent executes (child span maybe annotated with what it did, like “LLM generation” as sub-span), then returns to orchestrator which ends the span. The context propagation requires passing trace headers in the HTTP calls. The code doesn’t explicitly show OpenTelemetry, but given modern practices and the mention of Jaeger reserved ports, we can assume an integration exists or is planned. Possibly an environment variable like JAEGER_AGENT_HOST would be set for the services to auto-export spans. If implemented, an engineer can trace e.g. “Task ID 123” from orchestrator to Agent A to Agent B if collaborative, seeing timings and any errors along the way. This is invaluable for debugging performance issues or race conditions.

Monitoring Service and Audits: The code references a MonitoringService that is used in the FastAPI router to log events
GitHub
GitHub
. This could be hooking into a more sophisticated monitoring pipeline (e.g. sending events to an ELK stack or an internal analytics DB). Possibly it logs user-level events too (like which user triggered what agent). Auditing wise, every action can be traced back: tasks have IDs, agents log what they did. If a particular output is incorrect or problematic, one can find in logs which agent produced it and what inputs it had. There might also be periodic “hygiene” checks run (the pipeline’s validate:hygiene runs a hygiene-enforcement-coordinator.py and hygiene-agent-orchestrator.py --check
GitHub
). These likely verify that documentation is up to date, that logs contain expected entries, or that no agent is misbehaving (maybe even scanning runtime logs for anomalies).

Performance Targets: The system aims to meet certain SLOs:

Task latency: For a typical task that doesn’t involve heavy computation, the orchestrator’s overhead should be minimal (a few milliseconds to route). The goal might be <50ms added latency by orchestrator. E.g., a performance test achieved 10 concurrent requests handled in ~1245ms, meaning each request ~125ms which likely includes model generation time
GitHub
. So overhead is low.

Throughput: The orchestrator (and coordinator) should handle bursts of tasks without crashing – queueing them and processing as fast as agents allow. Memory usage should remain controlled (since tasks are mostly small JSON).

Resource usage: Each agent ideally uses under a certain amount of CPU and memory in steady state. For example, orchestrator might target <1 CPU core usage when idle. The arbitrator’s policies mention not oversubscribing CPU beyond 80% to keep system responsive
GitHub
.

Reliability: The system should have near-100% uptime for orchestrator and critical agents. The health checks and multi-instance possibilities aim for redundancy (e.g., one can run multiple orchestrators in active-passive or active-active if needed).

The coverage of tests (95% etc.) indicates reliability and correctness is highly valued.

Alerts & Automated Recovery: If an agent goes down, orchestrator marks it unreachable and could optionally trigger an auto-restart (if orchestrator or agent manager integrated with Docker/K8s control – but usually we rely on Docker auto-restart or K8s liveness probe to restart the container). For example, if an agent process segfaults, K8s will restart it and orchestrator will notice it’s healthy again on next check. If orchestrator itself fails, in Kubernetes a new pod is started (ensuring orchestrator’s state is mostly ephemeral except what it reconstructs from agents on start). Running orchestrator in HA (two instances) might require leader election to avoid duplicate registrations, but since heartbeats come in, possibly both can consume but only one should respond to external requests (the gateway can route to a primary).

Debugging and Profiling: For engineers, detailed logs can be turned on per agent by setting env LOG_LEVEL=DEBUG on that container to get step-by-step logs (like every HTTP request and response perhaps). For performance profiling, one could enable tools or metrics such as timing each step in orchestrator (maybe using middleware or the log of execution_time for tasks
GitHub
). The orchestrator currently measures execution time of tasks (in AgentManager’s execute_task, it logs execution_time and updates metrics)
GitHub
. These can be aggregated to find slow tasks or bottleneck agents.

In essence, the system is designed to be observable: nearly every significant action is logged or measured, and those data are collected in a central monitoring stack. Operating the system involves watching dashboards for any anomalies (spikes in errors, agent offline, queue growth) and reacting, often automatically (like auto-scaling or restart on failure). The aim is to detect issues before users do (for example, if an agent is not healthy, an alert can be sent to engineers or an automated script can spin up a replacement).

Error Handling and Resilience

No complex system is immune to errors, but SutazAI’s agent network is built to fail gracefully and recover automatically wherever possible. Here we outline how errors are handled at various levels and what mechanisms are in place for resilience:

Agent-Level Errors: If an agent encounters an error while executing a task (this could be an exception in code, inability to fulfill request, or model error):

The agent’s API will catch the exception and return an HTTP 500 with an error message in JSON (our FastAPI error handler does this by default, and we also explicitly catch and log errors in endpoints like /task)
GitHub
. For instance, if an agent is asked to open a file that doesn’t exist, it might raise an IOError; the endpoint catches it and responds with {"detail": "Error processing task: <error message>"}. The orchestrator upon receiving a non-200 or a response containing "error" will mark the task failed.

The orchestrator wraps the error in its response to the client. For example, AgentOrchestrator.execute_task returns a dict with "status": "failed", "error": "Agent X is not available"} if the agent was not healthy
GitHub
 or if the execution threw exception it returns an error field with the exception message
GitHub
. So the client (or MCP) always gets a structured indication rather than silence.

The agent also logs the error internally
GitHub
 and increases its error_count metric. The AgentManager’s monitoring thread looks at agent metrics and status; if an agent’s status goes to ERROR and error_count is still below threshold, the manager triggers a recovery attempt
GitHub
GitHub
. Recovery might mean re-initializing the agent instance or resetting some state. In the code, _handle_agent_error (not fully shown above, but references indicate it marks status ERROR and increments error_count, then in a recovery thread it sees error_count < max_retries and tries a simulated recovery)
GitHub
GitHub
. For example, if a Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real TestAgent crashed, the manager might create a new one to replace it (since these are local in that mode).

If the error is persistent (error_count exceeds max_retries or something fundamentally broken), the manager will mark the agent as FAILED or STOPPED. The orchestrator will then exclude that agent from selection (because its status is not healthy). In containers deployment, typically if an agent keeps erroring (e.g. failing health checks), Kubernetes will restart it automatically. So recovery often means just waiting for the restart (which orchestrator will detect as a fresh agent or via regained health).

Agent Unavailability: If an agent service goes offline unexpectedly (container crash, network issue):

The orchestrator’s health monitor will mark it "unreachable" after a failed ping
GitHub
. Any new tasks that would have gone to that agent are either queued (hoping agent returns) or rerouted to an alternative agent if possible. For example, if PrivateGPT agent is down, maybe BigAGI or LlamaIndex can handle a user question – the orchestrator might choose another capable agent if available. If not, it fails tasks targeting that agent with an error.

The orchestrator continues to ping; when the agent comes back up, a successful health check will mark it healthy again
GitHub
. The orchestrator also recalculates healthy_agents count which reflects in /health output
GitHub
 (so you can see how many are offline).

If an agent remains offline beyond a threshold (e.g. stale_agent_threshold 120s)
GitHub
, orchestrator could remove it from the registry. In current design, we typically leave it but marked unreachable.

With multiple orchestrator instances (if any), Consul or a leader election would manage failover. If orchestrator itself fails, Kubernetes will restart it within seconds. During orchestrator downtime, clients can’t submit tasks – but since tasks are ephemeral, this just results in a brief outage. On restart, orchestrator may need to rebuild state: it can query each agent’s /health and possibly ask them for current tasks (though in stateless design, it can consider tasks lost if orchestrator went down mid-task, unless tasks are persisted in a durable queue like RabbitMQ which orchestrator on restart could re-consume. There’s mention in future of Redis-backed state
GitHub
 to persist tasks and agent registry so orchestrator can recover state after a crash).

Task Failures and Retries: A task can fail for various reasons – agent error, external call failure, or a planned failure (like task precondition not met). The platform’s approach:

The orchestrator (or AgentManager) will retry a task automatically if it’s safe to do so. The AgentTask model has max_retries (default 3) and tracks retry_count
GitHub
. If a task fails (status=FAILED), and retry_count < max_retries, then can_retry() returns True
GitHub
. In such case, orchestrator might requeue the task (potentially after a backoff delay).

Exponential Backoff: While not explicitly in the code snippet, the AgentAuthUpdater script’s config for retries uses a backoff_factor: 2 and timeout: 30 as an example
GitHub
. In general, if an agent is busy or an external resource fails, the orchestrator should wait a moment and try again. One could implement a geometric backoff (like wait 1s, then 2s, then 4s) for successive retries to avoid hammering an agent that might be in trouble. The config’s global retry_delay of 5 seconds in AgentManager suggests a constant delay in their current recovery loop for agent errors
GitHub
.

Some tasks are not retryable (if the error is not transient). The developer can mark those in logic, or simply orchestrator will attempt up to max_retries and then give up. When giving up, it logs and sets task status to FAILED permanently.

For multi-agent workflows, if one subtask fails, the orchestrator might attempt a different agent for that subtask. For example, if the primary code generation agent fails to produce output, orchestrator could fall back to a simpler approach or another agent (maybe try a second attempt with Aider if GPT-Engineer failed). These contingency paths can be configured in task_routing preferences as multiple preferred_agents entries
GitHub
GitHub
.

At the client side, the presence of error messages means the UI (Claude or others) can inform the user and possibly prompt them to adjust input or try again.

Isolation and Fault Containment: Each agent runs separately, so a crash in one does not directly crash others. This isolation is key – heavy tasks like large model inference might run out of memory and crash their container, but orchestrator and others keep running. The orchestrator’s design ensures that if an agent is down, tasks for it fail but do not cascade. The AgentManager has locks around agent operations
GitHub
GitHub
 to avoid race conditions in multi-thread scenarios.

Fail-Safe Defaults: If the orchestrator cannot find a matching agent for a task (like a completely unknown task type), it defaults to AutoGPT (a generalist)
GitHub
. This is a safe fallback so that the system at least attempts something. Similarly, if the Multi-Agent Coordinator is unavailable for a collaborative task, orchestrator will sequentially query each agent as a fallback
GitHub
. These defaults ensure that the system tries to do something even in unexpected cases.

Circuit Breakers and Timeouts: Each agent call is wrapped with a timeout (the orchestrator uses httpx AsyncClient with timeout 60.0s by default for /execute calls)
GitHub
. If an agent doesn’t respond in 60s, orchestrator treats it as error (and likely marks agent unreachable). This prevents the orchestrator from hanging indefinitely on a bad call. The agent’s own operations often have timeouts (the LLM calls are given 30s in base agent)
GitHub
. For very long tasks, the orchestrator’s timeout_seconds in AgentTask or config (default 300s global)
GitHub
 ensures that if a task exceeds 5 minutes, it’s considered failed (the coordinator or manager can have a watchdog to check tasks’ start time vs now and cancel if too long).
A circuit breaker pattern could be implemented such that if an agent is failing repeatedly, orchestrator stops sending new tasks to it for a cool-off period. For now, marking it unhealthy achieves a similar effect.

Graceful Shutdown: If the system needs to shut down (for deploy or maintenance), orchestrator should ideally stop accepting new tasks and allow in-flight tasks to complete. A shutdown() method is provided in orchestrator to do a graceful stop
GitHub
. This would cancel or finish running tasks and mark orchestrator not initialized (so health endpoint might report "initializing" or "down"). In Kubernetes, one can set a preStop hook to call orchestrator’s /shutdown (if exposed) before killing it, giving it a chance to clean up. Agents similarly could have shutdown hooks (though currently probably just rely on process termination). The goal is to avoid cutting off tasks mid-way – in practice, short tasks make this less of an issue, but for long tasks, a strategy might be to quiesce, let tasks finish, then stop.

Data Persistence and Recovery: The orchestrator and agents largely operate in-memory with ephemeral data (aside from any domain-specific data stored in vector DB or Postgres via MCP). This means recovery mostly means recomputing state. For tasks, if orchestrator goes down, tasks not completed could be lost unless re-submitted. A future improvement might have orchestrator push pending tasks into a durable queue (like a RabbitMQ task queue) so that on restart it can resume. Right now, the assumption is orchestrator is quite stable or tasks can be reissued by clients if a failure occurs. For agent state, because each is independent, if one crashes it loses any in-memory context (for example, an AutoGPT agent's chain-of-thought if not saved externally is gone). To mitigate, agents could periodically checkpoint important state to the SharedMemory or a file. E.g., an agent doing a lengthy research might save intermediate results so that if restarted, it can attempt to continue. This is agent-specific though.

Testing for Resilience: The test suite likely includes scenarios of agent failures and restarts to ensure orchestrator handles them. Engineers should consider edge cases like network partitions (what if orchestrator can't reach an agent due to network blip? It will mark unreachable, then reachable again when back – and tasks might fail in between). Also consider scenario of partial results: if multi-agent collaboration where one agent fails, orchestrator should either attempt to continue with remaining or abort cleanly.

Preemption and Cancellation: The platform policies also consider preemption – if a high-priority task comes in, the coordinator or arbitrator might cancel or pause a lower priority task. For example, if resource arbitrator sees CPU fully used by a low-pri training job and a critical request arrives, it could signal that agent to pause or stop. The current code doesn’t show actual preemption in action, but the concept is mentioned (priority-based preemption)
GitHub
. Implementation might involve the arbitrator sending a message to the low-priority agent to gracefully halt (if possible). Agents would need to periodically check if they should abort (cooperative cancellation). In any case, the orchestrator and coordinator will not start new low priority tasks if high priority ones are waiting (that's the easier guarantee to implement).

Rollbacks: In deployment context, if a new version of an agent is causing errors (e.g. its error_count spikes), the quickest remedy is to trigger rollback (which the CI pipeline can do automatically if health checks fail). Meanwhile, orchestrator sees it unhealthy and stops using it, so the impact is minimized. Blue-green deployment means the old version is still up to take tasks while the new one is tested.

Fail-open vs Fail-closed: The system tends to fail closed for security – i.e., if something’s wrong, it errs on not performing an action rather than doing something potentially harmful. E.g., if auth token is invalid, the call is refused (not allowed to maybe run without auth). If an agent’s capabilities are unknown, orchestrator chooses a conservative default rather than risking an inappropriate agent. If conflict resolution isn't implemented, the orchestrator simply doesn't attempt it (so worst case some concurrent interactions might conflict, but currently conflict detection is off so it doesn’t block anything incorrectly either)
GitHub
GitHub
.

In essence, resilience in SutazAI’s agent system comes from redundancy (many agents, can substitute for each other to a degree), isolation (one crash doesn’t kill all), continuous monitoring (detect and respond quickly), and automation (retries, restarts). As the system evolves, more sophisticated self-healing (like auto scaling, predictive failure handling) can be added, but the foundation ensures a robust operation. Operators should still keep an eye on logs and alerts, but many common failures are handled gracefully without manual intervention.

Testing, Quality Assurance, and CI/CD Practices

All agent services in SutazAI are developed under strict quality controls. Here we summarize the testing and CI/CD norms that ensure the system remains reliable and maintainable as it grows:

Code Style and Linting: The project adheres to PEP8 style (with slight modifications) and uses automated formatters/linters:

Black is used for code formatting; the CI pipeline runs black --check on relevant directories (e.g., backend/, agents/) to ensure no formatting drift
GitHub
. This means developers should run Black before committing to avoid CI failures.

Flake8 is run with a max line length of 88 and ignoring some warnings (E203, W503 which are common with Black’s formatting)
GitHub
. Flake8 catches unused imports, undefined variables, etc., to keep the code clean.

Mypy (static type checker) enforces type hints consistency
GitHub
. The presence of many type annotations in code indicates it’s been considered. In CI, mypy backend/app is run to ensure type safety (with possibly ignoring third-party imports issues)
GitHub
.

Pylint may also be used (it’s installed in the CI environment) to catch more complex issues or style inconsistencies. The pipeline installs it
GitHub
, though the sample doesn’t show its invocation explicitly (it might be run as part of validate step).

These tools ensure a baseline quality: no obvious bugs like NameErrors or indentation mistakes pass through, and the codebase remains uniform in style.

Unit Tests: Each component (orchestrator, each type of agent that has logic, protocols, etc.) has corresponding unit tests. For instance:

Orchestrator tests would simulate registering agents, sending tasks (maybe with a dummy httpx client stub), and verifying correct agent selection and result aggregation.

Agent Factory and Agent Manager tests ensure that creating, starting, stopping agents behaves as expected and edge cases (creating an unknown type raises ValueError, etc.)
GitHub
.

The testing likely uses pytest (as in requirements)
GitHub
 with fixtures to simulate RabbitMQ or Redis if needed (or tests might use a test double).

Integration tests might spin up a subset of the system and run actual HTTP calls. The MCP_SERVER_INSTALLATION_GUIDE.md suggests a test suite exists (npm test for MCP server, and in our context possibly tests/run_tests.sh on the whole system)
GitHub
. They mention expected output from tests including performance tests results
GitHub
.

Test coverage is measured; the CI possibly produces a coverage report. The Implementation Guide numbers (95% orchestrator, etc.)
GitHub
 show a strong emphasis on thorough testing. Developers should aim to keep coverage high – adding tests for any new logic in an agent or orchestrator. The CI may enforce a minimum coverage threshold.

Continuous Integration (CI): The integrated pipeline automatically runs all validations and tests on each merge request and on main branch commits
GitHub
. Only code that passes all checks can be merged (likely enforced by branch protection). The CI also builds the Docker images for any changes to ensure build integrity (catching any Dockerfile issues early).

Continuous Deployment (CD): Merges to main (or a release tag) trigger the deployment stages, meaning new agent versions can be deployed perhaps multiple times a day in staging. However, deployment to production may be manual gated or require passing the staging tests (the pipeline likely waits for approval or success in staging before continuing to prod). The blue-green and canary strategies ensure that deploying updates doesn’t cause downtime or regressions:

For example, if an update to the orchestrator is deployed, it is brought up in parallel (with a different service name or behind a flag), then traffic is switched. If something goes wrong (new orchestrator not functioning), the system can revert to the old orchestrator quickly within the rollback timeout.

For agents, because each agent is separate, one can deploy updated agents one by one without affecting others. If a new agent type is added, deploying it doesn’t impact existing ones at all until orchestrator is aware of it. For an updated agent, orchestrator can continue to use an old instance until the new one signals readiness (health checks passing) – at which point orchestrator might start routing tasks to the new version (especially if it registered as the same agent ID – in a rolling update scenario, orchestrator would just see one agent bouncing in and out of health).

The blue-green approach might involve labeling all new pods as “green”, then update a service to point to green instead of blue. The orchestrator’s internal registry might need to update if agent addresses change, but if hostnames remain (e.g. using Kubernetes service names that stay constant), the orchestrator might not even notice (the health might just see a brief fail then recovery as pods switch).

Quality Gates: Apart from testing, additional checks:

Security scanning – e.g. using Bandit (Python security linter) or dependency scanning. Possibly part of "security" stage.

Coverage enforcement – maybe the pipeline fails if coverage drops below a threshold.

Hygiene enforcement – the hygiene-enforcement-coordinator.py could be a script that ensures certain standards: e.g., no TODO comments left, all agents have corresponding docs or configs, etc.

Documentation – The repository has architecture docs, and perhaps a requirement that they be updated with changes. This very AGENTS.md would be part of documentation to keep updated. Possibly there's a test to ensure that capabilities listed in code match those in docs or config (just speculation, but could be).

Local Development and Testing: Engineers can run agents locally (each agent directory likely can be run via uvicorn main:app). For orchestrator (which may be part of the backend app), one can run the FastAPI app directly (the backend/app/main.py likely includes orchestrator startup if ORCHESTRATOR_AVAILABLE)
GitHub
. The team might use tools like pytest with docker-compose to spin up a testing environment that includes dummy RabbitMQ/Redis and then run tests against it. Also, there's a Makefile to simplify common tasks (like make test, make lint, etc.).

Canary Testing of Agents: In addition to automated tests, new agents or major changes might go through a canary evaluation. Because the platform involves AI, automated tests can’t cover subjective quality of outputs. So, for example, a new version of the Code Generation agent might be deployed in staging and tested on a set of example tasks (some sort of evaluation script or even manual testing by engineers). Similarly, performance tests check that latency or memory usage hasn’t regressed significantly.

Coverage of Scenarios: The multi-agent nature yields many possible interaction patterns. Likely, tests include scenarios like:

Orchestrator routes a code generation task and then a security scan of that output using two different agents, verifying the pipeline works.

Stress tests: e.g., spawn 50 tasks concurrently, see that they queue and complete, no crashes or deadlocks.

Failure injection: simulate an agent’s /health returning 500 or not responding, confirm orchestrator marks it unreachable and tasks get retried or failed properly (maybe using a stub agent during test).

Long-running tasks: ensure that timeouts kick in and the orchestrator returns a failure after the timeout.

Race conditions: maybe test pausing an agent mid-task if possible, or two orchestrators running (if supporting HA) to see no double-processing.

Maintaining Agents: For each agent service repository code, there should be:

Up-to-date README or documentation describing how to run and test it (for developers working on that agent).

Unit tests covering its core logic (if it has any beyond calling an LLM).

Possibly sample inputs/outputs for manual verification.

CI Integration with Issue Tracking: The pipeline might also update GitHub/GitLab issues or PRs with test results, and use conventional commit messages or similar to automatically deploy certain branches. That might be beyond scope here, but it's worth noting as part of the polished development lifecycle.

In short, the development lifecycle in SutazAI ensures that by the time code is deployed to production:

It is formatted, linted, and statically typed correctly.

It passes extensive automated tests.

It doesn’t introduce known security holes or performance regressions (checked via specialized pipeline stages).

It can be deployed with minimal risk (blue-green and rollbacks).

Monitoring will catch any issue post-deployment and roll back if necessary.

All engineers contributing must follow these norms, and new agents should come with tests and documentation. By standardizing this process, the addition of dozens of specialized agents remains manageable and the system’s reliability remains high even as it scales up.

Adding and Extending Agents

One of the key strengths of the SutazAI architecture is its extensibility – engineers can create new agents or extend existing ones with minimal friction, thanks to standardized patterns. This section provides a step-by-step guide for adding a new agent service to the system, ensuring it meets all requirements from development through deployment and monitoring.

1. Design the Agent: First clarify the agent’s purpose, capabilities, and scope. Determine:

The agent’s name (use a descriptive, kebab-case name, e.g. "database-manager" or "quantum-optimizer").

Its capabilities list – these should be short keywords indicating what it can do. Try to reuse existing capability terms if applicable (check agents.yaml for common terms) or introduce new ones if needed.

The tasks or use cases it will handle. Identify if they fit existing task_routing categories or if new task types should be added. For example, if adding an "Image Processing Agent", you might introduce tasks like "analyze_image" or "generate_image".

Any dependencies or external tools it requires (e.g., does it need OpenCV library? GPU access?).

Security considerations: what tools should it have access to (file system read/write? internet? etc.), and what scopes in service account terms.

2. Create Agent Scaffold: Use the provided script or template to generate the basic file structure for the agent. The repository includes a script scripts/prepare-20-agents.py which demonstrates creating Dockerfile, main.py, requirements.txt for new agents
GitHub
GitHub
. You can adapt this or simply follow the same pattern:

Make a new directory under agents/ (if the codebase is structured that way) or under a similar path. For example, agents/image-processor/.

Create a main.py using the MAIN_PY_TEMPLATE from the script as guidance
GitHub
GitHub
. Edit the placeholder fields:

Set agent_name constant (same as your agent’s name).

Title and description to something concise.

The endpoint logic in process_task should be filled with at least a placeholder implementation that returns a success message. Initially, you might put a TODO and return a dummy result. (But if the agent’s logic is simple and you know it, implement it directly).

Ensure to include a HealthResponse and /health as provided.

Create a Dockerfile for the agent. Use the template from the script: base on Python 3.11 slim, copy code, install requirements, set AGENT_NAME env, healthcheck, run as non-root user, and entrypoint to uvicorn
GitHub
GitHub
. If your agent requires additional system packages (apt-get something like libopencv), add those in the Dockerfile.

Add any needed Python dependencies to requirements.txt in that agent folder (or if none special, it can remain mostly empty aside from a comment)
GitHub
. Also consider if it needs any of the optional packages (for example, for image processing you might add Pillow, etc.).

3. Define Capabilities and Routing: Open config/agents.yaml:

Add an entry for your agent under agents:. For example:

new_image_processor:
  id: new_image_processor
  queue: agent.image_processor
  capabilities:
    - image_analysis
    - image_generation
  max_concurrent_tasks: 3
  priority: 3
  health_check_interval: 60
  timeout_seconds: 300


Use appropriate values: queue name for messaging (even if not used now), a sane max_concurrent_tasks (if it’s CPU intensive, keep low), priority (critical=1, high=2, normal=3, etc. – how important tasks for this agent are relative to others), health check interval (usually 30 or 60 seconds), and default timeout (how long a typical task might take maximum).

Under task_routing: add mappings for any new task types to this agent. E.g.,

analyze_image:
  required_capabilities:
    - image_analysis
  preferred_agents:
    - new_image_processor
generate_image:
  required_capabilities:
    - image_generation
  preferred_agents:
    - new_image_processor


If your agent shares tasks with others, you can add it to existing preferred_agents list or create alternatives.

If needed, update assignment_strategies or global_settings but typically not necessary for adding one agent.

Double-check capabilities names do not clash inadvertently and represent what you want.

4. Implement Agent Logic: With skeleton in place, flesh out the agent’s process_task (or if it has more endpoints, implement those too). For AI-driven agents using LLM:

You can utilize the base classes like OllamaLocalAgent if appropriate. For instance, if your agent will mostly send prompts to the LLM, you might subclass OllamaLocalAgent (found in agents/base_agent.py) or just instantiate one inside your endpoint.

If using that base, ensure your container has access to OLLAMA_BASE_URL env (which it will, if the environment is configured cluster-wide).

If your agent performs specific computations (e.g., database queries, vector searches), implement them. Use asyncio if possible (the FastAPI endpoints can be async def and await I/O).

Handle errors gracefully: wrap risky operations in try/except and log errors. If something is not implemented yet, you might return {"message": "not implemented"} but better to raise HTTPException(500) so that orchestrator knows it failed.

Add security checks if needed. For example, if the agent should not do certain things without auth, but since everything passes through orchestrator with auth, you might be fine. If the agent calls external APIs, consider using stored API keys from env or vault.

Write docstrings and comments to explain your logic, since other team members or the orchestrator maintainers may need to understand it.

5. Test Locally: Before integrating into the main system:

Run uvicorn main:app --port 8080 in the agent’s directory to start it locally. Hit http://localhost:8080/health in a browser or curl to ensure it returns healthy JSON.

Test the POST /task endpoint with a sample payload. For example:

curl -X POST http://localhost:8080/task -H "Content-Type: application/json" -d '{"type": "analyze_image", "data": {"image_url": "http://example.com/pic.jpg"}}'


See that you get a reasonable response (like status success and some dummy result). This is essentially unit testing the agent in isolation.

Write unit tests for the agent’s internal functions if it has any complex logic. For simple ones relying on LLM, you might skip heavy Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Testing but at least test that the agent responds to certain inputs correctly (maybe by stubbing the LLM calls).

If possible, integrate with orchestrator locally: run the orchestrator (which might require running the entire backend FastAPI). Possibly use docker-compose to spin up orchestrator, a dummy Redis/Rabbit, and your agent container to see them working together. If not, at least simulate orchestrator’s call by making sure your agent returns expected output given a typical input.

6. Documentation and Persona (for AI logic agents): If your agent is knowledge or conversation-based (like those defined in .md persona files), create a markdown file for it:

Follow the format shown in .claude/agents/agent-expert.md etc. Start with YAML frontmatter with name, description (when to use this agent), maybe tools if applicable, and optionally examples.

Then write a system message: "You are a [Expert in X]. Your job is to...". Include guidelines, steps, etc., focusing on the domain.

Save this as <agent_name>.md perhaps in a agents/ folder that gets into the container at runtime (the base code looks for /agents/<name>.md). If using the prepare script, note it copies COPY . . which might include these files
GitHub
. Ensure it's included in Docker build (if not, adjust Dockerfile to copy it).

The format_agent_prompt in base_agent will incorporate this description automatically for you
GitHub
. So, writing a good persona file improves your agent’s output quality significantly.

7. Integration in Orchestrator: The orchestrator static registry (in backend/app/agent_orchestrator.py) might need updating if dynamic reg is not fully functional yet:

Add an entry in _register_agents for your new agent type. For example:

self.agents[AgentType.IMAGEPROCESSOR] = Agent(
    name="ImageProcessor",
    type=AgentType.IMAGEPROCESSOR,
    url="http://image-processor",  # hostname in docker compose or k8s service
    port=8080,
    capabilities=["image_analysis", "image_generation"]
)


And also add the type in the AgentType Enum (e.g. IMAGEPROCESSOR = "imageprocessor"). This ensures the orchestrator knows about it at startup and pings it. If you skip this, the orchestrator will not route tasks to it unless you implement dynamic registration (like sending a registration message).

Update any selection logic if needed. For instance, perhaps in _select_agent you’d add:

elif "image" in task_type or "image" in task_desc:
    return AgentType.IMAGEPROCESSOR


so that for tasks related to images, it picks your agent. Or rely on the default capability matching via the coordinator.

If using dynamic approach, you might instead rely on an AgentRegistrationMessage being sent by your agent on startup. But to keep it simple, modifying orchestrator’s list is fine.

8. Add to Port Registry and Compose/Helm: In config/port-registry.yaml, assign a port for your agent service:

Find the appropriate range (11020-11044 for specialized agents of medium priority, or 11000-11019 if you consider it critical). Choose the next free port number.

Add an entry:

11030:
  service: image-processor
  container: sutazai-image-processor
  description: "Image processing agent"
  internal_port: 8080
  priority: high


The container name here is what your Docker image will be labeled as (they usually prefix with sutazai-). Also reflect the priority.

This port is used by docker-compose or k8s to route traffic.

Add your service to docker-compose.yml or Helm charts: ensure the new agent container is defined, using the built image and the correct environment variables (likely just needs the basics and any model or API keys if required).

If using Kubernetes, add a Deployment and Service for it. The naming convention should match others (the CI's build job suggests images are named by agent directory, which is good if you followed that).

Also update any relevant documentation or lists of agents (like this AGENTS.md or any overview doc that enumerates agents).

9. CI Pipeline Adjustments: If needed, add your agent to the CI build matrix:

The .gitlab-ci.yml has groups for agents. If your agent fits one (like 'security' or 'monitoring'), add its name in that list. E.g., in the snippet
GitHub
, if adding to specialized, add image-processor in that loop. Or create a new category if these don't fit (e.g., if the groups are just to parallelize, you can slot it anywhere, but keep balance).

Ensure tests are included: if you created agents/youragent/tests/test_something.py, the pytest discovery should pick it up.

The pipeline’s TOTAL_AGENTS might need update if that number is explicitly used.

10. Testing in Integration: Deploy to a dev/staging environment:

Confirm the new agent comes up and registers (if statically configured, orchestrator logs should show it initialized and health check passed for it).

Use orchestrator’s API or MCP to send a test task of the type it handles. Check that orchestrator selects it and gets a response.

Check Grafana that metrics from your agent appear (if applicable) and logs show up in central log.

Perhaps run a load test or two on the new agent to ensure it can handle expected load and see if it respects memory/CPU limits.

11. Documentation & Maintenance: Write documentation for the agent:

Purpose and usage of the agent (for internal wiki or user-facing docs if needed).

Update this AGENTS.md in the relevant section for new capabilities and any new endpoints if you added custom ones.

Ensure security is covered (like if it needs special secrets, document how to provision them e.g. "This agent requires an API key in Vault at secret/path X").

12. Observability for New Agent: Add any specific metrics or traces:

If your agent does something significant, you might instrument extra Prom metrics. E.g., count how many images processed.

Add a log line for when a task is done, to easily grep logs for it.

If needed, create a Grafana panel for it (though if using consistent metrics names, maybe not needed specifically).

By following these steps, the new agent will integrate smoothly:

The orchestrator will begin sending it tasks it’s meant for.

The agent will operate under the same monitoring and security regime as others.

CI will automatically include it in future builds and quality checks.

Future developers will see it documented and tested, making maintenance easier.

Example: Suppose we add "Database Manager" agent to manage database migrations or queries. We would:

Implement an agent that listens for tasks like "run_migration" or "backup_database".

Maybe it calls a script or uses Python to connect to Postgres. We add psycopg2 to its requirements and ensure a DB connection string is provided via env.

The orchestrator might not spontaneously assign tasks to it unless instructed, so usage might be primarily via explicit calls (like an operator triggers a backup via orchestrator).

We thoroughly test it because messing with databases is sensitive.

We give it limited scope (perhaps read-only queries unless specifically allowed for migrations).

This disciplined approach ensures the system can keep expanding (to, say, 50 or 100 agent services) without chaos: everything is catalogued, standardized, and observable.

Conclusion

With the information in this manual, an engineer or operator should be able to understand every detail of SutazAI’s agent services architecture and manage it confidently. We covered how the multi-agent orchestration works, how each agent is built and behaves, the communication patterns and protocols they follow, and the rigorous practices around security, testing, and deployment that keep the system robust. Always ensure that any changes to the system align with these standards – maintain clear capabilities, proper health checks, thorough tests, and secure configurations. The goal is a self-coordinating AI system that is powerful yet predictable, complex yet controlled. By adhering to this guide, one can extend the system’s capabilities (adding new “skills” via agents or improving existing ones) without compromising stability or clarity.

The SutazAI agent ecosystem is essentially a digital organization of specialized AI workers with the orchestrator as their manager. Just as a real organization needs structure, communication, and oversight, so does SutazAI – and this document serves as the operating handbook for it. Use it to ensure every agent and service operates in concert, delivering intelligent automation with reliability and transparency.