#!/usr/bin/env node

/**
 * SutazAI MCP Server - Model Context Protocol Integration
 * Provides comprehensive access to SutazAI AGI/ASI system capabilities
 */

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ErrorCode,
  ListResourcesRequestSchema,
  ListToolsRequestSchema,
  McpError,
  ReadResourceRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import pg from 'pg';
import redis from 'redis';
import axios from 'axios';

const { Client } = pg;

// Configuration
const CONFIG = {
  DATABASE_URL: process.env.DATABASE_URL || 'postgresql://sutazai:sutazai_password@localhost:5432/sutazai',
  REDIS_URL: process.env.REDIS_URL || 'redis://localhost:6379',
  BACKEND_API_URL: process.env.BACKEND_API_URL || 'http://localhost:8000',
  OLLAMA_URL: process.env.OLLAMA_URL || 'http://localhost:11434',
  CHROMADB_URL: process.env.CHROMADB_URL || 'http://localhost:8000',
  QDRANT_URL: process.env.QDRANT_URL || 'http://localhost:6333',
};

// Database connections
let pgClient;
let redisClient;

// Initialize database connections
async function initializeConnections() {
  try {
    // PostgreSQL connection
    pgClient = new Client({
      connectionString: CONFIG.DATABASE_URL,
    });
    await pgClient.connect();
    console.error(JSON.stringify({
      type: "info",
      message: "Connected to PostgreSQL database"
    }));

    // Redis connection
    redisClient = redis.createClient({
      url: CONFIG.REDIS_URL
    });
    await redisClient.connect();
    console.error(JSON.stringify({
      type: "info",
      message: "Connected to Redis cache"
    }));

  } catch (error) {
    console.error(JSON.stringify({
      type: "error",
      message: "Database connection failed",
      error: error.message
    }));
  }
}

// Create MCP server instance
const server = new Server(
  {
    name: "sutazai-mcp-server",
    version: "1.0.0",
  },
  {
    capabilities: {
      resources: {},
      tools: {},
    },
  }
);

// ===========================================
// RESOURCE HANDLERS
// ===========================================

server.setRequestHandler(ListResourcesRequestSchema, async () => {
  return {
    resources: [
      {
        uri: "sutazai://agents/list",
        mimeType: "application/json",
        name: "AI Agents",
        description: "List of all registered AI agents and their status"
      },
      {
        uri: "sutazai://models/available",
        mimeType: "application/json", 
        name: "Available Models",
        description: "All AI models available through Ollama"
      },
      {
        uri: "sutazai://agents/tasks",
        mimeType: "application/json",
        name: "Agent Tasks",
        description: "Current and completed agent tasks"
      },
      {
        uri: "sutazai://system/metrics",
        mimeType: "application/json",
        name: "System Metrics",
        description: "Real-time system performance and health metrics"
      },
      {
        uri: "sutazai://knowledge/embeddings",
        mimeType: "application/json",
        name: "Knowledge Base",
        description: "Document embeddings and knowledge base entries"
      },
      {
        uri: "sutazai://agents/workspaces",
        mimeType: "application/json",
        name: "Agent Workspaces",
        description: "Agent workspace data and outputs"
      }
    ],
  };
});

server.setRequestHandler(ReadResourceRequestSchema, async (request) => {
  const { uri } = request.params;

  try {
    switch (uri) {
      case "sutazai://agents/list":
        return await getAgentsList();
      
      case "sutazai://models/available":
        return await getAvailableModels();
        
      case "sutazai://agents/tasks":
        return await getAgentTasks();
        
      case "sutazai://system/metrics":
        return await getSystemMetrics();
        
      case "sutazai://knowledge/embeddings":
        return await getKnowledgeBase();
        
      case "sutazai://agents/workspaces":
        return await getAgentWorkspaces();
        
      default:
        throw new McpError(ErrorCode.InvalidRequest, `Unknown resource: ${uri}`);
    }
  } catch (error) {
    throw new McpError(ErrorCode.InternalError, `Resource error: ${error.message}`);
  }
});

// ===========================================
// TOOL HANDLERS  
// ===========================================

server.setRequestHandler(ListToolsRequestSchema, async () => {
  return {
    tools: [
      {
        name: "deploy_agent",
        description: "Deploy a new AI agent with specified configuration",
        inputSchema: {
          type: "object",
          properties: {
            agent_type: {
              type: "string",
              enum: ["autogpt", "crewai", "localagi", "tabbyml", "semgrep", "browser_use", "skyvern", "documind", "finrobot", "gpt_engineer", "aider", "bigagi", "agentzero", "langflow", "dify", "autogen", "agentgpt", "privategpt", "llamaindex", "flowise", "shellgpt", "pentestgpt"],
              description: "Type of AI agent to deploy"
            },
            name: {
              type: "string",
              description: "Unique name for the agent instance"
            },
            capabilities: {
              type: "array",
              items: { type: "string" },
              description: "List of capabilities for the agent"
            },
            config: {
              type: "object",
              description: "Agent-specific configuration parameters"
            }
          },
          required: ["agent_type", "name"]
        }
      },
      {
        name: "execute_agent_task",
        description: "Execute a task using a specific AI agent",
        inputSchema: {
          type: "object",
          properties: {
            agent_name: {
              type: "string",
              description: "Name of the agent to use"
            },
            task: {
              type: "string", 
              description: "Task description or instruction"
            },
            context: {
              type: "object",
              description: "Additional context for the task"
            },
            priority: {
              type: "string",
              enum: ["low", "normal", "high", "urgent"],
              default: "normal"
            }
          },
          required: ["agent_name", "task"]
        }
      },
      {
        name: "manage_model",
        description: "Manage AI models through Ollama",
        inputSchema: {
          type: "object",
          properties: {
            action: {
              type: "string",
              enum: ["pull", "delete", "list", "run", "stop"],
              description: "Action to perform on the model"
            },
            model_name: {
              type: "string",
              description: "Name of the model (e.g., 'tinyllama')"
            },
            parameters: {
              type: "object",
              description: "Additional parameters for the action"
            }
          },
          required: ["action"]
        }
      },
      {
        name: "query_knowledge_base",
        description: "Query the vector knowledge base for relevant information",
        inputSchema: {
          type: "object",
          properties: {
            query: {
              type: "string",
              description: "Search query for the knowledge base"
            },
            collection: {
              type: "string",
              description: "Specific collection to search (optional)"
            },
            limit: {
              type: "integer",
              default: 10,
              description: "Maximum number of results to return"
            },
            similarity_threshold: {
              type: "number",
              default: 0.7,
              description: "Minimum similarity score for results"
            }
          },
          required: ["query"]
        }
      },
      {
        name: "monitor_system",
        description: "Monitor system health and performance metrics",
        inputSchema: {
          type: "object",
          properties: {
            metric_type: {
              type: "string",
              enum: ["cpu", "memory", "disk", "network", "containers", "agents", "models"],
              description: "Type of metrics to retrieve"
            },
            time_range: {
              type: "string",
              enum: ["1h", "6h", "24h", "7d"],
              default: "1h",
              description: "Time range for metrics"
            }
          },
          required: ["metric_type"]
        }
      },
      {
        name: "manage_agent_workspace",
        description: "Manage agent workspaces and data persistence",
        inputSchema: {
          type: "object",
          properties: {
            action: {
              type: "string",
              enum: ["create", "delete", "backup", "restore", "list"],
              description: "Action to perform on workspace"
            },
            agent_name: {
              type: "string",
              description: "Agent name for workspace operations"
            },
            workspace_data: {
              type: "object",
              description: "Workspace configuration or data"
            }
          },
          required: ["action"]
        }
      },
      {
        name: "orchestrate_multi_agent",
        description: "Orchestrate complex tasks across multiple AI agents",
        inputSchema: {
          type: "object",
          properties: {
            task_description: {
              type: "string",
              description: "Overall task description"
            },
            agents: {
              type: "array",
              items: {
                type: "object",
                properties: {
                  agent_name: { type: "string" },
                  role: { type: "string" },
                  subtask: { type: "string" }
                }
              },
              description: "List of agents and their roles"
            },
            coordination_strategy: {
              type: "string",
              enum: ["sequential", "parallel", "hierarchical", "collaborative"],
              default: "collaborative"
            }
          },
          required: ["task_description", "agents"]
        }
      }
    ],
  };
});

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;

  try {
    switch (name) {
      case "deploy_agent":
        return await deployAgent(args);
        
      case "execute_agent_task":
        return await executeAgentTask(args);
        
      case "manage_model":
        return await manageModel(args);
        
      case "query_knowledge_base":
        return await queryKnowledgeBase(args);
        
      case "monitor_system":
        return await monitorSystem(args);
        
      case "manage_agent_workspace":
        return await manageAgentWorkspace(args);
        
      case "orchestrate_multi_agent":
        return await orchestrateMultiAgent(args);
        
      default:
        throw new McpError(ErrorCode.MethodNotFound, `Unknown tool: ${name}`);
    }
  } catch (error) {
    throw new McpError(ErrorCode.InternalError, `Tool execution error: ${error.message}`);
  }
});

// ===========================================
// RESOURCE IMPLEMENTATION FUNCTIONS
// ===========================================

async function getAgentsList() {
  try {
    // Get agents from database
    const result = await pgClient.query(`
      SELECT agent_name, agent_type, status, capabilities, last_seen, config
      FROM agents 
      ORDER BY last_seen DESC
    `);
    
    // Also get live status from backend API
    let liveStatus = {};
    try {
      const response = await axios.get(`${CONFIG.BACKEND_API_URL}/api/v1/agents/status`);
      liveStatus = response.data;
    } catch (error) {
      console.error("Failed to get live agent status:", error.message);
    }

    const agents = result.rows.map(row => ({
      ...row,
      live_status: liveStatus[row.agent_name] || 'unknown'
    }));

    return {
      contents: [
        {
          uri: "sutazai://agents/list",
          mimeType: "application/json",
          text: JSON.stringify({
            agents,
            total_count: agents.length,
            active_count: agents.filter(a => a.status === 'active').length,
            timestamp: new Date().toISOString()
          }, null, 2)
        }
      ]
    };
  } catch (error) {
    throw new Error(`Failed to get agents list: ${error.message}`);
  }
}

async function getAvailableModels() {
  try {
    const response = await axios.get(`${CONFIG.OLLAMA_URL}/api/tags`);
    const models = response.data.models || [];

    return {
      contents: [
        {
          uri: "sutazai://models/available", 
          mimeType: "application/json",
          text: JSON.stringify({
            models: models.map(model => ({
              name: model.name,
              size: model.size,
              modified_at: model.modified_at,
              digest: model.digest,
              details: model.details
            })),
            total_count: models.length,
            timestamp: new Date().toISOString()
          }, null, 2)
        }
      ]
    };
  } catch (error) {
    throw new Error(`Failed to get available models: ${error.message}`);
  }
}

async function getAgentTasks() {
  try {
    const result = await pgClient.query(`
      SELECT task_id, agent_name, task_description, status, created_at, 
             completed_at, result, priority, context
      FROM agent_tasks 
      ORDER BY created_at DESC 
      LIMIT 100
    `);

    return {
      contents: [
        {
          uri: "sutazai://agents/tasks",
          mimeType: "application/json", 
          text: JSON.stringify({
            tasks: result.rows,
            total_count: result.rows.length,
            timestamp: new Date().toISOString()
          }, null, 2)
        }
      ]
    };
  } catch (error) {
    throw new Error(`Failed to get agent tasks: ${error.message}`);
  }
}

async function getSystemMetrics() {
  try {
    // Get system metrics from Redis cache or database
    const metricsData = await redisClient.get('system:metrics:latest');
    let metrics = {};
    
    if (metricsData) {
      metrics = JSON.parse(metricsData);
    } else {
      // Fallback to basic metrics
      metrics = {
        cpu_usage: 0,
        memory_usage: 0,
        disk_usage: 0,
        active_agents: 0,
        running_models: 0,
        timestamp: new Date().toISOString()
      };
    }

    return {
      contents: [
        {
          uri: "sutazai://system/metrics",
          mimeType: "application/json",
          text: JSON.stringify(metrics, null, 2)
        }
      ]
    };
  } catch (error) {
    throw new Error(`Failed to get system metrics: ${error.message}`);
  }
}

async function getKnowledgeBase() {
  try {
    const result = await pgClient.query(`
      SELECT document_id, title, content_preview, embedding_model, 
             created_at, metadata, collection_name
      FROM knowledge_documents 
      ORDER BY created_at DESC 
      LIMIT 50
    `);

    return {
      contents: [
        {
          uri: "sutazai://knowledge/embeddings",
          mimeType: "application/json",
          text: JSON.stringify({
            documents: result.rows,
            total_count: result.rows.length,
            timestamp: new Date().toISOString()
          }, null, 2)
        }
      ]
    };
  } catch (error) {
    throw new Error(`Failed to get knowledge base: ${error.message}`);
  }
}

async function getAgentWorkspaces() {
  try {
    const result = await pgClient.query(`
      SELECT workspace_id, agent_name, workspace_path, size_mb, 
             last_modified, backup_status
      FROM agent_workspaces 
      ORDER BY last_modified DESC
    `);

    return {
      contents: [
        {
          uri: "sutazai://agents/workspaces",
          mimeType: "application/json",
          text: JSON.stringify({
            workspaces: result.rows,
            total_count: result.rows.length,
            total_size_mb: result.rows.reduce((sum, w) => sum + (w.size_mb || 0), 0),
            timestamp: new Date().toISOString()
          }, null, 2)
        }
      ]
    };
  } catch (error) {
    throw new Error(`Failed to get agent workspaces: ${error.message}`);
  }
}

// ===========================================
// TOOL IMPLEMENTATION FUNCTIONS
// ===========================================

async function deployAgent(args) {
  const { agent_type, name, capabilities = [], config = {} } = args;
  
  try {
    // Call backend API to deploy agent
    const response = await axios.post(`${CONFIG.BACKEND_API_URL}/api/v1/agents/deploy`, {
      agent_type,
      name,
      capabilities,
      config
    });

    // Store in database
    await pgClient.query(`
      INSERT INTO agents (agent_name, agent_type, status, capabilities, config, created_at)
      VALUES ($1, $2, $3, $4, $5, NOW())
      ON CONFLICT (agent_name) 
      DO UPDATE SET agent_type = $2, capabilities = $4, config = $5, updated_at = NOW()
    `, [name, agent_type, 'deploying', JSON.stringify(capabilities), JSON.stringify(config)]);

    return {
      content: [
        {
          type: "text",
          text: JSON.stringify({
            success: true,
            message: `Agent '${name}' of type '${agent_type}' deployed successfully`,
            agent_id: response.data.agent_id,
            deployment_status: response.data.status,
            timestamp: new Date().toISOString()
          }, null, 2)
        }
      ]
    };
  } catch (error) {
    return {
      content: [
        {
          type: "text", 
          text: JSON.stringify({
            success: false,
            error: `Failed to deploy agent: ${error.message}`,
            timestamp: new Date().toISOString()
          }, null, 2)
        }
      ]
    };
  }
}

async function executeAgentTask(args) {
  const { agent_name, task, context = {}, priority = 'normal' } = args;
  
  try {
    // Create task in database
    const taskResult = await pgClient.query(`
      INSERT INTO agent_tasks (agent_name, task_description, status, priority, context, created_at)
      VALUES ($1, $2, 'pending', $3, $4, NOW())
      RETURNING task_id
    `, [agent_name, task, priority, JSON.stringify(context)]);
    
    const taskId = taskResult.rows[0].task_id;

    // Submit task to backend API
    const response = await axios.post(`${CONFIG.BACKEND_API_URL}/api/v1/agents/execute`, {
      agent_name,
      task_id: taskId,
      task,
      context,
      priority
    });

    return {
      content: [
        {
          type: "text",
          text: JSON.stringify({
            success: true,
            task_id: taskId,
            message: `Task submitted to agent '${agent_name}' successfully`,
            execution_status: response.data.status,
            estimated_completion: response.data.estimated_completion,
            timestamp: new Date().toISOString()
          }, null, 2)
        }
      ]
    };
  } catch (error) {
    return {
      content: [
        {
          type: "text",
          text: JSON.stringify({
            success: false,
            error: `Failed to execute agent task: ${error.message}`,
            timestamp: new Date().toISOString()
          }, null, 2)
        }
      ]
    };
  }
}

async function manageModel(args) {
  const { action, model_name, parameters = {} } = args;
  
  try {
    let response;
    
    switch (action) {
      case 'list':
        response = await axios.get(`${CONFIG.OLLAMA_URL}/api/tags`);
        break;
        
      case 'pull':
        if (!model_name) throw new Error('model_name required for pull action');
        response = await axios.post(`${CONFIG.OLLAMA_URL}/api/pull`, {
          name: model_name,
          ...parameters
        });
        break;
        
      case 'delete':
        if (!model_name) throw new Error('model_name required for delete action');
        response = await axios.delete(`${CONFIG.OLLAMA_URL}/api/delete`, {
          data: { name: model_name }
        });
        break;
        
      case 'run':
        if (!model_name) throw new Error('model_name required for run action');
        response = await axios.post(`${CONFIG.OLLAMA_URL}/api/generate`, {
          model: model_name,
          prompt: parameters.prompt || "Hello",
          stream: false,
          ...parameters
        });
        break;
        
      default:
        throw new Error(`Unknown action: ${action}`);
    }

    return {
      content: [
        {
          type: "text",
          text: JSON.stringify({
            success: true,
            action,
            model_name,
            result: response.data,
            timestamp: new Date().toISOString()
          }, null, 2)
        }
      ]
    };
  } catch (error) {
    return {
      content: [
        {
          type: "text",
          text: JSON.stringify({
            success: false,
            action,
            model_name,
            error: error.message,
            timestamp: new Date().toISOString()
          }, null, 2)
        }
      ]
    };
  }
}

async function queryKnowledgeBase(args) {
  const { query, collection, limit = 10, similarity_threshold = 0.7 } = args;
  
  try {
    // Query vector database through backend API
    const response = await axios.post(`${CONFIG.BACKEND_API_URL}/api/v1/knowledge/search`, {
      query,
      collection,
      limit,
      similarity_threshold
    });

    return {
      content: [
        {
          type: "text",
          text: JSON.stringify({
            success: true,
            query,
            results: response.data.results,
            total_found: response.data.total_found,
            processing_time_ms: response.data.processing_time_ms,
            timestamp: new Date().toISOString()
          }, null, 2)
        }
      ]
    };
  } catch (error) {
    return {
      content: [
        {
          type: "text",
          text: JSON.stringify({
            success: false,
            query,
            error: `Failed to query knowledge base: ${error.message}`,
            timestamp: new Date().toISOString()
          }, null, 2)
        }
      ]
    };
  }
}

async function monitorSystem(args) {
  const { metric_type, time_range = '1h' } = args;
  
  try {
    // Get metrics from monitoring API
    const response = await axios.get(`${CONFIG.BACKEND_API_URL}/api/v1/monitoring/metrics`, {
      params: { type: metric_type, range: time_range }
    });

    return {
      content: [
        {
          type: "text",
          text: JSON.stringify({
            success: true,
            metric_type,
            time_range,
            metrics: response.data.metrics,
            summary: response.data.summary,
            timestamp: new Date().toISOString()
          }, null, 2)
        }
      ]
    };
  } catch (error) {
    return {
      content: [
        {
          type: "text",
          text: JSON.stringify({
            success: false,
            metric_type,
            error: `Failed to get system metrics: ${error.message}`,
            timestamp: new Date().toISOString()
          }, null, 2)
        }
      ]
    };
  }
}

async function manageAgentWorkspace(args) {
  const { action, agent_name, workspace_data = {} } = args;
  
  try {
    const response = await axios.post(`${CONFIG.BACKEND_API_URL}/api/v1/agents/workspace`, {
      action,
      agent_name,
      workspace_data
    });

    return {
      content: [
        {
          type: "text",
          text: JSON.stringify({
            success: true,
            action,
            agent_name,
            result: response.data,
            timestamp: new Date().toISOString()
          }, null, 2)
        }
      ]
    };
  } catch (error) {
    return {
      content: [
        {
          type: "text",
          text: JSON.stringify({
            success: false,
            action,
            agent_name,
            error: `Failed to manage workspace: ${error.message}`,
            timestamp: new Date().toISOString()
          }, null, 2)
        }
      ]
    };
  }
}

async function orchestrateMultiAgent(args) {
  const { task_description, agents, coordination_strategy = 'collaborative' } = args;
  
  try {
    // Submit multi-agent orchestration request
    const response = await axios.post(`${CONFIG.BACKEND_API_URL}/api/v1/orchestration/multi-agent`, {
      task_description,
      agents,
      coordination_strategy
    });

    // Store orchestration session
    const sessionResult = await pgClient.query(`
      INSERT INTO orchestration_sessions (task_description, agents, strategy, status, created_at)
      VALUES ($1, $2, $3, 'active', NOW())
      RETURNING session_id
    `, [task_description, JSON.stringify(agents), coordination_strategy]);

    return {
      content: [
        {
          type: "text",
          text: JSON.stringify({
            success: true,
            session_id: sessionResult.rows[0].session_id,
            orchestration_id: response.data.orchestration_id,
            task_description,
            agents_count: agents.length,
            coordination_strategy,
            estimated_completion: response.data.estimated_completion,
            timestamp: new Date().toISOString()
          }, null, 2)
        }
      ]
    };
  } catch (error) {
    return {
      content: [
        {
          type: "text",
          text: JSON.stringify({
            success: false,
            task_description,
            error: `Failed to orchestrate multi-agent task: ${error.message}`,
            timestamp: new Date().toISOString()
          }, null, 2)
        }
      ]
    };
  }
}

// ===========================================
// SERVER STARTUP
// ===========================================

async function main() {
  try {
    // Initialize database connections
    await initializeConnections();
    
    // Start MCP server
    const transport = new StdioServerTransport();
    await server.connect(transport);

    console.error(JSON.stringify({
      type: "info",
      message: "SutazAI MCP Server started successfully",
      capabilities: {
        agents: "✓ Deploy and manage AI agents",
        models: "✓ Manage Ollama models", 
        knowledge: "✓ Query vector knowledge base",
        orchestration: "✓ Multi-agent coordination",
        monitoring: "✓ System health monitoring",
        workspaces: "✓ Agent workspace management"
      },
      timestamp: new Date().toISOString()
    }));

  } catch (error) {
    console.error(JSON.stringify({
      type: "error",
      message: "Failed to start SutazAI MCP Server",
      error: error.message,
      timestamp: new Date().toISOString()
    }));
    process.exit(1);
  }
}

// Handle graceful shutdown
process.on('SIGINT', async () => {
  console.error(JSON.stringify({
    type: "info", 
    message: "Shutting down SutazAI MCP Server..."
  }));
  
  try {
    if (pgClient) await pgClient.end();
    if (redisClient) await redisClient.disconnect();
  } catch (error) {
    console.error(JSON.stringify({
      type: "error",
      message: "Error during shutdown",
      error: error.message
    }));
  }
  
  process.exit(0);
});

// Start the server
main().catch((error) => {
  console.error(JSON.stringify({
    type: "error",
    message: "Server startup failed", 
    error: error.message
  }));
  process.exit(1);
}); 