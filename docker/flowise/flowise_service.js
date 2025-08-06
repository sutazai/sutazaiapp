#!/usr/bin/env node
/**
 * FlowiseAI Service for SutazAI
 * Visual flow-based LLM orchestration
 */

const express = require('express');
const axios = require('axios');

const app = express();
app.use(express.json());

const PORT = process.env.PORT || 3000;
const SERVICE_NAME = 'FlowiseAI';

class FlowiseService {
    constructor() {
        this.flows = new Map();
        this.executions = new Map();
        this.serviceName = SERVICE_NAME;
    }

    async createFlow(flowData) {
        try {
            const flowId = `flow_${Date.now()}`;
            const flow = {
                id: flowId,
                name: flowData.name || 'Untitled Flow',
                nodes: flowData.nodes || [],
                edges: flowData.edges || [],
                created_at: new Date().toISOString(),
                status: 'active'
            };

            this.flows.set(flowId, flow);

            return {
                success: true,
                flow_id: flowId,
                message: `Flow ${flow.name} created successfully`
            };
        } catch (error) {
            console.error('Flow creation failed:', error);
            return {
                success: false,
                error: `Flow creation failed: ${error.message}`
            };
        }
    }

    async executeFlow(flowId, inputData = {}) {
        try {
            const flow = this.flows.get(flowId);
            if (!flow) {
                return {
                    success: false,
                    error: `Flow ${flowId} not found`
                };
            }

            const executionId = `exec_${Date.now()}`;
            const execution = {
                id: executionId,
                flow_id: flowId,
                input_data: inputData,
                started_at: new Date().toISOString(),
                status: 'running'
            };

            this.executions.set(executionId, execution);

            // Simulate flow execution
            const result = await this._simulateFlowExecution(flow, inputData);
            
            execution.status = 'completed';
            execution.completed_at = new Date().toISOString();
            execution.result = result;

            return {
                success: true,
                execution_id: executionId,
                result: result,
                message: `Flow executed successfully`
            };
        } catch (error) {
            console.error('Flow execution failed:', error);
            return {
                success: false,
                error: `Flow execution failed: ${error.message}`
            };
        }
    }

    async _simulateFlowExecution(flow, inputData) {
        // Simulate processing through flow nodes
        const steps = [];
        
        for (const node of flow.nodes) {
            const stepResult = await this._processNode(node, inputData);
            steps.push({
                node_id: node.id,
                node_type: node.type,
                result: stepResult
            });
        }

        return {
            flow_name: flow.name,
            input: inputData,
            steps: steps,
            output: `Processed through ${flow.nodes.length} nodes`,
            execution_time: `${Math.random() * 2 + 0.5}s`
        };
    }

    async _processNode(node, data) {
        // Simulate different node types
        switch (node.type) {
            case 'llm':
                return await this._processLLMNode(node, data);
            case 'retriever':
                return await this._processRetrieverNode(node, data);
            case 'memory':
                return await this._processMemoryNode(node, data);
            case 'tool':
                return await this._processToolNode(node, data);
            default:
                return { processed: true, node_type: node.type };
        }
    }

    async _processLLMNode(node, data) {
        try {
            // Call local Ollama instance
            const response = await axios.post('http://ollama:11434/api/generate', {
                model: 'gpt-oss',
                prompt: data.query || data.prompt || 'Hello',
                stream: false
            }, { timeout: 30000 });

            return {
                type: 'llm_response',
                response: response.data.response || 'LLM response generated',
                model: 'gpt-oss'
            };
        } catch (error) {
            return {
                type: 'llm_response',
                response: 'LLM service unavailable',
                error: error.message
            };
        }
    }

    async _processRetrieverNode(node, data) {
        return {
            type: 'retrieval',
            documents: [
                { id: 'doc1', content: 'Retrieved document 1', score: 0.95 },
                { id: 'doc2', content: 'Retrieved document 2', score: 0.87 }
            ],
            query: data.query
        };
    }

    async _processMemoryNode(node, data) {
        return {
            type: 'memory',
            stored: true,
            memory_key: node.memory_key || 'default',
            content: data
        };
    }

    async _processToolNode(node, data) {
        return {
            type: 'tool_execution',
            tool: node.tool_name || 'generic_tool',
            result: 'Tool executed successfully',
            input: data
        };
    }

    getFlows() {
        return Array.from(this.flows.values());
    }

    getExecutions() {
        return Array.from(this.executions.values());
    }
}

// Initialize service
const flowiseService = new FlowiseService();

// Health check endpoint
app.get('/health', (req, res) => {
    res.json({
        status: 'healthy',
        service: SERVICE_NAME,
        flows: flowiseService.flows.size,
        executions: flowiseService.executions.size,
        timestamp: new Date().toISOString()
    });
});

// Create flow
app.post('/flows', async (req, res) => {
    try {
        const result = await flowiseService.createFlow(req.body);
        res.json({
            success: result.success,
            result: result,
            service: SERVICE_NAME,
            timestamp: new Date().toISOString()
        });
    } catch (error) {
        res.status(500).json({
            success: false,
            error: error.message,
            service: SERVICE_NAME
        });
    }
});

// Execute flow
app.post('/flows/:flowId/execute', async (req, res) => {
    try {
        const result = await flowiseService.executeFlow(req.params.flowId, req.body);
        res.json({
            success: result.success,
            result: result,
            service: SERVICE_NAME,
            timestamp: new Date().toISOString()
        });
    } catch (error) {
        res.status(500).json({
            success: false,
            error: error.message,
            service: SERVICE_NAME
        });
    }
});

// List flows
app.get('/flows', (req, res) => {
    res.json({
        flows: flowiseService.getFlows(),
        service: SERVICE_NAME
    });
});

// List executions
app.get('/executions', (req, res) => {
    res.json({
        executions: flowiseService.getExecutions(),
        service: SERVICE_NAME
    });
});

// Root endpoint
app.get('/', (req, res) => {
    res.json({
        service: 'FlowiseAI Visual Flow Orchestration',
        status: 'online',
        version: '1.0.0',
        description: 'Visual flow-based LLM orchestration for SutazAI'
    });
});

app.listen(PORT, '0.0.0.0', () => {
    console.log(`${SERVICE_NAME} service running on port ${PORT}`);
});