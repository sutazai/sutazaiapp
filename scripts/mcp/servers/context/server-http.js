#!/usr/bin/env node
const express = require('express');
const fs = require('fs').promises;
const path = require('path');

const app = express();
app.use(express.json());

const PORT = process.env.PORT || 3004;
const CONTEXT_PATH = process.env.CONTEXT_PATH || '/data/context';

// Context storage
let contextStore = {
  contexts: {},
  embeddings: {},
  relationships: {}
};

// Load context from disk on startup
async function loadContext() {
  try {
    await fs.mkdir(CONTEXT_PATH, { recursive: true });
    const dbFile = path.join(CONTEXT_PATH, 'context.json');
    try {
      const data = await fs.readFile(dbFile, 'utf-8');
      contextStore = JSON.parse(data);
      console.log(`Loaded ${Object.keys(contextStore.contexts).length} context entries`);
    } catch (err) {
      console.log('No existing context database, starting fresh');
    }
  } catch (error) {
    console.error('Error loading context:', error);
  }
}

// Save context to disk
async function saveContext() {
  try {
    const dbFile = path.join(CONTEXT_PATH, 'context.json');
    await fs.writeFile(dbFile, JSON.stringify(contextStore, null, 2));
  } catch (error) {
    console.error('Error saving context:', error);
  }
}

// MCP Protocol endpoints
app.get('/health', (req, res) => {
  res.json({ 
    status: 'healthy', 
    server: 'mcp-context-server', 
    version: '1.0.0',
    contexts: Object.keys(contextStore.contexts).length 
  });
});

// List available tools
app.get('/tools', (req, res) => {
  res.json({
    tools: [
      {
        name: 'store_context',
        description: 'Store a context entry with semantic information',
        inputSchema: {
          type: 'object',
          properties: {
            id: { type: 'string', description: 'Unique identifier for the context' },
            content: { type: 'string', description: 'Context content' },
            metadata: { type: 'object', description: 'Additional metadata' },
            tags: { 
              type: 'array', 
              items: { type: 'string' },
              description: 'Tags for categorization' 
            }
          },
          required: ['id', 'content']
        }
      },
      {
        name: 'retrieve_context',
        description: 'Retrieve a context entry by ID',
        inputSchema: {
          type: 'object',
          properties: {
            id: { type: 'string', description: 'Context ID to retrieve' }
          },
          required: ['id']
        }
      },
      {
        name: 'search_context',
        description: 'Search contexts by content or tags',
        inputSchema: {
          type: 'object',
          properties: {
            query: { type: 'string', description: 'Search query' },
            tags: { 
              type: 'array', 
              items: { type: 'string' },
              description: 'Tags to filter by' 
            },
            limit: { type: 'number', description: 'Maximum results', default: 10 }
          }
        }
      },
      {
        name: 'link_contexts',
        description: 'Create a relationship between contexts',
        inputSchema: {
          type: 'object',
          properties: {
            source_id: { type: 'string', description: 'Source context ID' },
            target_id: { type: 'string', description: 'Target context ID' },
            relationship: { type: 'string', description: 'Relationship type' }
          },
          required: ['source_id', 'target_id', 'relationship']
        }
      },
      {
        name: 'get_related_contexts',
        description: 'Get contexts related to a given context',
        inputSchema: {
          type: 'object',
          properties: {
            id: { type: 'string', description: 'Context ID' },
            relationship: { type: 'string', description: 'Filter by relationship type' }
          },
          required: ['id']
        }
      }
    ]
  });
});

// Execute tool
app.post('/tools/:toolName', async (req, res) => {
  const { toolName } = req.params;
  const args = req.body;

  try {
    let result;
    switch (toolName) {
      case 'store_context':
        contextStore.contexts[args.id] = {
          content: args.content,
          metadata: args.metadata || {},
          tags: args.tags || [],
          timestamp: new Date().toISOString()
        };
        await saveContext();
        result = { success: true, message: `Context stored: ${args.id}` };
        break;
      
      case 'retrieve_context':
        const context = contextStore.contexts[args.id];
        if (!context) {
          return res.status(404).json({ error: `Context not found: ${args.id}` });
        }
        result = context;
        break;
      
      case 'search_context':
        const results = [];
        const limit = args.limit || 10;
        
        for (const [id, context] of Object.entries(contextStore.contexts)) {
          // Check tags
          if (args.tags && args.tags.length > 0) {
            const contextTags = context.tags || [];
            if (!args.tags.some(tag => contextTags.includes(tag))) {
              continue;
            }
          }
          
          // Check query
          if (args.query) {
            const content = context.content.toLowerCase();
            if (!content.includes(args.query.toLowerCase())) {
              continue;
            }
          }
          
          results.push({ id, ...context });
          if (results.length >= limit) break;
        }
        
        result = { results };
        break;
      
      case 'link_contexts':
        if (!contextStore.relationships[args.source_id]) {
          contextStore.relationships[args.source_id] = [];
        }
        contextStore.relationships[args.source_id].push({
          target: args.target_id,
          relationship: args.relationship,
          created: new Date().toISOString()
        });
        await saveContext();
        result = { success: true, message: 'Contexts linked successfully' };
        break;
      
      case 'get_related_contexts':
        const relations = contextStore.relationships[args.id] || [];
        const filtered = args.relationship 
          ? relations.filter(r => r.relationship === args.relationship)
          : relations;
        result = { relations: filtered };
        break;
      
      default:
        return res.status(404).json({ error: `Unknown tool: ${toolName}` });
    }
    
    res.json(result);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// List resources
app.get('/resources', (req, res) => {
  const ids = Object.keys(contextStore.contexts);
  res.json({
    resources: ids.slice(0, 100).map(id => ({
      uri: `context://${id}`,
      name: id,
      type: 'context'
    }))
  });
});

// Initialize and start server
loadContext().then(() => {
  app.listen(PORT, '0.0.0.0', () => {
    console.log(`MCP Context Server listening on port ${PORT}`);
    console.log(`Context database path: ${CONTEXT_PATH}`);
    console.log(`Loaded ${Object.keys(contextStore.contexts).length} context entries`);
  });
});