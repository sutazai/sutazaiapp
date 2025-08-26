#!/usr/bin/env node
const express = require('express');
const fs = require('fs').promises;
const path = require('path');

const app = express();
app.use(express.json());

const PORT = process.env.PORT || 3009;
const MEMORY_DB_PATH = process.env.MEMORY_DB_PATH || '/data/memory';

// In-memory storage with persistence
let memoryStore = {};

// Load memory from disk on startup
async function loadMemory() {
  try {
    await fs.mkdir(MEMORY_DB_PATH, { recursive: true });
    const dbFile = path.join(MEMORY_DB_PATH, 'memory.json');
    try {
      const data = await fs.readFile(dbFile, 'utf-8');
      memoryStore = JSON.parse(data);
      console.log(`Loaded ${Object.keys(memoryStore).length} memory entries`);
    } catch (err) {
      console.log('No existing memory database, starting fresh');
    }
  } catch (error) {
    console.error('Error loading memory:', error);
  }
}

// Save memory to disk
async function saveMemory() {
  try {
    const dbFile = path.join(MEMORY_DB_PATH, 'memory.json');
    await fs.writeFile(dbFile, JSON.stringify(memoryStore, null, 2));
  } catch (error) {
    console.error('Error saving memory:', error);
  }
}

// MCP Protocol endpoints
app.get('/health', (req, res) => {
  res.json({ 
    status: 'healthy', 
    server: 'mcp-memory-server', 
    version: '1.0.0',
    entries: Object.keys(memoryStore).length 
  });
});

// List available tools
app.get('/tools', (req, res) => {
  res.json({
    tools: [
      {
        name: 'store_memory',
        description: 'Store a memory entry',
        inputSchema: {
          type: 'object',
          properties: {
            key: { type: 'string', description: 'Unique key for the memory' },
            value: { type: 'object', description: 'Memory content to store' },
            tags: { 
              type: 'array', 
              items: { type: 'string' },
              description: 'Tags for categorization' 
            }
          },
          required: ['key', 'value']
        }
      },
      {
        name: 'retrieve_memory',
        description: 'Retrieve a memory entry',
        inputSchema: {
          type: 'object',
          properties: {
            key: { type: 'string', description: 'Key of the memory to retrieve' }
          },
          required: ['key']
        }
      },
      {
        name: 'search_memories',
        description: 'Search memories by tags or pattern',
        inputSchema: {
          type: 'object',
          properties: {
            tags: { 
              type: 'array', 
              items: { type: 'string' },
              description: 'Tags to filter by' 
            },
            pattern: { type: 'string', description: 'Pattern to search in memory content' },
            limit: { type: 'number', description: 'Maximum number of results', default: 10 }
          }
        }
      },
      {
        name: 'delete_memory',
        description: 'Delete a memory entry',
        inputSchema: {
          type: 'object',
          properties: {
            key: { type: 'string', description: 'Key of the memory to delete' }
          },
          required: ['key']
        }
      },
      {
        name: 'list_memories',
        description: 'List all memory keys',
        inputSchema: {
          type: 'object',
          properties: {
            prefix: { type: 'string', description: 'Prefix to filter keys' },
            limit: { type: 'number', description: 'Maximum number of results', default: 100 }
          }
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
      case 'store_memory':
        memoryStore[args.key] = {
          value: args.value,
          tags: args.tags || [],
          timestamp: new Date().toISOString(),
          version: Date.now()
        };
        await saveMemory();
        result = { success: true, message: `Memory stored: ${args.key}` };
        break;
      
      case 'retrieve_memory':
        const entry = memoryStore[args.key];
        if (!entry) {
          return res.status(404).json({ error: `Memory not found: ${args.key}` });
        }
        result = entry;
        break;
      
      case 'search_memories':
        const results = [];
        const limit = args.limit || 10;
        
        for (const [key, value] of Object.entries(memoryStore)) {
          // Check tags
          if (args.tags && args.tags.length > 0) {
            const entryTags = value.tags || [];
            if (!args.tags.some(tag => entryTags.includes(tag))) {
              continue;
            }
          }
          
          // Check pattern
          if (args.pattern) {
            const content = JSON.stringify(value.value);
            if (!content.includes(args.pattern)) {
              continue;
            }
          }
          
          results.push({ key, ...value });
          if (results.length >= limit) break;
        }
        
        result = { results };
        break;
      
      case 'delete_memory':
        if (!memoryStore[args.key]) {
          return res.status(404).json({ error: `Memory not found: ${args.key}` });
        }
        delete memoryStore[args.key];
        await saveMemory();
        result = { success: true, message: `Memory deleted: ${args.key}` };
        break;
      
      case 'list_memories':
        const keys = Object.keys(memoryStore);
        const filteredKeys = args.prefix 
          ? keys.filter(k => k.startsWith(args.prefix))
          : keys;
        result = { 
          keys: filteredKeys.slice(0, args.limit || 100),
          total: filteredKeys.length 
        };
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
  const keys = Object.keys(memoryStore);
  res.json({
    resources: keys.slice(0, 100).map(key => ({
      uri: `memory://${key}`,
      name: key,
      type: 'memory'
    }))
  });
});

// Read resource
app.get('/resources/read', (req, res) => {
  const { uri } = req.query;
  
  if (!uri || !uri.startsWith('memory://')) {
    return res.status(400).json({ error: 'Invalid URI' });
  }
  
  const key = uri.slice(9);
  const entry = memoryStore[key];
  
  if (!entry) {
    return res.status(404).json({ error: `Memory not found: ${key}` });
  }
  
  res.json({ content: entry, uri });
});

// Initialize and start server
loadMemory().then(() => {
  app.listen(PORT, '0.0.0.0', () => {
    console.log(`MCP Memory Server listening on port ${PORT}`);
    console.log(`Memory database path: ${MEMORY_DB_PATH}`);
    console.log(`Loaded ${Object.keys(memoryStore).length} memory entries`);
  });
});