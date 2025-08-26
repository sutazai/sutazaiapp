#!/usr/bin/env node
const { Server } = require('@modelcontextprotocol/sdk/server/index.js');
const { StdioServerTransport } = require('@modelcontextprotocol/sdk/server/stdio.js');
const { Level } = require('level');
const path = require('path');

// Real MCP Memory Server Implementation
class MemoryServer {
  constructor() {
    this.server = new Server(
      {
        name: 'mcp-memory-server',
        version: '1.0.0',
      },
      {
        capabilities: {
          tools: {
            list: true,
            call: true,
          },
          resources: {
            list: true,
            read: true,
          },
        },
      }
    );

    // Initialize persistent storage
    this.dbPath = process.env.MEMORY_DB_PATH || '/tmp/mcp-memory';
    this.db = new Level(this.dbPath, { valueEncoding: 'json' });
    this.setupHandlers();
  }

  setupHandlers() {
    // List available tools
    this.server.setRequestHandler('tools/list', async () => {
      return {
        tools: [
          {
            name: 'store_memory',
            description: 'Store a memory entry',
            inputSchema: {
              type: 'object',
              properties: {
                key: {
                  type: 'string',
                  description: 'Unique key for the memory',
                },
                value: {
                  type: 'object',
                  description: 'Memory content to store',
                },
                tags: {
                  type: 'array',
                  items: { type: 'string' },
                  description: 'Tags for categorization',
                },
              },
              required: ['key', 'value'],
            },
          },
          {
            name: 'retrieve_memory',
            description: 'Retrieve a memory entry',
            inputSchema: {
              type: 'object',
              properties: {
                key: {
                  type: 'string',
                  description: 'Key of the memory to retrieve',
                },
              },
              required: ['key'],
            },
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
                  description: 'Tags to filter by',
                },
                pattern: {
                  type: 'string',
                  description: 'Pattern to search in memory content',
                },
                limit: {
                  type: 'number',
                  description: 'Maximum number of results',
                  default: 10,
                },
              },
            },
          },
          {
            name: 'delete_memory',
            description: 'Delete a memory entry',
            inputSchema: {
              type: 'object',
              properties: {
                key: {
                  type: 'string',
                  description: 'Key of the memory to delete',
                },
              },
              required: ['key'],
            },
          },
          {
            name: 'list_memories',
            description: 'List all memory keys',
            inputSchema: {
              type: 'object',
              properties: {
                prefix: {
                  type: 'string',
                  description: 'Prefix to filter keys',
                },
                limit: {
                  type: 'number',
                  description: 'Maximum number of results',
                  default: 100,
                },
              },
            },
          },
        ],
      };
    });

    // Handle tool calls
    this.server.setRequestHandler('tools/call', async (request) => {
      const { name, arguments: args } = request.params;

      switch (name) {
        case 'store_memory':
          return await this.storeMemory(args.key, args.value, args.tags);
        case 'retrieve_memory':
          return await this.retrieveMemory(args.key);
        case 'search_memories':
          return await this.searchMemories(args.tags, args.pattern, args.limit);
        case 'delete_memory':
          return await this.deleteMemory(args.key);
        case 'list_memories':
          return await this.listMemories(args.prefix, args.limit);
        default:
          throw new Error(`Unknown tool: ${name}`);
      }
    });

    // List available resources
    this.server.setRequestHandler('resources/list', async () => {
      try {
        const keys = await this.getAllKeys();
        return {
          resources: keys.slice(0, 100).map(key => ({
            uri: `memory://${key}`,
            name: key,
            mimeType: 'application/json',
            description: `Memory entry: ${key}`,
          })),
        };
      } catch (error) {
        console.error('Error listing resources:', error);
        return { resources: [] };
      }
    });

    // Read a specific resource
    this.server.setRequestHandler('resources/read', async (request) => {
      const uri = request.params.uri;
      if (!uri.startsWith('memory://')) {
        throw new Error('Invalid URI scheme');
      }
      
      const key = uri.slice(9);
      try {
        const value = await this.db.get(key);
        return {
          contents: [
            {
              uri: uri,
              mimeType: 'application/json',
              text: JSON.stringify(value, null, 2),
            },
          ],
        };
      } catch (error) {
        throw new Error(`Failed to read memory: ${error.message}`);
      }
    });
  }

  async storeMemory(key, value, tags = []) {
    try {
      const entry = {
        value,
        tags,
        timestamp: new Date().toISOString(),
        version: Date.now(),
      };
      await this.db.put(key, entry);
      return {
        content: [
          {
            type: 'text',
            text: `Memory stored successfully: ${key}`,
          },
        ],
      };
    } catch (error) {
      throw new Error(`Failed to store memory: ${error.message}`);
    }
  }

  async retrieveMemory(key) {
    try {
      const entry = await this.db.get(key);
      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify(entry, null, 2),
          },
        ],
      };
    } catch (error) {
      if (error.code === 'LEVEL_NOT_FOUND') {
        return {
          content: [
            {
              type: 'text',
              text: `Memory not found: ${key}`,
            },
          ],
        };
      }
      throw new Error(`Failed to retrieve memory: ${error.message}`);
    }
  }

  async searchMemories(tags, pattern, limit = 10) {
    try {
      const results = [];
      const stream = this.db.iterator();
      
      for await (const [key, value] of stream) {
        // Check tags
        if (tags && tags.length > 0) {
          const entryTags = value.tags || [];
          if (!tags.some(tag => entryTags.includes(tag))) {
            continue;
          }
        }
        
        // Check pattern
        if (pattern) {
          const content = JSON.stringify(value.value);
          if (!content.includes(pattern)) {
            continue;
          }
        }
        
        results.push({ key, ...value });
        
        if (results.length >= limit) {
          break;
        }
      }
      
      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify(results, null, 2),
          },
        ],
      };
    } catch (error) {
      throw new Error(`Failed to search memories: ${error.message}`);
    }
  }

  async deleteMemory(key) {
    try {
      await this.db.del(key);
      return {
        content: [
          {
            type: 'text',
            text: `Memory deleted successfully: ${key}`,
          },
        ],
      };
    } catch (error) {
      throw new Error(`Failed to delete memory: ${error.message}`);
    }
  }

  async listMemories(prefix, limit = 100) {
    try {
      const keys = [];
      const stream = this.db.keys();
      
      for await (const key of stream) {
        if (!prefix || key.startsWith(prefix)) {
          keys.push(key);
          if (keys.length >= limit) {
            break;
          }
        }
      }
      
      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify(keys, null, 2),
          },
        ],
      };
    } catch (error) {
      throw new Error(`Failed to list memories: ${error.message}`);
    }
  }

  async getAllKeys() {
    const keys = [];
    for await (const key of this.db.keys()) {
      keys.push(key);
    }
    return keys;
  }

  async run() {
    const transport = new StdioServerTransport();
    await this.server.connect(transport);
    console.error('MCP Memory Server started successfully');
  }
}

// Start the server
const server = new MemoryServer();
server.run().catch(console.error);