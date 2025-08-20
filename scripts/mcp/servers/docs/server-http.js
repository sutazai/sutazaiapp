#!/usr/bin/env node
const express = require('express');
const fs = require('fs').promises;
const path = require('path');

const app = express();
app.use(express.json());

const PORT = process.env.PORT || 3017;
const DOCS_PATH = process.env.DOCS_PATH || '/data/docs';

// Documentation storage
let docsStore = {
  documents: {},
  categories: {},
  metadata: {
    totalDocs: 0,
    lastUpdated: null
  }
};

// Load docs from disk
async function loadDocs() {
  try {
    await fs.mkdir(DOCS_PATH, { recursive: true });
    const dbFile = path.join(DOCS_PATH, 'docs.json');
    try {
      const data = await fs.readFile(dbFile, 'utf-8');
      docsStore = JSON.parse(data);
      console.log(`Loaded ${docsStore.metadata.totalDocs} documents`);
    } catch (err) {
      console.log('No existing docs database, starting fresh');
    }
  } catch (error) {
    console.error('Error loading docs:', error);
  }
}

// Save docs to disk
async function saveDocs() {
  try {
    const dbFile = path.join(DOCS_PATH, 'docs.json');
    await fs.writeFile(dbFile, JSON.stringify(docsStore, null, 2));
  } catch (error) {
    console.error('Error saving docs:', error);
  }
}

// MCP Protocol endpoints
app.get('/health', (req, res) => {
  res.json({ 
    status: 'healthy', 
    server: 'mcp-docs-server', 
    version: '1.0.0',
    documents: docsStore.metadata.totalDocs
  });
});

// List available tools
app.get('/tools', (req, res) => {
  res.json({
    tools: [
      {
        name: 'store_doc',
        description: 'Store a documentation entry',
        inputSchema: {
          type: 'object',
          properties: {
            id: { type: 'string', description: 'Document ID' },
            title: { type: 'string', description: 'Document title' },
            content: { type: 'string', description: 'Document content' },
            category: { type: 'string', description: 'Document category' },
            tags: { 
              type: 'array', 
              items: { type: 'string' },
              description: 'Document tags' 
            },
            metadata: { type: 'object', description: 'Additional metadata' }
          },
          required: ['id', 'title', 'content']
        }
      },
      {
        name: 'get_doc',
        description: 'Retrieve a documentation entry',
        inputSchema: {
          type: 'object',
          properties: {
            id: { type: 'string', description: 'Document ID' }
          },
          required: ['id']
        }
      },
      {
        name: 'list_docs',
        description: 'List documentation entries',
        inputSchema: {
          type: 'object',
          properties: {
            category: { type: 'string', description: 'Filter by category' },
            tags: { 
              type: 'array', 
              items: { type: 'string' },
              description: 'Filter by tags' 
            },
            limit: { type: 'number', description: 'Maximum results', default: 20 }
          }
        }
      },
      {
        name: 'search_docs',
        description: 'Search documentation',
        inputSchema: {
          type: 'object',
          properties: {
            query: { type: 'string', description: 'Search query' },
            category: { type: 'string', description: 'Filter by category' },
            limit: { type: 'number', description: 'Maximum results', default: 10 }
          },
          required: ['query']
        }
      },
      {
        name: 'list_categories',
        description: 'List all documentation categories',
        inputSchema: {
          type: 'object',
          properties: {}
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
      case 'store_doc':
        docsStore.documents[args.id] = {
          title: args.title,
          content: args.content,
          category: args.category || 'general',
          tags: args.tags || [],
          metadata: args.metadata || {},
          created: new Date().toISOString(),
          updated: new Date().toISOString()
        };
        
        // Update category index
        const category = args.category || 'general';
        if (!docsStore.categories[category]) {
          docsStore.categories[category] = [];
        }
        if (!docsStore.categories[category].includes(args.id)) {
          docsStore.categories[category].push(args.id);
        }
        
        // Update metadata
        docsStore.metadata.totalDocs = Object.keys(docsStore.documents).length;
        docsStore.metadata.lastUpdated = new Date().toISOString();
        
        await saveDocs();
        result = { success: true, message: `Document stored: ${args.id}` };
        break;
      
      case 'get_doc':
        const doc = docsStore.documents[args.id];
        if (!doc) {
          return res.status(404).json({ error: `Document not found: ${args.id}` });
        }
        result = { id: args.id, ...doc };
        break;
      
      case 'list_docs':
        let docs = Object.entries(docsStore.documents);
        
        // Filter by category
        if (args.category) {
          docs = docs.filter(([_, doc]) => doc.category === args.category);
        }
        
        // Filter by tags
        if (args.tags && args.tags.length > 0) {
          docs = docs.filter(([_, doc]) => 
            args.tags.some(tag => doc.tags.includes(tag))
          );
        }
        
        // Limit results
        docs = docs.slice(0, args.limit || 20);
        
        result = {
          documents: docs.map(([id, doc]) => ({ id, ...doc })),
          total: docs.length
        };
        break;
      
      case 'search_docs':
        const query = args.query.toLowerCase();
        let searchResults = [];
        
        for (const [id, doc] of Object.entries(docsStore.documents)) {
          // Filter by category if specified
          if (args.category && doc.category !== args.category) {
            continue;
          }
          
          // Search in title and content
          const titleMatch = doc.title.toLowerCase().includes(query);
          const contentMatch = doc.content.toLowerCase().includes(query);
          
          if (titleMatch || contentMatch) {
            searchResults.push({
              id,
              ...doc,
              relevance: titleMatch ? 2 : 1
            });
          }
        }
        
        // Sort by relevance and limit
        searchResults.sort((a, b) => b.relevance - a.relevance);
        searchResults = searchResults.slice(0, args.limit || 10);
        
        result = { results: searchResults, total: searchResults.length };
        break;
      
      case 'list_categories':
        const categories = Object.keys(docsStore.categories).map(cat => ({
          name: cat,
          count: docsStore.categories[cat].length
        }));
        result = { categories };
        break;
      
      default:
        return res.status(404).json({ error: `Unknown tool: ${toolName}` });
    }
    
    res.json(result);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Initialize and start server
loadDocs().then(() => {
  app.listen(PORT, '0.0.0.0', () => {
    console.log(`MCP Docs Server listening on port ${PORT}`);
    console.log(`Docs path: ${DOCS_PATH}`);
    console.log(`Total documents: ${docsStore.metadata.totalDocs}`);
  });
});