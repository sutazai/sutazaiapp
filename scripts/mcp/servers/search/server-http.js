#!/usr/bin/env node
const express = require('express');
const fs = require('fs').promises;
const path = require('path');

const app = express();
app.use(express.json());

const PORT = process.env.PORT || 3006;
const INDEX_PATH = process.env.INDEX_PATH || '/data/search';

// Search index storage
let searchIndex = {
  documents: {},
  index: {},
  stats: {
    totalDocuments: 0,
    totalTerms: 0,
    lastUpdated: null
  }
};

// Load index from disk
async function loadIndex() {
  try {
    await fs.mkdir(INDEX_PATH, { recursive: true });
    const indexFile = path.join(INDEX_PATH, 'index.json');
    try {
      const data = await fs.readFile(indexFile, 'utf-8');
      searchIndex = JSON.parse(data);
      console.log(`Loaded index with ${searchIndex.stats.totalDocuments} documents`);
    } catch (err) {
      console.log('No existing search index, starting fresh');
    }
  } catch (error) {
    console.error('Error loading index:', error);
  }
}

// Save index to disk
async function saveIndex() {
  try {
    const indexFile = path.join(INDEX_PATH, 'index.json');
    await fs.writeFile(indexFile, JSON.stringify(searchIndex, null, 2));
  } catch (error) {
    console.error('Error saving index:', error);
  }
}

// Tokenize text for indexing
function tokenize(text) {
  return text.toLowerCase()
    .replace(/[^\w\s]/g, ' ')
    .split(/\s+/)
    .filter(token => token.length > 2);
}

// MCP Protocol endpoints
app.get('/health', (req, res) => {
  res.json({ 
    status: 'healthy', 
    server: 'mcp-search-server', 
    version: '1.0.0',
    documents: searchIndex.stats.totalDocuments,
    terms: searchIndex.stats.totalTerms
  });
});

// List available tools
app.get('/tools', (req, res) => {
  res.json({
    tools: [
      {
        name: 'index_document',
        description: 'Index a document for searching',
        inputSchema: {
          type: 'object',
          properties: {
            id: { type: 'string', description: 'Document ID' },
            title: { type: 'string', description: 'Document title' },
            content: { type: 'string', description: 'Document content' },
            metadata: { type: 'object', description: 'Additional metadata' }
          },
          required: ['id', 'content']
        }
      },
      {
        name: 'search',
        description: 'Search indexed documents',
        inputSchema: {
          type: 'object',
          properties: {
            query: { type: 'string', description: 'Search query' },
            limit: { type: 'number', description: 'Maximum results', default: 10 }
          },
          required: ['query']
        }
      },
      {
        name: 'get_document',
        description: 'Get a document by ID',
        inputSchema: {
          type: 'object',
          properties: {
            id: { type: 'string', description: 'Document ID' }
          },
          required: ['id']
        }
      },
      {
        name: 'delete_document',
        description: 'Delete a document from the index',
        inputSchema: {
          type: 'object',
          properties: {
            id: { type: 'string', description: 'Document ID' }
          },
          required: ['id']
        }
      },
      {
        name: 'get_stats',
        description: 'Get search index statistics',
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
      case 'index_document':
        // Store document
        searchIndex.documents[args.id] = {
          title: args.title || '',
          content: args.content,
          metadata: args.metadata || {},
          indexed: new Date().toISOString()
        };
        
        // Index tokens
        const tokens = tokenize(args.content + ' ' + (args.title || ''));
        tokens.forEach(token => {
          if (!searchIndex.index[token]) {
            searchIndex.index[token] = [];
          }
          if (!searchIndex.index[token].includes(args.id)) {
            searchIndex.index[token].push(args.id);
          }
        });
        
        // Update stats
        searchIndex.stats.totalDocuments = Object.keys(searchIndex.documents).length;
        searchIndex.stats.totalTerms = Object.keys(searchIndex.index).length;
        searchIndex.stats.lastUpdated = new Date().toISOString();
        
        await saveIndex();
        result = { success: true, message: `Document indexed: ${args.id}` };
        break;
      
      case 'search':
        const queryTokens = tokenize(args.query);
        const scores = {};
        
        // Calculate document scores
        queryTokens.forEach(token => {
          const docIds = searchIndex.index[token] || [];
          docIds.forEach(docId => {
            scores[docId] = (scores[docId] || 0) + 1;
          });
        });
        
        // Sort by score and get top results
        const sortedDocs = Object.entries(scores)
          .sort((a, b) => b[1] - a[1])
          .slice(0, args.limit || 10)
          .map(([docId, score]) => ({
            id: docId,
            score,
            ...searchIndex.documents[docId]
          }));
        
        result = { results: sortedDocs, total: sortedDocs.length };
        break;
      
      case 'get_document':
        const doc = searchIndex.documents[args.id];
        if (!doc) {
          return res.status(404).json({ error: `Document not found: ${args.id}` });
        }
        result = { id: args.id, ...doc };
        break;
      
      case 'delete_document':
        if (!searchIndex.documents[args.id]) {
          return res.status(404).json({ error: `Document not found: ${args.id}` });
        }
        
        // Remove from documents
        delete searchIndex.documents[args.id];
        
        // Remove from index
        for (const token in searchIndex.index) {
          searchIndex.index[token] = searchIndex.index[token].filter(id => id !== args.id);
          if (searchIndex.index[token].length === 0) {
            delete searchIndex.index[token];
          }
        }
        
        // Update stats
        searchIndex.stats.totalDocuments = Object.keys(searchIndex.documents).length;
        searchIndex.stats.totalTerms = Object.keys(searchIndex.index).length;
        searchIndex.stats.lastUpdated = new Date().toISOString();
        
        await saveIndex();
        result = { success: true, message: `Document deleted: ${args.id}` };
        break;
      
      case 'get_stats':
        result = searchIndex.stats;
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
loadIndex().then(() => {
  app.listen(PORT, '0.0.0.0', () => {
    console.log(`MCP Search Server listening on port ${PORT}`);
    console.log(`Index path: ${INDEX_PATH}`);
    console.log(`Index stats:`, searchIndex.stats);
  });
});