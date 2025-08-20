#!/usr/bin/env node
const express = require('express');
const fs = require('fs').promises;
const path = require('path');
const glob = require('glob');

const app = express();
app.use(express.json());

const PORT = process.env.PORT || 3003;
const PROJECT_DIR = process.env.PROJECT_DIR || '/workspace';

// MCP Protocol endpoints
app.get('/health', (req, res) => {
  res.json({ status: 'healthy', server: 'mcp-files-server', version: '1.0.0' });
});

// List available tools
app.get('/tools', (req, res) => {
  res.json({
    tools: [
      {
        name: 'read_file',
        description: 'Read the contents of a file',
        inputSchema: {
          type: 'object',
          properties: {
            path: { type: 'string', description: 'Path to the file to read' }
          },
          required: ['path']
        }
      },
      {
        name: 'write_file',
        description: 'Write content to a file',
        inputSchema: {
          type: 'object',
          properties: {
            path: { type: 'string', description: 'Path to the file to write' },
            content: { type: 'string', description: 'Content to write' }
          },
          required: ['path', 'content']
        }
      },
      {
        name: 'list_directory',
        description: 'List contents of a directory',
        inputSchema: {
          type: 'object',
          properties: {
            path: { type: 'string', description: 'Path to the directory' },
            pattern: { type: 'string', description: 'Glob pattern to filter files' }
          },
          required: ['path']
        }
      },
      {
        name: 'create_directory',
        description: 'Create a new directory',
        inputSchema: {
          type: 'object',
          properties: {
            path: { type: 'string', description: 'Path to the directory to create' }
          },
          required: ['path']
        }
      },
      {
        name: 'delete_file',
        description: 'Delete a file',
        inputSchema: {
          type: 'object',
          properties: {
            path: { type: 'string', description: 'Path to the file to delete' }
          },
          required: ['path']
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
      case 'read_file':
        const content = await fs.readFile(args.path, 'utf-8');
        result = { content };
        break;
      
      case 'write_file':
        await fs.writeFile(args.path, args.content, 'utf-8');
        result = { success: true, message: `File written: ${args.path}` };
        break;
      
      case 'list_directory':
        const files = await fs.readdir(args.path);
        result = { files };
        break;
      
      case 'create_directory':
        await fs.mkdir(args.path, { recursive: true });
        result = { success: true, message: `Directory created: ${args.path}` };
        break;
      
      case 'delete_file':
        await fs.unlink(args.path);
        result = { success: true, message: `File deleted: ${args.path}` };
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
app.get('/resources', async (req, res) => {
  try {
    const files = await new Promise((resolve, reject) => {
      glob('**/*', { cwd: PROJECT_DIR, nodir: true, ignore: 'node_modules/**' }, (err, files) => {
        if (err) reject(err);
        else resolve(files);
      });
    });
    
    res.json({
      resources: files.slice(0, 100).map(file => ({
        uri: `file://${path.join(PROJECT_DIR, file)}`,
        name: path.basename(file),
        path: file
      }))
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Read resource
app.get('/resources/read', async (req, res) => {
  const { uri } = req.query;
  
  if (!uri || !uri.startsWith('file://')) {
    return res.status(400).json({ error: 'Invalid URI' });
  }
  
  try {
    const filePath = uri.slice(7);
    const content = await fs.readFile(filePath, 'utf-8');
    res.json({ content, uri });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.listen(PORT, '0.0.0.0', () => {
  console.log(`MCP Files Server listening on port ${PORT}`);
  console.log(`Project directory: ${PROJECT_DIR}`);
});