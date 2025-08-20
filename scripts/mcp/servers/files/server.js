#!/usr/bin/env node
const { Server } = require('@modelcontextprotocol/sdk/server/index.js');
const { StdioServerTransport } = require('@modelcontextprotocol/sdk/server/stdio.js');
const fs = require('fs').promises;
const path = require('path');
const glob = require('glob');
const chokidar = require('chokidar');

// Real MCP File System Server Implementation
class FileSystemServer {
  constructor() {
    this.server = new Server(
      {
        name: 'mcp-files-server',
        version: '1.0.0',
      },
      {
        capabilities: {
          resources: {
            list: true,
            read: true,
            subscribe: true,
          },
          tools: {
            list: true,
            call: true,
          },
        },
      }
    );

    this.projectDir = process.env.PROJECT_DIR || '/opt/sutazaiapp';
    this.setupHandlers();
  }

  setupHandlers() {
    // List available resources
    this.server.setRequestHandler('resources/list', async () => {
      try {
        const files = await this.listFiles(this.projectDir);
        return {
          resources: files.map(file => ({
            uri: `file://${file}`,
            name: path.basename(file),
            mimeType: this.getMimeType(file),
            description: `File: ${file}`,
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
      if (!uri.startsWith('file://')) {
        throw new Error('Invalid URI scheme');
      }
      
      const filePath = uri.slice(7);
      try {
        const content = await fs.readFile(filePath, 'utf-8');
        return {
          contents: [
            {
              uri: uri,
              mimeType: this.getMimeType(filePath),
              text: content,
            },
          ],
        };
      } catch (error) {
        throw new Error(`Failed to read file: ${error.message}`);
      }
    });

    // List available tools
    this.server.setRequestHandler('tools/list', async () => {
      return {
        tools: [
          {
            name: 'read_file',
            description: 'Read the contents of a file',
            inputSchema: {
              type: 'object',
              properties: {
                path: {
                  type: 'string',
                  description: 'Path to the file to read',
                },
              },
              required: ['path'],
            },
          },
          {
            name: 'write_file',
            description: 'Write content to a file',
            inputSchema: {
              type: 'object',
              properties: {
                path: {
                  type: 'string',
                  description: 'Path to the file to write',
                },
                content: {
                  type: 'string',
                  description: 'Content to write to the file',
                },
              },
              required: ['path', 'content'],
            },
          },
          {
            name: 'list_directory',
            description: 'List contents of a directory',
            inputSchema: {
              type: 'object',
              properties: {
                path: {
                  type: 'string',
                  description: 'Path to the directory',
                },
                pattern: {
                  type: 'string',
                  description: 'Glob pattern to filter files',
                },
              },
              required: ['path'],
            },
          },
          {
            name: 'create_directory',
            description: 'Create a new directory',
            inputSchema: {
              type: 'object',
              properties: {
                path: {
                  type: 'string',
                  description: 'Path to the directory to create',
                },
              },
              required: ['path'],
            },
          },
          {
            name: 'delete_file',
            description: 'Delete a file',
            inputSchema: {
              type: 'object',
              properties: {
                path: {
                  type: 'string',
                  description: 'Path to the file to delete',
                },
              },
              required: ['path'],
            },
          },
        ],
      };
    });

    // Handle tool calls
    this.server.setRequestHandler('tools/call', async (request) => {
      const { name, arguments: args } = request.params;

      switch (name) {
        case 'read_file':
          return await this.readFile(args.path);
        case 'write_file':
          return await this.writeFile(args.path, args.content);
        case 'list_directory':
          return await this.listDirectory(args.path, args.pattern);
        case 'create_directory':
          return await this.createDirectory(args.path);
        case 'delete_file':
          return await this.deleteFile(args.path);
        default:
          throw new Error(`Unknown tool: ${name}`);
      }
    });

    // Subscribe to resource changes
    this.server.setRequestHandler('resources/subscribe', async (request) => {
      const uri = request.params.uri;
      if (!uri.startsWith('file://')) {
        throw new Error('Invalid URI scheme');
      }
      
      const filePath = uri.slice(7);
      const watcher = chokidar.watch(filePath, {
        persistent: true,
        ignoreInitial: true,
      });

      watcher.on('change', () => {
        this.server.sendResourceUpdated({ uri });
      });

      return { subscribed: true };
    });
  }

  async listFiles(dir, pattern = '**/*') {
    return new Promise((resolve, reject) => {
      glob(pattern, { cwd: dir, nodir: true }, (err, files) => {
        if (err) reject(err);
        else resolve(files.map(f => path.join(dir, f)));
      });
    });
  }

  async readFile(filePath) {
    try {
      const content = await fs.readFile(filePath, 'utf-8');
      return {
        content: [
          {
            type: 'text',
            text: content,
          },
        ],
      };
    } catch (error) {
      throw new Error(`Failed to read file: ${error.message}`);
    }
  }

  async writeFile(filePath, content) {
    try {
      await fs.writeFile(filePath, content, 'utf-8');
      return {
        content: [
          {
            type: 'text',
            text: `File written successfully: ${filePath}`,
          },
        ],
      };
    } catch (error) {
      throw new Error(`Failed to write file: ${error.message}`);
    }
  }

  async listDirectory(dirPath, pattern) {
    try {
      const files = pattern 
        ? await this.listFiles(dirPath, pattern)
        : await fs.readdir(dirPath);
      
      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify(files, null, 2),
          },
        ],
      };
    } catch (error) {
      throw new Error(`Failed to list directory: ${error.message}`);
    }
  }

  async createDirectory(dirPath) {
    try {
      await fs.mkdir(dirPath, { recursive: true });
      return {
        content: [
          {
            type: 'text',
            text: `Directory created: ${dirPath}`,
          },
        ],
      };
    } catch (error) {
      throw new Error(`Failed to create directory: ${error.message}`);
    }
  }

  async deleteFile(filePath) {
    try {
      await fs.unlink(filePath);
      return {
        content: [
          {
            type: 'text',
            text: `File deleted: ${filePath}`,
          },
        ],
      };
    } catch (error) {
      throw new Error(`Failed to delete file: ${error.message}`);
    }
  }

  getMimeType(filePath) {
    const ext = path.extname(filePath).toLowerCase();
    const mimeTypes = {
      '.txt': 'text/plain',
      '.md': 'text/markdown',
      '.js': 'application/javascript',
      '.json': 'application/json',
      '.py': 'text/x-python',
      '.html': 'text/html',
      '.css': 'text/css',
      '.yaml': 'text/yaml',
      '.yml': 'text/yaml',
    };
    return mimeTypes[ext] || 'application/octet-stream';
  }

  async run() {
    const transport = new StdioServerTransport();
    await this.server.connect(transport);
    console.error('MCP Files Server started successfully');
  }
}

// Start the server
const server = new FileSystemServer();
server.run().catch(console.error);