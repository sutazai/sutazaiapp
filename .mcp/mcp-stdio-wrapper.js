#!/usr/bin/env node
/**
 * MCP STDIO Wrapper for Claude Integration
 * Provides proper STDIO communication for MCP servers
 */

const { spawn } = require('child_process');
const readline = require('readline');

const serverMap = {
  'filesystem': ['npx', '-y', '@modelcontextprotocol/server-filesystem', '/opt/sutazaiapp'],
  'github': ['npx', '-y', '@modelcontextprotocol/server-github'],
  'postgres': ['npx', '-y', '@modelcontextprotocol/server-postgres', process.env.DATABASE_URL],
  'sutazai': ['node', '/opt/sutazaiapp/docker/mcp-services/real-mcp-server/dist/server.js']
};

const serverName = process.argv[2] || 'sutazai';
const serverCmd = serverMap[serverName];

if (!serverCmd) {
  console.error(`Unknown server: ${serverName}`);
  process.exit(1);
}

const server = spawn(serverCmd[0], serverCmd.slice(1), {
  stdio: ['pipe', 'pipe', 'pipe']
});

// Forward stdin to server
process.stdin.pipe(server.stdin);

// Forward server stdout to our stdout
server.stdout.pipe(process.stdout);

// Forward server stderr to our stderr
server.stderr.pipe(process.stderr);

// Handle server exit
server.on('exit', (code) => {
  process.exit(code);
});

// Handle errors
server.on('error', (err) => {
  console.error('Server error:', err);
  process.exit(1);
});
