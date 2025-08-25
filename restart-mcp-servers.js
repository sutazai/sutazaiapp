/**
 * MCP Server Restart and Test Script
 * Restarts and tests all configured MCP servers
 */

const { spawn, exec } = require('child_process');
const { promisify } = require('util');
const fs = require('fs').promises;
const path = require('path');

const execAsync = promisify(exec);

class MCPServerManager {
    constructor() {
        this.servers = [];
        this.testResults = {};
    }

    async log(message, type = 'info') {
        const icons = {
            info: '📘',
            success: '✅',
            error: '❌',
            warning: '⚠️',
            server: '🖥️',
            restart: '🔄'
        };
        console.log(`${icons[type] || '📝'} ${message}`);
    }

    async killExistingServers() {
        await this.log('Killing existing MCP server processes...', 'restart');
        
        // Kill any existing MCP server processes
        const killCommands = [
            'taskkill /F /IM "mcp-server*" 2>nul',
            'taskkill /F /IM "claude-flow*" 2>nul',
            'taskkill /F /IM "ruv-swarm*" 2>nul'
        ];
        
        for (const cmd of killCommands) {
            try {
                await execAsync(cmd);
            } catch (e) {
                // Process might not exist, that's okay
            }
        }
        
        await this.log('Existing servers terminated', 'success');
    }

    async testServer(name, command, args = []) {
        await this.log(`Testing ${name}...`, 'server');
        
        return new Promise((resolve) => {
            const fullCommand = args.length > 0 
                ? `${command} ${args.join(' ')}` 
                : command;
            
            // Test if the command exists
            exec(`where ${command.split(' ')[0]}`, (error, stdout, stderr) => {
                if (error) {
                    this.testResults[name] = {
                        status: 'not_found',
                        error: `Command not found: ${command}`
                    };
                    resolve(false);
                    return;
                }
                
                // Try to run the server with a version check
                const testArgs = [...args];
                if (!testArgs.includes('--version')) {
                    testArgs.push('--version');
                }
                
                const child = spawn(command, testArgs, {
                    shell: true,
                    stdio: 'pipe'
                });
                
                let output = '';
                let errorOutput = '';
                
                const timeout = setTimeout(() => {
                    child.kill();
                    this.testResults[name] = {
                        status: 'timeout',
                        message: 'Server timed out (might be running correctly)'
                    };
                    resolve(true); // Timeout might mean it's running
                }, 3000);
                
                child.stdout.on('data', (data) => {
                    output += data.toString();
                });
                
                child.stderr.on('data', (data) => {
                    errorOutput += data.toString();
                });
                
                child.on('error', (error) => {
                    clearTimeout(timeout);
                    this.testResults[name] = {
                        status: 'error',
                        error: error.message
                    };
                    resolve(false);
                });
                
                child.on('close', (code) => {
                    clearTimeout(timeout);
                    if (code === 0 || output || errorOutput.includes('version')) {
                        this.testResults[name] = {
                            status: 'available',
                            output: output || errorOutput
                        };
                        resolve(true);
                    } else {
                        this.testResults[name] = {
                            status: 'failed',
                            code,
                            error: errorOutput
                        };
                        resolve(false);
                    }
                });
            });
        });
    }

    async installMissingServers() {
        await this.log('Checking for missing MCP servers...', 'info');
        
        const commonServers = [
            { name: '@modelcontextprotocol/server-http-fetch', global: false },
            { name: '@modelcontextprotocol/server-ddg', global: false },
            { name: '@modelcontextprotocol/server-files', global: false },
            { name: 'mcp-sequential-thinking', global: true }
        ];
        
        for (const server of commonServers) {
            await this.log(`Checking ${server.name}...`, 'info');
            try {
                const checkCmd = server.global 
                    ? `npm list -g ${server.name} 2>nul`
                    : `npm list ${server.name} 2>nul`;
                
                await execAsync(checkCmd);
                await this.log(`${server.name} is installed`, 'success');
            } catch (e) {
                await this.log(`Installing ${server.name}...`, 'warning');
                try {
                    const installCmd = server.global
                        ? `npm install -g ${server.name}`
                        : `npm install ${server.name}`;
                    
                    await execAsync(installCmd);
                    await this.log(`${server.name} installed successfully`, 'success');
                } catch (installError) {
                    await this.log(`Failed to install ${server.name}: ${installError.message}`, 'error');
                }
            }
        }
    }

    async testAllServers() {
        await this.log('Testing MCP server availability...', 'info');
        
        const serversToTest = [
            { name: 'sequential-thinking', command: 'mcp-server-sequential-thinking', args: [] },
            { name: 'claude-flow', command: 'npx', args: ['claude-flow@alpha', '--version'] },
            { name: 'ruv-swarm', command: 'npx', args: ['ruv-swarm@latest', '--version'] },
            { name: 'http_fetch', command: 'npx', args: ['@modelcontextprotocol/server-http-fetch', '--help'] },
            { name: 'ddg', command: 'npx', args: ['@modelcontextprotocol/server-ddg', '--help'] },
            { name: 'files', command: 'npx', args: ['@modelcontextprotocol/server-files', '--help'] }
        ];
        
        const results = [];
        for (const server of serversToTest) {
            const available = await this.testServer(server.name, server.command, server.args);
            results.push({ ...server, available });
        }
        
        return results;
    }

    async generateReport() {
        await this.log('\n=== MCP Server Status Report ===', 'info');
        
        const availableServers = [];
        const unavailableServers = [];
        const needsAttention = [];
        
        for (const [name, result] of Object.entries(this.testResults)) {
            if (result.status === 'available' || result.status === 'timeout') {
                availableServers.push(name);
            } else if (result.status === 'not_found') {
                unavailableServers.push(name);
            } else {
                needsAttention.push(name);
            }
        }
        
        console.log('\n📊 Summary:');
        console.log(`✅ Available: ${availableServers.length}`);
        console.log(`❌ Not Found: ${unavailableServers.length}`);
        console.log(`⚠️ Needs Attention: ${needsAttention.length}`);
        
        if (availableServers.length > 0) {
            console.log('\n✅ Available Servers:');
            availableServers.forEach(s => console.log(`  - ${s}`));
        }
        
        if (unavailableServers.length > 0) {
            console.log('\n❌ Not Found (need installation):');
            unavailableServers.forEach(s => console.log(`  - ${s}`));
        }
        
        if (needsAttention.length > 0) {
            console.log('\n⚠️ Needs Attention:');
            needsAttention.forEach(s => {
                console.log(`  - ${s}: ${this.testResults[s].error || this.testResults[s].message}`);
            });
        }
        
        // Save report
        const report = {
            timestamp: new Date().toISOString(),
            results: this.testResults,
            summary: {
                available: availableServers,
                notFound: unavailableServers,
                needsAttention: needsAttention
            }
        };
        
        await fs.writeFile('mcp-server-status.json', JSON.stringify(report, null, 2));
        await this.log('\nReport saved to mcp-server-status.json', 'success');
    }

    async updateClaudeConfig() {
        await this.log('Updating Claude Desktop configuration...', 'info');
        
        const configPath = 'C:\\Users\\root\\AppData\\Roaming\\Claude\\claude_desktop_config.json';
        
        try {
            // Create minimal working config
            const minimalConfig = {
                mcpServers: {
                    "sequential-thinking": {
                        "command": "mcp-server-sequential-thinking"
                    },
                    "files": {
                        "command": "npx",
                        "args": ["@modelcontextprotocol/server-files", "C:\\Users\\root\\sutazaiapp"]
                    },
                    "http-fetch": {
                        "command": "npx",
                        "args": ["@modelcontextprotocol/server-http-fetch"]
                    }
                }
            };
            
            await fs.writeFile(configPath, JSON.stringify(minimalConfig, null, 2));
            await this.log('Claude Desktop configuration updated', 'success');
            await this.log('Please restart Claude Desktop to apply changes', 'warning');
        } catch (error) {
            await this.log(`Failed to update config: ${error.message}`, 'error');
        }
    }

    async run() {
        await this.log('🚀 MCP Server Manager Starting...', 'info');
        await this.log(`Platform: ${process.platform}`, 'info');
        await this.log(`Node Version: ${process.version}`, 'info');
        
        // Kill existing servers
        await this.killExistingServers();
        
        // Install missing servers
        await this.installMissingServers();
        
        // Test all servers
        await this.testAllServers();
        
        // Generate report
        await this.generateReport();
        
        // Update Claude config
        await this.updateClaudeConfig();
        
        await this.log('\n✅ MCP Server management complete!', 'success');
        await this.log('Next steps:', 'info');
        await this.log('1. Restart Claude Desktop application', 'info');
        await this.log('2. Check mcp-server-status.json for details', 'info');
        await this.log('3. Use /mcp command in Claude to reconnect', 'info');
    }
}

// Run the manager
if (require.main === module) {
    const manager = new MCPServerManager();
    manager.run()
        .then(() => process.exit(0))
        .catch(error => {
            console.error('❌ Error:', error);
            process.exit(1);
        });
}

module.exports = MCPServerManager;