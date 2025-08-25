/**
 * MCP Integration Test Script for Windows/VS Code
 * Tests available MCP servers and their functionality
 */

const { spawn } = require('child_process');
const readline = require('readline');

class MCPTester {
    constructor() {
        this.servers = {
            'sequential-thinking': {
                command: 'node',
                args: ['C:\\Users\\root\\AppData\\Roaming\\npm\\node_modules\\mcp-sequential-thinking\\dist\\index.js']
            },
            'claude-flow': {
                command: 'npx',
                args: ['claude-flow@alpha', 'mcp', 'start', '--stdio']
            },
            'ruv-swarm': {
                command: 'npx',
                args: ['ruv-swarm@latest', 'mcp', '--stdio']
            }
        };
    }

    async testServer(name, config) {
        console.log(`\n🔧 Testing MCP Server: ${name}`);
        console.log(`   Command: ${config.command} ${config.args.join(' ')}`);
        
        return new Promise((resolve) => {
            const child = spawn(config.command, config.args, {
                stdio: ['pipe', 'pipe', 'pipe'],
                shell: true
            });

            let output = '';
            let errorOutput = '';
            
            // Set timeout
            const timeout = setTimeout(() => {
                child.kill();
                console.log(`   ⏱️ Timeout reached, killing process`);
                resolve({
                    server: name,
                    status: 'timeout',
                    output,
                    error: errorOutput
                });
            }, 5000);

            child.stdout.on('data', (data) => {
                output += data.toString();
            });

            child.stderr.on('data', (data) => {
                errorOutput += data.toString();
            });

            child.on('error', (error) => {
                clearTimeout(timeout);
                console.log(`   ❌ Error: ${error.message}`);
                resolve({
                    server: name,
                    status: 'error',
                    error: error.message
                });
            });

            child.on('close', (code) => {
                clearTimeout(timeout);
                console.log(`   📊 Process exited with code: ${code}`);
                resolve({
                    server: name,
                    status: code === 0 ? 'success' : 'failed',
                    code,
                    output,
                    error: errorOutput
                });
            });

            // Send initial MCP handshake
            setTimeout(() => {
                const initMessage = JSON.stringify({
                    jsonrpc: '2.0',
                    id: 1,
                    method: 'initialize',
                    params: {
                        clientInfo: {
                            name: 'mcp-tester',
                            version: '1.0.0'
                        }
                    }
                }) + '\n';
                
                child.stdin.write(initMessage);
                console.log(`   📤 Sent initialize message`);
            }, 1000);
        });
    }

    async runTests() {
        console.log('🚀 Starting MCP Integration Tests for Windows/VS Code\n');
        console.log('Environment:');
        console.log(`  Platform: ${process.platform}`);
        console.log(`  Node Version: ${process.version}`);
        console.log(`  Current Directory: ${process.cwd()}`);
        
        const results = [];
        
        for (const [name, config] of Object.entries(this.servers)) {
            const result = await this.testServer(name, config);
            results.push(result);
        }
        
        console.log('\n📊 Test Results Summary:');
        console.log('========================');
        
        for (const result of results) {
            const icon = result.status === 'success' ? '✅' : 
                        result.status === 'timeout' ? '⏱️' : '❌';
            console.log(`${icon} ${result.server}: ${result.status}`);
            
            if (result.output && result.output.trim()) {
                console.log(`   Output: ${result.output.substring(0, 100)}...`);
            }
            if (result.error && result.error.trim()) {
                console.log(`   Error: ${result.error.substring(0, 100)}...`);
            }
        }
        
        return results;
    }
}

// Run tests
if (require.main === module) {
    const tester = new MCPTester();
    tester.runTests()
        .then(results => {
            console.log('\n✅ MCP Integration tests completed');
            process.exit(0);
        })
        .catch(error => {
            console.error('❌ Test failed:', error);
            process.exit(1);
        });
}

module.exports = MCPTester;