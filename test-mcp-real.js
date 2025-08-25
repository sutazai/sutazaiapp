/**
 * Real MCP Server Test - Actual Connection Testing
 * Tests MCP servers with proper stdio communication
 */

const { spawn } = require('child_process');
const readline = require('readline');

class RealMCPTester {
    constructor() {
        this.testResults = {};
        this.activeServers = {};
    }

    log(message, type = 'info') {
        const icons = {
            info: '📘',
            success: '✅',
            error: '❌',
            warning: '⚠️',
            test: '🧪',
            server: '🖥️'
        };
        const timestamp = new Date().toISOString().split('T')[1].split('.')[0];
        console.log(`[${timestamp}] ${icons[type] || '📝'} ${message}`);
    }

    async testMCPServer(name, command, args = []) {
        return new Promise((resolve) => {
            this.log(`Testing ${name}...`, 'test');
            
            const startTime = Date.now();
            const server = spawn(command, args, {
                stdio: ['pipe', 'pipe', 'pipe'],
                shell: true,
                env: { ...process.env, NODE_ENV: 'production' }
            });

            let initialized = false;
            let capabilities = null;
            let errorMessages = [];
            let outputBuffer = '';

            // Handle timeout
            const timeout = setTimeout(() => {
                if (!initialized) {
                    server.kill();
                    const duration = Date.now() - startTime;
                    this.testResults[name] = {
                        status: 'timeout',
                        duration: `${duration}ms`,
                        error: 'Server did not respond within 10 seconds'
                    };
                    this.log(`${name}: Timeout after ${duration}ms`, 'error');
                    resolve(false);
                }
            }, 10000);

            // Send initialization message
            const initMessage = JSON.stringify({
                jsonrpc: '2.0',
                id: 1,
                method: 'initialize',
                params: {
                    protocolVersion: '2024-11-05',
                    clientInfo: {
                        name: 'mcp-tester',
                        version: '1.0.0'
                    }
                }
            }) + '\n';

            server.stdin.write(initMessage);
            this.log(`Sent initialization to ${name}`, 'info');

            // Process stdout
            const rl = readline.createInterface({
                input: server.stdout,
                crlfDelay: Infinity
            });

            rl.on('line', (line) => {
                outputBuffer += line + '\n';
                try {
                    const message = JSON.parse(line);
                    
                    if (message.id === 1 && message.result) {
                        // Initialization response
                        initialized = true;
                        capabilities = message.result;
                        clearTimeout(timeout);
                        
                        const duration = Date.now() - startTime;
                        this.testResults[name] = {
                            status: 'success',
                            duration: `${duration}ms`,
                            capabilities: capabilities,
                            serverName: capabilities.serverInfo?.name,
                            version: capabilities.serverInfo?.version
                        };
                        
                        this.log(`${name}: Connected successfully in ${duration}ms`, 'success');
                        this.log(`  Server: ${capabilities.serverInfo?.name} v${capabilities.serverInfo?.version}`, 'info');
                        
                        // Test listing tools
                        const listToolsMessage = JSON.stringify({
                            jsonrpc: '2.0',
                            id: 2,
                            method: 'tools/list'
                        }) + '\n';
                        
                        server.stdin.write(listToolsMessage);
                        
                        // Give time for tools response then close
                        setTimeout(() => {
                            server.kill();
                            resolve(true);
                        }, 2000);
                    } else if (message.id === 2 && message.result) {
                        // Tools list response
                        const tools = message.result.tools || [];
                        this.testResults[name].toolCount = tools.length;
                        this.log(`  Tools available: ${tools.length}`, 'info');
                        if (tools.length > 0) {
                            this.log(`  Sample tools: ${tools.slice(0, 3).map(t => t.name).join(', ')}`, 'info');
                        }
                    }
                } catch (e) {
                    // Not JSON, ignore
                }
            });

            // Process stderr
            server.stderr.on('data', (data) => {
                const message = data.toString();
                errorMessages.push(message);
                if (message.includes('ERROR') || message.includes('Error')) {
                    this.log(`${name} error: ${message.substring(0, 100)}...`, 'warning');
                }
            });

            // Handle server exit
            server.on('exit', (code, signal) => {
                clearTimeout(timeout);
                if (!initialized) {
                    const duration = Date.now() - startTime;
                    this.testResults[name] = {
                        status: 'failed',
                        duration: `${duration}ms`,
                        exitCode: code,
                        signal: signal,
                        errors: errorMessages.join('\n').substring(0, 500)
                    };
                    this.log(`${name}: Failed with exit code ${code}`, 'error');
                    resolve(false);
                }
            });

            // Handle errors
            server.on('error', (error) => {
                clearTimeout(timeout);
                const duration = Date.now() - startTime;
                this.testResults[name] = {
                    status: 'error',
                    duration: `${duration}ms`,
                    error: error.message
                };
                this.log(`${name}: Error - ${error.message}`, 'error');
                resolve(false);
            });
        });
    }

    async runAllTests() {
        this.log('🚀 Starting Real MCP Server Tests', 'info');
        this.log(`Platform: ${process.platform}`, 'info');
        this.log(`Node: ${process.version}`, 'info');
        this.log(`Directory: ${process.cwd()}`, 'info');
        console.log('='.repeat(60));

        const servers = [
            {
                name: 'sequential-thinking',
                command: 'mcp-server-sequential-thinking',
                args: []
            },
            {
                name: 'filesystem',
                command: 'npx',
                args: ['-y', '@modelcontextprotocol/server-filesystem', process.cwd()]
            },
            {
                name: 'claude-flow',
                command: 'npx',
                args: ['claude-flow@alpha', 'mcp', 'start', '--stdio']
            },
            {
                name: 'ruv-swarm',
                command: 'npx',
                args: ['ruv-swarm@latest', 'mcp', 'start', '--stdio']
            }
        ];

        const results = [];
        for (const server of servers) {
            const success = await this.testMCPServer(server.name, server.command, server.args);
            results.push({ ...server, success });
        }

        // Generate report
        console.log('\n' + '='.repeat(60));
        this.log('📊 MCP Server Test Report', 'info');
        console.log('='.repeat(60));

        let successCount = 0;
        let failureCount = 0;

        for (const [name, result] of Object.entries(this.testResults)) {
            const icon = result.status === 'success' ? '✅' : 
                        result.status === 'timeout' ? '⏱️' : '❌';
            
            console.log(`\n${icon} ${name}`);
            console.log(`  Status: ${result.status}`);
            console.log(`  Duration: ${result.duration}`);
            
            if (result.status === 'success') {
                successCount++;
                if (result.serverName) {
                    console.log(`  Server: ${result.serverName} v${result.version}`);
                }
                if (result.toolCount !== undefined) {
                    console.log(`  Tools: ${result.toolCount}`);
                }
                if (result.capabilities) {
                    const caps = result.capabilities.capabilities || {};
                    console.log(`  Capabilities: ${Object.keys(caps).filter(k => caps[k]).join(', ')}`);
                }
            } else {
                failureCount++;
                if (result.error) {
                    console.log(`  Error: ${result.error}`);
                }
                if (result.exitCode !== undefined) {
                    console.log(`  Exit Code: ${result.exitCode}`);
                }
            }
        }

        console.log('\n' + '='.repeat(60));
        console.log(`Summary: ${successCount} passed, ${failureCount} failed`);
        console.log(`Success Rate: ${((successCount / (successCount + failureCount)) * 100).toFixed(0)}%`);

        // Save detailed report
        const report = {
            timestamp: new Date().toISOString(),
            platform: process.platform,
            nodeVersion: process.version,
            summary: {
                total: successCount + failureCount,
                passed: successCount,
                failed: failureCount,
                successRate: `${((successCount / (successCount + failureCount)) * 100).toFixed(0)}%`
            },
            servers: this.testResults
        };

        require('fs').writeFileSync(
            'mcp-real-test-report.json',
            JSON.stringify(report, null, 2)
        );

        console.log('\nDetailed report saved to: mcp-real-test-report.json');
        
        return report;
    }
}

// Run tests
if (require.main === module) {
    const tester = new RealMCPTester();
    tester.runAllTests()
        .then(() => process.exit(0))
        .catch(error => {
            console.error('Test failed:', error);
            process.exit(1);
        });
}

module.exports = RealMCPTester;