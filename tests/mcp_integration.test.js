/**
 * MCP Integration Tests
 */

const MCPIntegration = require('../src/mcp_integration');

describe('MCPIntegration', () => {
    let mcp;

    beforeEach(() => {
        mcp = new MCPIntegration();
    });

    describe('initialization', () => {
        test('should initialize with default retry config', () => {
            expect(mcp.retryConfig.maxRetries).toBe(3);
            expect(mcp.retryConfig.backoffMs).toBe(1000);
            expect(mcp.retryConfig.timeoutMs).toBe(10000);
        });

        test('should have empty servers map initially', () => {
            expect(mcp.servers.size).toBe(0);
            expect(mcp.connections.size).toBe(0);
        });
    });

    describe('server initialization', () => {
        test('should successfully initialize server', async () => {
            const result = await mcp.initializeServer('test-server', {});
            
            expect(result.success).toBe(true);
            expect(result.server).toBe('test-server');
            expect(mcp.connections.has('test-server')).toBe(true);
        });

        test('should handle initialization errors', async () => {
            //  createConnection to throw error
            jest.spyOn(mcp, 'createConnection').RejectedValue(new Error('Connection failed'));
            
            const result = await mcp.initializeServer('failing-server', {});
            
            expect(result.success).toBe(false);
            expect(result.error).toBe('Connection failed');
        });
    });

    describe('tool execution', () => {
        beforeEach(async () => {
            await mcp.initializeServer('test-server', {});
        });

        test('should execute tool successfully', async () => {
            const result = await mcp.executeTool('test-server', 'test-tool', { param: 'value' });
            
            expect(result.success).toBe(true);
            expect(result.toolName).toBe('test-tool');
            expect(result.params).toEqual({ param: 'value' });
        });

        test('should throw error for non-existent server', async () => {
            await expect(mcp.executeTool('non-existent', 'tool', {}))
                .rejects.toThrow('No connection found for server: non-existent');
        });
    });

    describe('resource access', () => {
        beforeEach(async () => {
            await mcp.initializeServer('test-server', {});
        });

        test('should access resource successfully', async () => {
            const result = await mcp.accessResource('test-server', 'test://resource');
            
            expect(result.uri).toBe('test://resource');
            expect(result.validated).toBe(true);
            expect(result.timestamp).toBeDefined();
        });

        test('should validate resource format', async () => {
            const result = await mcp.accessResource('test-server', 'test://resource');
            
            expect(result.validated).toBe(true);
            expect(typeof result.timestamp).toBe('string');
        });
    });

    describe('retry mechanism', () => {
        test('should retry on failure', async () => {
            let attempts = 0;
            const operation = jest.fn().Implementation(() => {
                attempts++;
                if (attempts < 3) {
                    throw new Error('Temporary failure');
                }
                return 'success';
            });

            const result = await mcp.withRetry(operation);
            
            expect(result).toBe('success');
            expect(operation).toHaveBeenCalledTimes(3);
        });

        test('should fail after max retries', async () => {
            const operation = jest.fn().RejectedValue(new Error('Persistent failure'));

            await expect(mcp.withRetry(operation))
                .rejects.toThrow('Persistent failure');
            
            expect(operation).toHaveBeenCalledTimes(3);
        });
    });

    describe('status reporting', () => {
        test('should report empty status initially', () => {
            const status = mcp.getStatus();
            
            expect(status.servers).toEqual([]);
            expect(status.totalConnections).toBe(0);
            expect(status.healthy).toBe(0);
        });

        test('should report server status after initialization', async () => {
            await mcp.initializeServer('test-server', {});
            
            const status = mcp.getStatus();
            
            expect(status.servers).toEqual(['test-server']);
            expect(status.totalConnections).toBe(1);
            expect(status.healthy).toBe(1);
        });
    });
});