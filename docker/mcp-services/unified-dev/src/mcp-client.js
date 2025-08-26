#!/usr/bin/env node
/**
 * MCP Client Integration for Unified Development Service
 * Provides seamless connectivity to 21 MCP servers in the infrastructure
 */

const axios = require('axios');
const EventEmitter = require('events');

class MCPClient extends EventEmitter {
  constructor(options = {}) {
    super();
    
    this.config = {
      backendUrl: options.backendUrl || 'http://localhost:10010',
      timeout: options.timeout || 30000,
      retryAttempts: options.retryAttempts || 3,
      retryDelay: options.retryDelay || 1000,
      enableMetrics: options.enableMetrics !== false
    };
    
    this.metrics = {
      requests: { total: 0, successful: 0, failed: 0 },
      response_times: [],
      errors: [],
      server_status: new Map()
    };
    
    this.circuitBreakers = new Map();
    this.logger = options.logger || console;
  }

  /**
   * Initialize MCP client and validate connectivity
   */
  async initialize() {
    try {
      this.logger.info('Initializing MCP client...');
      
      // Check backend connectivity
      const healthCheck = await this.checkBackendHealth();
      if (!healthCheck.success) {
        throw new Error(`Backend health check failed: ${healthCheck.error}`);
      }
      
      // Load server registry
      await this.loadServerRegistry();
      
      // Initialize circuit breakers
      this.initializeCircuitBreakers();
      
      this.logger.info('MCP client initialized successfully');
      this.emit('ready');
      
      return { success: true };
      
    } catch (error) {
      this.logger.error('MCP client initialization failed:', error.message);
      this.emit('error', error);
      throw error;
    }
  }

  /**
   * Check backend API health
   */
  async checkBackendHealth() {
    const startTime = Date.now();
    
    try {
      const response = await axios.get(`${this.config.backendUrl}/api/v1/mcp/health`, {
        timeout: this.config.timeout
      });
      
      const duration = Date.now() - startTime;
      this.updateMetrics('health_check', duration, true);
      
      return {
        success: true,
        data: response.data,
        responseTime: duration
      };
      
    } catch (error) {
      const duration = Date.now() - startTime;
      this.updateMetrics('health_check', duration, false, error);
      
      return {
        success: false,
        error: error.message,
        responseTime: duration
      };
    }
  }

  /**
   * Load MCP server registry from backend
   */
  async loadServerRegistry() {
    try {
      const healthResponse = await axios.get(`${this.config.backendUrl}/api/v1/mcp/health`, {
        timeout: this.config.timeout
      });
      
      const services = healthResponse.data.services || {};
      
      // Update server status metrics
      for (const [serverName, status] of Object.entries(services)) {
        this.metrics.server_status.set(serverName, {
          healthy: status.healthy,
          available: status.available,
          lastCheck: new Date().toISOString()
        });
      }
      
      this.logger.info(`Loaded ${Object.keys(services).length} MCP servers`);
      
      return services;
      
    } catch (error) {
      this.logger.error('Failed to load server registry:', error.message);
      throw error;
    }
  }

  /**
   * Initialize circuit breakers for each MCP server
   */
  initializeCircuitBreakers() {
    for (const serverName of this.metrics.server_status.keys()) {
      this.circuitBreakers.set(serverName, {
        state: 'closed', // closed, open, half-open
        failureCount: 0,
        lastFailureTime: null,
        threshold: 5,
        timeout: 60000 // 1 minute
      });
    }
  }

  /**
   * Execute MCP tool call with circuit breaker protection
   */
  async callMCPTool(serverName, toolName, arguments_obj = {}) {
    const startTime = Date.now();
    
    try {
      // Check circuit breaker
      if (!this.isCircuitBreakerClosed(serverName)) {
        throw new Error(`Circuit breaker open for server: ${serverName}`);
      }
      
      const url = `${this.config.backendUrl}/api/v1/mcp/${serverName}/tools/${toolName}`;
      
      const response = await this.executeWithRetry(async () => {
        return await axios.post(url, arguments_obj, {
          timeout: this.config.timeout,
          headers: {
            'Content-Type': 'application/json'
          }
        });
      });
      
      const duration = Date.now() - startTime;
      this.updateMetrics('tool_call', duration, true);
      this.resetCircuitBreaker(serverName);
      
      return {
        success: true,
        data: response.data,
        server: serverName,
        tool: toolName,
        responseTime: duration
      };
      
    } catch (error) {
      const duration = Date.now() - startTime;
      this.updateMetrics('tool_call', duration, false, error);
      this.recordCircuitBreakerFailure(serverName);
      
      return {
        success: false,
        error: error.message,
        server: serverName,
        tool: toolName,
        responseTime: duration
      };
    }
  }

  /**
   * Access MCP resource with error handling
   */
  async accessMCPResource(serverName, resourceUri) {
    const startTime = Date.now();
    
    try {
      if (!this.isCircuitBreakerClosed(serverName)) {
        throw new Error(`Circuit breaker open for server: ${serverName}`);
      }
      
      const url = `${this.config.backendUrl}/api/v1/mcp/${serverName}/resources`;
      
      const response = await this.executeWithRetry(async () => {
        return await axios.get(url, {
          params: { uri: resourceUri },
          timeout: this.config.timeout
        });
      });
      
      const duration = Date.now() - startTime;
      this.updateMetrics('resource_access', duration, true);
      this.resetCircuitBreaker(serverName);
      
      return {
        success: true,
        data: response.data,
        server: serverName,
        resource: resourceUri,
        responseTime: duration
      };
      
    } catch (error) {
      const duration = Date.now() - startTime;
      this.updateMetrics('resource_access', duration, false, error);
      this.recordCircuitBreakerFailure(serverName);
      
      return {
        success: false,
        error: error.message,
        server: serverName,
        resource: resourceUri,
        responseTime: duration
      };
    }
  }

  /**
   * Execute request with retry logic
   */
  async executeWithRetry(operation) {
    let lastError;
    
    for (let attempt = 1; attempt <= this.config.retryAttempts; attempt++) {
      try {
        return await operation();
      } catch (error) {
        lastError = error;
        
        if (attempt < this.config.retryAttempts) {
          await this.sleep(this.config.retryDelay * attempt);
        }
      }
    }
    
    throw lastError;
  }

  /**
   * Circuit breaker management
   */
  isCircuitBreakerClosed(serverName) {
    const breaker = this.circuitBreakers.get(serverName);
    if (!breaker) return true;
    
    const now = Date.now();
    
    if (breaker.state === 'open') {
      if (now - breaker.lastFailureTime > breaker.timeout) {
        breaker.state = 'half-open';
        return true;
      }
      return false;
    }
    
    return breaker.state !== 'open';
  }

  recordCircuitBreakerFailure(serverName) {
    const breaker = this.circuitBreakers.get(serverName);
    if (!breaker) return;
    
    breaker.failureCount++;
    breaker.lastFailureTime = Date.now();
    
    if (breaker.failureCount >= breaker.threshold) {
      breaker.state = 'open';
      this.logger.warn(`Circuit breaker opened for server: ${serverName}`);
      this.emit('circuit_breaker_opened', { serverName, failures: breaker.failureCount });
    }
  }

  resetCircuitBreaker(serverName) {
    const breaker = this.circuitBreakers.get(serverName);
    if (!breaker) return;
    
    if (breaker.state === 'half-open') {
      breaker.state = 'closed';
      this.logger.info(`Circuit breaker closed for server: ${serverName}`);
      this.emit('circuit_breaker_closed', { serverName });
    }
    
    breaker.failureCount = 0;
    breaker.lastFailureTime = null;
  }

  /**
   * Update performance metrics
   */
  updateMetrics(operation, duration, success, error = null) {
    if (!this.config.enableMetrics) return;
    
    this.metrics.requests.total++;
    
    if (success) {
      this.metrics.requests.successful++;
    } else {
      this.metrics.requests.failed++;
      if (error) {
        this.metrics.errors.push({
          operation,
          error: error.message,
          timestamp: new Date().toISOString()
        });
      }
    }
    
    this.metrics.response_times.push({
      operation,
      duration,
      success,
      timestamp: new Date().toISOString()
    });
    
    // Keep only last 1000 entries
    if (this.metrics.response_times.length > 1000) {
      this.metrics.response_times.shift();
    }
    
    if (this.metrics.errors.length > 100) {
      this.metrics.errors.shift();
    }
  }

  /**
   * Get performance metrics
   */
  getMetrics() {
    const responseTimes = this.metrics.response_times.map(r => r.duration);
    
    return {
      requests: this.metrics.requests,
      performance: {
        avg_response_time: responseTimes.length ? responseTimes.reduce((a, b) => a + b, 0) / responseTimes.length : 0,
        min_response_time: responseTimes.length ? Math.min(...responseTimes) : 0,
        max_response_time: responseTimes.length ? Math.max(...responseTimes) : 0,
        p95_response_time: this.calculatePercentile(responseTimes, 95),
        p99_response_time: this.calculatePercentile(responseTimes, 99)
      },
      servers: Object.fromEntries(this.metrics.server_status),
      circuit_breakers: Object.fromEntries(
        Array.from(this.circuitBreakers.entries()).map(([name, breaker]) => [
          name,
          { state: breaker.state, failures: breaker.failureCount }
        ])
      ),
      recent_errors: this.metrics.errors.slice(-10)
    };
  }

  /**
   * Calculate percentile from array of numbers
   */
  calculatePercentile(arr, percentile) {
    if (arr.length === 0) return 0;
    
    const sorted = arr.slice().sort((a, b) => a - b);
    const index = Math.ceil((percentile / 100) * sorted.length) - 1;
    return sorted[index] || 0;
  }

  /**
   * Utility sleep function
   */
  sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Graceful shutdown
   */
  async shutdown() {
    this.logger.info('Shutting down MCP client...');
    this.removeAllListeners();
    return { success: true };
  }
}

module.exports = MCPClient;