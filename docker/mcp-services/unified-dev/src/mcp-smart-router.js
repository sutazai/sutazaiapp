#!/usr/bin/env node
/**
 * MCP Smart Router - Enterprise Load Balancing & Intelligent Routing
 * Provides advanced routing capabilities for optimal MCP server utilization
 */

const EventEmitter = require('events');

class MCPSmartRouter extends EventEmitter {
  constructor(options = {}) {
    super();
    
    this.config = {
      routingStrategy: options.routingStrategy || 'weighted_round_robin',
      healthCheckInterval: options.healthCheckInterval || 10000,
      enablePredictiveRouting: options.enablePredictiveRouting !== false,
      enableCaching: options.enableCaching !== false,
      maxRetries: options.maxRetries || 3,
      timeoutMs: options.timeoutMs || 30000
    };
    
    this.serverPools = new Map();
    this.routingTable = new Map();
    this.performanceMetrics = new Map();
    this.cache = new Map();
    this.logger = options.logger || console;
    
    // Initialize routing algorithms
    this.initializeRoutingAlgorithms();
    
    // Start performance monitoring
    this.startPerformanceMonitoring();
  }

  /**
   * Initialize different routing algorithms
   */
  initializeRoutingAlgorithms() {
    this.routingAlgorithms = {
      round_robin: this.roundRobinRouter.bind(this),
      weighted_round_robin: this.weightedRoundRobinRouter.bind(this),
      least_connections: this.leastConnectionsRouter.bind(this),
      fastest_response: this.fastestResponseRouter.bind(this),
      resource_based: this.resourceBasedRouter.bind(this),
      predictive: this.predictiveRouter.bind(this),
      geographic: this.geographicRouter.bind(this)
    };
  }

  /**
   * Register MCP server pool with load balancing
   */
  registerServerPool(serviceName, servers) {
    const pool = {
      servers: servers.map(server => ({
        ...server,
        id: server.id || `${serviceName}-${Date.now()}-${Math.random()}`,
        weight: server.weight || 1,
        connections: 0,
        totalRequests: 0,
        successfulRequests: 0,
        averageResponseTime: 0,
        lastResponseTime: 0,
        healthy: true,
        region: server.region || 'default',
        priority: server.priority || 1
      })),
      currentIndex: 0,
      serviceName
    };
    
    this.serverPools.set(serviceName, pool);
    this.routingTable.set(serviceName, this.config.routingStrategy);
    
    this.logger.info(`Registered server pool for ${serviceName}:`, {
      serverCount: pool.servers.length,
      strategy: this.config.routingStrategy
    });
    
    return pool;
  }

  /**
   * Smart route selection based on configured strategy
   */
  async selectServer(serviceName, requestContext = {}) {
    const pool = this.serverPools.get(serviceName);
    if (!pool) {
      throw new Error(`No server pool found for service: ${serviceName}`);
    }
    
    const strategy = this.routingTable.get(serviceName) || this.config.routingStrategy;
    const algorithm = this.routingAlgorithms[strategy];
    
    if (!algorithm) {
      throw new Error(`Unknown routing strategy: ${strategy}`);
    }
    
    // Check cache first if enabled
    if (this.config.enableCaching) {
      const cached = this.checkCache(serviceName, requestContext);
      if (cached) {
        this.logger.debug(`Cache hit for ${serviceName}`);
        return cached;
      }
    }
    
    const selectedServer = algorithm(pool, requestContext);
    
    if (!selectedServer) {
      throw new Error(`No healthy servers available for service: ${serviceName}`);
    }
    
    // Increment connection count
    selectedServer.connections++;
    selectedServer.totalRequests++;
    
    // Cache selection if enabled
    if (this.config.enableCaching) {
      this.cacheSelection(serviceName, requestContext, selectedServer);
    }
    
    return selectedServer;
  }

  /**
   * Round Robin routing algorithm
   */
  roundRobinRouter(pool) {
    const healthyServers = pool.servers.filter(s => s.healthy);
    if (healthyServers.length === 0) return null;
    
    pool.currentIndex = (pool.currentIndex + 1) % healthyServers.length;
    return healthyServers[pool.currentIndex];
  }

  /**
   * Weighted Round Robin routing algorithm
   */
  weightedRoundRobinRouter(pool) {
    const healthyServers = pool.servers.filter(s => s.healthy);
    if (healthyServers.length === 0) return null;
    
    const totalWeight = healthyServers.reduce((sum, server) => sum + server.weight, 0);
    let randomWeight = Math.random() * totalWeight;
    
    for (const server of healthyServers) {
      randomWeight -= server.weight;
      if (randomWeight <= 0) {
        return server;
      }
    }
    
    return healthyServers[0]; // Fallback
  }

  /**
   * Least Connections routing algorithm
   */
  leastConnectionsRouter(pool) {
    const healthyServers = pool.servers.filter(s => s.healthy);
    if (healthyServers.length === 0) return null;
    
    return healthyServers.reduce((least, current) => 
      current.connections < least.connections ? current : least
    );
  }

  /**
   * Fastest Response routing algorithm
   */
  fastestResponseRouter(pool) {
    const healthyServers = pool.servers.filter(s => s.healthy);
    if (healthyServers.length === 0) return null;
    
    return healthyServers.reduce((fastest, current) => 
      current.averageResponseTime < fastest.averageResponseTime ? current : fastest
    );
  }

  /**
   * Resource-based routing algorithm
   */
  resourceBasedRouter(pool, requestContext) {
    const healthyServers = pool.servers.filter(s => s.healthy);
    if (healthyServers.length === 0) return null;
    
    // Score servers based on CPU, memory, and load
    const scoredServers = healthyServers.map(server => ({
      ...server,
      score: this.calculateResourceScore(server, requestContext)
    }));
    
    return scoredServers.reduce((best, current) => 
      current.score > best.score ? current : best
    );
  }

  /**
   * Predictive routing using ML-style prediction
   */
  predictiveRouter(pool, requestContext) {
    if (!this.config.enablePredictiveRouting) {
      return this.weightedRoundRobinRouter(pool);
    }
    
    const healthyServers = pool.servers.filter(s => s.healthy);
    if (healthyServers.length === 0) return null;
    
    // Predict server performance based on historical data
    const predictions = healthyServers.map(server => ({
      ...server,
      predictedLatency: this.predictLatency(server, requestContext),
      predictedSuccess: this.predictSuccessRate(server, requestContext)
    }));
    
    // Select server with best predicted performance
    return predictions.reduce((best, current) => {
      const bestScore = best.predictedSuccess * 100 - best.predictedLatency;
      const currentScore = current.predictedSuccess * 100 - current.predictedLatency;
      return currentScore > bestScore ? current : best;
    });
  }

  /**
   * Geographic routing algorithm
   */
  geographicRouter(pool, requestContext) {
    const healthyServers = pool.servers.filter(s => s.healthy);
    if (healthyServers.length === 0) return null;
    
    const clientRegion = requestContext.region || 'default';
    
    // Prefer servers in same region
    const sameRegionServers = healthyServers.filter(s => s.region === clientRegion);
    if (sameRegionServers.length > 0) {
      return this.weightedRoundRobinRouter({ servers: sameRegionServers });
    }
    
    // Fallback to weighted round robin
    return this.weightedRoundRobinRouter(pool);
  }

  /**
   * Calculate resource-based score for server selection
   */
  calculateResourceScore(server, requestContext) {
    const cpuScore = (100 - (server.cpuUsage || 0)) / 100;
    const memoryScore = (100 - (server.memoryUsage || 0)) / 100;
    const loadScore = Math.max(0, (100 - server.connections * 10)) / 100;
    const responseScore = Math.max(0, (1000 - server.averageResponseTime)) / 1000;
    
    return (cpuScore * 0.3 + memoryScore * 0.2 + loadScore * 0.3 + responseScore * 0.2);
  }

  /**
   * Predict latency using historical data
   */
  predictLatency(server, requestContext) {
    // Simple prediction based on recent performance
    const baseLatency = server.averageResponseTime || 100;
    const loadFactor = Math.max(1, server.connections / 10);
    const timeFactor = this.getTimeOfDayFactor();
    
    return baseLatency * loadFactor * timeFactor;
  }

  /**
   * Predict success rate using historical data
   */
  predictSuccessRate(server, requestContext) {
    if (server.totalRequests === 0) return 0.95; // Default for new servers
    
    return server.successfulRequests / server.totalRequests;
  }

  /**
   * Get time of day factor for load prediction
   */
  getTimeOfDayFactor() {
    const hour = new Date().getHours();
    // Peak hours: 9-17, off-peak: 0-8, 18-23
    if (hour >= 9 && hour <= 17) return 1.5; // Peak load
    if (hour >= 0 && hour <= 6) return 0.7;  // Low load
    return 1.0; // Normal load
  }

  /**
   * Record request completion and update metrics
   */
  recordRequestCompletion(server, responseTime, success) {
    server.connections = Math.max(0, server.connections - 1);
    server.lastResponseTime = responseTime;
    
    if (success) {
      server.successfulRequests++;
    }
    
    // Update rolling average response time
    const alpha = 0.1; // Smoothing factor
    server.averageResponseTime = server.averageResponseTime * (1 - alpha) + responseTime * alpha;
    
    // Update performance metrics
    this.updatePerformanceMetrics(server, responseTime, success);
  }

  /**
   * Update server health status
   */
  updateServerHealth(serviceName, serverId, isHealthy, metrics = {}) {
    const pool = this.serverPools.get(serviceName);
    if (!pool) return;
    
    const server = pool.servers.find(s => s.id === serverId);
    if (!server) return;
    
    const wasHealthy = server.healthy;
    server.healthy = isHealthy;
    
    // Update resource metrics
    if (metrics.cpuUsage !== undefined) server.cpuUsage = metrics.cpuUsage;
    if (metrics.memoryUsage !== undefined) server.memoryUsage = metrics.memoryUsage;
    if (metrics.diskUsage !== undefined) server.diskUsage = metrics.diskUsage;
    
    if (wasHealthy !== isHealthy) {
      this.emit('server_health_changed', {
        serviceName,
        serverId,
        healthy: isHealthy,
        server
      });
      
      this.logger.info(`Server health changed: ${serviceName}/${serverId} -> ${isHealthy ? 'healthy' : 'unhealthy'}`);
    }
  }

  /**
   * Check cache for server selection
   */
  checkCache(serviceName, requestContext) {
    if (!this.config.enableCaching) return null;
    
    const cacheKey = this.generateCacheKey(serviceName, requestContext);
    const cached = this.cache.get(cacheKey);
    
    if (cached && Date.now() - cached.timestamp < 60000) { // 1 minute TTL
      return cached.server;
    }
    
    return null;
  }

  /**
   * Cache server selection
   */
  cacheSelection(serviceName, requestContext, server) {
    if (!this.config.enableCaching) return;
    
    const cacheKey = this.generateCacheKey(serviceName, requestContext);
    this.cache.set(cacheKey, {
      server,
      timestamp: Date.now()
    });
    
    // Cleanup old cache entries
    if (this.cache.size > 1000) {
      const oldest = Array.from(this.cache.entries())
        .sort((a, b) => a[1].timestamp - b[1].timestamp)
        .slice(0, 100);
      
      oldest.forEach(([key]) => this.cache.delete(key));
    }
  }

  /**
   * Generate cache key for request context
   */
  generateCacheKey(serviceName, requestContext) {
    const keyParts = [
      serviceName,
      requestContext.operation || 'default',
      requestContext.region || 'default',
      requestContext.priority || 'normal'
    ];
    
    return keyParts.join(':');
  }

  /**
   * Update performance metrics
   */
  updatePerformanceMetrics(server, responseTime, success) {
    const key = `${server.serviceName}:${server.id}`;
    
    if (!this.performanceMetrics.has(key)) {
      this.performanceMetrics.set(key, {
        samples: [],
        successCount: 0,
        totalCount: 0
      });
    }
    
    const metrics = this.performanceMetrics.get(key);
    metrics.totalCount++;
    
    if (success) {
      metrics.successCount++;
    }
    
    metrics.samples.push({
      responseTime,
      success,
      timestamp: Date.now()
    });
    
    // Keep only last 100 samples
    if (metrics.samples.length > 100) {
      metrics.samples.shift();
    }
  }

  /**
   * Start performance monitoring
   */
  startPerformanceMonitoring() {
    setInterval(() => {
      this.cleanupCache();
      this.analyzePerformancePatterns();
      this.adjustRoutingStrategies();
    }, this.config.healthCheckInterval);
  }

  /**
   * Cleanup expired cache entries
   */
  cleanupCache() {
    const now = Date.now();
    for (const [key, value] of this.cache.entries()) {
      if (now - value.timestamp > 300000) { // 5 minutes
        this.cache.delete(key);
      }
    }
  }

  /**
   * Analyze performance patterns
   */
  analyzePerformancePatterns() {
    for (const [key, metrics] of this.performanceMetrics.entries()) {
      if (metrics.samples.length < 10) continue;
      
      const recentSamples = metrics.samples.slice(-10);
      const avgResponseTime = recentSamples.reduce((sum, s) => sum + s.responseTime, 0) / recentSamples.length;
      const successRate = recentSamples.filter(s => s.success).length / recentSamples.length;
      
      // Emit performance events for monitoring
      this.emit('performance_analysis', {
        serverKey: key,
        averageResponseTime: avgResponseTime,
        successRate: successRate,
        sampleCount: metrics.samples.length
      });
    }
  }

  /**
   * Adjust routing strategies based on performance
   */
  adjustRoutingStrategies() {
    for (const [serviceName, pool] of this.serverPools.entries()) {
      const totalRequests = pool.servers.reduce((sum, s) => sum + s.totalRequests, 0);
      
      if (totalRequests < 100) continue; // Not enough data
      
      const avgSuccessRate = pool.servers.reduce((sum, s) => 
        sum + (s.totalRequests > 0 ? s.successfulRequests / s.totalRequests : 0), 0
      ) / pool.servers.length;
      
      const avgResponseTime = pool.servers.reduce((sum, s) => sum + s.averageResponseTime, 0) / pool.servers.length;
      
      // Switch to predictive routing if performance is declining
      if (avgSuccessRate < 0.95 || avgResponseTime > 1000) {
        if (this.routingTable.get(serviceName) !== 'predictive') {
          this.routingTable.set(serviceName, 'predictive');
          this.logger.info(`Switched ${serviceName} to predictive routing due to performance issues`);
        }
      }
    }
  }

  /**
   * Get routing statistics
   */
  getRoutingStats() {
    const stats = {};
    
    for (const [serviceName, pool] of this.serverPools.entries()) {
      const serverStats = pool.servers.map(server => ({
        id: server.id,
        healthy: server.healthy,
        connections: server.connections,
        totalRequests: server.totalRequests,
        successfulRequests: server.successfulRequests,
        averageResponseTime: Math.round(server.averageResponseTime),
        successRate: server.totalRequests > 0 ? 
          Math.round((server.successfulRequests / server.totalRequests) * 100) / 100 : 0
      }));
      
      stats[serviceName] = {
        strategy: this.routingTable.get(serviceName),
        servers: serverStats,
        totalServers: pool.servers.length,
        healthyServers: pool.servers.filter(s => s.healthy).length
      };
    }
    
    return {
      services: stats,
      cacheSize: this.cache.size,
      totalMetrics: this.performanceMetrics.size,
      timestamp: new Date().toISOString()
    };
  }

  /**
   * Graceful shutdown
   */
  async shutdown() {
    this.logger.info('Shutting down MCP Smart Router...');
    this.cache.clear();
    this.performanceMetrics.clear();
    this.removeAllListeners();
    return { success: true };
  }
}

module.exports = MCPSmartRouter;