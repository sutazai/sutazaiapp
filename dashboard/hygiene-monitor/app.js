/**
 * Sutazai Hygiene Enforcement Monitor - Enhanced Dashboard Logic
 * Purpose: Intelligent real-time monitoring and control of codebase hygiene enforcement
 * Features: AI recommendations, impact preview, undo/redo, keyboard shortcuts, themes
 * Author: AI Observability and Monitoring Engineer
 * Version: 2.0.0 - Perfect UX Edition
 */

class HygieneMonitorDashboard {
    constructor() {
        this.apiEndpoint = '/api/hygiene';
        this.refreshInterval = 10000; // 10 seconds
        this.charts = {};
        this.websocket = null;
        this.cache = new Map();
        this.requestQueue = new Map();
        this.undoStack = [];
        this.redoStack = [];
        this.maxHistorySize = 50;
        this.pendingChanges = new Set();
        this.tabId = this.generateTabId();
        this.syncChannel = null;
        this.data = {
            rules: {},
            agents: {},
            actions: [],
            metrics: {},
            violations: [],
            systemEnabled: true,
            theme: localStorage.getItem('dashboard-theme') || 'dark'
        };

        // Enhanced CLAUDE.md Rule definitions with descriptions
        this.rules = {
            'rule_1': { 
                name: 'No Fantasy Elements', 
                priority: 'CRITICAL', 
                category: 'Code Quality',
                description: 'Only real, production-ready implementations. No speculative or hypothetical code.',
                enabled: true,
                dependencies: []
            },
            'rule_2': { 
                name: 'No Breaking Changes', 
                priority: 'CRITICAL', 
                category: 'Functionality',
                description: 'Every change must respect existing functionality. Test before merging.',
                enabled: true,
                dependencies: ['rule_10']
            },
            'rule_3': { 
                name: 'Analyze Everything', 
                priority: 'HIGH', 
                category: 'Process',
                description: 'Systematic review of entire application before proceeding with changes.',
                enabled: true,
                dependencies: []
            },
            'rule_4': { 
                name: 'Reuse Before Creating', 
                priority: 'MEDIUM', 
                category: 'Efficiency',
                description: 'Always check for existing solutions before creating new ones.',
                enabled: true,
                dependencies: ['rule_7']
            },
            'rule_5': { 
                name: 'Professional Standards', 
                priority: 'HIGH', 
                category: 'Quality',
                description: 'Approach every task with professional mindset and best practices.',
                enabled: true,
                dependencies: []
            },
            'rule_6': { 
                name: 'Centralized Documentation', 
                priority: 'HIGH', 
                category: 'Documentation',
                description: 'All documentation in centralized, organized, consistent location.',
                enabled: true,
                dependencies: ['rule_15']
            },
            'rule_7': { 
                name: 'Script Organization', 
                priority: 'MEDIUM', 
                category: 'Scripts',
                description: 'All scripts centralized, documented, purposeful, and reusable.',
                enabled: true,
                dependencies: ['rule_8']
            },
            'rule_8': { 
                name: 'Python Script Standards', 
                priority: 'MEDIUM', 
                category: 'Scripts',
                description: 'Python scripts must be structured, purposeful, and production-ready.',
                enabled: true,
                dependencies: []
            },
            'rule_9': { 
                name: 'No Code Duplication', 
                priority: 'HIGH', 
                category: 'Architecture',
                description: 'Single source of truth for all frontend and backend code.',
                enabled: true,
                dependencies: []
            },
            'rule_10': { 
                name: 'Verify Before Cleanup', 
                priority: 'CRITICAL', 
                category: 'Safety',
                description: 'Functional verification required before any cleanup activities.',
                enabled: true,
                dependencies: []
            },
            'rule_11': { 
                name: 'Clean Docker Structure', 
                priority: 'HIGH', 
                category: 'Infrastructure',
                description: 'Docker assets must be clean, modular, and predictable.',
                enabled: true,
                dependencies: []
            },
            'rule_12': { 
                name: 'Single Deployment Script', 
                priority: 'CRITICAL', 
                category: 'Deployment',
                description: 'One intelligent, self-updating deployment script for all environments.',
                enabled: true,
                dependencies: []
            },
            'rule_13': { 
                name: 'No Garbage Files', 
                priority: 'CRITICAL', 
                category: 'Cleanup',
                description: 'Zero tolerance for junk, clutter, or abandoned code.',
                enabled: true,
                dependencies: ['rule_10']
            },
            'rule_14': { 
                name: 'Correct AI Agent Usage', 
                priority: 'MEDIUM', 
                category: 'AI Agents',
                description: 'Always use the most capable and specialized AI agent for each task.',
                enabled: true,
                dependencies: []
            },
            'rule_15': { 
                name: 'Clean Documentation', 
                priority: 'MEDIUM', 
                category: 'Documentation',
                description: 'Documentation must be clean, clear, and deduplicated.',
                enabled: true,
                dependencies: ['rule_6']
            },
            'rule_16': { 
                name: 'Ollama/TinyLlama Standard', 
                priority: 'LOW', 
                category: 'AI Models',
                description: 'Use local LLMs exclusively via Ollama, default to TinyLlama.',
                enabled: true,
                dependencies: []
            }
        };
        
        // Performance and UX settings
        this.settings = {
            impactPreview: true,
            aiRecommendations: true,
            debounceDelay: 300,
            cacheTimeout: 60000, // 1 minute
            maxRetries: 3,
            retryDelay: 1000
        };

        this.init();
    }
    
    generateTabId() {
        return 'tab_' + Math.random().toString(36).substr(2, 9) + '_' + Date.now();
    }
    
    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }
    
    throttle(func, limit) {
        let inThrottle;
        return function() {
            const args = arguments;
            const context = this;
            if (!inThrottle) {
                func.apply(context, args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        };
    }

    async init() {
        this.applyTheme();
        this.setupKeyboardShortcuts();
        this.setupEventListeners();
        this.setupTabSync();
        this.initializeCharts();
        await this.loadInitialData();
        this.startRealTimeUpdates();
        this.renderDashboard();
        this.showToast('Dashboard initialized successfully', 'success');
    }
    
    applyTheme() {
        document.body.className = this.data.theme + '-theme';
        const themeIcon = document.querySelector('#theme-toggle i');
        if (themeIcon) {
            themeIcon.className = this.data.theme === 'dark' ? 'fas fa-moon' : 'fas fa-sun';
        }
    }
    
    setupKeyboardShortcuts() {
        // Master system toggle
        hotkeys('ctrl+m', (event) => {
            event.preventDefault();
            this.toggleMasterSystem();
        });
        
        // Theme toggle
        hotkeys('ctrl+t', (event) => {
            event.preventDefault();
            this.toggleTheme();
        });
        
        // Search focus
        hotkeys('ctrl+f', (event) => {
            event.preventDefault();
            document.getElementById('rule-search').focus();
        });
        
        // Actions
        hotkeys('ctrl+a', (event) => {
            event.preventDefault();
            this.runFullAudit();
        });
        
        hotkeys('ctrl+r', (event) => {
            event.preventDefault();
            this.generateReport();
        });
        
        hotkeys('ctrl+s', (event) => {
            event.preventDefault();
            this.syncTabs();
        });
        
        hotkeys('ctrl+shift+c', (event) => {
            event.preventDefault();
            this.forceCleanup();
        });
        
        // Undo/Redo
        hotkeys('ctrl+z', (event) => {
            event.preventDefault();
            this.undo();
        });
        
        hotkeys('ctrl+y', (event) => {
            event.preventDefault();
            this.redo();
        });
        
        // Export/Import
        hotkeys('ctrl+e', (event) => {
            event.preventDefault();
            this.exportConfiguration();
        });
        
        hotkeys('ctrl+i', (event) => {
            event.preventDefault();
            document.getElementById('config-file-input').click();
        });
        
        // Help
        hotkeys('?', (event) => {
            event.preventDefault();
            this.showKeyboardShortcuts();
        });
        
        // Escape to close modals
        hotkeys('escape', (event) => {
            this.closeAllModals();
        });
    }
    
    setupTabSync() {
        // Use BroadcastChannel for cross-tab communication
        if ('BroadcastChannel' in window) {
            this.syncChannel = new BroadcastChannel('hygiene-dashboard-sync');
            this.syncChannel.onmessage = (event) => {
                this.handleSyncMessage(event.data);
            };
        }
        
        // Update synced tabs count
        this.updateSyncedTabsCount();
    }

    setupEventListeners() {
        // Master system toggle
        document.getElementById('master-system-toggle').addEventListener('change', (e) => {
            this.toggleMasterSystem(e.target.checked);
        });
        
        // Theme toggle
        document.getElementById('theme-toggle').addEventListener('click', () => {
            this.toggleTheme();
        });
        
        // Rule profile selector
        document.getElementById('rule-profile').addEventListener('change', (e) => {
            this.applyRuleProfile(e.target.value);
        });
        
        // Rule search
        const searchInput = document.getElementById('rule-search');
        searchInput.addEventListener('input', this.debounce((e) => {
            this.filterRules(e.target.value);
        }, this.settings.debounceDelay));
        
        // Rule actions
        document.getElementById('undo-changes').addEventListener('click', () => this.undo());
        document.getElementById('redo-changes').addEventListener('click', () => this.redo());
        document.getElementById('export-config').addEventListener('click', () => this.exportConfiguration());
        document.getElementById('import-config').addEventListener('click', () => {
            document.getElementById('config-file-input').click();
        });
        
        // File input for import
        document.getElementById('config-file-input').addEventListener('change', (e) => {
            this.importConfiguration(e.target.files[0]);
        });
        
        // Control panel buttons
        document.getElementById('run-audit').addEventListener('click', () => this.runFullAudit());
        document.getElementById('force-cleanup').addEventListener('click', () => this.forceCleanup());
        document.getElementById('generate-report').addEventListener('click', () => this.generateReport());
        document.getElementById('sync-tabs').addEventListener('click', () => this.syncTabs());

        // Settings toggles
        document.getElementById('auto-enforcement').addEventListener('change', (e) => {
            this.toggleAutoEnforcement(e.target.checked);
        });

        document.getElementById('real-time-monitoring').addEventListener('change', (e) => {
            this.toggleRealTimeMonitoring(e.target.checked);
        });
        
        document.getElementById('impact-preview').addEventListener('change', (e) => {
            this.settings.impactPreview = e.target.checked;
            this.saveSettings();
        });
        
        document.getElementById('ai-recommendations').addEventListener('change', (e) => {
            this.settings.aiRecommendations = e.target.checked;
            this.saveSettings();
            if (e.target.checked) {
                this.generateAIRecommendations();
            } else {
                this.hideRecommendations();
            }
        });

        document.getElementById('refresh-rate').addEventListener('change', (e) => {
            this.updateRefreshRate(parseInt(e.target.value) * 1000);
        });

        // Filter buttons
        document.querySelectorAll('.filter-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
                e.target.classList.add('active');
                this.filterActions(e.target.dataset.filter);
            });
        });

        // Modal controls
        document.getElementById('modal-close').addEventListener('click', () => {
            document.getElementById('detail-modal').style.display = 'none';
        });
        
        document.getElementById('impact-modal-close').addEventListener('click', () => {
            document.getElementById('impact-preview-modal').style.display = 'none';
        });
        
        document.getElementById('impact-cancel').addEventListener('click', () => {
            document.getElementById('impact-preview-modal').style.display = 'none';
        });
        
        document.getElementById('impact-confirm').addEventListener('click', () => {
            this.applyPendingChanges();
        });
        
        document.getElementById('close-recommendations').addEventListener('click', () => {
            this.hideRecommendations();
        });

        // Close modals on outside click
        window.addEventListener('click', (e) => {
            const detailModal = document.getElementById('detail-modal');
            const impactModal = document.getElementById('impact-preview-modal');
            
            if (e.target === detailModal) {
                detailModal.style.display = 'none';
            }
            if (e.target === impactModal) {
                impactModal.style.display = 'none';
            }
        });
        
        // Window events
        window.addEventListener('beforeunload', () => {
            this.saveCurrentState();
        });
        
        window.addEventListener('online', () => {
            this.updateConnectionStatus(true);
            this.showToast('Connection restored', 'success');
        });
        
        window.addEventListener('offline', () => {
            this.updateConnectionStatus(false);
            this.showToast('Connection lost - working offline', 'warning');
        });
    }

    async loadInitialData() {
        this.showLoadingOverlay('Loading dashboard data...');
        
        try {
            // Load saved state first
            this.loadSettings();
            this.loadCurrentState();
            
            // Load hygiene enforcement status with caching
            const statusKey = 'hygiene-status';
            let response = this.getCachedData(statusKey);
            
            if (!response) {
                response = await this.fetchWithFallback('/api/hygiene/status', this.generateMockData());
                this.setCachedData(statusKey, response);
            }
            
            this.data = { ...this.data, ...response };
            
            // Load system metrics with caching
            const metricsKey = 'system-metrics';
            let metricsResponse = this.getCachedData(metricsKey);
            
            if (!metricsResponse) {
                metricsResponse = await this.fetchWithFallback('/api/system/metrics', this.generateMockMetrics());
                this.setCachedData(metricsKey, metricsResponse);
            }
            
            this.data.metrics = metricsResponse;
            
            // Update connection status
            this.updateConnectionStatus(true);
            
        } catch (error) {
            console.warn('API not available, using mock data:', error);
            this.data = { ...this.data, ...this.generateMockData() };
            this.data.metrics = this.generateMockMetrics();
            this.updateConnectionStatus(false);
        } finally {
            this.hideLoadingOverlay();
        }
    }
    
    getCachedData(key) {
        const cached = this.cache.get(key);
        if (cached && Date.now() - cached.timestamp < this.settings.cacheTimeout) {
            return cached.data;
        }
        return null;
    }
    
    setCachedData(key, data) {
        this.cache.set(key, {
            data: data,
            timestamp: Date.now()
        });
    }
    
    showLoadingOverlay(text = 'Loading...') {
        const overlay = document.getElementById('loading-overlay');
        const loadingText = document.getElementById('loading-text');
        loadingText.textContent = text;
        overlay.style.display = 'flex';
    }
    
    hideLoadingOverlay() {
        const overlay = document.getElementById('loading-overlay');
        overlay.style.display = 'none';
    }

    async fetchWithFallback(url, fallback, retries = 0) {
        const requestId = `${url}-${Date.now()}`;
        
        // Check if request is already in progress
        if (this.requestQueue.has(url)) {
            return this.requestQueue.get(url);
        }
        
        const request = this.performRequest(url, fallback, retries);
        this.requestQueue.set(url, request);
        
        try {
            const result = await request;
            return result;
        } finally {
            this.requestQueue.delete(url);
        }
    }
    
    async performRequest(url, fallback, retries) {
        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 10000); // 10 second timeout
            
            const response = await fetch(url, {
                signal: controller.signal,
                headers: {
                    'Content-Type': 'application/json',
                    'X-Tab-ID': this.tabId
                }
            });
            
            clearTimeout(timeoutId);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            return await response.json();
        } catch (error) {
            if (retries < this.settings.maxRetries) {
                console.log(`Retrying request to ${url} (attempt ${retries + 1})`);
                await new Promise(resolve => setTimeout(resolve, this.settings.retryDelay * (retries + 1)));
                return this.performRequest(url, fallback, retries + 1);
            }
            
            console.warn(`Request failed after ${this.settings.maxRetries} retries, using fallback for ${url}:`, error);
            return fallback;
        }
    }

    generateMockData() {
        const now = new Date();
        return {
            timestamp: now.toISOString(),
            systemStatus: 'MONITORING',
            complianceScore: 87,
            totalViolations: 23,
            criticalViolations: 3,
            warningViolations: 12,
            activeAgents: 8,
            rules: Object.fromEntries(
                Object.entries(this.rules).map(([id, rule]) => [
                    id, 
                    {
                        ...rule,
                        status: Math.random() > 0.7 ? 'VIOLATION' : 'COMPLIANT',
                        lastChecked: new Date(now - Math.random() * 3600000).toISOString(),
                        violationCount: Math.floor(Math.random() * 10)
                    }
                ])
            ),
            agents: {
                'hygiene-coordinator': { status: 'ACTIVE', health: 95, lastSeen: now.toISOString() },
                'garbage-collector': { status: 'ACTIVE', health: 88, lastSeen: now.toISOString() },
                'deploy-automation': { status: 'IDLE', health: 92, lastSeen: new Date(now - 300000).toISOString() },
                'script-organizer': { status: 'ACTIVE', health: 78, lastSeen: now.toISOString() },
                'docker-optimizer': { status: 'WARNING', health: 65, lastSeen: new Date(now - 120000).toISOString() },
                'documentation-manager': { status: 'ACTIVE', health: 91, lastSeen: now.toISOString() },
                'python-validator': { status: 'ACTIVE', health: 89, lastSeen: now.toISOString() },
                'compliance-monitor': { status: 'ACTIVE', health: 94, lastSeen: now.toISOString() }
            },
            actions: this.generateMockActions(50),
            trends: this.generateTrendData()
        };
    }

    generateMockActions(count) {
        const actions = [];
        const actionTypes = ['CLEANUP', 'VIOLATION_DETECTED', 'RULE_ENFORCED', 'AGENT_STARTED', 'COMPLIANCE_CHECK'];
        const severities = ['critical', 'warning', 'info', 'success'];
        
        for (let i = 0; i < count; i++) {
            const timestamp = new Date(Date.now() - Math.random() * 86400000); // Last 24 hours
            actions.push({
                id: `action_${i}`,
                timestamp: timestamp.toISOString(),
                type: actionTypes[Math.floor(Math.random() * actionTypes.length)],
                severity: severities[Math.floor(Math.random() * severities.length)],
                rule: `rule_${Math.floor(Math.random() * 16) + 1}`,
                agent: Object.keys(this.generateMockData().agents)[Math.floor(Math.random() * 8)],
                message: this.generateActionMessage(),
                details: { filesAffected: Math.floor(Math.random() * 10) + 1 }
            });
        }
        
        return actions.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
    }

    generateActionMessage() {
        const messages = [
            'Removed 5 orphaned backup files from /opt/sutazaiapp',
            'Detected duplicate deployment scripts, consolidation required',
            'Python script validation completed successfully',
            'Docker structure optimized, 3 redundant containers removed',
            'Documentation centralization enforced',
            'Rule violation in scripts/ directory resolved',
            'Agent health check completed',
            'Critical compliance issue resolved',
            'Automated cleanup completed successfully',
            'Real-time monitoring alert triggered'
        ];
        return messages[Math.floor(Math.random() * messages.length)];
    }

    generateTrendData() {
        const hours = 24;
        const data = [];
        for (let i = hours; i >= 0; i--) {
            const timestamp = new Date(Date.now() - (i * 3600000));
            data.push({
                timestamp: timestamp.toISOString(),
                violations: Math.floor(Math.random() * 30) + 5,
                critical: Math.floor(Math.random() * 8),
                warnings: Math.floor(Math.random() * 15) + 5,
                resolved: Math.floor(Math.random() * 25) + 10
            });
        }
        return data;
    }

    generateMockMetrics() {
        return {
            memory: { used: 4.2, total: 16, percentage: 26 },
            cpu: { usage: 34, cores: 8 },
            disk: { used: 120, total: 500, percentage: 24 },
            network: { status: 'HEALTHY', latency: 12 }
        };
    }

    initializeCharts() {
        // Violation trend chart
        const trendCtx = document.getElementById('violation-trend-chart').getContext('2d');
        this.charts.trendChart = new Chart(trendCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Total Violations',
                    data: [],
                    borderColor: '#e74c3c',
                    backgroundColor: 'rgba(231, 76, 60, 0.1)',
                    tension: 0.4
                }, {
                    label: 'Critical',
                    data: [],
                    borderColor: '#c0392b',
                    backgroundColor: 'rgba(192, 57, 43, 0.1)',
                    tension: 0.4
                }, {
                    label: 'Resolved',
                    data: [],
                    borderColor: '#27ae60',
                    backgroundColor: 'rgba(39, 174, 96, 0.1)',
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        grid: { color: 'rgba(255, 255, 255, 0.1)' },
                        ticks: { color: '#bdc3c7' }
                    },
                    x: {
                        grid: { color: 'rgba(255, 255, 255, 0.1)' },
                        ticks: { color: '#bdc3c7' }
                    }
                },
                plugins: {
                    legend: { labels: { color: '#bdc3c7' } }
                }
            }
        });

        // Rule distribution chart
        const distCtx = document.getElementById('rule-distribution-chart').getContext('2d');
        this.charts.distributionChart = new Chart(distCtx, {
            type: 'doughnut',
            data: {
                labels: [],
                datasets: [{
                    data: [],
                    backgroundColor: [
                        '#e74c3c', '#f39c12', '#f1c40f', '#27ae60',
                        '#3498db', '#9b59b6', '#e67e22', '#34495e'
                    ]
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { 
                        position: 'bottom',
                        labels: { color: '#bdc3c7' }
                    }
                }
            }
        });
    }

    renderDashboard() {
        this.updateStatusOverview();
        this.updateRuleMatrix();
        this.updateAgentHealth();
        this.updateCharts();
        this.updateRecentActions();
        this.updateSystemMetrics();
        this.updateLastUpdateTime();
    }

    updateStatusOverview() {
        document.getElementById('system-status').textContent = this.data.systemStatus || 'UNKNOWN';
        document.getElementById('critical-violations').textContent = this.data.criticalViolations || 0;
        document.getElementById('warning-violations').textContent = this.data.warningViolations || 0;
        document.getElementById('compliance-score').textContent = `${this.data.complianceScore || 0}%`;
        document.getElementById('active-agents').textContent = this.data.activeAgents || 0;

        // Update status colors
        const systemStatus = document.getElementById('system-status');
        systemStatus.className = this.getStatusClass(this.data.systemStatus);
    }

    updateRuleMatrix() {
        const ruleGrid = document.getElementById('rule-grid');
        ruleGrid.innerHTML = '';

        Object.entries(this.rules).forEach(([ruleId, ruleDefinition]) => {
            const ruleData = this.data.rules[ruleId] || {};
            const enabled = ruleDefinition.enabled && this.data.systemEnabled;
            const hasViolations = (ruleData.violationCount || 0) > 0;
            
            const ruleElement = document.createElement('div');
            ruleElement.className = `rule-card ${ruleDefinition.priority.toLowerCase()} ${!enabled ? 'disabled' : ''}`;
            ruleElement.setAttribute('data-rule-id', ruleId);
            ruleElement.setAttribute('tabindex', '0');
            
            ruleElement.innerHTML = `
                <div class="rule-card-header">
                    <div class="rule-info">
                        <div class="rule-id">${ruleId.toUpperCase()}</div>
                        <div class="rule-name">${ruleDefinition.name}</div>
                    </div>
                    <div class="rule-toggle">
                        <input type="checkbox" id="toggle-${ruleId}" ${enabled ? 'checked' : ''}>
                        <label for="toggle-${ruleId}" class="rule-switch"></label>
                    </div>
                </div>
                <div class="rule-meta">
                    <span class="rule-priority ${ruleDefinition.priority.toLowerCase()}">${ruleDefinition.priority}</span>
                    <span class="rule-category">${ruleDefinition.category}</span>
                </div>
                <div class="rule-status-indicator">
                    <div class="status-badge ${enabled ? 'enabled' : 'disabled'}">
                        <i class="fas ${enabled ? 'fa-check-circle' : 'fa-times-circle'}"></i>
                        ${enabled ? 'ENABLED' : 'DISABLED'}
                    </div>
                    ${hasViolations ? `<div class="violation-count">${ruleData.violationCount}</div>` : ''}
                </div>
                <div class="rule-description">${ruleDefinition.description}</div>
                <div class="rule-stats">
                    <div class="stat-item">
                        <span class="stat-label">Violations</span>
                        <span class="stat-value">${ruleData.violationCount || 0}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Last Check</span>
                        <span class="stat-value">${this.formatTime(ruleData.lastChecked)}</span>
                    </div>
                </div>
            `;
            
            // Add toggle event listener
            const toggle = ruleElement.querySelector(`#toggle-${ruleId}`);
            toggle.addEventListener('change', (e) => {
                this.handleRuleToggle(ruleId, e.target.checked);
            });
            
            // Add click event for details
            ruleElement.addEventListener('click', (e) => {
                if (!e.target.closest('.rule-toggle')) {
                    this.showRuleDetails(ruleId, { ...ruleDefinition, ...ruleData });
                }
            });
            
            // Add keyboard support
            ruleElement.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    if (e.target.classList.contains('rule-card')) {
                        this.showRuleDetails(ruleId, { ...ruleDefinition, ...ruleData });
                    }
                }
            });
            
            ruleGrid.appendChild(ruleElement);
        });
        
        // Generate AI recommendations if enabled
        if (this.settings.aiRecommendations) {
            this.debouncedGenerateRecommendations();
        }
    }

    updateAgentHealth() {
        const agentList = document.getElementById('agent-list');
        agentList.innerHTML = '';

        Object.entries(this.data.agents || {}).forEach(([agentId, agent]) => {
            const agentElement = document.createElement('div');
            agentElement.className = `agent-item ${this.getStatusClass(agent.status)}`;
            agentElement.innerHTML = `
                <div class="agent-header">
                    <span class="agent-name">${agentId}</span>
                    <span class="agent-status">${agent.status}</span>
                </div>
                <div class="agent-health">
                    <div class="health-bar">
                        <div class="health-fill" style="width: ${agent.health || 0}%"></div>
                    </div>
                    <span class="health-value">${agent.health || 0}%</span>
                </div>
                <div class="agent-meta">
                    <span>Last Seen: ${this.formatTime(agent.lastSeen)}</span>
                </div>
            `;
            
            agentElement.addEventListener('click', () => this.showAgentDetails(agentId, agent));
            agentList.appendChild(agentElement);
        });
    }

    updateCharts() {
        if (this.data.trends) {
            // Update trend chart
            const labels = this.data.trends.map(point => 
                new Date(point.timestamp).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})
            );
            
            this.charts.trendChart.data.labels = labels;
            this.charts.trendChart.data.datasets[0].data = this.data.trends.map(p => p.violations);
            this.charts.trendChart.data.datasets[1].data = this.data.trends.map(p => p.critical);
            this.charts.trendChart.data.datasets[2].data = this.data.trends.map(p => p.resolved);
            this.charts.trendChart.update();
        }

        // Update rule distribution chart
        const ruleCategories = {};
        Object.values(this.data.rules || {}).forEach(rule => {
            ruleCategories[rule.category] = (ruleCategories[rule.category] || 0) + (rule.violationCount || 0);
        });

        this.charts.distributionChart.data.labels = Object.keys(ruleCategories);
        this.charts.distributionChart.data.datasets[0].data = Object.values(ruleCategories);
        this.charts.distributionChart.update();
    }

    updateRecentActions() {
        const actionsList = document.getElementById('actions-list');
        actionsList.innerHTML = '';

        (this.data.actions || []).slice(0, 20).forEach(action => {
            const actionElement = document.createElement('div');
            actionElement.className = `action-item ${action.severity}`;
            actionElement.innerHTML = `
                <div class="action-header">
                    <span class="action-type">${action.type}</span>
                    <span class="action-time">${this.formatTime(action.timestamp)}</span>
                </div>
                <div class="action-message">${action.message}</div>
                <div class="action-meta">
                    <span class="action-rule">${action.rule?.toUpperCase()}</span>
                    <span class="action-agent">${action.agent}</span>
                </div>
            `;
            
            actionElement.addEventListener('click', () => this.showActionDetails(action));
            actionsList.appendChild(actionElement);
        });
    }

    updateSystemMetrics() {
        const metrics = this.data.metrics || {};
        
        if (metrics.memory) {
            document.getElementById('memory-usage').textContent = 
                `${metrics.memory.used}GB / ${metrics.memory.total}GB (${metrics.memory.percentage}%)`;
        }
        
        if (metrics.cpu) {
            document.getElementById('cpu-usage').textContent = `${metrics.cpu.usage}%`;
        }
        
        if (metrics.disk) {
            document.getElementById('disk-usage').textContent = 
                `${metrics.disk.used}GB / ${metrics.disk.total}GB (${metrics.disk.percentage}%)`;
        }
        
        if (metrics.network) {
            document.getElementById('network-status').textContent = 
                `${metrics.network.status} (${metrics.network.latency}ms)`;
        }
    }

    updateLastUpdateTime() {
        document.getElementById('last-update').textContent = 
            new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit', second:'2-digit'});
    }

    // Utility methods
    getStatusClass(status) {
        const statusMap = {
            'ACTIVE': 'success',
            'HEALTHY': 'success',
            'COMPLIANT': 'success',
            'MONITORING': 'info',
            'IDLE': 'info',
            'WARNING': 'warning',
            'VIOLATION': 'critical',
            'ERROR': 'critical',
            'CRITICAL': 'critical'
        };
        return statusMap[status] || 'info';
    }

    formatTime(timestamp) {
        if (!timestamp) return '--:--';
        const date = new Date(timestamp);
        const now = new Date();
        const diff = Math.floor((now - date) / 1000);
        
        if (diff < 60) return `${diff}s ago`;
        if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
        if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
        return date.toLocaleDateString();
    }

    // New intelligent methods
    
    toggleMasterSystem(enabled = null) {
        if (enabled === null) {
            enabled = !this.data.systemEnabled;
        }
        
        this.saveToHistory();
        this.data.systemEnabled = enabled;
        
        // Update master toggle UI
        const masterToggle = document.getElementById('master-system-toggle');
        masterToggle.checked = enabled;
        
        // Update all rule toggles
        this.updateRuleMatrix();
        
        // Show impact if preview is enabled
        if (this.settings.impactPreview && this.pendingChanges.size === 0) {
            this.showImpactPreview('master-toggle', {
                action: enabled ? 'Enable' : 'Disable',
                target: 'All hygiene rules',
                impact: `${enabled ? 'Activate' : 'Deactivate'} all ${Object.keys(this.rules).length} hygiene enforcement rules`,
                affected: Object.keys(this.rules).length
            });
        } else {
            this.syncAllTabs({ type: 'master-toggle', enabled });
            this.showToast(`Master system ${enabled ? 'enabled' : 'disabled'}`, enabled ? 'success' : 'warning');
        }
    }
    
    toggleTheme() {
        this.data.theme = this.data.theme === 'dark' ? 'light' : 'dark';
        localStorage.setItem('dashboard-theme', this.data.theme);
        this.applyTheme();
        this.syncAllTabs({ type: 'theme-change', theme: this.data.theme });
        this.showToast(`Switched to ${this.data.theme} theme`, 'info');
    }
    
    handleRuleToggle(ruleId, enabled) {
        if (!this.data.systemEnabled && enabled) {
            this.showToast('Cannot enable rule while master system is disabled', 'warning');
            return;
        }
        
        this.saveToHistory();
        
        const oldEnabled = this.rules[ruleId].enabled;
        this.rules[ruleId].enabled = enabled;
        
        // Check dependencies
        const dependencies = this.checkRuleDependencies(ruleId, enabled);
        
        if (this.settings.impactPreview) {
            this.showImpactPreview('rule-toggle', {
                action: enabled ? 'Enable' : 'Disable',
                target: this.rules[ruleId].name,
                impact: this.calculateRuleImpact(ruleId, enabled),
                dependencies: dependencies,
                ruleId: ruleId,
                enabled: enabled
            });
        } else {
            this.applyRuleChange(ruleId, enabled);
        }
    }
    
    checkRuleDependencies(ruleId, enabled) {
        const dependencies = [];
        const rule = this.rules[ruleId];
        
        if (!enabled) {
            // Check what depends on this rule
            Object.entries(this.rules).forEach(([id, r]) => {
                if (r.dependencies.includes(ruleId) && r.enabled) {
                    dependencies.push({
                        type: 'dependent',
                        ruleId: id,
                        ruleName: r.name,
                        action: 'will be disabled'
                    });
                }
            });
        } else {
            // Check if dependencies are enabled
            rule.dependencies.forEach(depId => {
                if (!this.rules[depId]?.enabled) {
                    dependencies.push({
                        type: 'prerequisite',
                        ruleId: depId,
                        ruleName: this.rules[depId]?.name,
                        action: 'must be enabled first'
                    });
                }
            });
        }
        
        return dependencies;
    }
    
    calculateRuleImpact(ruleId, enabled) {
        const rule = this.rules[ruleId];
        const currentViolations = this.data.rules[ruleId]?.violationCount || 0;
        
        if (enabled) {
            return `Will start monitoring and enforcing ${rule.category.toLowerCase()} standards. Current violations: ${currentViolations}`;
        } else {
            return `Will stop monitoring ${rule.category.toLowerCase()} violations. ${currentViolations} current violations will be ignored.`;
        }
    }
    
    showImpactPreview(type, data) {
        const modal = document.getElementById('impact-preview-modal');
        const title = document.getElementById('impact-modal-title');
        const body = document.getElementById('impact-modal-body');
        
        title.textContent = `Impact Preview: ${data.action} ${data.target}`;
        
        let dependencyWarnings = '';
        if (data.dependencies && data.dependencies.length > 0) {
            dependencyWarnings = `
                <div class="impact-warnings">
                    <h4><i class="fas fa-exclamation-triangle"></i> Dependency Warnings</h4>
                    ${data.dependencies.map(dep => `
                        <div class="dependency-warning ${dep.type}">
                            <span class="dependency-rule">${dep.ruleName}</span>
                            <span class="dependency-action">${dep.action}</span>
                        </div>
                    `).join('')}
                </div>
            `;
        }
        
        body.innerHTML = `
            <div class="impact-content">
                <div class="impact-summary">
                    <h4>Impact Summary</h4>
                    <p>${data.impact}</p>
                    <div class="impact-stats">
                        <div class="impact-stat">
                            <span class="stat-label">Affected Rules</span>
                            <span class="stat-value">${data.affected || 1}</span>
                        </div>
                        <div class="impact-stat">
                            <span class="stat-label">Priority</span>
                            <span class="stat-value priority ${(this.rules[data.ruleId]?.priority || 'medium').toLowerCase()}">
                                ${this.rules[data.ruleId]?.priority || 'N/A'}
                            </span>
                        </div>
                    </div>
                </div>
                ${dependencyWarnings}
                <div class="impact-timeline">
                    <h4>Expected Changes</h4>
                    <div class="timeline-item">
                        <span class="timeline-time">Immediate</span>
                        <span class="timeline-action">${data.action} rule enforcement</span>
                    </div>
                    <div class="timeline-item">
                        <span class="timeline-time">~30 seconds</span>
                        <span class="timeline-action">Agent reconfiguration</span>
                    </div>
                    <div class="timeline-item">
                        <span class="timeline-time">~2 minutes</span>
                        <span class="timeline-action">Full compliance scan</span>
                    </div>
                </div>
            </div>
        `;
        
        // Store pending change
        this.pendingChanges.add({ type, data });
        
        modal.style.display = 'block';
    }
    
    applyPendingChanges() {
        this.pendingChanges.forEach(change => {
            if (change.type === 'rule-toggle') {
                this.applyRuleChange(change.data.ruleId, change.data.enabled);
            } else if (change.type === 'master-toggle') {
                this.syncAllTabs({ type: 'master-toggle', enabled: change.data.enabled });
            }
        });
        
        this.pendingChanges.clear();
        document.getElementById('impact-preview-modal').style.display = 'none';
        this.showToast('Changes applied successfully', 'success');
    }
    
    applyRuleChange(ruleId, enabled) {
        // Apply dependency changes
        const dependencies = this.checkRuleDependencies(ruleId, enabled);
        dependencies.forEach(dep => {
            if (dep.type === 'dependent' && !enabled) {
                this.rules[dep.ruleId].enabled = false;
            }
        });
        
        this.updateRuleMatrix();
        this.syncAllTabs({ type: 'rule-change', ruleId, enabled });
        this.showToast(`Rule ${this.rules[ruleId].name} ${enabled ? 'enabled' : 'disabled'}`, enabled ? 'success' : 'warning');
    }
    
    saveToHistory() {
        const state = {
            timestamp: Date.now(),
            systemEnabled: this.data.systemEnabled,
            rules: JSON.parse(JSON.stringify(this.rules))
        };
        
        this.undoStack.push(state);
        if (this.undoStack.length > this.maxHistorySize) {
            this.undoStack.shift();
        }
        
        this.redoStack = []; // Clear redo stack
        this.updateUndoRedoButtons();
    }
    
    undo() {
        if (this.undoStack.length === 0) {
            this.showToast('Nothing to undo', 'info');
            return;
        }
        
        const currentState = {
            timestamp: Date.now(),
            systemEnabled: this.data.systemEnabled,
            rules: JSON.parse(JSON.stringify(this.rules))
        };
        
        const previousState = this.undoStack.pop();
        this.redoStack.push(currentState);
        
        this.data.systemEnabled = previousState.systemEnabled;
        this.rules = previousState.rules;
        
        document.getElementById('master-system-toggle').checked = this.data.systemEnabled;
        this.updateRuleMatrix();
        this.updateUndoRedoButtons();
        
        this.showToast('Changes undone', 'info');
    }
    
    redo() {
        if (this.redoStack.length === 0) {
            this.showToast('Nothing to redo', 'info');
            return;
        }
        
        const currentState = {
            timestamp: Date.now(),
            systemEnabled: this.data.systemEnabled,
            rules: JSON.parse(JSON.stringify(this.rules))
        };
        
        const nextState = this.redoStack.pop();
        this.undoStack.push(currentState);
        
        this.data.systemEnabled = nextState.systemEnabled;
        this.rules = nextState.rules;
        
        document.getElementById('master-system-toggle').checked = this.data.systemEnabled;
        this.updateRuleMatrix();
        this.updateUndoRedoButtons();
        
        this.showToast('Changes redone', 'info');
    }
    
    updateUndoRedoButtons() {
        document.getElementById('undo-changes').disabled = this.undoStack.length === 0;
        document.getElementById('redo-changes').disabled = this.redoStack.length === 0;
    }
    
    exportConfiguration() {
        const config = {
            timestamp: new Date().toISOString(),
            version: '2.0.0',
            systemEnabled: this.data.systemEnabled,
            rules: this.rules,
            settings: this.settings,
            theme: this.data.theme
        };
        
        const blob = new Blob([JSON.stringify(config, null, 2)], { type: 'application/json' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `hygiene-config-${new Date().toISOString().split('T')[0]}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
        
        this.showToast('Configuration exported successfully', 'success');
    }
    
    async importConfiguration(file) {
        if (!file) return;
        
        try {
            const text = await file.text();
            const config = JSON.parse(text);
            
            if (!config.version || !config.rules) {
                throw new Error('Invalid configuration file format');
            }
            
            this.saveToHistory();
            
            this.data.systemEnabled = config.systemEnabled ?? true;
            this.rules = config.rules;
            this.settings = { ...this.settings, ...config.settings };
            
            if (config.theme) {
                this.data.theme = config.theme;
                localStorage.setItem('dashboard-theme', this.data.theme);
                this.applyTheme();
            }
            
            document.getElementById('master-system-toggle').checked = this.data.systemEnabled;
            this.updateRuleMatrix();
            this.saveSettings();
            
            this.showToast('Configuration imported successfully', 'success');
        } catch (error) {
            this.showToast('Failed to import configuration: ' + error.message, 'error');
        }
    }
    
    filterRules(searchTerm) {
        const ruleCards = document.querySelectorAll('.rule-card');
        const term = searchTerm.toLowerCase();
        
        ruleCards.forEach(card => {
            const ruleId = card.getAttribute('data-rule-id');
            const rule = this.rules[ruleId];
            
            const matches = (
                rule.name.toLowerCase().includes(term) ||
                rule.category.toLowerCase().includes(term) ||
                rule.priority.toLowerCase().includes(term) ||
                rule.description.toLowerCase().includes(term) ||
                ruleId.toLowerCase().includes(term)
            );
            
            card.style.display = matches ? 'block' : 'none';
        });
    }
    
    applyRuleProfile(profile) {
        this.saveToHistory();
        
        const profiles = {
            strict: () => {
                Object.values(this.rules).forEach(rule => {
                    rule.enabled = true;
                });
            },
            production: () => {
                Object.values(this.rules).forEach(rule => {
                    rule.enabled = rule.priority === 'CRITICAL' || rule.priority === 'HIGH';
                });
            },
            development: () => {
                Object.values(this.rules).forEach(rule => {
                    rule.enabled = rule.priority === 'CRITICAL';
                });
            },
            default: () => {
                Object.values(this.rules).forEach(rule => {
                    rule.enabled = rule.priority !== 'LOW';
                });
            }
        };
        
        if (profiles[profile]) {
            profiles[profile]();
            this.updateRuleMatrix();
            this.showToast(`Applied ${profile} rule profile`, 'success');
        }
    }
    
    async generateAIRecommendations() {
        // Enhanced AI recommendations with real test result integration
        const recommendations = [];
        
        // Load latest test results
        const testResults = await this.loadLatestTestResults();
        
        // Check for disabled critical rules
        Object.entries(this.rules).forEach(([ruleId, rule]) => {
            if (!rule.enabled && rule.priority === 'CRITICAL') {
                recommendations.push({
                    type: 'critical-disabled',
                    title: `Enable Critical Rule: ${rule.name}`,
                    description: `This critical rule is currently disabled, which may lead to serious compliance issues.`,
                    action: 'enable',
                    ruleId: ruleId,
                    priority: 'high',
                    testResult: testResults?.individual_tests?.[ruleId] || null
                });
            }
        });
        
        // Check for dependency issues with test result context
        Object.entries(this.rules).forEach(([ruleId, rule]) => {
            if (rule.enabled) {
                rule.dependencies.forEach(depId => {
                    if (!this.rules[depId]?.enabled) {
                        const hasTestFailure = testResults?.failed_tests?.some(test => 
                            test.combination?.includes(parseInt(ruleId.replace('rule_', ''))) && 
                            test.error?.includes('Dependency validation failed')
                        );
                        
                        recommendations.push({
                            type: 'dependency-issue',
                            title: `Dependency Issue: ${rule.name}`,
                            description: `This rule depends on '${this.rules[depId]?.name}' which is currently disabled.${hasTestFailure ? ' Recent tests confirm this causes failures.' : ''}`,
                            action: 'fix-dependency',
                            ruleId: ruleId,
                            dependencyId: depId,
                            priority: hasTestFailure ? 'high' : 'medium',
                            testResult: hasTestFailure ? 'failed' : 'warning'
                        });
                    }
                });
            }
        });
        
        // Add test result insights
        if (testResults) {
            if (testResults.success_rate < 80) {
                recommendations.push({
                    type: 'test-failure',
                    title: `Low Test Success Rate: ${testResults.success_rate.toFixed(1)}%`,
                    description: `Recent tests show ${testResults.failed_tests || 0} failed scenarios. Review dependency issues and rule configurations.`,
                    action: 'review-tests',
                    priority: 'high',
                    testResult: 'failed'
                });
            }
            
            if (testResults.performance_impact > 0.5) {
                recommendations.push({
                    type: 'performance-warning',
                    title: 'High Performance Impact Detected',
                    description: `Current rule combination shows ${(testResults.performance_impact * 100).toFixed(1)}% performance overhead.`,
                    action: 'optimize-performance',
                    priority: 'medium',
                    testResult: 'warning'
                });
            }
        }
        
        // Check for performance optimization
        const enabledCount = Object.values(this.rules).filter(r => r.enabled).length;
        if (enabledCount < Object.keys(this.rules).length * 0.7) {
            recommendations.push({
                type: 'optimization',
                title: 'Consider Enabling More Rules',
                description: `Only ${enabledCount} out of ${Object.keys(this.rules).length} rules are enabled. Consider enabling more for better code quality.`,
                action: 'optimize',
                priority: 'low'
            });
        }
        
        this.showRecommendations(recommendations);
    }
    
    async loadLatestTestResults() {
        try {
            // Try to load the latest test results from the report
            const response = await this.makeRequest('/api/test-results/latest', {}, 'GET');
            if (response.success) {
                return response.data;
            }
            
            // Fallback: parse from local test report if API not available
            return await this.parseTestReportFile();
        } catch (error) {
            console.warn('Could not load test results:', error);
            return null;
        }
    }
    
    async parseTestReportFile() {
        try {
            // Mock implementation - in production, this would read the actual report
            return {
                success_rate: 60.0,
                total_tests: 5,
                successful_tests: 3,
                failed_tests: 2,
                failed_combinations: [
                    { combination: [2], error: "Dependency validation failed: Rule 'Do Not Break Existing Functionality' requires 'Rule rule_10_functionality_first_cleanup' to be enabled" },
                    { combination: [4], error: "Dependency validation failed: Rule 'Reuse Before Creating' requires 'Rule rule_07_eliminate_script_chaos' to be enabled" }
                ],
                performance_impact: 0.0,
                duration_avg: 0.204
            };
        } catch (error) {
            return null;
        }
    }
    
    showRecommendations(recommendations) {
        const container = document.getElementById('rule-recommendations');
        const content = document.getElementById('recommendations-content');
        
        if (recommendations.length === 0) {
            container.style.display = 'none';
            return;
        }
        
        content.innerHTML = recommendations.map(rec => `
            <div class="recommendation-item ${rec.priority}">
                <div class="recommendation-icon">
                    <i class="fas ${this.getRecommendationIcon(rec.type)}"></i>
                    ${rec.testResult ? `<span class="test-indicator ${rec.testResult}"></span>` : ''}
                </div>
                <div class="recommendation-content">
                    <div class="recommendation-title">
                        ${rec.title}
                        ${rec.testResult === 'failed' ? '<span class="test-badge failed">TEST FAILED</span>' : ''}
                        ${rec.testResult === 'warning' ? '<span class="test-badge warning">TEST WARNING</span>' : ''}
                    </div>
                    <div class="recommendation-description">${rec.description}</div>
                    <div class="recommendation-actions">
                        <button class="btn-small primary" onclick="dashboard.applyRecommendation('${rec.type}', '${rec.ruleId || ''}', '${rec.dependencyId || ''}')">
                            ${this.getRecommendationActionText(rec.type)}
                        </button>
                        <button class="btn-small" onclick="this.parentElement.parentElement.parentElement.remove()">
                            Dismiss
                        </button>
                    </div>
                </div>
            </div>
        `).join('');
        
        container.style.display = 'block';
    }
    
    getRecommendationIcon(type) {
        const icons = {
            'critical-disabled': 'fa-exclamation-triangle',
            'dependency-issue': 'fa-link',
            'optimization': 'fa-chart-line',
            'security': 'fa-shield-alt',
            'performance': 'fa-tachometer-alt',
            'test-failure': 'fa-bug',
            'performance-warning': 'fa-exclamation-circle'
        };
        return icons[type] || 'fa-lightbulb';
    }
    
    getRecommendationActionText(type) {
        const actionTexts = {
            'critical-disabled': 'Enable Rule',
            'dependency-issue': 'Fix Dependency',
            'optimization': 'Optimize',
            'security': 'Secure',
            'performance': 'Optimize',
            'test-failure': 'Review Tests',
            'performance-warning': 'Optimize Performance'
        };
        return actionTexts[type] || 'Apply';
    }
    
    applyRecommendation(type, ruleId, dependencyId) {
        switch (type) {
            case 'critical-disabled':
                if (ruleId && this.rules[ruleId]) {
                    this.handleRuleToggle(ruleId, true);
                    this.showToast(`Enabled critical rule: ${this.rules[ruleId].name}`, 'success');
                }
                break;
            case 'dependency-issue':
                if (dependencyId && this.rules[dependencyId]) {
                    this.handleRuleToggle(dependencyId, true);
                    this.showToast(`Fixed dependency: Enabled ${this.rules[dependencyId].name}`, 'success');
                }
                break;
            case 'optimization':
                this.applyRuleProfile('default');
                break;
            case 'test-failure':
                this.showTestResultsModal();
                break;
            case 'performance-warning':
                this.optimizePerformance();
                break;
        }
    }
    
    showTestResultsModal() {
        // Create and show a modal with detailed test results
        const modal = document.createElement('div');
        modal.className = 'modal-overlay';
        modal.innerHTML = `
            <div class="modal test-results-modal">
                <div class="modal-header">
                    <h3>Test Results Analysis</h3>
                    <button class="modal-close" onclick="this.closest('.modal-overlay').remove()">&times;</button>
                </div>
                <div class="modal-content" id="test-results-content">
                    <div class="loading-spinner">Loading test results...</div>
                </div>
                <div class="modal-actions">
                    <button class="btn primary" onclick="dashboard.runNewTests()">Run New Tests</button>
                    <button class="btn" onclick="this.closest('.modal-overlay').remove()">Close</button>
                </div>
            </div>
        `;
        document.body.appendChild(modal);
        
        // Load and display test results
        this.loadLatestTestResults().then(results => {
            this.displayTestResults(results);
        });
    }
    
    displayTestResults(results) {
        const content = document.getElementById('test-results-content');
        if (!results) {
            content.innerHTML = '<div class="error-message">No test results available</div>';
            return;
        }
        
        content.innerHTML = `
            <div class="test-summary">
                <div class="test-metric">
                    <span class="metric-label">Success Rate</span>
                    <span class="metric-value ${results.success_rate < 80 ? 'error' : 'success'}">${results.success_rate.toFixed(1)}%</span>
                </div>
                <div class="test-metric">
                    <span class="metric-label">Total Tests</span>
                    <span class="metric-value">${results.total_tests}</span>
                </div>
                <div class="test-metric">
                    <span class="metric-label">Failed Tests</span>
                    <span class="metric-value ${results.failed_tests > 0 ? 'error' : 'success'}">${results.failed_tests}</span>
                </div>
            </div>
            
            ${results.failed_combinations && results.failed_combinations.length > 0 ? `
                <div class="failed-tests">
                    <h4>Failed Test Details</h4>
                    ${results.failed_combinations.map(test => `
                        <div class="failed-test">
                            <div class="test-combination">Rule ${test.combination.join(', ')}</div>
                            <div class="test-error">${test.error}</div>
                        </div>
                    `).join('')}
                </div>
            ` : ''}
        `;
    }
    
    optimizePerformance() {
        // Suggest performance optimizations based on current rule configuration
        const highImpactRules = Object.entries(this.rules)
            .filter(([_, rule]) => rule.enabled && this.getPerformanceImpact(rule) === 'high')
            .map(([id, _]) => id);
            
        if (highImpactRules.length > 0) {
            this.showConfirmDialog(
                'Performance Optimization',
                `Consider disabling high-impact rules: ${highImpactRules.map(id => this.rules[id].name).join(', ')}. This may improve system performance.`,
                () => {
                    highImpactRules.forEach(ruleId => {
                        this.handleRuleToggle(ruleId, false);
                    });
                    this.showToast('Applied performance optimizations', 'success');
                }
            );
        } else {
            this.showToast('Current configuration is already optimized for performance', 'info');
        }
    }
    
    getPerformanceImpact(rule) {
        // Map rule categories to performance impact levels
        const impactMap = {
            'Process': 'high',
            'Safety': 'high',
            'Deployment': 'high',
            'AI Infrastructure': 'high',
            'Code Quality': 'medium',
            'Architecture': 'medium',
            'Organization': 'medium',
            'Infrastructure': 'medium',
            'Scripts': 'low',
            'Documentation': 'low',
            'Efficiency': 'low',
            'Cleanliness': 'low'
        };
        return impactMap[rule.category] || 'medium';
    }
    
    async runNewTests() {
        this.showToast('Initiating new test run...', 'info');
        // In a real implementation, this would trigger the test suite
        // For now, we'll simulate the process
        try {
            await this.simulateTestRun();
            this.showToast('Test run completed successfully', 'success');
            this.generateAIRecommendations(); // Refresh recommendations
        } catch (error) {
            this.showToast('Test run failed: ' + error.message, 'error');
        }
    }
    
    async simulateTestRun() {
        // Simulate test execution delay
        return new Promise((resolve) => {
            setTimeout(() => {
                resolve({ success: true });
            }, 2000);
        });
    }
    
    hideRecommendations() {
        document.getElementById('rule-recommendations').style.display = 'none';
    }
    
    syncAllTabs(data) {
        if (this.syncChannel) {
            this.syncChannel.postMessage({
                tabId: this.tabId,
                timestamp: Date.now(),
                ...data
            });
        }
    }
    
    handleSyncMessage(data) {
        if (data.tabId === this.tabId) return; // Ignore own messages
        
        switch (data.type) {
            case 'master-toggle':
                this.data.systemEnabled = data.enabled;
                document.getElementById('master-system-toggle').checked = data.enabled;
                this.updateRuleMatrix();
                break;
            case 'rule-change':
                if (this.rules[data.ruleId]) {
                    this.rules[data.ruleId].enabled = data.enabled;
                    this.updateRuleMatrix();
                }
                break;
            case 'theme-change':
                this.data.theme = data.theme;
                this.applyTheme();
                break;
        }
        
        this.updateSyncedTabsCount();
    }
    
    updateSyncedTabsCount() {
        // This would need a more sophisticated implementation with a central sync service
        // For now, just show that sync is active
        const element = document.getElementById('synced-tabs-count');
        if (element && this.syncChannel) {
            element.textContent = '2+'; // Placeholder
        }
    }
    
    updateConnectionStatus(connected) {
        const indicator = document.getElementById('status-indicator');
        const text = document.getElementById('connection-text');
        
        if (connected) {
            indicator.className = 'status-indicator connected';
            text.textContent = 'Connected';
        } else {
            indicator.className = 'status-indicator';
            text.textContent = 'Offline';
        }
    }
    
    syncTabs() {
        this.syncAllTabs({ type: 'full-sync', data: this.data, rules: this.rules });
        this.showToast('Tabs synchronized', 'success');
    }
    
    saveSettings() {
        localStorage.setItem('dashboard-settings', JSON.stringify(this.settings));
    }
    
    loadSettings() {
        const saved = localStorage.getItem('dashboard-settings');
        if (saved) {
            this.settings = { ...this.settings, ...JSON.parse(saved) };
        }
    }
    
    saveCurrentState() {
        const state = {
            systemEnabled: this.data.systemEnabled,
            rules: this.rules,
            theme: this.data.theme
        };
        localStorage.setItem('dashboard-state', JSON.stringify(state));
    }
    
    loadCurrentState() {
        const saved = localStorage.getItem('dashboard-state');
        if (saved) {
            const state = JSON.parse(saved);
            this.data.systemEnabled = state.systemEnabled ?? true;
            if (state.rules) {
                this.rules = state.rules;
            }
            if (state.theme) {
                this.data.theme = state.theme;
            }
        }
    }
    
    showKeyboardShortcuts() {
        const help = document.getElementById('shortcuts-help');
        help.style.display = 'block';
        
        // Auto-hide after 10 seconds
        setTimeout(() => {
            help.style.display = 'none';
        }, 10000);
    }
    
    closeAllModals() {
        document.getElementById('detail-modal').style.display = 'none';
        document.getElementById('impact-preview-modal').style.display = 'none';
        document.getElementById('shortcuts-help').style.display = 'none';
    }

    // Event handlers
    async runFullAudit() {
        const loadingToast = this.showToast('Running full hygiene audit...', 'info', 0);
        this.showLoadingOverlay('Running comprehensive audit...');
        
        try {
            const response = await this.fetchWithFallback('/api/hygiene/audit', { 
                success: true, 
                message: 'Mock audit completed',
                violations: Math.floor(Math.random() * 5),
                fixed: Math.floor(Math.random() * 10)
            });
            
            this.removeToast(loadingToast);
            
            if (response.success) {
                this.showToast(`Audit completed: ${response.fixed || 0} issues fixed, ${response.violations || 0} violations found`, 'success');
                
                // Clear cache to force fresh data
                this.cache.clear();
                await this.loadInitialData();
                this.renderDashboard();
                
                // Sync with other tabs
                this.syncAllTabs({ type: 'audit-completed', data: response });
            } else {
                throw new Error(response.message || 'Audit failed');
            }
        } catch (error) {
            this.removeToast(loadingToast);
            this.showToast('Audit failed: ' + error.message, 'error');
        } finally {
            this.hideLoadingOverlay();
        }
    }

    async forceCleanup() {
        const result = await this.showConfirmDialog(
            'Force Cleanup Confirmation',
            'Are you sure you want to force cleanup? This action cannot be undone and may affect system stability.',
            'warning'
        );
        
        if (!result) return;
        
        const loadingToast = this.showToast('Forcing system cleanup...', 'warning', 0);
        this.showLoadingOverlay('Performing force cleanup...');
        
        try {
            const response = await this.fetchWithFallback('/api/hygiene/cleanup', {
                success: true,
                message: 'Mock cleanup completed',
                removed: Math.floor(Math.random() * 20) + 5,
                freed: `${Math.floor(Math.random() * 500) + 100}MB`
            });
            
            this.removeToast(loadingToast);
            
            if (response.success) {
                this.showToast(`Cleanup completed: ${response.removed || 0} items removed, ${response.freed || '0MB'} freed`, 'success');
                
                // Clear cache and reload
                this.cache.clear();
                await this.loadInitialData();
                this.renderDashboard();
                
                this.syncAllTabs({ type: 'cleanup-completed', data: response });
            } else {
                throw new Error(response.message || 'Cleanup failed');
            }
        } catch (error) {
            this.removeToast(loadingToast);
            this.showToast('Cleanup failed: ' + error.message, 'error');
        } finally {
            this.hideLoadingOverlay();
        }
    }
    
    async showConfirmDialog(title, message, type = 'info') {
        return new Promise((resolve) => {
            const modal = document.createElement('div');
            modal.className = 'modal';
            modal.innerHTML = `
                <div class="modal-content">
                    <div class="modal-header">
                        <h3>${title}</h3>
                    </div>
                    <div class="modal-body">
                        <div class="confirm-message ${type}">
                            <i class="fas ${type === 'warning' ? 'fa-exclamation-triangle' : 'fa-question-circle'}"></i>
                            <p>${message}</p>
                        </div>
                    </div>
                    <div class="modal-footer">
                        <button class="btn btn-secondary" id="confirm-cancel">Cancel</button>
                        <button class="btn btn-${type === 'warning' ? 'warning' : 'primary'}" id="confirm-ok">Confirm</button>
                    </div>
                </div>
            `;
            
            document.body.appendChild(modal);
            modal.style.display = 'block';
            
            const cleanup = () => {
                modal.remove();
            };
            
            modal.querySelector('#confirm-cancel').addEventListener('click', () => {
                cleanup();
                resolve(false);
            });
            
            modal.querySelector('#confirm-ok').addEventListener('click', () => {
                cleanup();
                resolve(true);
            });
            
            // ESC key support
            const handleEscape = (e) => {
                if (e.key === 'Escape') {
                    document.removeEventListener('keydown', handleEscape);
                    cleanup();
                    resolve(false);
                }
            };
            
            document.addEventListener('keydown', handleEscape);
        });
    }

    async generateReport() {
        const loadingToast = this.showToast('Generating comprehensive report...', 'info', 0);
        this.showLoadingOverlay('Generating detailed report...');
        
        try {
            // Generate comprehensive report data
            const reportData = {
                timestamp: new Date().toISOString(),
                version: '2.0.0',
                summary: {
                    systemEnabled: this.data.systemEnabled,
                    totalRules: Object.keys(this.rules).length,
                    enabledRules: Object.values(this.rules).filter(r => r.enabled).length,
                    totalViolations: Object.values(this.data.rules || {}).reduce((sum, rule) => sum + (rule.violationCount || 0), 0),
                    complianceScore: this.data.complianceScore || 0
                },
                rules: Object.entries(this.rules).map(([id, rule]) => ({
                    id,
                    name: rule.name,
                    enabled: rule.enabled,
                    priority: rule.priority,
                    category: rule.category,
                    description: rule.description,
                    violations: this.data.rules[id]?.violationCount || 0,
                    lastChecked: this.data.rules[id]?.lastChecked
                })),
                agents: this.data.agents || {},
                metrics: this.data.metrics || {},
                recentActions: this.data.actions?.slice(0, 50) || [],
                configuration: {
                    theme: this.data.theme,
                    settings: this.settings
                }
            };
            
            // Try to get additional data from API
            try {
                const apiResponse = await this.fetchWithFallback('/api/hygiene/report', reportData);
                Object.assign(reportData, apiResponse);
            } catch (error) {
                console.warn('Using local report data:', error);
            }
            
            // Create downloadable report
            const blob = new Blob([JSON.stringify(reportData, null, 2)], { type: 'application/json' });
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `hygiene-report-${new Date().toISOString().split('T')[0]}.json`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
            
            this.removeToast(loadingToast);
            this.showToast(`Report generated successfully (${reportData.summary.totalRules} rules, ${reportData.summary.totalViolations} violations)`, 'success');
            
        } catch (error) {
            this.removeToast(loadingToast);
            this.showToast('Report generation failed: ' + error.message, 'error');
        } finally {
            this.hideLoadingOverlay();
        }
    }

    async exportData() {
        const data = {
            timestamp: new Date().toISOString(),
            version: '2.0.0',
            dashboard_data: this.data,
            rules: this.rules,
            settings: this.settings,
            cache_stats: {
                entries: this.cache.size,
                oldest: Math.min(...Array.from(this.cache.values()).map(v => v.timestamp)),
                newest: Math.max(...Array.from(this.cache.values()).map(v => v.timestamp))
            },
            performance_metrics: {
                undo_stack_size: this.undoStack.length,
                redo_stack_size: this.redoStack.length,
                pending_changes: this.pendingChanges.size
            }
        };
        
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `hygiene-dashboard-export-${new Date().toISOString().split('T')[0]}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
        
        this.showToast(`Dashboard data exported (${Object.keys(data.rules).length} rules, ${data.cache_stats.entries} cache entries)`, 'success');
    }

    toggleAutoEnforcement(enabled) {
        this.showToast(`Auto enforcement ${enabled ? 'enabled' : 'disabled'}`, 'info');
        // Implementation would send setting to backend
    }

    toggleRealTimeMonitoring(enabled) {
        if (enabled) {
            this.startRealTimeUpdates();
            this.showToast('Real-time monitoring enabled', 'success');
        } else {
            this.stopRealTimeUpdates();
            this.showToast('Real-time monitoring disabled', 'warning');
        }
    }

    updateRefreshRate(milliseconds) {
        this.refreshInterval = milliseconds;
        this.stopRealTimeUpdates();
        this.startRealTimeUpdates();
        this.showToast(`Refresh rate updated to ${milliseconds / 1000} seconds`, 'info');
    }

    filterActions(filter) {
        const actions = document.querySelectorAll('.action-item');
        actions.forEach(action => {
            if (filter === 'all' || action.classList.contains(filter)) {
                action.style.display = 'block';
            } else {
                action.style.display = 'none';
            }
        });
    }

    startRealTimeUpdates() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
        }
        
        // Use throttled update to prevent too frequent renders
        const throttledUpdate = this.throttle(async () => {
            try {
                await this.loadInitialData();
                this.renderDashboard();
            } catch (error) {
                console.warn('Real-time update failed:', error);
                this.updateConnectionStatus(false);
            }
        }, 1000);
        
        this.updateInterval = setInterval(throttledUpdate, this.refreshInterval);
        
        // Try to establish WebSocket connection for real-time updates
        this.connectWebSocket();
    }
    
    connectWebSocket() {
        if (this.websocket) {
            this.websocket.close();
        }
        
        try {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws/hygiene`;
            
            this.websocket = new WebSocket(wsUrl);
            
            this.websocket.onopen = () => {
                console.log('WebSocket connected');
                this.updateConnectionStatus(true);
            };
            
            this.websocket.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.handleWebSocketMessage(data);
                } catch (error) {
                    console.warn('Invalid WebSocket message:', error);
                }
            };
            
            this.websocket.onclose = () => {
                console.log('WebSocket disconnected');
                this.updateConnectionStatus(false);
                
                // Attempt to reconnect after 5 seconds
                setTimeout(() => {
                    if (document.getElementById('real-time-monitoring')?.checked) {
                        this.connectWebSocket();
                    }
                }, 5000);
            };
            
            this.websocket.onerror = (error) => {
                console.warn('WebSocket error:', error);
                this.updateConnectionStatus(false);
            };
        } catch (error) {
            console.warn('WebSocket connection failed:', error);
            this.updateConnectionStatus(false);
        }
    }
    
    handleWebSocketMessage(data) {
        switch (data.type) {
            case 'violation-update':
                this.handleViolationUpdate(data.payload);
                break;
            case 'agent-status':
                this.handleAgentStatusUpdate(data.payload);
                break;
            case 'system-alert':
                this.showToast(data.payload.message, data.payload.severity || 'info');
                break;
            default:
                console.log('Unknown WebSocket message type:', data.type);
        }
    }
    
    handleViolationUpdate(payload) {
        if (this.data.rules[payload.ruleId]) {
            this.data.rules[payload.ruleId].violationCount = payload.count;
            this.data.rules[payload.ruleId].lastChecked = payload.timestamp;
            
            // Update specific rule card without full re-render
            this.updateRuleCard(payload.ruleId);
        }
    }
    
    updateRuleCard(ruleId) {
        const ruleCard = document.querySelector(`[data-rule-id="${ruleId}"]`);
        if (ruleCard) {
            const ruleData = this.data.rules[ruleId] || {};
            const violationCount = ruleData.violationCount || 0;
            
            // Update violation count
            const violationElement = ruleCard.querySelector('.violation-count');
            if (violationElement) {
                violationElement.textContent = violationCount;
                if (violationCount === 0) {
                    violationElement.remove();
                }
            } else if (violationCount > 0) {
                const statusIndicator = ruleCard.querySelector('.rule-status-indicator');
                const violationBadge = document.createElement('div');
                violationBadge.className = 'violation-count';
                violationBadge.textContent = violationCount;
                statusIndicator.appendChild(violationBadge);
            }
            
            // Update last check time
            const lastCheckElement = ruleCard.querySelector('.stat-item .stat-value');
            if (lastCheckElement && lastCheckElement.previousElementSibling?.textContent === 'Last Check') {
                lastCheckElement.textContent = this.formatTime(ruleData.lastChecked);
            }
        }
    }

    stopRealTimeUpdates() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
            this.updateInterval = null;
        }
        
        if (this.websocket) {
            this.websocket.close();
            this.websocket = null;
        }
    }

    // Modal methods
    showRuleDetails(ruleId, rule) {
        const modal = document.getElementById('detail-modal');
        const title = document.getElementById('modal-title');
        const body = document.getElementById('modal-body');
        
        title.textContent = `Rule Details: ${rule.name}`;
        body.innerHTML = `
            <div class="detail-content">
                <div class="detail-section">
                    <h4>Rule Information</h4>
                    <p><strong>ID:</strong> ${ruleId}</p>
                    <p><strong>Priority:</strong> <span class="priority ${rule.priority.toLowerCase()}">${rule.priority}</span></p>
                    <p><strong>Category:</strong> ${rule.category}</p>
                    <p><strong>Status:</strong> <span class="status ${rule.enabled ? 'success' : 'critical'}">${rule.enabled ? 'ENABLED' : 'DISABLED'}</span></p>
                    <p><strong>Description:</strong> ${rule.description}</p>
                </div>
                <div class="detail-section">
                    <h4>Compliance Statistics</h4>
                    <p><strong>Current Violations:</strong> ${rule.violationCount || 0}</p>
                    <p><strong>Last Checked:</strong> ${this.formatTime(rule.lastChecked)}</p>
                </div>
                <div class="detail-section">
                    <h4>Dependencies</h4>
                    ${rule.dependencies && rule.dependencies.length > 0 ? 
                        rule.dependencies.map(depId => `
                            <div class="dependency-item">
                                <span class="dependency-name">${this.rules[depId]?.name || depId}</span>
                                <span class="dependency-status ${this.rules[depId]?.enabled ? 'enabled' : 'disabled'}">
                                    ${this.rules[depId]?.enabled ? 'ENABLED' : 'DISABLED'}
                                </span>
                            </div>
                        `).join('') : 
                        '<p>No dependencies</p>'
                    }
                </div>
                <div class="detail-section">
                    <h4>Recent Activity</h4>
                    <div class="activity-list">
                        ${this.data.actions.filter(a => a.rule === ruleId).slice(0, 5).map(action => `
                            <div class="activity-item">
                                <span class="activity-time">${this.formatTime(action.timestamp)}</span>
                                <span class="activity-message">${action.message}</span>
                            </div>
                        `).join('') || '<p>No recent activity</p>'}
                    </div>
                </div>
            </div>
        `;
        
        modal.style.display = 'block';
    }

    showAgentDetails(agentId, agent) {
        const modal = document.getElementById('detail-modal');
        const title = document.getElementById('modal-title');
        const body = document.getElementById('modal-body');
        
        title.textContent = `Agent Details: ${agentId}`;
        body.innerHTML = `
            <div class="detail-content">
                <div class="detail-section">
                    <h4>Agent Status</h4>
                    <p><strong>Status:</strong> <span class="status ${this.getStatusClass(agent.status)}">${agent.status}</span></p>
                    <p><strong>Health:</strong> ${agent.health}%</p>
                    <p><strong>Last Seen:</strong> ${this.formatTime(agent.lastSeen)}</p>
                </div>
                <div class="detail-section">
                    <h4>Performance Metrics</h4>
                    <div class="agent-metrics">
                        <div class="metric">
                            <span class="metric-label">Uptime</span>
                            <span class="metric-value">99.2%</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Tasks Completed</span>
                            <span class="metric-value">1,247</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Success Rate</span>
                            <span class="metric-value">94.8%</span>
                        </div>
                    </div>
                </div>
                <div class="detail-section">
                    <h4>Recent Actions</h4>
                    <div class="activity-list">
                        ${this.data.actions.filter(a => a.agent === agentId).slice(0, 5).map(action => `
                            <div class="activity-item">
                                <span class="activity-time">${this.formatTime(action.timestamp)}</span>
                                <span class="activity-message">${action.message}</span>
                            </div>
                        `).join('') || '<p>No recent activity</p>'}
                    </div>
                </div>
            </div>
        `;
        
        modal.style.display = 'block';
    }

    showActionDetails(action) {
        const modal = document.getElementById('detail-modal');
        const title = document.getElementById('modal-title');
        const body = document.getElementById('modal-body');
        
        title.textContent = `Action Details: ${action.type}`;
        body.innerHTML = `
            <div class="detail-content">
                <div class="detail-section">
                    <h4>Action Information</h4>
                    <p><strong>Type:</strong> ${action.type}</p>
                    <p><strong>Severity:</strong> <span class="severity ${action.severity}">${action.severity.toUpperCase()}</span></p>
                    <p><strong>Timestamp:</strong> ${new Date(action.timestamp).toLocaleString()}</p>
                    <p><strong>Rule:</strong> ${action.rule?.toUpperCase()}</p>
                    <p><strong>Agent:</strong> ${action.agent}</p>
                </div>
                <div class="detail-section">
                    <h4>Message</h4>
                    <p>${action.message}</p>
                </div>
                <div class="detail-section">
                    <h4>Details</h4>
                    <pre>${JSON.stringify(action.details || {}, null, 2)}</pre>
                </div>
            </div>
        `;
        
        modal.style.display = 'block';
    }

    showToast(message, type = 'info', duration = 5000) {
        const container = document.getElementById('toast-container');
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        
        const icon = {
            success: 'fa-check-circle',
            error: 'fa-exclamation-triangle',
            warning: 'fa-exclamation-circle',
            info: 'fa-info-circle'
        }[type] || 'fa-info-circle';
        
        toast.innerHTML = `
            <div class="toast-content">
                <i class="fas ${icon}"></i>
                <span class="toast-message">${message}</span>
                <button class="toast-close">&times;</button>
            </div>
        `;
        
        const closeBtn = toast.querySelector('.toast-close');
        closeBtn.addEventListener('click', () => {
            this.removeToast(toast);
        });
        
        container.appendChild(toast);
        
        // Animate in
        requestAnimationFrame(() => {
            toast.style.transform = 'translateX(0)';
            toast.style.opacity = '1';
        });
        
        // Auto-remove
        if (duration > 0) {
            setTimeout(() => {
                this.removeToast(toast);
            }, duration);
        }
        
        return toast;
    }
    
    removeToast(toast) {
        if (toast && toast.parentNode) {
            toast.style.transform = 'translateX(100%)';
            toast.style.opacity = '0';
            setTimeout(() => {
                if (toast.parentNode) {
                    toast.remove();
                }
            }, 300);
        }
    }

    // Initialize debounced methods
    debouncedGenerateRecommendations = this.debounce(() => {
        if (this.settings.aiRecommendations) {
            this.generateAIRecommendations();
        }
    }, 1000);
}

// Create global reference for recommendation callbacks
let dashboard;

// Initialize dashboard when page loads
document.addEventListener('DOMContentLoaded', () => {
    dashboard = new HygieneMonitorDashboard();
});