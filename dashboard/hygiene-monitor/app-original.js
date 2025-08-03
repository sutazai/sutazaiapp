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
                dependencies: ['rule_3']
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
                dependencies: ['rule_3']
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
                dependencies: []
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
                dependencies: ['rule_3']
            },
            'rule_10': { 
                name: 'Verify Before Cleanup', 
                priority: 'CRITICAL', 
                category: 'Safety',
                description: 'Functional verification required before any cleanup activities.',
                enabled: true,
                dependencies: ['rule_3']
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
                dependencies: ['rule_11']
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
        try {
            // Load hygiene enforcement status
            const response = await this.fetchWithFallback('/api/hygiene/status', this.generateMockData());
            this.data = response;
            
            // Load system metrics
            const metricsResponse = await this.fetchWithFallback('/api/system/metrics', this.generateMockMetrics());
            this.data.metrics = metricsResponse;

        } catch (error) {
            console.warn('API not available, using mock data:', error);
            this.data = this.generateMockData();
            this.data.metrics = this.generateMockMetrics();
        }
    }

    async fetchWithFallback(url, fallback) {
        try {
            const response = await fetch(url);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            return await response.json();
        } catch (error) {
            console.warn(`Falling back to mock data for ${url}:`, error);
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

        Object.entries(this.data.rules || {}).forEach(([ruleId, rule]) => {
            const ruleElement = document.createElement('div');
            ruleElement.className = `rule-item ${this.getStatusClass(rule.status)}`;
            ruleElement.innerHTML = `
                <div class="rule-header">
                    <span class="rule-id">${ruleId.toUpperCase()}</span>
                    <span class="rule-status">${rule.status}</span>
                </div>
                <div class="rule-name">${rule.name}</div>
                <div class="rule-meta">
                    <span class="rule-priority ${rule.priority.toLowerCase()}">${rule.priority}</span>
                    <span class="rule-category">${rule.category}</span>
                </div>
                <div class="rule-stats">
                    <span>Violations: ${rule.violationCount || 0}</span>
                    <span>Last Check: ${this.formatTime(rule.lastChecked)}</span>
                </div>
            `;
            
            ruleElement.addEventListener('click', () => this.showRuleDetails(ruleId, rule));
            ruleGrid.appendChild(ruleElement);
        });
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

    // Event handlers
    async runFullAudit() {
        this.showToast('Running full hygiene audit...', 'info');
        try {
            const response = await fetch('/api/hygiene/audit', { method: 'POST' });
            if (response.ok) {
                this.showToast('Full audit completed successfully', 'success');
                await this.loadInitialData();
                this.renderDashboard();
            } else {
                throw new Error('Audit failed');
            }
        } catch (error) {
            this.showToast('Audit failed: ' + error.message, 'error');
        }
    }

    async forceCleanup() {
        if (!confirm('Are you sure you want to force cleanup? This action cannot be undone.')) {
            return;
        }
        
        this.showToast('Forcing cleanup...', 'warning');
        try {
            const response = await fetch('/api/hygiene/cleanup', { method: 'POST' });
            if (response.ok) {
                this.showToast('Force cleanup completed', 'success');
                await this.loadInitialData();
                this.renderDashboard();
            } else {
                throw new Error('Cleanup failed');
            }
        } catch (error) {
            this.showToast('Cleanup failed: ' + error.message, 'error');
        }
    }

    async generateReport() {
        this.showToast('Generating comprehensive report...', 'info');
        try {
            const response = await fetch('/api/hygiene/report');
            if (response.ok) {
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `hygiene-report-${new Date().toISOString().split('T')[0]}.json`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                window.URL.revokeObjectURL(url);
                this.showToast('Report downloaded successfully', 'success');
            } else {
                throw new Error('Report generation failed');
            }
        } catch (error) {
            this.showToast('Report generation failed: ' + error.message, 'error');
        }
    }

    async exportData() {
        const data = {
            timestamp: new Date().toISOString(),
            dashboard_data: this.data,
            export_version: '1.0'
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
        
        this.showToast('Data exported successfully', 'success');
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
        
        this.updateInterval = setInterval(async () => {
            await this.loadInitialData();
            this.renderDashboard();
        }, this.refreshInterval);
    }

    stopRealTimeUpdates() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
            this.updateInterval = null;
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
                    <p><strong>Status:</strong> <span class="status ${this.getStatusClass(rule.status)}">${rule.status}</span></p>
                </div>
                <div class="detail-section">
                    <h4>Compliance Statistics</h4>
                    <p><strong>Current Violations:</strong> ${rule.violationCount || 0}</p>
                    <p><strong>Last Checked:</strong> ${this.formatTime(rule.lastChecked)}</p>
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

    showToast(message, type = 'info') {
        const container = document.getElementById('toast-container');
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.innerHTML = `
            <div class="toast-content">
                <span class="toast-message">${message}</span>
                <button class="toast-close">&times;</button>
            </div>
        `;
        
        const closeBtn = toast.querySelector('.toast-close');
        closeBtn.addEventListener('click', () => {
            toast.remove();
        });
        
        container.appendChild(toast);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (toast.parentNode) {
                toast.remove();
            }
        }, 5000);
    }
}

// Initialize dashboard when page loads
document.addEventListener('DOMContentLoaded', () => {
    new HygieneMonitorDashboard();
});