"""
Agent Management Panel Component for SutazAI Streamlit Interface
"""

import streamlit as st
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class AgentPanel:
    """Advanced Agent Management Panel with comprehensive controls"""
    
    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator
        self.agent_configs = self._load_agent_configurations()
    
    def _load_agent_configurations(self) -> Dict[str, Dict]:
        """Load agent configurations and capabilities"""
        return {
            'AutoGPT': {
                'description': 'Autonomous GPT agent for complex task execution',
                'capabilities': ['Web browsing', 'File operations', 'Code execution', 'API calls'],
                'category': 'General Purpose',
                'status': 'active',
                'version': '1.0.0',
                'dependencies': ['openai', 'selenium'],
                'config_options': {
                    'max_iterations': {'type': 'int', 'default': 10, 'range': [1, 50]},
                    'temperature': {'type': 'float', 'default': 0.7, 'range': [0.0, 2.0]},
                    'model': {'type': 'select', 'options': ['gpt-4', 'gpt-3.5-turbo'], 'default': 'gpt-4'}
                }
            },
            'LocalAGI': {
                'description': 'Local AGI agent for privacy-focused general intelligence',
                'capabilities': ['Local inference', 'Memory management', 'Task planning', 'Learning'],
                'category': 'Privacy-First',
                'status': 'active',
                'version': '1.0.0',
                'dependencies': ['transformers', 'torch'],
                'config_options': {
                    'model_size': {'type': 'select', 'options': ['small', 'medium', 'large'], 'default': 'medium'},
                    'memory_limit': {'type': 'int', 'default': 1000, 'range': [100, 10000]},
                    'learning_rate': {'type': 'float', 'default': 0.001, 'range': [0.0001, 0.1]}
                }
            },
            'AutoGen': {
                'description': 'Multi-agent conversation framework for collaborative AI',
                'capabilities': ['Multi-agent chat', 'Role assignment', 'Collaborative problem solving'],
                'category': 'Collaborative',
                'status': 'beta',
                'version': '0.9.0',
                'dependencies': ['autogen', 'openai'],
                'config_options': {
                    'max_agents': {'type': 'int', 'default': 5, 'range': [2, 20]},
                    'conversation_rounds': {'type': 'int', 'default': 10, 'range': [1, 100]},
                    'termination_condition': {'type': 'text', 'default': 'TERMINATE'}
                }
            },
            'BigAGI': {
                'description': 'Advanced artificial general intelligence with multiple domains',
                'capabilities': ['Multi-domain intelligence', 'Meta-learning', 'Strategic planning'],
                'category': 'Advanced AI',
                'status': 'experimental',
                'version': '0.8.0',
                'dependencies': ['tensorflow', 'pytorch', 'transformers'],
                'config_options': {
                    'intelligence_domains': {'type': 'multiselect', 'options': ['linguistic', 'mathematical', 'spatial', 'musical', 'interpersonal'], 'default': ['linguistic', 'mathematical']},
                    'meta_learning_enabled': {'type': 'bool', 'default': True},
                    'planning_horizon': {'type': 'int', 'default': 100, 'range': [10, 1000]}
                }
            },
            'AgentZero': {
                'description': 'Universal problem-solving with zero-shot learning capabilities',
                'capabilities': ['Zero-shot learning', 'Adaptive reasoning', 'Knowledge transfer'],
                'category': 'Learning',
                'status': 'active',
                'version': '1.1.0',
                'dependencies': ['transformers', 'faiss'],
                'config_options': {
                    'adaptation_speed': {'type': 'float', 'default': 0.5, 'range': [0.1, 1.0]},
                    'knowledge_retention': {'type': 'bool', 'default': True},
                    'transfer_threshold': {'type': 'float', 'default': 0.8, 'range': [0.5, 1.0]}
                }
            },
            'BrowserUse': {
                'description': 'Intelligent web automation with form filling and data extraction',
                'capabilities': ['Web automation', 'Form filling', 'Data extraction', 'Session management'],
                'category': 'Automation',
                'status': 'active',
                'version': '1.0.0',
                'dependencies': ['selenium', 'beautifulsoup4', 'requests'],
                'config_options': {
                    'browser_type': {'type': 'select', 'options': ['chrome', 'firefox', 'edge'], 'default': 'chrome'},
                    'headless_mode': {'type': 'bool', 'default': True},
                    'wait_timeout': {'type': 'int', 'default': 30, 'range': [5, 120]}
                }
            },
            'Skyvern': {
                'description': 'Advanced AI-powered web automation with visual recognition',
                'capabilities': ['Visual element recognition', 'Adaptive workflows', 'Screenshot analysis'],
                'category': 'Advanced Automation',
                'status': 'beta',
                'version': '0.9.0',
                'dependencies': ['opencv-python', 'pillow', 'selenium'],
                'config_options': {
                    'visual_recognition_threshold': {'type': 'float', 'default': 0.9, 'range': [0.5, 1.0]},
                    'adaptive_learning': {'type': 'bool', 'default': True},
                    'screenshot_quality': {'type': 'select', 'options': ['low', 'medium', 'high'], 'default': 'medium'}
                }
            },
            'OpenWebUI': {
                'description': 'Web interface management with chat control and plugin system',
                'capabilities': ['Interface management', 'Chat control', 'Plugin management', 'Real-time interaction'],
                'category': 'Interface',
                'status': 'active',
                'version': '1.0.0',
                'dependencies': ['streamlit', 'flask', 'socketio'],
                'config_options': {
                    'theme': {'type': 'select', 'options': ['light', 'dark', 'auto'], 'default': 'auto'},
                    'max_chat_history': {'type': 'int', 'default': 1000, 'range': [100, 10000]},
                    'real_time_updates': {'type': 'bool', 'default': True}
                }
            },
            'TabbyML': {
                'description': 'AI-powered code completion with context awareness',
                'capabilities': ['Code completion', 'Context analysis', 'Multi-language support', 'Quality suggestions'],
                'category': 'Development',
                'status': 'active',
                'version': '1.0.0',
                'dependencies': ['transformers', 'tree-sitter'],
                'config_options': {
                    'languages': {'type': 'multiselect', 'options': ['python', 'javascript', 'java', 'cpp', 'rust'], 'default': ['python', 'javascript']},
                    'completion_confidence': {'type': 'float', 'default': 0.8, 'range': [0.5, 1.0]},
                    'max_suggestions': {'type': 'int', 'default': 5, 'range': [1, 20]}
                }
            },
            'Semgrep': {
                'description': 'Advanced static code analysis with security vulnerability detection',
                'capabilities': ['Static analysis', 'Security scanning', 'Compliance checking', 'Custom rules'],
                'category': 'Security',
                'status': 'active',
                'version': '1.0.0',
                'dependencies': ['semgrep'],
                'config_options': {
                    'rule_sets': {'type': 'multiselect', 'options': ['security', 'performance', 'bugs', 'style'], 'default': ['security', 'bugs']},
                    'severity_threshold': {'type': 'select', 'options': ['info', 'warning', 'error'], 'default': 'warning'},
                    'custom_rules_enabled': {'type': 'bool', 'default': False}
                }
            }
        }
    
    def render_agent_overview(self):
        """Render the main agent overview panel"""
        st.header("🤖 Agent Management Dashboard")
        
        # Quick stats
        total_agents = len(self.agent_configs)
        active_agents = len([a for a in self.agent_configs.values() if a['status'] == 'active'])
        categories = set(a['category'] for a in self.agent_configs.values())
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Agents", total_agents)
        with col2:
            st.metric("Active Agents", active_agents)
        with col3:
            st.metric("Categories", len(categories))
        with col4:
            st.metric("Running", len(st.session_state.get('active_agents', [])))
        
        # Agent category filter
        selected_category = st.selectbox(
            "Filter by Category",
            ["All"] + sorted(list(categories)),
            key="agent_category_filter"
        )
        
        # Agent grid
        self._render_agent_grid(selected_category)
    
    def _render_agent_grid(self, category_filter: str):
        """Render the agent grid with cards"""
        filtered_agents = self.agent_configs.items()
        
        if category_filter != "All":
            filtered_agents = [(name, config) for name, config in filtered_agents 
                             if config['category'] == category_filter]
        
        # Create grid layout
        cols_per_row = 3
        rows = [list(filtered_agents)[i:i + cols_per_row] 
                for i in range(0, len(list(filtered_agents)), cols_per_row)]
        
        for row in rows:
            cols = st.columns(cols_per_row)
            
            for idx, (agent_name, agent_config) in enumerate(row):
                with cols[idx]:
                    self._render_agent_card(agent_name, agent_config)
    
    def _render_agent_card(self, agent_name: str, agent_config: Dict):
        """Render individual agent card"""
        with st.container():
            # Card header with status indicator
            status_color = {
                'active': '🟢',
                'beta': '🟡', 
                'experimental': '🟠',
                'inactive': '🔴'
            }
            
            st.markdown(f"""
            <div style="border: 1px solid #ddd; border-radius: 8px; padding: 1rem; margin: 0.5rem 0; background: white;">
                <h4>{status_color.get(agent_config['status'], '⚪')} {agent_name}</h4>
                <p><strong>Category:</strong> {agent_config['category']}</p>
                <p><strong>Version:</strong> {agent_config['version']}</p>
                <p><strong>Status:</strong> {agent_config['status'].title()}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Description
            st.write(agent_config['description'])
            
            # Capabilities
            with st.expander("🔧 Capabilities"):
                for capability in agent_config['capabilities']:
                    st.write(f"• {capability}")
            
            # Agent controls
            col1, col2, col3 = st.columns(3)
            
            is_running = agent_name in st.session_state.get('active_agents', [])
            
            with col1:
                if st.button("▶️ Start", key=f"start_{agent_name}", disabled=is_running):
                    self._start_agent(agent_name)
            
            with col2:
                if st.button("⏹️ Stop", key=f"stop_{agent_name}", disabled=not is_running):
                    self._stop_agent(agent_name)
            
            with col3:
                if st.button("⚙️ Config", key=f"config_{agent_name}"):
                    st.session_state[f'show_config_{agent_name}'] = True
            
            # Configuration modal
            if st.session_state.get(f'show_config_{agent_name}', False):
                self._render_agent_config_modal(agent_name, agent_config)
    
    def _render_agent_config_modal(self, agent_name: str, agent_config: Dict):
        """Render agent configuration modal"""
        with st.modal(f"Configure {agent_name}"):
            st.header(f"⚙️ {agent_name} Configuration")
            
            config_options = agent_config.get('config_options', {})
            
            if not config_options:
                st.info("No configuration options available for this agent.")
                return
            
            # Render configuration options
            new_config = {}
            
            for option_name, option_config in config_options.items():
                option_type = option_config['type']
                default_value = option_config['default']
                
                if option_type == 'int':
                    min_val, max_val = option_config['range']
                    new_config[option_name] = st.slider(
                        option_name.replace('_', ' ').title(),
                        min_value=min_val,
                        max_value=max_val,
                        value=default_value,
                        key=f"{agent_name}_{option_name}"
                    )
                
                elif option_type == 'float':
                    min_val, max_val = option_config['range']
                    new_config[option_name] = st.slider(
                        option_name.replace('_', ' ').title(),
                        min_value=min_val,
                        max_value=max_val,
                        value=default_value,
                        step=0.1,
                        key=f"{agent_name}_{option_name}"
                    )
                
                elif option_type == 'bool':
                    new_config[option_name] = st.checkbox(
                        option_name.replace('_', ' ').title(),
                        value=default_value,
                        key=f"{agent_name}_{option_name}"
                    )
                
                elif option_type == 'select':
                    new_config[option_name] = st.selectbox(
                        option_name.replace('_', ' ').title(),
                        options=option_config['options'],
                        index=option_config['options'].index(default_value),
                        key=f"{agent_name}_{option_name}"
                    )
                
                elif option_type == 'multiselect':
                    new_config[option_name] = st.multiselect(
                        option_name.replace('_', ' ').title(),
                        options=option_config['options'],
                        default=default_value,
                        key=f"{agent_name}_{option_name}"
                    )
                
                elif option_type == 'text':
                    new_config[option_name] = st.text_input(
                        option_name.replace('_', ' ').title(),
                        value=default_value,
                        key=f"{agent_name}_{option_name}"
                    )
            
            # Configuration actions
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("💾 Save Config"):
                    self._save_agent_config(agent_name, new_config)
                    st.success("Configuration saved!")
            
            with col2:
                if st.button("🔄 Reset to Defaults"):
                    self._reset_agent_config(agent_name)
                    st.success("Configuration reset!")
            
            with col3:
                if st.button("❌ Close"):
                    st.session_state[f'show_config_{agent_name}'] = False
                    st.rerun()
    
    def render_agent_monitoring(self):
        """Render agent monitoring and performance panel"""
        st.header("📊 Agent Monitoring")
        
        # Active agents status
        active_agents = st.session_state.get('active_agents', [])
        
        if not active_agents:
            st.info("No agents are currently running. Start some agents to see monitoring data.")
            return
        
        # Performance metrics for each active agent
        for agent_name in active_agents:
            with st.expander(f"📈 {agent_name} Performance", expanded=True):
                col1, col2, col3, col4 = st.columns(4)
                
                # Generate sample metrics
                import random
                
                with col1:
                    st.metric("Tasks Completed", random.randint(10, 100))
                
                with col2:
                    st.metric("Success Rate", f"{random.uniform(85, 99):.1f}%")
                
                with col3:
                    st.metric("Avg Response Time", f"{random.uniform(0.5, 3.0):.2f}s")
                
                with col4:
                    st.metric("Memory Usage", f"{random.uniform(20, 80):.1f}%")
                
                # Activity log
                st.subheader("Recent Activity")
                activities = [
                    f"Task completed: Document analysis",
                    f"Request processed: Web search query",
                    f"Action executed: File operation",
                    f"Response generated: User query",
                    f"Error handled: Connection timeout"
                ]
                
                for activity in activities[:3]:
                    st.write(f"• {activity}")
    
    def render_agent_logs(self):
        """Render agent logs panel"""
        st.header("📋 Agent Logs")
        
        # Log level filter
        log_level = st.selectbox(
            "Log Level",
            ["ALL", "DEBUG", "INFO", "WARNING", "ERROR"],
            index=2
        )
        
        # Agent filter
        available_agents = list(self.agent_configs.keys())
        selected_agent = st.selectbox(
            "Agent",
            ["ALL"] + available_agents
        )
        
        # Time range filter
        time_range = st.selectbox(
            "Time Range",
            ["Last Hour", "Last 6 Hours", "Last 24 Hours", "Last Week"]
        )
        
        # Sample log entries
        sample_logs = [
            {"timestamp": "2024-01-15 10:30:15", "level": "INFO", "agent": "AutoGPT", "message": "Task execution started"},
            {"timestamp": "2024-01-15 10:30:20", "level": "DEBUG", "agent": "AutoGPT", "message": "Parsing user request"},
            {"timestamp": "2024-01-15 10:30:25", "level": "WARNING", "agent": "BrowserUse", "message": "Slow response from target website"},
            {"timestamp": "2024-01-15 10:30:30", "level": "ERROR", "agent": "Semgrep", "message": "Failed to load custom rules"},
            {"timestamp": "2024-01-15 10:30:35", "level": "INFO", "agent": "TabbyML", "message": "Code completion suggestion generated"}
        ]
        
        # Filter logs
        filtered_logs = sample_logs
        if log_level != "ALL":
            filtered_logs = [log for log in filtered_logs if log['level'] == log_level]
        if selected_agent != "ALL":
            filtered_logs = [log for log in filtered_logs if log['agent'] == selected_agent]
        
        # Display logs
        for log_entry in filtered_logs:
            level_color = {
                'DEBUG': '🔍',
                'INFO': 'ℹ️',
                'WARNING': '⚠️',
                'ERROR': '❌'
            }
            
            st.markdown(f"""
            {level_color.get(log_entry['level'], '📝')} **{log_entry['timestamp']}** 
            [{log_entry['level']}] {log_entry['agent']}: {log_entry['message']}
            """)
    
    def _start_agent(self, agent_name: str):
        """Start a specific agent"""
        try:
            if 'active_agents' not in st.session_state:
                st.session_state.active_agents = []
            
            if agent_name not in st.session_state.active_agents:
                st.session_state.active_agents.append(agent_name)
                st.success(f"✅ Started {agent_name} agent")
                logger.info(f"Started agent: {agent_name}")
            else:
                st.warning(f"⚠️ {agent_name} agent is already running")
                
        except Exception as e:
            st.error(f"❌ Failed to start {agent_name}: {str(e)}")
            logger.error(f"Failed to start agent {agent_name}: {e}")
    
    def _stop_agent(self, agent_name: str):
        """Stop a specific agent"""
        try:
            if 'active_agents' in st.session_state and agent_name in st.session_state.active_agents:
                st.session_state.active_agents.remove(agent_name)
                st.success(f"⏹️ Stopped {agent_name} agent")
                logger.info(f"Stopped agent: {agent_name}")
            else:
                st.warning(f"⚠️ {agent_name} agent is not running")
                
        except Exception as e:
            st.error(f"❌ Failed to stop {agent_name}: {str(e)}")
            logger.error(f"Failed to stop agent {agent_name}: {e}")
    
    def _save_agent_config(self, agent_name: str, config: Dict):
        """Save agent configuration"""
        try:
            if 'agent_configs' not in st.session_state:
                st.session_state.agent_configs = {}
            
            st.session_state.agent_configs[agent_name] = config
            logger.info(f"Saved configuration for agent: {agent_name}")
            
        except Exception as e:
            logger.error(f"Failed to save config for agent {agent_name}: {e}")
    
    def _reset_agent_config(self, agent_name: str):
        """Reset agent configuration to defaults"""
        try:
            if 'agent_configs' in st.session_state and agent_name in st.session_state.agent_configs:
                del st.session_state.agent_configs[agent_name]
            
            logger.info(f"Reset configuration for agent: {agent_name}")
            
        except Exception as e:
            logger.error(f"Failed to reset config for agent {agent_name}: {e}")
    
    def render_bulk_actions(self):
        """Render bulk agent actions panel"""
        st.header("🔧 Bulk Agent Actions")
        
        available_agents = list(self.agent_configs.keys())
        
        # Select agents for bulk actions
        selected_agents = st.multiselect(
            "Select Agents for Bulk Actions",
            available_agents,
            key="bulk_action_agents"
        )
        
        if not selected_agents:
            st.info("Select one or more agents to perform bulk actions.")
            return
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("▶️ Start All Selected"):
                for agent in selected_agents:
                    self._start_agent(agent)
                st.success(f"Started {len(selected_agents)} agents")
        
        with col2:
            if st.button("⏹️ Stop All Selected"):
                for agent in selected_agents:
                    self._stop_agent(agent)
                st.success(f"Stopped {len(selected_agents)} agents")
        
        with col3:
            if st.button("🔄 Restart All Selected"):
                for agent in selected_agents:
                    self._stop_agent(agent)
                    self._start_agent(agent)
                st.success(f"Restarted {len(selected_agents)} agents")
        
        with col4:
            if st.button("⚙️ Configure All Selected"):
                st.info("Bulk configuration feature coming soon!")
    
    def render(self):
        """Render the complete agent panel"""
        tab1, tab2, tab3, tab4 = st.tabs([
            "🏠 Overview",
            "📊 Monitoring", 
            "📋 Logs",
            "🔧 Bulk Actions"
        ])
        
        with tab1:
            self.render_agent_overview()
        
        with tab2:
            self.render_agent_monitoring()
        
        with tab3:
            self.render_agent_logs()
        
        with tab4:
            self.render_bulk_actions()