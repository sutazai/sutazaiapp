# Frontend Component Structure

## Overview

The SutazAI frontend is built using Streamlit, providing an intuitive web interface for interacting with the AI agent system. The application follows a modular component architecture for maintainability and extensibility.

## Directory Structure

```
frontend/
├── app.py                    # Main application entry point
├── agent_dashboard.py        # Agent management dashboard
├── agent_health_dashboard.py # Health monitoring interface
├── minimal_app.py           # Minimal version for low-resource deployments
├── components/              # Reusable UI components
│   ├── enhanced_ui.py       # Advanced UI components
│   ├── enter_key_handler.py # Keyboard interaction handling
│   ├── charts.py            # Data visualization components
│   └── widgets.py           # Custom widget implementations
├── pages/                   # Multi-page application structure
│   ├── agents.py           # Agent management page
│   ├── tasks.py            # Task management page
│   ├── monitoring.py       # System monitoring page
│   └── settings.py         # Configuration page
├── static/                 # Static assets
│   ├── css/               # Custom CSS styles
│   ├── js/                # JavaScript enhancements
│   └── images/            # Image assets
├── utils/                 # Utility functions
│   ├── api_client.py      # Backend API communication
│   ├── data_processing.py # Data manipulation utilities
│   └── helpers.py         # Common helper functions
└── requirements.txt       # Python dependencies
```

## Main Application Structure

### Entry Point (`app.py`)
```python
import streamlit as st
from typing import Dict, List, Any, Optional
import asyncio
import httpx

# Page configuration
st.set_page_config(
    page_title="SutazAI Task Automation System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main application class
class SutazAIApp:
    def __init__(self):
        self.api_client = APIClient()
        self.session_state = SessionState()
        self.ui_components = UIComponents()
    
    def run(self):
        """Main application flow"""
        self.render_sidebar()
        self.render_main_content()
        self.render_footer()
    
    def render_sidebar(self):
        """Render navigation sidebar"""
        
    def render_main_content(self):
        """Render main content area"""
        
    def render_footer(self):
        """Render application footer"""
```

## Component Architecture

### 1. Core Components

#### Navigation Component
```python
class NavigationComponent:
    """Main navigation and routing"""
    
    def __init__(self):
        self.pages = {
            "Dashboard": "🏠",
            "Agents": "🤖", 
            "Tasks": "📋",
            "Monitoring": "📊",
            "Settings": "⚙️"
        }
    
    def render_sidebar(self):
        """Render sidebar navigation"""
        st.sidebar.title("🚀 SutazAI")
        
        # Navigation menu
        selected_page = st.sidebar.selectbox(
            "Navigate to:",
            list(self.pages.keys()),
            format_func=lambda x: f"{self.pages[x]} {x}"
        )
        
        return selected_page
    
    def render_breadcrumb(self, current_page: str):
        """Render breadcrumb navigation"""
        st.markdown(
            f"**🏠 SutazAI** → **{self.pages[current_page]} {current_page}**"
        )
```

#### Agent List Component
```python
class AgentListComponent:
    """Agent discovery and management interface"""
    
    def __init__(self, api_client):
        self.api_client = api_client
    
    def render_agent_grid(self, agents: List[Dict]):
        """Render agents in grid layout"""
        
        # Filter controls
        col1, col2, col3 = st.columns(3)
        with col1:
            category_filter = st.selectbox(
                "Category",
                ["All"] + list(set(agent["category"] for agent in agents))
            )
        
        with col2:
            status_filter = st.selectbox(
                "Status", 
                ["All", "Active", "Inactive", "Error"]
            )
        
        with col3:
            search_term = st.text_input("🔍 Search agents")
        
        # Agent grid
        filtered_agents = self.filter_agents(agents, category_filter, status_filter, search_term)
        
        cols = st.columns(3)
        for i, agent in enumerate(filtered_agents):
            with cols[i % 3]:
                self.render_agent_card(agent)
    
    def render_agent_card(self, agent: Dict):
        """Render individual agent card"""
        with st.container():
            st.markdown(f"""
            <div class="agent-card">
                <div class="agent-header">
                    <h4>{agent['name']}</h4>
                    <span class="status-{agent['status'].lower()}">{agent['status']}</span>
                </div>
                <p class="agent-description">{agent['description']}</p>
                <div class="agent-capabilities">
                    {', '.join(agent['capabilities'])}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button(f"Execute", key=f"exec_{agent['id']}"):
                    self.show_task_dialog(agent)
            
            with col2:
                if st.button(f"Details", key=f"details_{agent['id']}"):
                    self.show_agent_details(agent)
```

### 2. Task Management Components

#### Task Creation Component
```python
class TaskCreationComponent:
    """Task creation and configuration interface"""
    
    def render_task_form(self, agent: Dict):
        """Render task creation form"""
        
        with st.form(f"task_form_{agent['id']}"):
            st.subheader(f"Create Task for {agent['name']}")
            
            # Task type selection
            task_type = st.selectbox(
                "Task Type",
                agent['supported_tasks'],
                help="Select the type of task to execute"
            )
            
            # Dynamic input fields based on task type
            inputs = self.render_dynamic_inputs(task_type)
            
            # Advanced configuration
            with st.expander("Advanced Configuration"):
                timeout = st.slider("Timeout (seconds)", 30, 600, 300)
                priority = st.selectbox("Priority", ["Low", "Normal", "High"])
                
            # Submit button
            submitted = st.form_submit_button("🚀 Execute Task")
            
            if submitted:
                task_data = {
                    "type": task_type,
                    "agent_id": agent['id'],
                    "input": inputs,
                    "config": {
                        "timeout": timeout,
                        "priority": priority.lower()
                    }
                }
                
                return self.submit_task(task_data)
    
    def render_dynamic_inputs(self, task_type: str) -> Dict:
        """Render inputs based on task type"""
        
        inputs = {}
        
        if task_type == "code_review":
            inputs['code'] = st.text_area(
                "Code to Review",
                height=200,
                help="Paste the code you want reviewed"
            )
            inputs['language'] = st.selectbox(
                "Programming Language",
                ["python", "javascript", "java", "cpp", "go"]
            )
            
        elif task_type == "security_scan":
            inputs['code'] = st.text_area("Code to Scan", height=200)
            inputs['scan_type'] = st.multiselect(
                "Scan Types",
                ["vulnerability", "dependency", "secrets", "compliance"]
            )
            
        elif task_type == "documentation":
            inputs['code'] = st.text_area("Code to Document", height=200)
            inputs['style'] = st.selectbox(
                "Documentation Style",
                ["sphinx", "google", "numpy", "markdown"]
            )
        
        return inputs
```

#### Task Status Component
```python
class TaskStatusComponent:
    """Real-time task monitoring and status display"""
    
    def __init__(self):
        self.status_colors = {
            "pending": "🟡",
            "processing": "🔵", 
            "completed": "🟢",
            "failed": "🔴",
            "cancelled": "⚫"
        }
    
    def render_task_list(self, tasks: List[Dict]):
        """Render list of tasks with status"""
        
        if not tasks:
            st.info("No tasks found")
            return
        
        # Create DataFrame for better display
        df = pd.DataFrame(tasks)
        
        # Format columns
        df['status_icon'] = df['status'].map(self.status_colors)
        df['created'] = pd.to_datetime(df['created_at']).dt.strftime('%Y-%m-%d %H:%M')
        
        # Display table
        st.dataframe(
            df[['status_icon', 'type', 'agent_name', 'created', 'progress']],
            column_config={
                "status_icon": st.column_config.TextColumn("Status"),
                "type": st.column_config.TextColumn("Task Type"),
                "agent_name": st.column_config.TextColumn("Agent"),
                "created": st.column_config.TextColumn("Created"),
                "progress": st.column_config.ProgressColumn("Progress", min_value=0, max_value=100)
            },
            use_container_width=True
        )
    
    def render_task_detail(self, task: Dict):
        """Render detailed task view"""
        
        # Header
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.markdown(f"### {self.status_colors[task['status']]} {task['type'].title()}")
        
        with col2:
            st.metric("Progress", f"{task['progress']}%")
        
        with col3:
            duration = self.calculate_duration(task)
            st.metric("Duration", duration)
        
        # Progress bar
        progress_bar = st.progress(task['progress'] / 100)
        
        # Task details
        with st.expander("Task Details", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.json(task['input'])
            
            with col2:
                if task['output']:
                    st.json(task['output'])
                else:
                    st.info("Task in progress...")
        
        # Real-time updates
        if task['status'] in ['pending', 'processing']:
            placeholder = st.empty()
            self.update_task_status(task['id'], placeholder)
```

### 3. Monitoring Components

#### System Health Component
```python
class SystemHealthComponent:
    """System health monitoring and metrics display"""
    
    def render_health_dashboard(self, health_data: Dict):
        """Render system health overview"""
        
        # Overall health status
        overall_status = health_data['status']
        status_color = {
            'healthy': '🟢',
            'degraded': '🟡', 
            'unhealthy': '🔴'
        }
        
        st.markdown(f"""
        ## System Health {status_color[overall_status]}
        **Status:** {overall_status.title()}
        """)
        
        # Service health grid
        services = health_data['services']
        cols = st.columns(4)
        
        for i, (service, status) in enumerate(services.items()):
            with cols[i % 4]:
                color = 'green' if status == 'healthy' else 'red'
                st.markdown(f"""
                <div style="padding: 1rem; border-left: 4px solid {color}; background: #f0f0f0;">
                    <h4>{service.title()}</h4>
                    <p>{status.title()}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Resource usage metrics
        self.render_resource_metrics(health_data.get('resources', {}))
    
    def render_resource_metrics(self, resources: Dict):
        """Render system resource usage"""
        
        st.subheader("Resource Usage")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            cpu_usage = resources.get('cpu_usage', 0)
            st.metric(
                "CPU Usage",
                f"{cpu_usage:.1f}%",
                delta=f"{cpu_usage - 50:.1f}%" if cpu_usage > 50 else None
            )
        
        with col2:
            memory_usage = resources.get('memory_usage', 0)
            st.metric(
                "Memory Usage", 
                f"{memory_usage:.1f}%",
                delta=f"{memory_usage - 60:.1f}%" if memory_usage > 60 else None
            )
        
        with col3:
            disk_usage = resources.get('disk_usage', 0)
            st.metric("Disk Usage", f"{disk_usage:.1f}%")
        
        with col4:
            active_connections = resources.get('active_connections', 0)
            st.metric("Active Connections", active_connections)
```

### 4. Data Visualization Components

#### Charts Component
```python
class ChartsComponent:
    """Data visualization and chart components"""
    
    def render_agent_performance_chart(self, agent_metrics: List[Dict]):
        """Render agent performance over time"""
        
        df = pd.DataFrame(agent_metrics)
        
        fig = px.line(
            df,
            x='timestamp',
            y='response_time',
            color='agent_name',
            title='Agent Response Time Trends',
            labels={'response_time': 'Response Time (s)', 'timestamp': 'Time'}
        )
        
        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Response Time (seconds)",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_task_distribution_chart(self, task_data: List[Dict]):
        """Render task type distribution"""
        
        df = pd.DataFrame(task_data)
        task_counts = df['type'].value_counts()
        
        fig = px.pie(
            values=task_counts.values,
            names=task_counts.index,
            title='Task Type Distribution'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_system_metrics_chart(self, metrics: Dict):
        """Render real-time system metrics"""
        
        # Create gauge charts for key metrics
        fig = go.Figure()
        
        # CPU gauge
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=metrics['cpu_usage'],
            domain={'x': [0, 0.3], 'y': [0, 1]},
            title={'text': "CPU Usage"},
            gauge={'axis': {'range': [None, 100]},
                   'bar': {'color': "darkblue"},
                   'steps': [
                       {'range': [0, 50], 'color': "lightgray"},
                       {'range': [50, 80], 'color': "yellow"},
                       {'range': [80, 100], 'color': "red"}
                   ],
                   'threshold': {'line': {'color': "red", 'width': 4},
                                'thickness': 0.75, 'value': 90}}
        ))
        
        st.plotly_chart(fig, use_container_width=True)
```

## State Management

### Session State Management
```python
class SessionState:
    """Centralized session state management"""
    
    def __init__(self):
        self.initialize_state()
    
    def initialize_state(self):
        """Initialize default session state"""
        
        if 'current_page' not in st.session_state:
            st.session_state.current_page = 'Dashboard'
        
        if 'selected_agent' not in st.session_state:
            st.session_state.selected_agent = None
        
        if 'active_tasks' not in st.session_state:
            st.session_state.active_tasks = []
        
        if 'user_preferences' not in st.session_state:
            st.session_state.user_preferences = {
                'theme': 'light',
                'auto_refresh': True,
                'refresh_interval': 30
            }
    
    def update_state(self, key: str, value: Any):
        """Update session state value"""
        st.session_state[key] = value
    
    def get_state(self, key: str, default=None):
        """Get session state value"""
        return st.session_state.get(key, default)
```

## API Integration

### API Client
```python
class APIClient:
    """Backend API communication layer"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = httpx.AsyncClient()
    
    async def get_agents(self) -> List[Dict]:
        """Fetch all available agents"""
        
        response = await self.session.get(f"{self.base_url}/api/v1/agents")
        response.raise_for_status()
        return response.json()['agents']
    
    async def create_task(self, task_data: Dict) -> Dict:
        """Create new task"""
        
        response = await self.session.post(
            f"{self.base_url}/api/v1/tasks",
            json=task_data
        )
        response.raise_for_status()
        return response.json()
    
    async def get_task_status(self, task_id: str) -> Dict:
        """Get task status"""
        
        response = await self.session.get(f"{self.base_url}/api/v1/tasks/{task_id}")
        response.raise_for_status()
        return response.json()
    
    async def get_system_health(self) -> Dict:
        """Get system health status"""
        
        response = await self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
```

## Responsive Design

### CSS Framework
```css
/* Custom styles for responsive design */
.agent-card {
    background: white;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    border-left: 4px solid #007bff;
}

.agent-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
}

.status-active { color: #28a745; }
.status-inactive { color: #6c757d; }
.status-error { color: #dc3545; }

@media (max-width: 768px) {
    .agent-card {
        margin: 0.25rem 0;
        padding: 0.75rem;
    }
}
```

## Error Handling

### Error Boundary Component
```python
class ErrorHandler:
    """Global error handling and user feedback"""
    
    @staticmethod
    def handle_api_error(error: Exception):
        """Handle API communication errors"""
        
        if isinstance(error, httpx.HTTPStatusError):
            if error.response.status_code == 404:
                st.error("Resource not found")
            elif error.response.status_code == 500:
                st.error("Server error. Please try again later.")
            else:
                st.error(f"API Error: {error.response.status_code}")
        else:
            st.error("Connection error. Please check your network.")
    
    @staticmethod
    def display_error(message: str, details: str = None):
        """Display user-friendly error message"""
        
        st.error(message)
        
        if details:
            with st.expander("Error Details"):
                st.code(details)
```

## Performance Optimization

### Caching Strategy
```python
@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_agents():
    """Cached agent data fetch"""
    return asyncio.run(api_client.get_agents())

@st.cache_data(ttl=60)  # Cache for 1 minute
def fetch_system_health():
    """Cached health data fetch"""
    return asyncio.run(api_client.get_system_health())
```

### Lazy Loading
```python
class LazyLoader:
    """Lazy loading for heavy components"""
    
    @staticmethod
    def load_component_on_demand(component_name: str):
        """Load component only when needed"""
        
        if component_name not in st.session_state:
            with st.spinner(f"Loading {component_name}..."):
                # Load component
                st.session_state[component_name] = True
```

This component structure provides a solid foundation for the SutazAI frontend, ensuring maintainability, performance, and user experience while supporting the complex multi-agent automation system.