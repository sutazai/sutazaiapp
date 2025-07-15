# SutazAI Development Guide

## Development Environment Setup

### Prerequisites
- Python 3.8+
- Docker 20.10+
- Git 2.30+
- Node.js 16+ (for frontend development)
- Visual Studio Code (recommended)

### Setting Up Development Environment

#### 1. Clone and Setup
```bash
# Clone repository
git clone https://github.com/sutazai/sutazaiapp.git
cd sutazaiapp

# Create development branch
git checkout -b feature/your-feature-name

# Setup virtual environment
python3 -m venv venv
source venv/bin/activate

# Install development dependencies
pip install -r requirements-dev.txt
```

#### 2. Development Configuration
```bash
# Copy development environment file
cp config/development.env .env

# Set development mode
export ENVIRONMENT=development
export DEBUG=true
export LOG_LEVEL=DEBUG
```

#### 3. Pre-commit Hooks
```bash
# Install pre-commit hooks
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

## Project Structure

```
sutazaiapp/
├── sutazai/                 # Core AI components
│   ├── core/               # Core modules (CGM, KG, ACM)
│   ├── nln/                # Neural Link Networks
│   ├── agents/             # AI agents
│   └── utils/              # Utility functions
├── backend/                # FastAPI backend
│   ├── api/                # API endpoints
│   ├── models/             # Database models
│   ├── services/           # Business logic
│   └── config/             # Configuration
├── frontend/               # Web interface
│   ├── components/         # React components
│   ├── pages/              # Page components
│   ├── hooks/              # Custom hooks
│   └── utils/              # Frontend utilities
├── tests/                  # Test suites
│   ├── unit/               # Unit tests
│   ├── integration/        # Integration tests
│   └── e2e/                # End-to-end tests
├── docs/                   # Documentation
├── scripts/                # Utility scripts
├── docker/                 # Docker configurations
└── deployment/             # Deployment configurations
```

## Core Components Development

### Code Generation Module (CGM)

#### Architecture
```python
# sutazai/core/cgm.py
class CodeGenerationModule:
    def __init__(self):
        self.neural_generator = NeuralCodeGenerator()
        self.meta_learner = MetaLearningModel()
        self.quality_assessor = CodeQualityAssessor()
    
    async def generate_code(self, prompt: str, context: dict = None):
        # Generate initial code
        code = await self.neural_generator.generate(prompt, context)
        
        # Assess quality
        quality_score = self.quality_assessor.assess(code)
        
        # Improve if needed
        if quality_score < 0.8:
            code = await self.improve_code(code, quality_score)
        
        return code
```

#### Adding New Generation Strategies
```python
# Register new strategy
@cgm.register_strategy("functional_programming")
class FunctionalProgrammingStrategy(GenerationStrategy):
    async def generate(self, prompt: str) -> str:
        # Implement functional programming strategy
        pass
```

### Knowledge Graph (KG)

#### Adding New Entity Types
```python
# sutazai/core/kg.py
class CustomEntity(Entity):
    def __init__(self, name: str, entity_type: str):
        super().__init__(name, entity_type)
        self.custom_attributes = {}
    
    def add_custom_relationship(self, target: str, relationship_type: str):
        # Implement custom relationship logic
        pass
```

#### Extending Search Capabilities
```python
class SemanticSearch:
    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    async def semantic_search(self, query: str, top_k: int = 10):
        # Implement semantic search using embeddings
        query_embedding = self.embedding_model.encode(query)
        # Search logic here
        return results
```

### Neural Link Networks (NLN)

#### Creating Custom Node Types
```python
# sutazai/nln/custom_nodes.py
class ReasoningNode(NeuralNode):
    def __init__(self, node_id: str):
        super().__init__(node_id, "reasoning")
        self.reasoning_capacity = 1.0
        self.logic_rules = []
    
    async def process_reasoning(self, input_data: dict):
        # Implement reasoning logic
        pass
```

#### Adding New Synapse Types
```python
class InhibitorySynapse(NeuralSynapse):
    def __init__(self, pre_node: str, post_node: str):
        super().__init__(pre_node, post_node, "inhibitory")
        self.inhibition_strength = 0.5
    
    def transmit_signal(self, signal: float) -> float:
        # Inhibitory transmission logic
        return -signal * self.inhibition_strength
```

## API Development

### Adding New Endpoints

#### 1. Define Models
```python
# backend/models/requests.py
from pydantic import BaseModel

class NewFeatureRequest(BaseModel):
    parameter1: str
    parameter2: int = 10
    optional_param: Optional[str] = None
```

#### 2. Implement Service
```python
# backend/services/new_feature.py
class NewFeatureService:
    async def process_request(self, request: NewFeatureRequest):
        # Implement business logic
        result = await self.process_data(request)
        return result
```

#### 3. Create Endpoint
```python
# backend/api/v1/new_feature.py
from fastapi import APIRouter, Depends

router = APIRouter(prefix="/new-feature", tags=["new-feature"])

@router.post("/process")
async def process_new_feature(
    request: NewFeatureRequest,
    service: NewFeatureService = Depends()
):
    result = await service.process_request(request)
    return {"result": result}
```

#### 4. Register Router
```python
# backend/main.py
from backend.api.v1.new_feature import router as new_feature_router

app.include_router(new_feature_router, prefix="/api/v1")
```

### Authentication and Authorization

#### Adding New Authentication Methods
```python
# backend/auth/oauth.py
class OAuthProvider:
    async def authenticate(self, token: str) -> User:
        # Implement OAuth authentication
        pass

# Register provider
auth_manager.register_provider("oauth", OAuthProvider())
```

#### Custom Authorization Decorators
```python
# backend/auth/decorators.py
def require_permission(permission: str):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Check permission
            if not current_user.has_permission(permission):
                raise PermissionDenied()
            return await func(*args, **kwargs)
        return wrapper
    return decorator

# Usage
@require_permission("admin")
async def admin_endpoint():
    pass
```

## AI Agent Development

### Creating Custom Agents

#### 1. Define Agent Class
```python
# sutazai/agents/custom_agent.py
from sutazai.agents.base import BaseAgent

class CustomAgent(BaseAgent):
    def __init__(self, agent_id: str):
        super().__init__(agent_id, "custom")
        self.capabilities = ["custom_task", "data_processing"]
    
    async def execute_task(self, task: Task) -> TaskResult:
        if task.type == "custom_task":
            return await self.handle_custom_task(task)
        else:
            return await super().execute_task(task)
    
    async def handle_custom_task(self, task: Task) -> TaskResult:
        # Implement custom task logic
        pass
```

#### 2. Register Agent
```python
# Register with agent manager
agent_manager.register_agent_type("custom", CustomAgent)

# Create agent instance
custom_agent = agent_manager.create_agent("custom", "custom_001")
```

### Agent Communication

#### Inter-Agent Messaging
```python
class AgentMessenger:
    async def send_message(self, from_agent: str, to_agent: str, 
                          message: dict):
        # Implement message routing
        target_agent = agent_manager.get_agent(to_agent)
        await target_agent.receive_message(from_agent, message)
    
    async def broadcast_message(self, from_agent: str, message: dict):
        # Broadcast to all agents
        for agent in agent_manager.get_all_agents():
            if agent.id != from_agent:
                await agent.receive_message(from_agent, message)
```

## Frontend Development

### React Component Development

#### Creating New Components
```jsx
// frontend/components/NewComponent.jsx
import React, { useState, useEffect } from 'react';
import { useApi } from '../hooks/useApi';

const NewComponent = ({ prop1, prop2 }) => {
    const [data, setData] = useState(null);
    const api = useApi();
    
    useEffect(() => {
        const fetchData = async () => {
            const result = await api.get('/api/v1/new-endpoint');
            setData(result.data);
        };
        
        fetchData();
    }, []);
    
    return (
        <div className="new-component">
            {/* Component JSX */}
        </div>
    );
};

export default NewComponent;
```

#### Custom Hooks
```jsx
// frontend/hooks/useNewFeature.js
import { useState, useCallback } from 'react';
import { useApi } from './useApi';

export const useNewFeature = () => {
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const api = useApi();
    
    const processFeature = useCallback(async (data) => {
        setLoading(true);
        setError(null);
        
        try {
            const result = await api.post('/api/v1/new-feature/process', data);
            return result.data;
        } catch (err) {
            setError(err.message);
            throw err;
        } finally {
            setLoading(false);
        }
    }, [api]);
    
    return { processFeature, loading, error };
};
```

### State Management

#### Adding New Redux Slices
```javascript
// frontend/store/slices/newFeatureSlice.js
import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';

export const fetchNewFeatureData = createAsyncThunk(
    'newFeature/fetchData',
    async (params, { rejectWithValue }) => {
        try {
            const response = await api.get('/api/v1/new-feature', { params });
            return response.data;
        } catch (error) {
            return rejectWithValue(error.message);
        }
    }
);

const newFeatureSlice = createSlice({
    name: 'newFeature',
    initialState: {
        data: null,
        loading: false,
        error: null
    },
    reducers: {
        clearError: (state) => {
            state.error = null;
        }
    },
    extraReducers: (builder) => {
        builder
            .addCase(fetchNewFeatureData.pending, (state) => {
                state.loading = true;
            })
            .addCase(fetchNewFeatureData.fulfilled, (state, action) => {
                state.loading = false;
                state.data = action.payload;
            })
            .addCase(fetchNewFeatureData.rejected, (state, action) => {
                state.loading = false;
                state.error = action.payload;
            });
    }
});

export const { clearError } = newFeatureSlice.actions;
export default newFeatureSlice.reducer;
```

## Testing

### Unit Testing

#### Testing Core Components
```python
# tests/unit/test_cgm.py
import pytest
from unittest.mock import Mock, AsyncMock
from sutazai.core.cgm import CodeGenerationModule

class TestCodeGenerationModule:
    @pytest.fixture
    def cgm(self):
        return CodeGenerationModule()
    
    @pytest.mark.asyncio
    async def test_generate_code(self, cgm):
        # Mock dependencies
        cgm.neural_generator.generate = AsyncMock(return_value="def test(): pass")
        cgm.quality_assessor.assess = Mock(return_value=0.9)
        
        result = await cgm.generate_code("create a test function")
        
        assert "def test():" in result
        cgm.neural_generator.generate.assert_called_once()
```

#### Testing API Endpoints
```python
# tests/unit/test_api.py
import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_generate_code_endpoint():
    request_data = {
        "prompt": "create a function",
        "language": "python"
    }
    response = client.post("/api/v1/generate/code", json=request_data)
    assert response.status_code == 200
    assert "generated_code" in response.json()
```

### Integration Testing

#### Testing Component Integration
```python
# tests/integration/test_ai_workflow.py
import pytest
from sutazai.core import SutazAI

class TestAIWorkflow:
    @pytest.mark.asyncio
    async def test_complete_generation_workflow(self):
        ai = SutazAI()
        
        # Test complete workflow
        result = await ai.generate_code("create a REST API")
        
        assert result.code is not None
        assert result.quality_score > 0.7
        assert "api" in result.code.lower()
```

### End-to-End Testing

#### Playwright E2E Tests
```javascript
// tests/e2e/test_ui.spec.js
const { test, expect } = require('@playwright/test');

test('complete user workflow', async ({ page }) => {
    // Navigate to application
    await page.goto('http://localhost:8000');
    
    // Login
    await page.fill('[data-testid="email"]', 'test@example.com');
    await page.fill('[data-testid="password"]', 'password');
    await page.click('[data-testid="login-button"]');
    
    // Test code generation
    await page.click('[data-testid="code-generation"]');
    await page.fill('[data-testid="prompt"]', 'create a function');
    await page.click('[data-testid="generate"]');
    
    // Verify result
    await expect(page.locator('[data-testid="generated-code"]')).toBeVisible();
});
```

## Code Quality and Standards

### Coding Standards

#### Python Code Style
```python
# Use type hints
def process_data(data: List[Dict[str, Any]]) -> ProcessedData:
    """Process input data and return processed result.
    
    Args:
        data: List of dictionaries containing raw data
        
    Returns:
        ProcessedData: Processed data object
        
    Raises:
        ValidationError: If data format is invalid
    """
    pass

# Use dataclasses for structured data
@dataclass
class ProcessedData:
    result: str
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)
```

#### JavaScript/React Style
```javascript
// Use TypeScript interfaces
interface ComponentProps {
    title: string;
    data?: Array<DataItem>;
    onUpdate?: (data: DataItem) => void;
}

// Use functional components with hooks
const MyComponent: React.FC<ComponentProps> = ({ 
    title, 
    data = [], 
    onUpdate 
}) => {
    const [loading, setLoading] = useState<boolean>(false);
    
    return (
        <div className="my-component">
            {/* Component content */}
        </div>
    );
};
```

### Code Review Guidelines

#### Review Checklist
- [ ] Code follows style guidelines
- [ ] Tests are included and pass
- [ ] Documentation is updated
- [ ] Security considerations addressed
- [ ] Performance impact evaluated
- [ ] Error handling implemented
- [ ] Logging added where appropriate

#### Pull Request Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
```

## Performance Optimization

### Profiling and Monitoring

#### Application Profiling
```python
# Profile performance
import cProfile
import pstats

def profile_function():
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Code to profile
    result = expensive_function()
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)
    
    return result
```

#### Memory Profiling
```python
# Memory profiling with memory_profiler
from memory_profiler import profile

@profile
def memory_intensive_function():
    # Function code here
    pass
```

### Database Optimization

#### Query Optimization
```python
# Use indexes
CREATE INDEX idx_user_email ON users(email);
CREATE INDEX idx_session_token ON sessions(session_token);

# Optimize queries
def get_user_sessions(user_id: str):
    # Use joins instead of multiple queries
    query = """
    SELECT u.email, s.session_token, s.created_at
    FROM users u
    JOIN sessions s ON u.id = s.user_id
    WHERE u.id = ?
    """
    return db.execute(query, (user_id,))
```

## Deployment

### Docker Development

#### Development Dockerfile
```dockerfile
# Dockerfile.dev
FROM python:3.9-slim

WORKDIR /app

# Install development dependencies
COPY requirements-dev.txt .
RUN pip install -r requirements-dev.txt

# Copy source code
COPY . .

# Expose port
EXPOSE 8000

# Development command
CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
```

#### Docker Compose for Development
```yaml
# docker-compose.dev.yml
version: '3.8'

services:
  sutazai-dev:
    build:
      context: .
      dockerfile: Dockerfile.dev
    ports:
      - "8000:8000"
    volumes:
      - .:/app
      - /app/venv
    environment:
      - ENVIRONMENT=development
      - DEBUG=true
    depends_on:
      - redis
      - postgres

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"

  postgres:
    image: postgres:13
    environment:
      POSTGRES_DB: sutazaidev
      POSTGRES_USER: dev
      POSTGRES_PASSWORD: devpass
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

### Continuous Integration

#### GitHub Actions Workflow
```yaml
# .github/workflows/ci.yml
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install -r requirements-dev.txt
    
    - name: Run tests
      run: |
        pytest tests/ --cov=sutazai
    
    - name: Run linting
      run: |
        flake8 sutazai/
        black --check sutazai/
    
    - name: Security scan
      run: |
        bandit -r sutazai/
```

## Contributing Guidelines

### Development Workflow

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/feature-name
   ```

2. **Make Changes**
   - Follow coding standards
   - Add tests
   - Update documentation

3. **Commit Changes**
   ```bash
   git add .
   git commit -m "feat: add new feature"
   ```

4. **Push and Create PR**
   ```bash
   git push origin feature/feature-name
   # Create pull request on GitHub
   ```

5. **Code Review**
   - Address review comments
   - Update tests if needed

6. **Merge**
   - Squash and merge
   - Delete feature branch

### Commit Message Format
```
type(scope): description

[optional body]

[optional footer]
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

## Resources

### Development Tools
- **IDE**: Visual Studio Code with Python extension
- **API Testing**: Postman or Insomnia
- **Database**: DB Browser for SQLite
- **Monitoring**: Grafana + Prometheus
- **Version Control**: Git with conventional commits

### Useful Libraries
- **FastAPI**: Web framework
- **SQLAlchemy**: ORM
- **Pytest**: Testing framework
- **Black**: Code formatting
- **Flake8**: Linting
- **Pre-commit**: Git hooks

### Documentation
- **Internal**: `/docs` directory
- **API Docs**: Generated by FastAPI
- **Code Docs**: Sphinx documentation
- **Architecture**: Lucidchart diagrams

---

Happy coding! 🚀 For questions, reach out to the development team on Discord or create an issue on GitHub.
