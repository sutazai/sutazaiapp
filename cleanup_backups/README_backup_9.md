# SutazAI Code Improvement Workflow

This directory contains practical workflows for automated code analysis and improvement using multiple AI agents.

## Features

### 1. Code Improvement Workflow (`code_improvement_workflow.py`)

A comprehensive workflow that analyzes code quality using three specialized AI agents:

- **Senior AI Engineer**: Analyzes ML/AI code patterns, model optimization, and neural architecture
- **Testing QA Validator**: Checks test coverage, error handling, and code quality
- **Infrastructure DevOps Manager**: Reviews Docker configurations, deployment setup, and infrastructure

### 2. API Integration

The workflow is integrated with the backend API through new endpoints:

- `POST /api/v1/agents/workflows/code-improvement` - Start a code improvement analysis
- `GET /api/v1/agents/workflows/{workflow_id}` - Check workflow status
- `GET /api/v1/agents/workflows/{workflow_id}/report` - Get the full analysis report
- `POST /api/v1/agents/consensus` - Get multi-agent consensus on decisions
- `POST /api/v1/agents/delegate` - Delegate tasks to appropriate agents

### 3. Analysis Capabilities

The workflow analyzes:

#### Code Quality Metrics
- Lines of code
- Complexity score
- Test coverage estimation
- Code duplication
- Documentation quality

#### Issue Detection
- **Security Issues**: Hardcoded secrets, vulnerable patterns
- **Performance Issues**: Missing GPU checks, unoptimized loops
- **Style Violations**: Missing docstrings, inconsistent formatting
- **Bug Detection**: Missing error handling, unchecked exceptions
- **Testing Gaps**: Missing test files, no assertions

#### Actionable Improvements
- Specific code fixes with line numbers
- Refactoring recommendations
- Architecture improvements
- Best practice suggestions

## Usage

### Command Line

```bash
# Analyze a directory
python workflows/code_improvement_workflow.py /path/to/code --output report.md

# Run the demo
python workflows/demo_workflow.py
```

### Via API

```python
import httpx

# Start workflow
response = await client.post(
    "http://localhost:8000/api/v1/agents/workflows/code-improvement",
    json={"directory": "/opt/sutazaiapp/backend/app"}
)
workflow_id = response.json()["workflow_id"]

# Check status
status = await client.get(f"/api/v1/agents/workflows/{workflow_id}")

# Get report when complete
report = await client.get(f"/api/v1/agents/workflows/{workflow_id}/report")
```

## Example Output

The workflow generates both Markdown and JSON reports with:

1. **Metrics Summary**: Overall code health indicators
2. **Issue List**: Categorized by severity (critical, high, medium, low)
3. **Specific Fixes**: Line-by-line suggestions for improvements
4. **Agent Recommendations**: High-level architecture and process improvements

## Sample Report

```markdown
# Code Improvement Report

## Code Metrics
- Lines of Code: 16,247
- Complexity Score: 6.44
- Security Issues: 0
- Performance Issues: 1
- Style Violations: 262

## Top Issues

### 1. [HIGH] Missing error handling
   - File: knowledge_manager.py:483
   - Fix: Add specific exception handling with proper error messages

### 2. [MEDIUM] Hardcoded configuration
   - File: config.py:25
   - Fix: Move to environment variables
```

## Integration with Coordinator

The workflow can optionally integrate with the `ContinuousCoordinator` to:
- Delegate analysis tasks to specific agents
- Track agent collaboration
- Monitor workflow progress
- Handle agent failures gracefully

## Testing

Run the test script to verify the workflow:

```bash
python workflows/test_code_improvement.py
```

This will:
1. List available agents
2. Start a code improvement workflow
3. Monitor progress
4. Display results
5. Test agent consensus
6. Test task delegation

## Architecture

```
CodeImprovementWorkflow
├── SeniorAIEngineerAnalyzer
│   ├── ML pattern detection
│   ├── Model optimization checks
│   └── Neural architecture review
├── TestingQAValidatorAnalyzer
│   ├── Test coverage analysis
│   ├── Error handling validation
│   └── Code quality checks
└── InfrastructureDevOpsAnalyzer
    ├── Docker best practices
    ├── Security configuration
    └── Deployment readiness
```

## Benefits

1. **Automated Analysis**: No manual code review needed
2. **Multi-Perspective**: Different agents catch different issues
3. **Actionable Results**: Specific fixes, not just problems
4. **Scalable**: Can analyze entire codebases
5. **Integrated**: Works with existing SutazAI infrastructure

## Future Enhancements

- Add more specialized analyzers (Frontend, Database, Security)
- Implement automatic fix application
- Add code smell detection
- Integrate with CI/CD pipelines
- Support for more languages beyond Python