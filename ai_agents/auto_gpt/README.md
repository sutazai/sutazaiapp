# AutoGPT Agent

An autonomous agent capable of executing complex tasks using language models and various tools.

## Features

- Task planning and execution
- Memory management with conversation history
- Tool registry for extensible functionality
- Error handling and recovery
- Progress monitoring and evaluation
- Persistent state management
- Configurable model parameters

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
# Create .env file
OPENAI_API_KEY=your_api_key_here
OPENAI_API_BASE=https://api.openai.com/v1  # Optional
```

## Usage

```python
from ai_agents.auto_gpt import AutoGPTAgent

# Create agent instance
agent = AutoGPTAgent(
    config={
        "model_config": {
            "model_name": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 2000
        },
        "memory_size": 10,
        "max_iterations": 25
    },
    name="my_agent",
    log_dir="logs/my_agent"
)

# Initialize agent
await agent.initialize()

# Execute a task
result = await agent.execute({
    "objective": "Research and summarize recent developments in quantum computing",
    "context": {
        "focus_areas": ["hardware", "algorithms", "applications"],
        "time_frame": "last 6 months"
    }
})

# Clean up
await agent.shutdown()
```

## Configuration

The agent accepts the following configuration options:

- `model_config`: Configuration for the language model
  - `model_name`: Name of the model to use (default: "gpt-4")
  - `temperature`: Sampling temperature (default: 0.7)
  - `max_tokens`: Maximum tokens per request (default: 2000)
  - `top_p`: Nucleus sampling parameter (default: 1.0)
  - `frequency_penalty`: Token frequency penalty (default: 0.0)
  - `presence_penalty`: Token presence penalty (default: 0.0)

- `memory_size`: Maximum number of messages to keep in memory (default: 10)
- `max_iterations`: Maximum number of iterations per task (default: 25)

## Tools

The agent comes with several built-in tools and supports registering custom tools:

```python
from ai_agents.auto_gpt.tools import Tool, ToolParameter

# Create a custom tool
@agent.register_tool(
    name="custom_tool",
    description="Description of what the tool does",
    parameters=[
        ToolParameter(
            name="param1",
            description="Description of parameter 1",
            type="str",
            required=True
        ),
        ToolParameter(
            name="param2",
            description="Description of parameter 2",
            type="int",
            required=False,
            default=42
        )
    ]
)
def custom_tool(param1: str, param2: int = 42):
    # Tool implementation
    return f"Processed {param1} with {param2}"
```

## Task Structure

Tasks can be specified as either a string (simple objective) or a dictionary with additional context:

```python
# Simple task
task = "Generate a Python script that processes CSV files"

# Complex task
task = {
    "objective": "Generate a Python script that processes CSV files",
    "context": {
        "input_format": "customer_data.csv with columns: id, name, email",
        "output_format": "processed_data.csv with additional columns: age_group, region",
        "requirements": ["data validation", "error handling", "progress logging"]
    },
    "max_steps": 10  # Override default max_iterations for this task
}
```

## Error Handling

The agent includes built-in error handling and recovery mechanisms:

- Automatic retry for transient errors
- Step-level error handling with failure recording
- Task-level error handling with plan adjustment
- Persistent state management for recovery

## Logging

Logs are stored in the specified `log_dir`:

- `memory.json`: Conversation history and context
- `task.json`: Current task state and progress
- `performance.log`: Performance metrics and timing data

## Development

1. Run tests:
```bash
pytest tests/ -v
```

2. Run with coverage:
```bash
pytest tests/ --cov=ai_agents/auto_gpt --cov-report=html
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 