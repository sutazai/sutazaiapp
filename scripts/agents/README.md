# Agent Scripts

This directory contains executable scripts for agent implementations.

## Scripts

### agentgpt_executor.py
- **Purpose**: Execute local tasks using Ollama TinyLlama model
- **Usage**: `python agentgpt_executor.py "task description" [--file FILE] [--action ACTION]`
- **Requirements**: Ollama installed, TinyLlama model downloaded

### comprehensive-analysis-agent.py
- **Purpose**: Comprehensive Analysis Agent - Enforces Rule 3: Analyze Everything—Every Time
- **Usage**: `python comprehensive-analysis-agent.py [--report-dir REPORT_DIR] [--format FORMAT] [--fix]`
- **Requirements**: Python 3.8+, standard library modules
- **Features**:
  - Systematic review of entire codebase
  - Analysis of 10 key categories: files, folders, scripts, code logic, dependencies, APIs, configuration, build/deploy, logs/monitoring, testing
  - Generates detailed JSON and Markdown reports
  - Calculates compliance scores
  - Optional automatic fixing of certain issues (file permissions, empty folders)
  - Issue severity classification (Critical, High, Medium, Low)

## Organization
All agent implementation scripts should be placed here, following these rules:
- One script per agent
- Clear, descriptive filenames
- Proper documentation headers
- No duplicates or variations