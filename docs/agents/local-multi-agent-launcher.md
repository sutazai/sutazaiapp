# Local Multi-Agent Launcher

Purpose: Launch multiple local SutazAI agent instances concurrently for development and validation.

## Overview

`scripts/launch_local_agents.py` spawns N independent agent processes using the existing `GenericAgentWithHealth` implementation. Each agent exposes a `/health` endpoint on a unique port and attempts non-fatal registration/heartbeat with your backend, if reachable.

## Requirements

- Python 3.8+
- Working checkout of this repository (no external APIs required)
- Optional: Local backend at `http://localhost:8000` and Ollama at `http://localhost:11434`

## Usage

Example: launch 20 agents on ports 8600–8619

```
python3 scripts/launch_local_agents.py --count 20 --base-port 8600 \
  --agent-type local-dev \
  --backend-url http://localhost:8000 \
  --ollama-url http://localhost:11434
```

Health endpoints:

- `http://localhost:8600/health` … `http://localhost:8619/health`

## Notes

- The launcher uses only local URLs you provide (Rule 16: local LLMs via Ollama).
- Backend registration/heartbeats are best-effort; failures are logged but not fatal.
- Ctrl+C cleanly terminates all agent processes.

