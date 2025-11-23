# agent-metrics

View agent performance metrics.

## Usage
```bash
npx removed agent metrics [options]
```

## Options
- `--agent-id <id>` - Specific agent
- `--period <time>` - Time period
- `--format <type>` - Output format

## Examples
```bash
# All agents metrics
npx removed agent metrics

# Specific agent
npx removed agent metrics --agent-id agent-001

# Last hour
npx removed agent metrics --period 1h
```
