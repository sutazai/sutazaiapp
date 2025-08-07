Append-only JSONL ledger for cleanup operation.

Files:
- `ledger.jsonl`: all change events, status updates, test runs
- `approvals.jsonl`: architect approvals per task

Event schema:
{
  "ts": "2025-08-07T00:00:00Z",
  "event": "status|change|test|approval",
  "agent_id": "Claude-001",
  "phase": "discovery|resolution|integration|validation",
  "task_id": "CLN-20250807-0001",
  "detail": {...}
}

