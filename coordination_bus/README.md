# Coordination Bus

Local, file-based coordination bus for agent status and directives.

- Messages channel: `coordination_bus/messages/` (JSON Lines, one file per topic)
- Agents roster: `coordination_bus/agents.csv`
- Architect directives: `coordination_bus/directives.jsonl`
- Heartbeats: `coordination_bus/heartbeats.jsonl`

Message schema (status updates):
{
  "ts": "2025-08-07T00:00:00Z",
  "agent_id": "Claude-042",
  "phase": "discovery|resolution|integration|validation",
  "task_id": "CLN-20250807-0001",
  "status": "queued|in_progress|blocked|done",
  "summary": "short description",
  "artifact": "path/to/file"
}

All entries are append-only to preserve an immutable audit trail.

