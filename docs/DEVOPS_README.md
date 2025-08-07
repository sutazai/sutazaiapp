# DevOps Health Verification & Smoke Tests

This suite verifies operational status of core services (latency and reachability) and provides building blocks for CI gating.

## Scripts

- `scripts/devops/check_services_health.sh`: Bash script performing TCP reachability and basic HTTP checks with timestamped logs. CLI flags avoid hardcoded values.
  - Example:
    - `scripts/devops/check_services_health.sh --ollama-host localhost --ollama-port 10104 --kong-host localhost --kong-port 10005 --consul-host localhost --consul-port 10006 --vector-start 10100 --vector-end 10103`
  - Output: per‑service latency and HTTP status summaries; exits non‑zero on reachability failures.

- `scripts/devops/check_services_health.py`: Python (argparse) variant; checks only services explicitly provided.
  - Example:
    - `python scripts/devops/check_services_health.py --ollama localhost:10104 --kong localhost:10005 --consul localhost:10006 --vector-range 10100-10103`
  - Output: timestamped logs; exits non‑zero on any failed checks.

## CI/CD Integration

- Add a pipeline step that runs the script and fails the job on non‑zero exit.
- Recommended frequency: on PRs to protected branches and nightly.

## Troubleshooting

- Verify ports align with the active compose profile (`docker-compose.*.yml`).
- Check Kong Admin URL via `KONG_ADMIN_URL` when using `scripts/configure_kong.sh`.
- Confirm Consul Agent availability; override via `CONSUL_HOST`/`CONSUL_PORT` env vars for `register_with_consul.py`.

## Related Utilities

- `scripts/register_with_consul.py` — idempotent service registration
- `scripts/configure_kong.sh` — idempotent Kong Service/Route configuration
