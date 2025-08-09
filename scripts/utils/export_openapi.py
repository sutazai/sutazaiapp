#!/usr/bin/env python3
"""
Export FastAPI OpenAPI schema from backend without starting a server.

Writes: docs/backend_openapi.json

Notes:
- Disables optional enterprise features via env to minimize import side-effects.
- Requires Python dependencies used by the backend to be installed.
"""
import os
import json
import sys
from pathlib import Path


def main() -> int:
    # Make backend importable (backend package uses 'app' module path)
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    # Disable optional features to reduce import footprint
    os.environ.setdefault("SUTAZAI_ENTERPRISE_FEATURES", "0")
    os.environ.setdefault("SUTAZAI_ENABLE_KNOWLEDGE_GRAPH", "0")
    os.environ.setdefault("SUTAZAI_ENABLE_COGNITIVE", "0")

    # Minimal required envs to avoid strict validations
    os.environ.setdefault("POSTGRES_HOST", "postgres")
    os.environ.setdefault("POSTGRES_USER", "sutazai")
    os.environ.setdefault("POSTGRES_PASSWORD", "sutazai_password")
    os.environ.setdefault("POSTGRES_DB", "sutazai")
    os.environ.setdefault("REDIS_HOST", "redis")
    os.environ.setdefault("OLLAMA_HOST", "http://ollama:10104")

    # Import the FastAPI app
    try:
        from backend.app.main import app  # type: ignore
    except Exception as e:
        print(f"[export_openapi] Failed to import backend app: {e}")
        return 2

    # Generate OpenAPI schema
    try:
        schema = app.openapi()
    except Exception as e:
        print(f"[export_openapi] Failed to build OpenAPI schema: {e}")
        return 3

    out_dir = repo_root / "docs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "backend_openapi.json"
    out_file.write_text(json.dumps(schema, indent=2))
    print(f"[export_openapi] Wrote {out_file}")
    print(f"[export_openapi] Paths: {len(schema.get('paths', {}))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

