#!/usr/bin/env python3
"""
Summarize backend OpenAPI (docs/backend_openapi.json) into Markdown.
Writes: docs/backend_endpoints.md
Groups by tag and lists method/path pairs.
"""
import json
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    openapi_path = root / "docs" / "backend_openapi.json"
    out_path = root / "docs" / "backend_endpoints.md"

    if not openapi_path.exists():
        print(f"OpenAPI file not found: {openapi_path}. Generate it first.")
        return 1

    data = json.loads(openapi_path.read_text())
    paths = data.get("paths", {})

    # Build tag -> list of (method, path)
    by_tag = {}
    for path, methods in paths.items():
        for method, op in methods.items():
            if method.lower() not in {"get", "post", "put", "delete", "patch", "options", "head"}:
                continue
            tags = op.get("tags") or ["untagged"]
            for tag in tags:
                by_tag.setdefault(tag, []).append((method.upper(), path))

    # Sort for stable output
    for tag in by_tag:
        by_tag[tag].sort(key=lambda x: (x[0], x[1]))

    lines = []
    lines.append("# Backend Endpoints (from OpenAPI)\n")
    lines.append(f"Total paths: {len(paths)}\n")
    lines.append("")
    for tag in sorted(by_tag.keys()):
        lines.append(f"## {tag} ({len(by_tag[tag])})")
        for method, p in by_tag[tag]:
            lines.append(f"- `{method}` {p}")
        lines.append("")

    out_path.write_text("\n".join(lines))
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

