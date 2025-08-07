#!/usr/bin/env python3
"""
Run Hygiene Suite (Wrapper)

Reuses existing, safe, file-only checks to produce a consolidated report:
- scripts/check_naming.py
- scripts/check_secrets.py
- scripts/validate-compliance.py
- scripts/verify_claude_rules.py (agent rules presence)

Outputs a JSON and Markdown summary under ./reports.
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path("/opt/sutazaiapp")


CHECKS = [
    ("naming", [sys.executable, "scripts/check_naming.py"]),
    ("secrets", [sys.executable, "scripts/check_secrets.py"]),
    ("compliance", [sys.executable, "scripts/validate-compliance.py"]),
    ("claude_rules", [sys.executable, "scripts/verify_claude_rules.py"]),
]


def run_check(name: str, cmd: list[str]) -> dict:
    try:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        return {
            "name": name,
            "cmd": " ".join(cmd),
            "returncode": proc.returncode,
            "stdout": proc.stdout[-10000:],  # keep last 10k chars
            "stderr": proc.stderr[-4000:],
            "status": "pass" if proc.returncode == 0 else "fail",
        }
    except Exception as e:
        return {"name": name, "error": str(e), "status": "error", "cmd": " ".join(cmd)}


def main() -> int:
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)

    results = [run_check(n, c) for n, c in CHECKS]

    summary = {
        "timestamp": timestamp,
        "overall_status": "pass" if all(r.get("status") == "pass" for r in results) else "fail",
        "results": results,
    }

    # Write JSON
    json_path = reports_dir / f"hygiene_suite_{timestamp}.json"
    json_path.write_text(json.dumps(summary, indent=2))

    # Write Markdown summary
    md_lines = [
        f"# Hygiene Suite Report - {timestamp}",
        f"Overall: {'✅ PASS' if summary['overall_status']=='pass' else '❌ FAIL'}",
        "",
    ]
    for r in results:
        md_lines.append(f"## {r['name']}")
        status = r.get("status", "unknown").upper()
        md_lines.append(f"- Status: {status}")
        md_lines.append(f"- Command: `{r['cmd']}`")
        if r.get("stdout"):
            md_lines.append("<details><summary>stdout</summary>\n\n```")
            md_lines.append(r["stdout"])
            md_lines.append("```\n</details>")
        if r.get("stderr"):
            md_lines.append("<details><summary>stderr</summary>\n\n```")
            md_lines.append(r["stderr"])
            md_lines.append("```\n</details>")
        md_lines.append("")

    md_path = reports_dir / f"hygiene_suite_{timestamp}.md"
    md_path.write_text("\n".join(md_lines))

    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")
    return 0 if summary["overall_status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

