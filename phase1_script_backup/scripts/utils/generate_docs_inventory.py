#!/usr/bin/env python3
import os
import json
import hashlib
from datetime import datetime, timezone

DOCS_DIR = os.path.join(os.getcwd(), 'docs')
OUT_DIR = os.path.join(os.getcwd(), 'reports', 'cleanup')
OUT_FILE = os.path.join(OUT_DIR, 'docs_inventory.json')


def file_sha1(path, chunk_size=65536):
    h = hashlib.sha1()
    with open(path, 'rb') as f:
        while True:
            data = f.read(chunk_size)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


def main():
    if not os.path.isdir(DOCS_DIR):
        raise SystemExit(f"Docs directory not found: {DOCS_DIR}")
    os.makedirs(OUT_DIR, exist_ok=True)

    inventory = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": DOCS_DIR,
        "files": [],
        "summary": {"count": 0, "bytes": 0, "by_ext": {}}
    }

    for root, _, files in os.walk(DOCS_DIR):
        for name in files:
            path = os.path.join(root, name)
            rel = os.path.relpath(path, DOCS_DIR)
            try:
                st = os.stat(path)
                size = st.st_size
                mtime = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat()
                ext = os.path.splitext(name)[1].lower() or "<none>"
                sha1 = file_sha1(path) if size < 5_000_000 else None  # skip huge files
            except OSError:
                continue

            inventory["files"].append({
                "path": rel,
                "bytes": size,
                "mtime": mtime,
                "ext": ext,
                "sha1": sha1,
            })

            inv_sum = inventory["summary"]
            inv_sum["count"] += 1
            inv_sum["bytes"] += size
            inv_sum["by_ext"].setdefault(ext, {"count": 0, "bytes": 0})
            inv_sum["by_ext"][ext]["count"] += 1
            inv_sum["by_ext"][ext]["bytes"] += size

    inventory["files"].sort(key=lambda x: x["path"]) 

    with open(OUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(inventory, f, indent=2)

    print(f"Wrote inventory: {OUT_FILE}")


if __name__ == '__main__':
    main()

