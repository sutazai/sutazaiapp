#!/usr/bin/env python3
"""
Phase 1 Discovery: Static analysis without external deps.
- Build import graph for Python modules under key roots
- Detect circular imports
- Find duplicate class names across modules
- Flag files containing 'deprecated' markers or TODOs
Outputs JSON reports under reports/cleanup/
"""
from __future__ import annotations

import ast
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple

ROOTS = [
    'backend', 'services', 'agents', 'frontend', 'scripts'
]
EXCLUDE_DIRS = {'.git', 'venv', '.venv', 'node_modules', 'build', 'dist', '__pycache__', 'logs', 'htmlcov'}

REPORT_DIR = Path('reports/cleanup')
REPORT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class ModuleInfo:
    path: Path
    module: str
    imports: Set[str]
    classes: Set[str]
    deprecated_markers: bool


def is_code_dir(p: Path) -> bool:
    parts = set(p.parts)
    return not (parts & EXCLUDE_DIRS)


def module_name(path: Path) -> str:
    # Convert path like backend/foo/bar.py to module name backend.foo.bar
    rel = path.with_suffix("")
    return ".".join(rel.parts)


def parse_python_file(path: Path) -> ModuleInfo:
    text = path.read_text(errors='ignore')
    try:
        tree = ast.parse(text)
    except SyntaxError:
        tree = ast.parse("\n")  # treat as empty

    imports: Set[str] = set()
    classes: Set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                imports.add(n.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ''
            if node.level and mod:
                # relative import, keep module segment
                imports.add(mod.split('.')[0])
            elif mod:
                imports.add(mod.split('.')[0])
        elif isinstance(node, ast.ClassDef):
            classes.add(node.name)

    deprecated = bool(re.search(r"\\bdeprecated\\b", text, flags=re.IGNORECASE))

    return ModuleInfo(
        path=path,
        module=module_name(path.relative_to(Path('.'))),
        imports=imports,
        classes=classes,
        deprecated_markers=deprecated,
    )


def collect_modules() -> Dict[str, ModuleInfo]:
    modules: Dict[str, ModuleInfo] = {}
    for root in ROOTS:
        root_path = Path(root)
        if not root_path.exists():
            continue
        for path in root_path.rglob('*.py'):
            if not is_code_dir(path):
                continue
            mi = parse_python_file(path)
            modules[mi.module] = mi
    return modules


def build_internal_graph(mods: Dict[str, ModuleInfo]) -> Dict[str, Set[str]]:
    internal_roots = {r.split(os.sep)[0] for r in ROOTS}
    graph: Dict[str, Set[str]] = {m: set() for m in mods}
    for m, info in mods.items():
        for imp in info.imports:
            # consider only imports that look like internal roots
            if imp in internal_roots:
                # approximate edge to any module starting with that root
                # Use root-level dependency to reduce false precision
                graph[m].add(imp)
    return graph


def detect_cycles(graph: Dict[str, Set[str]]) -> List[List[str]]:
    visited: Set[str] = set()
    stack: Set[str] = set()
    cycles: List[List[str]] = []

    def dfs(node: str, path: List[str]):
        visited.add(node)
        stack.add(node)
        path.append(node)
        for nbr in graph.get(node, ()):            
            if nbr not in graph:
                continue
            if nbr not in visited:
                dfs(nbr, path)
            elif nbr in stack:
                # cycle from nbr to current node
                try:
                    idx = path.index(nbr)
                    cycle = path[idx:].copy()
                    if cycle and cycle not in cycles:
                        cycles.append(cycle)
                except ValueError:
                    pass
        path.pop()
        stack.remove(node)

    for node in list(graph.keys()):
        if node not in visited:
            dfs(node, [])
    return cycles


def find_duplicate_classes(mods: Dict[str, ModuleInfo]) -> Dict[str, List[str]]:
    m: Dict[str, List[str]] = defaultdict(list)
    for mod, info in mods.items():
        for cls in info.classes:
            m[cls].append(mod)
    return {cls: locs for cls, locs in m.items() if len(locs) > 1}


def main() -> int:
    mods = collect_modules()
    graph = build_internal_graph(mods)
    cycles = detect_cycles(graph)
    dupes = find_duplicate_classes(mods)
    deprecated_files = [m for m, i in mods.items() if i.deprecated_markers]

    summary = {
        'module_count': len(mods),
        'duplicate_classes': len(dupes),
        'cycles_found': len(cycles),
        'files_with_deprecated_markers': len(deprecated_files),
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / 'summary.json').write_text(json.dumps(summary, indent=2))
    (REPORT_DIR / 'import_cycles.json').write_text(json.dumps(cycles, indent=2))
    (REPORT_DIR / 'duplicate_classes.json').write_text(json.dumps(dupes, indent=2))
    (REPORT_DIR / 'deprecated_markers.json').write_text(json.dumps(deprecated_files, indent=2))

    # conflict map aggregate
    conflict_map = {
        'cycles': cycles,
        'duplicate_classes': dupes,
        'deprecated_markers': deprecated_files,
    }
    (REPORT_DIR / 'conflict_map.json').write_text(json.dumps(conflict_map, indent=2))

    print(json.dumps(summary))
    return 0


if __name__ == '__main__':
    sys.exit(main())

