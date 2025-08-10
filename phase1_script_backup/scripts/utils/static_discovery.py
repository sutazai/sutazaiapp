#!/usr/bin/env python3
import os
import re
import ast
import json
import hashlib
from typing import Dict, List, Set, Tuple
from datetime import datetime, timezone

ROOT = os.getcwd()
SRC_DIRS = [
    'backend', 'services', 'agents', 'frontend', 'scripts', 'src'
]
REPORT_DIR = os.path.join(ROOT, 'reports', 'cleanup')
DEP_GRAPH_FILE = os.path.join(REPORT_DIR, 'dependency_graph.json')
CONFLICT_MAP_FILE = os.path.join(REPORT_DIR, 'conflict_map.json')

DEPRECATED_PATTERNS = [
    r"\bimp\.",
    r"\bdistutils\b",
    r"asyncio\.get_event_loop\(",
    r"logger\.warn\(",
    r"setDaemon\(",
]


def file_sha1(path: str) -> str:
    h = hashlib.sha1()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def norm_module_from_path(path: str) -> str:
    rel = os.path.relpath(path, ROOT)
    if rel.endswith('.py'):
        rel = rel[:-3]
    return rel.replace(os.sep, '.')


def ast_without_pos(node: ast.AST) -> ast.AST:
    for n in ast.walk(node):
        for attr in ('lineno', 'col_offset', 'end_lineno', 'end_col_offset'):
            if hasattr(n, attr):
                setattr(n, attr, None)
    return node


def hash_def(node: ast.AST) -> str:
    cleaned = ast_without_pos(ast.fix_missing_locations(node))
    dumped = ast.dump(cleaned, include_attributes=False)
    return hashlib.sha1(dumped.encode('utf-8')).hexdigest()


def scan_python_file(path: str):
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        src = f.read()
    try:
        tree = ast.parse(src, filename=path)
    except SyntaxError:
        return None
    imports: Set[str] = set()
    dups: List[Tuple[str, str]] = []  # list of (kind:name, hash)
    stubs: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split('.')[0])
        elif isinstance(node, ast.FunctionDef):
            h = hash_def(node)
            dups.append((f"func:{node.name}", h))
            # stub heuristics
            body = node.body or []
            if len(body) == 1:
                b0 = body[0]
                if isinstance(b0, ast.Pass):
                    stubs.append(node.name)
                elif isinstance(b0, ast.Expr) and isinstance(b0.value, ast.Constant) and isinstance(b0.value.value, str):
                    # Only docstring
                    stubs.append(node.name)
                elif isinstance(b0, ast.Raise) and isinstance(b0.exc, ast.Name) and b0.exc.id in {'NotImplementedError'}:
                    stubs.append(node.name)
        elif isinstance(node, ast.ClassDef):
            h = hash_def(node)
            dups.append((f"class:{node.name}", h))
    # deprecated usage
    deprecated_hits: List[str] = []
    for pat in DEPRECATED_PATTERNS:
        if re.search(pat, src):
            deprecated_hits.append(pat)
    return {
        'imports': sorted(imports),
        'dups': dups,
        'stubs': sorted(set(stubs)),
        'deprecated': sorted(set(deprecated_hits)),
    }


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    files: List[str] = []
    for base in SRC_DIRS:
        root = os.path.join(ROOT, base)
        if not os.path.isdir(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            for name in filenames:
                if name.endswith('.py'):
                    files.append(os.path.join(dirpath, name))

    dep_graph = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'nodes': [],
        'edges': [],
        'summary': {}
    }
    node_index: Dict[str, int] = {}
    edges_set: Set[Tuple[str, str]] = set()

    # For conflicts
    hash_index: Dict[str, List[Dict]] = {}
    deprecations: Dict[str, List[str]] = {}
    stubs_index: Dict[str, List[str]] = {}

    for path in files:
        rel = os.path.relpath(path, ROOT)
        mod = norm_module_from_path(path)
        node_index[mod] = len(dep_graph['nodes'])
        dep_graph['nodes'].append({'id': mod, 'path': rel, 'sha1': file_sha1(path)})

        res = scan_python_file(path)
        if not res:
            continue
        for imp in res['imports']:
            edges_set.add((mod, imp))
        for kind_name, h in res['dups']:
            hash_index.setdefault(h, []).append({'symbol': kind_name, 'module': mod, 'path': rel})
        if res['deprecated']:
            deprecations[mod] = res['deprecated']
        if res['stubs']:
            stubs_index[mod] = res['stubs']

    # Build edges
    dep_graph['edges'] = [{'from': a, 'to': b} for (a, b) in sorted(edges_set)]
    dep_graph['summary'] = {
        'files_scanned': len(files),
        'modules': len(dep_graph['nodes']),
        'edges': len(dep_graph['edges'])
    }

    with open(DEP_GRAPH_FILE, 'w', encoding='utf-8') as f:
        json.dump(dep_graph, f, indent=2)

    # Conflicts: duplicates (same AST hash used in >1 module), deprecations, stubs
    duplicates: List[Dict] = []
    for h, occurrences in hash_index.items():
        if len(occurrences) > 1:
            symbols = sorted(set(o['symbol'] for o in occurrences))
            duplicates.append({'hash': h, 'symbols': symbols, 'occurrences': occurrences})

    # Brief cycle detection on edges among local modules only
    # Limit targets to known modules to avoid stdlib noise
    local_modules = set(node_index.keys())
    graph: Dict[str, Set[str]] = {}
    for a, b in edges_set:
        if a not in local_modules:
            continue
        # Only consider edges to local modules (by prefix matching on known top-levels)
        if any(b.startswith(prefix + '.') or b == prefix for prefix in ['backend', 'services', 'agents', 'frontend', 'scripts', 'src']):
            graph.setdefault(a, set()).add(b)

    def dfs_cycle(start: str) -> List[List[str]]:
        visited: Set[str] = set()
        stack: List[str] = []
        cycles: List[List[str]] = []

        def visit(n: str):
            if n in stack:
                i = stack.index(n)
                cycles.append(stack[i:] + [n])
                return
            if n in visited:
                return
            visited.add(n)
            stack.append(n)
            for m in graph.get(n, []):
                visit(m)
            stack.pop()

        visit(start)
        return cycles

    cycles: List[List[str]] = []
    for n in graph.keys():
        c = dfs_cycle(n)
        for cycle in c:
            # Normalize cycle representation
            if len(cycle) > 2:
                rotated = min([cycle[i:] + cycle[:i] for i in range(len(cycle)-1)])
                if rotated not in cycles:
                    cycles.append(rotated)

    conflict_map = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'duplicates': duplicates,
        'deprecated_usage': deprecations,
        'stub_implementations': stubs_index,
        'circular_imports': cycles,
        'summary': {
            'duplicate_groups': len(duplicates),
            'files_with_deprecations': len(deprecations),
            'files_with_stubs': len(stubs_index),
            'circular_components': len(cycles)
        }
    }

    with open(CONFLICT_MAP_FILE, 'w', encoding='utf-8') as f:
        json.dump(conflict_map, f, indent=2)

    print(f"Wrote: {DEP_GRAPH_FILE}")
    print(f"Wrote: {CONFLICT_MAP_FILE}")


if __name__ == '__main__':
    main()

