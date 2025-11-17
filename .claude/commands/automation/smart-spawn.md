# smart-spawn

Intelligently spawn agents based on workload analysis.

## Usage
```bash
npx removed automation smart-spawn [options]
```

## Options
- `--analyze` - Analyze before spawning
- `--threshold <n>` - Spawn threshold
- `--topology <type>` - Preferred topology

## Examples
```bash
# Smart spawn with analysis
npx removed automation smart-spawn --analyze

# Set spawn threshold
npx removed automation smart-spawn --threshold 5

# Force topology
npx removed automation smart-spawn --topology hierarchical
```
