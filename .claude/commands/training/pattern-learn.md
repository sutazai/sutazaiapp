# pattern-learn

Learn patterns from successful operations.

## Usage
```bash
npx removed training pattern-learn [options]
```

## Options
- `--source <type>` - Pattern source
- `--threshold <score>` - Success threshold
- `--save <name>` - Save pattern set

## Examples
```bash
# Learn from all ops
npx removed training pattern-learn

# High success only
npx removed training pattern-learn --threshold 0.9

# Save patterns
npx removed training pattern-learn --save optimal-patterns
```
