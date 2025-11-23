# memory-search

Search through stored memory.

## Usage
```bash
npx removed memory search [options]
```

## Options
- `--query <text>` - Search query
- `--pattern <regex>` - Pattern matching
- `--limit <n>` - Result limit

## Examples
```bash
# Search memory
npx removed memory search --query "authentication"

# Pattern search
npx removed memory search --pattern "api-.*"

# Limited results
npx removed memory search --query "config" --limit 10
```
