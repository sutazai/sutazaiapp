# memory-persist

Persist memory across sessions.

## Usage
```bash
npx removed memory persist [options]
```

## Options
- `--export <file>` - Export to file
- `--import <file>` - Import from file
- `--compress` - Compress memory data

## Examples
```bash
# Export memory
npx removed memory persist --export memory-backup.json

# Import memory
npx removed memory persist --import memory-backup.json

# Compressed export
npx removed memory persist --export memory.gz --compress
```
