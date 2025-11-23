# workflow-create

Create reusable workflow templates.

## Usage
```bash
npx removed workflow create [options]
```

## Options
- `--name <name>` - Workflow name
- `--from-history` - Create from history
- `--interactive` - Interactive creation

## Examples
```bash
# Create workflow
npx removed workflow create --name "deploy-api"

# From history
npx removed workflow create --name "test-suite" --from-history

# Interactive mode
npx removed workflow create --interactive
```
