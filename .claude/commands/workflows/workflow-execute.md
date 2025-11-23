# workflow-execute

Execute saved workflows.

## Usage
```bash
npx removed workflow execute [options]
```

## Options
- `--name <name>` - Workflow name
- `--params <json>` - Workflow parameters
- `--dry-run` - Preview execution

## Examples
```bash
# Execute workflow
npx removed workflow execute --name "deploy-api"

# With parameters
npx removed workflow execute --name "test-suite" --params '{"env": "staging"}'

# Dry run
npx removed workflow execute --name "deploy-api" --dry-run
```
