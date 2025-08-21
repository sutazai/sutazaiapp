# Critical Rules

## 🚨 ANTI-HALLUCINATION
1. **VERIFY**: Check files before claims
2. **QUOTE**: Use exact quotes + line numbers
3. **ADMIT**: Say "I need to verify" when unsure
4. **GROUND**: Only reference confirmed features

## 🚨 CONCURRENT EXECUTION
**GOLDEN RULE**: 1 MESSAGE = ALL OPERATIONS

### MANDATORY
- Batch ALL todos (5-10+ minimum)
- Spawn ALL agents together
- Batch ALL file operations
- Batch ALL bash commands

### FILE ORGANIZATION
**NEVER save to root. Use:**
- `/src` - Source code
- `/tests` - Tests
- `/docs` - Documentation
- `/config` - Configuration

## Claude Code vs MCP
**Claude Code**: File ops, coding, bash, git, testing
**MCP**: Coordination, memory, neural, swarm

## Hooks Protocol
```bash
# Before
npx claude-flow@alpha hooks pre-task --description "[task]"
# During
npx claude-flow@alpha hooks post-edit --file "[file]"
# After
npx claude-flow@alpha hooks post-task --task-id "[task]"
```