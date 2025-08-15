# Detailed Waste Investigation - Phase 1: Configuration Files

## Investigation Date: 2025-08-15
## Investigator: rules-enforcer

## 1. ENVIRONMENT FILES INVESTIGATION

### File: .env (PRIMARY)
- **Size**: 19 lines
- **Purpose**: Generated production configuration
- **References**: 557,777 (heavily used)
- **Git History**: Multiple updates, last at v67
- **Status**: ACTIVE - PRIMARY CONFIGURATION
- **Decision**: KEEP - Primary configuration file

### File: .env.example  
- **Size**: 19 lines
- **Purpose**: Template for environment setup
- **References**: 20
- **Content**: DIFFERENT from .env (template vs actual values)
- **Status**: ACTIVE - Documentation/template
- **Decision**: KEEP - Needed for onboarding

### File: .env.secure
- **Size**: 231 lines
- **Purpose**: Secure production settings
- **References**: 28
- **Status**: POTENTIALLY ACTIVE
- **Investigation Needed**: Check if actively used in production

### File: .env.production.secure
- **Size**: 128 lines  
- **Purpose**: Production-specific secure settings
- **References**: 12
- **Duplication**: Likely overlaps with .env.secure
- **Status**: POSSIBLE DUPLICATE
- **Investigation Needed**: Compare with .env.secure for consolidation

### File: .env.agents
- **Size**: 24 lines
- **Purpose**: Agent-specific configuration
- **References**: 9 (low usage)
- **Content**: Ollama and OpenAI settings
- **Status**: QUESTIONABLE - Low usage
- **Investigation Needed**: Check if agents actually use this

### File: .env.ollama
- **Size**: 50 lines
- **Purpose**: Ollama-specific configuration
- **References**: 3 (very low usage)
- **Status**: LIKELY UNUSED
- **Investigation Needed**: Verify if Ollama uses this or main .env

## 2. DOCKER FILES INVESTIGATION

### Archived Docker Files (/docker/archived/)
- **Count**: 6 files
- **Purpose**: Old Ollama configurations
- **Files**:
  - docker-compose.ollama-final.yml
  - docker-compose.ollama-optimized.yml
  - docker-compose.ollama-performance.yml
  - docker-compose.ollama-ultrafix.yml
  - backend-Dockerfile.optimized
  - backend-Dockerfile.secure.broken
- **Status**: ARCHIVED - Likely obsolete
- **Investigation Needed**: Confirm no active usage

### Active Docker Compose Files
- **Total**: 28 files
- **Categories**:
  - Base: docker-compose.yml, docker-compose.base.yml
  - Overrides: .skyvern, .documind, .public-images
  - Environment: .secure, .minimal, .ultra-performance
  - Deployment: .blue-green
  - MCP: docker-compose.mcp.yml
- **Investigation Needed**: Map dependencies and usage

## 3. INVESTIGATION PRIORITIES

### HIGH PRIORITY (Immediate Investigation)
1. **.env.secure vs .env.production.secure** - Likely duplicates (359 lines combined)
2. **Archived Docker files** - 6 files likely obsolete
3. **.env.agents and .env.ollama** - Low usage, possible consolidation

### MEDIUM PRIORITY  
4. **Docker compose variants** - Understand purpose of each
5. **Agent configurations** - 159 files need deduplication analysis

### LOW PRIORITY
6. **TODO/FIXME comments** - 9,720 instances (separate cleanup task)

## NEXT INVESTIGATION STEPS

1. Compare .env.secure with .env.production.secure for overlap
2. Verify archived Docker files are truly unused
3. Check if .env.agents and .env.ollama can be consolidated
4. Map Docker compose file dependencies
5. Analyze agent configuration duplication