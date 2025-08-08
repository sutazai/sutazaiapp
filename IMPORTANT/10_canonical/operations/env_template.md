# Environment Variables Template

Copy these into your deployment environment (e.g., `.env`) with secure values.

- Database
  - `POSTGRES_HOST=postgres`
  - `POSTGRES_PORT=5432`
  - `POSTGRES_DB=sutazai`
  - `POSTGRES_USER=sutazai`
  - `POSTGRES_PASSWORD=<GENERATE_SECURE_PASSWORD>`

- Redis
  - `REDIS_HOST=redis`
  - `REDIS_PORT=6379`
  - `REDIS_PASSWORD=<GENERATE_SECURE_PASSWORD>`

- Neo4j
  - `NEO4J_HOST=neo4j`
  - `NEO4J_BOLT_PORT=7687`
  - `NEO4J_PASSWORD=<GENERATE_SECURE_PASSWORD>`

- JWT / App Secrets
  - `JWT_SECRET=<GENERATE_256BIT_SECRET>`
  - `SECRET_KEY=<GENERATE_256BIT_SECRET>`

- Vector DB (Chroma)
  - `CHROMADB_HOST=chromadb`
  - `CHROMADB_PORT=8000`
  - `CHROMADB_API_KEY=<GENERATE_API_KEY>`

- Ollama
  - `OLLAMA_URL=http://ollama:11434`
  - `DEFAULT_MODEL=tinyllama`

- Gateway
  - `KONG_ADMIN_URL=http://localhost:10007`
  - `KONG_PROXY_URL=http://localhost:10005`

Generation commands (examples)
- `openssl rand -base64 32 | tr -d '\n'` for passwords
- `openssl rand -base64 64 | tr -d '\n'` for JWT/SECRET_KEY
