// Neo4j Database Optimization Script
// Creates indexes and constraints for production performance
// Safe to run multiple times (CREATE IF NOT EXISTS)

// ============================================================
// Agent Node Optimizations
// ============================================================

// Create constraint for unique agent names
CREATE CONSTRAINT agent_name_unique IF NOT EXISTS
FOR (a:Agent) REQUIRE a.name IS UNIQUE;

// Create index on agent status
CREATE INDEX agent_status_idx IF NOT EXISTS
FOR (a:Agent) ON (a.status);

// Create index on agent type
CREATE INDEX agent_type_idx IF NOT EXISTS
FOR (a:Agent) ON (a.type);

// Create index on agent capability tags
CREATE INDEX agent_capabilities_idx IF NOT EXISTS
FOR (a:Agent) ON (a.capabilities);

// ============================================================
// User Node Optimizations
// ============================================================

// Create constraint for unique user IDs
CREATE CONSTRAINT user_id_unique IF NOT EXISTS
FOR (u:User) REQUIRE u.user_id IS UNIQUE;

// Create index on user email
CREATE INDEX user_email_idx IF NOT EXISTS
FOR (u:User) ON (u.email);

// ============================================================
// Conversation/Session Optimizations
// ============================================================

// Create constraint for unique session IDs
CREATE CONSTRAINT session_id_unique IF NOT EXISTS
FOR (s:Session) REQUIRE s.session_id IS UNIQUE;

// Create index on session created timestamp
CREATE INDEX session_created_idx IF NOT EXISTS
FOR (s:Session) ON (s.created_at);

// Create index on session status
CREATE INDEX session_status_idx IF NOT EXISTS
FOR (s:Session) ON (s.status);

// ============================================================
// Message/Interaction Optimizations
// ============================================================

// Create index on message timestamps
CREATE INDEX message_timestamp_idx IF NOT EXISTS
FOR (m:Message) ON (m.timestamp);

// Create index on message type
CREATE INDEX message_type_idx IF NOT EXISTS
FOR (m:Message) ON (m.type);

// ============================================================
// Knowledge Graph Optimizations
// ============================================================

// Create constraint for unique document IDs
CREATE CONSTRAINT document_id_unique IF NOT EXISTS
FOR (d:Document) REQUIRE d.document_id IS UNIQUE;

// Create full-text index on document content (if using text search)
CREATE FULLTEXT INDEX document_content_fulltext IF NOT EXISTS
FOR (d:Document) ON EACH [d.content, d.title];

// Create index on document types
CREATE INDEX document_type_idx IF NOT EXISTS
FOR (d:Document) ON (d.type);

// ============================================================
// Relationship Optimizations
// ============================================================

// Note: Neo4j automatically indexes relationship types
// These queries optimize common traversal patterns

// Create composite index for user-agent interactions
CREATE INDEX user_agent_interaction_idx IF NOT EXISTS
FOR ()-[r:INTERACTS_WITH]-() ON (r.timestamp);

// Create index on agent collaboration relationships
CREATE INDEX agent_collaboration_idx IF NOT EXISTS
FOR ()-[r:COLLABORATES_WITH]-() ON (r.created_at);

// ============================================================
// Query Pattern Optimizations
// ============================================================

// Example: Find recent conversations for a user
// MATCH (u:User {user_id: $userId})-[:HAS_SESSION]->(s:Session)
// WHERE s.created_at > datetime() - duration('P7D')
// RETURN s ORDER BY s.created_at DESC
// Uses: user_id_unique constraint, session_created_idx

// Example: Find active agents with specific capability
// MATCH (a:Agent {status: 'active'})
// WHERE 'code_generation' IN a.capabilities
// RETURN a
// Uses: agent_status_idx, agent_capabilities_idx

// Example: Get conversation history with messages
// MATCH (s:Session {session_id: $sessionId})-[:CONTAINS]->(m:Message)
// RETURN m ORDER BY m.timestamp
// Uses: session_id_unique constraint, message_timestamp_idx

// ============================================================
// Performance Recommendations
// ============================================================

// 1. Always use direction hints in relationships for better performance:
//    MATCH (a:Agent)-[:HANDLES]->(t:Task)  // Explicit direction
//    Instead of: MATCH (a:Agent)-[:HANDLES]-(t:Task)

// 2. Use labels to reduce search space:
//    MATCH (a:Agent) WHERE a.name = 'Jarvis'  // Good
//    Instead of: MATCH (a) WHERE a.name = 'Jarvis'  // Bad

// 3. Use LIMIT to prevent large result sets:
//    MATCH (m:Message) RETURN m ORDER BY m.timestamp DESC LIMIT 100

// 4. Use EXPLAIN and PROFILE to analyze query plans:
//    PROFILE MATCH (a:Agent)-[:HANDLES]->(t:Task) RETURN count(t)

// ============================================================
// Verify Index Creation
// ============================================================

// Show all indexes
SHOW INDEXES;

// Show all constraints
SHOW CONSTRAINTS;

// Database statistics
CALL db.stats.retrieve('GRAPH COUNTS');
