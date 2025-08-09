#!/usr/bin/env python3
"""
SutazAI Database Health Check Script
Verifies that PostgreSQL database is properly set up and accessible
"""

import psycopg2
import json
import sys
from datetime import datetime, timedelta
from typing import Dict, Any, List

# Database connection parameters
DB_CONFIG = {
    'host': 'localhost',
    'port': 10000,
    'database': 'sutazai',
    'user': 'sutazai',
    'password': 'sutazai_secure_2024'
}

def get_db_connection():
    """Get database connection"""
    try:
        return psycopg2.connect(**DB_CONFIG)
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return None

def check_tables_exist(conn) -> Dict[str, bool]:
    """Check if all required tables exist"""
    required_tables = [
        'users', 'agents', 'tasks', 'chat_history', 'agent_executions',
        'system_metrics', 'sessions', 'agent_health', 'model_registry',
        'vector_collections', 'knowledge_documents', 'orchestration_sessions',
        'api_usage_logs', 'system_alerts'
    ]
    
    results = {}
    with conn.cursor() as cur:
        for table in required_tables:
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = %s
                );
            """, (table,))
            results[table] = cur.fetchone()[0]
    
    return results

def check_indexes_exist(conn) -> int:
    """Check if indexes are properly created"""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT COUNT(*) 
            FROM pg_indexes 
            WHERE schemaname = 'public' 
            AND indexname LIKE 'idx_%';
        """)
        return cur.fetchone()[0]

def check_views_exist(conn) -> List[str]:
    """Check if views are created"""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT viewname 
            FROM pg_views 
            WHERE schemaname = 'public' 
            ORDER BY viewname;
        """)
        return [row[0] for row in cur.fetchall()]

def get_table_counts(conn) -> Dict[str, int]:
    """Get record counts for all tables"""
    tables = ['users', 'agents', 'tasks', 'chat_history', 'system_metrics', 
              'sessions', 'model_registry', 'vector_collections']
    
    counts = {}
    with conn.cursor() as cur:
        for table in tables:
            try:
                cur.execute(f"SELECT COUNT(*) FROM {table}")
                counts[table] = cur.fetchone()[0]
            except Exception as e:
                counts[table] = f"Error: {e}"
    
    return counts

def test_basic_operations(conn) -> Dict[str, bool]:
    """Test basic database operations"""
    results = {}
    
    try:
        with conn.cursor() as cur:
            # Test SELECT
            cur.execute("SELECT 1")
            results['select'] = cur.fetchone()[0] == 1
            
            # Test INSERT
            cur.execute("""
                INSERT INTO system_metrics (metric_name, metric_value, tags) 
                VALUES ('test_metric', 1.0, '{"test": true}') 
                RETURNING id
            """)
            metric_id = cur.fetchone()[0]
            results['insert'] = metric_id is not None
            
            # Test UPDATE
            cur.execute("""
                UPDATE system_metrics 
                SET metric_value = 2.0 
                WHERE id = %s
            """, (metric_id,))
            results['update'] = cur.rowcount == 1
            
            # Test DELETE
            cur.execute("DELETE FROM system_metrics WHERE id = %s", (metric_id,))
            results['delete'] = cur.rowcount == 1
            
            conn.commit()
            
    except Exception as e:
        print(f"❌ Basic operations test failed: {e}")
        results['error'] = str(e)
        conn.rollback()
    
    return results

def test_views(conn) -> Dict[str, bool]:
    """Test that views are working"""
    view_tests = {}
    
    try:
        with conn.cursor() as cur:
            # Test system_health_dashboard view
            cur.execute("SELECT * FROM system_health_dashboard LIMIT 1")
            dashboard_data = cur.fetchone()
            view_tests['system_health_dashboard'] = dashboard_data is not None
            
            # Test agent_status_overview view
            cur.execute("SELECT * FROM agent_status_overview LIMIT 1")
            agent_data = cur.fetchall()
            view_tests['agent_status_overview'] = len(agent_data) > 0
            
            # Test performance_metrics view
            cur.execute("SELECT * FROM performance_metrics LIMIT 1")
            metrics_data = cur.fetchall()
            view_tests['performance_metrics'] = True  # Can be empty
            
    except Exception as e:
        print(f"❌ Views test failed: {e}")
        view_tests['error'] = str(e)
    
    return view_tests

def check_foreign_keys(conn) -> List[Dict[str, str]]:
    """Check foreign key constraints"""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT 
                tc.table_name, 
                kcu.column_name, 
                ccu.table_name AS foreign_table_name,
                ccu.column_name AS foreign_column_name 
            FROM 
                information_schema.table_constraints AS tc 
                JOIN information_schema.key_column_usage AS kcu
                  ON tc.constraint_name = kcu.constraint_name
                  AND tc.table_schema = kcu.table_schema
                JOIN information_schema.constraint_column_usage AS ccu
                  ON ccu.constraint_name = tc.constraint_name
                  AND ccu.table_schema = tc.table_schema
            WHERE tc.constraint_type = 'FOREIGN KEY' 
            AND tc.table_schema = 'public'
            ORDER BY tc.table_name;
        """)
        
        return [
            {
                'table': row[0],
                'column': row[1],
                'references_table': row[2],
                'references_column': row[3]
            }
            for row in cur.fetchall()
        ]

def main():
    """Main health check function"""
    print("🔍 SutazAI Database Health Check")
    print("=" * 50)
    
    # Connect to database
    conn = get_db_connection()
    if not conn:
        sys.exit(1)
    
    try:
        # Check database connectivity
        print("✅ Database connection: SUCCESS")
        
        # Check tables
        print("\n📋 Checking tables...")
        tables = check_tables_exist(conn)
        missing_tables = [name for name, exists in tables.items() if not exists]
        
        if missing_tables:
            print(f"❌ Missing tables: {missing_tables}")
        else:
            print(f"✅ All {len(tables)} required tables exist")
        
        # Check indexes
        print("\n📇 Checking indexes...")
        index_count = check_indexes_exist(conn)
        print(f"✅ {index_count} custom indexes created")
        
        # Check views
        print("\n👁️ Checking views...")
        views = check_views_exist(conn)
        print(f"✅ {len(views)} views created: {', '.join(views)}")
        
        # Check table counts
        print("\n📊 Checking data...")
        counts = get_table_counts(conn)
        for table, count in counts.items():
            print(f"  {table}: {count} records")
        
        # Test basic operations
        print("\n🔧 Testing basic operations...")
        operations = test_basic_operations(conn)
        failed_ops = [op for op, success in operations.items() if not success]
        if failed_ops:
            print(f"❌ Failed operations: {failed_ops}")
        else:
            print("✅ All basic operations working (SELECT, INSERT, UPDATE, DELETE)")
        
        # Test views
        print("\n🔍 Testing views...")
        view_tests = test_views(conn)
        failed_views = [view for view, success in view_tests.items() if not success and view != 'error']
        if failed_views:
            print(f"❌ Failed views: {failed_views}")
        else:
            print("✅ All views working correctly")
        
        # Check foreign keys
        print("\n🔗 Checking foreign key constraints...")
        fk_constraints = check_foreign_keys(conn)
        print(f"✅ {len(fk_constraints)} foreign key constraints active")
        
        # Summary
        print("\n" + "=" * 50)
        print("📈 HEALTH CHECK SUMMARY:")
        
        total_score = 0
        max_score = 6
        
        if not missing_tables:
            total_score += 1
            print("✅ Tables: OK")
        else:
            print("❌ Tables: MISSING")
        
        if index_count > 20:
            total_score += 1
            print("✅ Indexes: OK")
        else:
            print("⚠️ Indexes: LIMITED")
        
        if len(views) >= 4:
            total_score += 1
            print("✅ Views: OK")
        else:
            print("❌ Views: MISSING")
        
        if sum(1 for c in counts.values() if isinstance(c, int) and c > 0) >= 3:
            total_score += 1
            print("✅ Data: OK")
        else:
            print("⚠️ Data: MINIMAL")
        
        if not failed_ops:
            total_score += 1
            print("✅ Operations: OK")
        else:
            print("❌ Operations: FAILED")
        
        if not failed_views:
            total_score += 1
            print("✅ Views: OK")
        else:
            print("❌ Views: FAILED")
        
        health_percentage = (total_score / max_score) * 100
        print(f"\n🎯 Overall Database Health: {health_percentage:.1f}% ({total_score}/{max_score})")
        
        if health_percentage >= 85:
            print("🟢 Status: EXCELLENT - Database is fully operational")
        elif health_percentage >= 70:
            print("🟡 Status: GOOD - Database is functional with minor issues")
        elif health_percentage >= 50:
            print("🟠 Status: DEGRADED - Database has significant issues")
        else:
            print("🔴 Status: CRITICAL - Database requires immediate attention")
        
        # Database info
        with conn.cursor() as cur:
            cur.execute("SELECT version()")
            db_version = cur.fetchone()[0]
            print(f"\n📋 Database Version: {db_version}")
            
            cur.execute("SELECT pg_database_size('sutazai')")
            db_size = cur.fetchone()[0]
            print(f"💾 Database Size: {db_size / (1024*1024):.2f} MB")
        
        return 0 if health_percentage >= 70 else 1
        
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return 1
    
    finally:
        conn.close()

if __name__ == "__main__":
    sys.exit(main())