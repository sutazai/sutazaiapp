#!/usr/bin/env python3
"""
Validation script for database migration
Verifies data integrity after migration
"""

import os
import sys
import json
import sqlite3
import psycopg2
from psycopg2.extras import RealDictCursor
import hashlib
import logging
from typing import Dict, List, Tuple
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MigrationValidator:
    """Validates the database migration"""
    
    def __init__(self, pg_config: Dict[str, str]):
        self.pg_config = pg_config
        self.pg_conn = None
        self.pg_cursor = None
        self.validation_results = {
            'total_checks': 0,
            'passed_checks': 0,
            'failed_checks': 0,
            'issues': []
        }
        
    def connect_postgres(self):
        """Connect to PostgreSQL"""
        try:
            self.pg_conn = psycopg2.connect(**self.pg_config)
            self.pg_cursor = self.pg_conn.cursor(cursor_factory=RealDictCursor)
            logger.info("Connected to PostgreSQL for validation")
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            sys.exit(1)
            
    def close_postgres(self):
        """Close PostgreSQL connection"""
        if self.pg_cursor:
            self.pg_cursor.close()
        if self.pg_conn:
            self.pg_conn.close()
            
    def validate_record_counts(self, sqlite_databases: List[str]) -> bool:
        """Validate that record counts match between SQLite and PostgreSQL"""
        logger.info("Validating record counts...")
        
        total_sqlite_records = 0
        issues = []
        
        # Count SQLite records
        for db_path in sqlite_databases:
            if not os.path.exists(db_path):
                continue
                
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM memory_entries")
                count = cursor.fetchone()[0]
                total_sqlite_records += count
                conn.close()
                
                # Check PostgreSQL has records from this source
                source_name = os.path.basename(os.path.dirname(db_path))
                self.pg_cursor.execute("""
                    SELECT COUNT(*) as count 
                    FROM unified_memory 
                    WHERE source_path = %s
                """, (db_path,))
                pg_count = self.pg_cursor.fetchone()['count']
                
                if pg_count != count:
                    issues.append(f"Count mismatch for {db_path}: SQLite={count}, PostgreSQL={pg_count}")
                    
            except Exception as e:
                issues.append(f"Error validating {db_path}: {e}")
        
        # Count PostgreSQL records (excluding extended memory)
        self.pg_cursor.execute("""
            SELECT COUNT(*) as count 
            FROM unified_memory 
            WHERE source_db != 'extended_memory'
        """)
        pg_total = self.pg_cursor.fetchone()['count']
        
        self.validation_results['total_checks'] += 1
        
        if pg_total == total_sqlite_records and len(issues) == 0:
            logger.info(f"✓ Record count validation passed: {total_sqlite_records} records")
            self.validation_results['passed_checks'] += 1
            return True
        else:
            logger.error(f"✗ Record count validation failed: SQLite={total_sqlite_records}, PostgreSQL={pg_total}")
            for issue in issues:
                logger.error(f"  - {issue}")
                self.validation_results['issues'].append(issue)
            self.validation_results['failed_checks'] += 1
            return False
            
    def validate_data_integrity(self, sample_size: int = 100) -> bool:
        """Spot check data integrity by sampling records"""
        logger.info(f"Validating data integrity (sample size: {sample_size})...")
        
        # Get random sample from PostgreSQL
        self.pg_cursor.execute("""
            SELECT key, namespace, source_path, value
            FROM unified_memory
            WHERE source_path IS NOT NULL
            ORDER BY RANDOM()
            LIMIT %s
        """, (sample_size,))
        
        pg_records = self.pg_cursor.fetchall()
        
        issues = []
        checked = 0
        matched = 0
        
        for pg_record in pg_records:
            if not os.path.exists(pg_record['source_path']):
                continue
                
            try:
                # Get original record from SQLite
                conn = sqlite3.connect(pg_record['source_path'])
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT value 
                    FROM memory_entries 
                    WHERE key = ? AND namespace = ?
                """, (pg_record['key'], pg_record['namespace']))
                
                sqlite_record = cursor.fetchone()
                conn.close()
                
                if sqlite_record:
                    checked += 1
                    # Compare values (accounting for JSON parsing)
                    try:
                        sqlite_value = json.loads(sqlite_record[0])
                    except:
                        sqlite_value = sqlite_record[0]
                        
                    if pg_record['value'] == sqlite_value or str(pg_record['value']) == str(sqlite_value):
                        matched += 1
                    else:
                        issues.append(f"Value mismatch for key={pg_record['key']}, namespace={pg_record['namespace']}")
                        
            except Exception as e:
                issues.append(f"Error checking record: {e}")
        
        self.validation_results['total_checks'] += 1
        
        if checked > 0:
            match_rate = (matched / checked) * 100
            if match_rate >= 95:  # Allow 5% tolerance for data conversion issues
                logger.info(f"✓ Data integrity validation passed: {match_rate:.1f}% match rate ({matched}/{checked})")
                self.validation_results['passed_checks'] += 1
                return True
            else:
                logger.error(f"✗ Data integrity validation failed: {match_rate:.1f}% match rate")
                for issue in issues[:10]:  # Show first 10 issues
                    logger.error(f"  - {issue}")
                self.validation_results['issues'].extend(issues)
                self.validation_results['failed_checks'] += 1
                return False
        else:
            logger.warning("No records to validate")
            return True
            
    def validate_indexes(self) -> bool:
        """Validate that all required indexes exist"""
        logger.info("Validating database indexes...")
        
        required_indexes = [
            'idx_unified_memory_namespace',
            'idx_unified_memory_expires',
            'idx_unified_memory_accessed',
            'idx_unified_memory_source',
            'idx_unified_memory_data_type',
            'idx_unified_memory_value_gin'
        ]
        
        self.pg_cursor.execute("""
            SELECT indexname 
            FROM pg_indexes 
            WHERE tablename = 'unified_memory'
        """)
        
        existing_indexes = [row['indexname'] for row in self.pg_cursor.fetchall()]
        
        missing_indexes = [idx for idx in required_indexes if idx not in existing_indexes]
        
        self.validation_results['total_checks'] += 1
        
        if len(missing_indexes) == 0:
            logger.info(f"✓ All {len(required_indexes)} required indexes exist")
            self.validation_results['passed_checks'] += 1
            return True
        else:
            logger.error(f"✗ Missing indexes: {', '.join(missing_indexes)}")
            self.validation_results['issues'].append(f"Missing indexes: {missing_indexes}")
            self.validation_results['failed_checks'] += 1
            return False
            
    def validate_namespaces(self) -> bool:
        """Validate namespace distribution"""
        logger.info("Validating namespace distribution...")
        
        self.pg_cursor.execute("""
            SELECT namespace, COUNT(*) as count
            FROM unified_memory
            GROUP BY namespace
            ORDER BY count DESC
        """)
        
        namespaces = self.pg_cursor.fetchall()
        
        self.validation_results['total_checks'] += 1
        
        if len(namespaces) > 0:
            logger.info(f"✓ Found {len(namespaces)} namespaces")
            for ns in namespaces[:5]:  # Show top 5
                logger.info(f"  - {ns['namespace']}: {ns['count']} records")
            self.validation_results['passed_checks'] += 1
            return True
        else:
            logger.error("✗ No namespaces found in PostgreSQL")
            self.validation_results['issues'].append("No namespaces found")
            self.validation_results['failed_checks'] += 1
            return False
            
    def validate_performance(self) -> bool:
        """Test query performance"""
        logger.info("Validating query performance...")
        
        import time
        
        tests = [
            ("Simple key lookup", """
                SELECT * FROM unified_memory 
                WHERE key = 'command-metrics-summary' 
                AND namespace = 'performance-metrics'
            """),
            ("Namespace query", """
                SELECT COUNT(*) FROM unified_memory 
                WHERE namespace = 'performance-metrics'
            """),
            ("Recent records", """
                SELECT * FROM unified_memory 
                ORDER BY updated_at DESC 
                LIMIT 10
            """),
            ("JSON search", """
                SELECT * FROM unified_memory 
                WHERE value @> '{"type": "active_swarm"}'::jsonb 
                LIMIT 5
            """)
        ]
        
        all_passed = True
        
        for test_name, query in tests:
            start = time.time()
            try:
                self.pg_cursor.execute(query)
                self.pg_cursor.fetchall()
                elapsed = time.time() - start
                
                if elapsed < 1.0:  # Should complete in under 1 second
                    logger.info(f"  ✓ {test_name}: {elapsed:.3f}s")
                else:
                    logger.warning(f"  ⚠ {test_name}: {elapsed:.3f}s (slow)")
                    all_passed = False
            except Exception as e:
                logger.error(f"  ✗ {test_name}: {e}")
                all_passed = False
                self.validation_results['issues'].append(f"Performance test failed: {test_name}")
        
        self.validation_results['total_checks'] += 1
        if all_passed:
            self.validation_results['passed_checks'] += 1
        else:
            self.validation_results['failed_checks'] += 1
            
        return all_passed
        
    def generate_report(self) -> Dict:
        """Generate validation report"""
        report = {
            'validation_summary': {
                'total_checks': self.validation_results['total_checks'],
                'passed': self.validation_results['passed_checks'],
                'failed': self.validation_results['failed_checks'],
                'success_rate': (self.validation_results['passed_checks'] / self.validation_results['total_checks'] * 100) if self.validation_results['total_checks'] > 0 else 0
            },
            'issues': self.validation_results['issues'],
            'recommendation': 'Migration successful' if self.validation_results['failed_checks'] == 0 else 'Review and fix issues before proceeding'
        }
        
        return report

def find_all_memory_databases(base_path: str = '/opt/sutazaiapp') -> List[str]:
    """Find all memory.db files"""
    databases = []
    for root, dirs, files in os.walk(base_path):
        if 'memory.db' in files:
            databases.append(os.path.join(root, 'memory.db'))
    return sorted(databases)

def main():
    # PostgreSQL configuration
    import subprocess
    try:
        # Try to get Docker container IP
        result = subprocess.run(['docker', 'inspect', 'sutazai-postgres', '-f', '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}'], 
                              capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            # Use Docker container IP
            pg_host = result.stdout.strip()
            pg_port = 5432  # Internal PostgreSQL port
        else:
            pg_host = 'localhost'
            pg_port = 10000
    except:
        pg_host = 'localhost'
        pg_port = 10000
    
    pg_config = {
        'host': pg_host,
        'port': pg_port,
        'database': 'sutazai',
        'user': 'sutazai',
        'password': 'change_me_secure'
    }
    
    validator = MigrationValidator(pg_config)
    validator.connect_postgres()
    
    try:
        # Find all SQLite databases
        sqlite_databases = find_all_memory_databases()
        
        # Run validations
        logger.info("=" * 60)
        logger.info("DATABASE MIGRATION VALIDATION")
        logger.info("=" * 60)
        
        validator.validate_record_counts(sqlite_databases)
        validator.validate_data_integrity()
        validator.validate_indexes()
        validator.validate_namespaces()
        validator.validate_performance()
        
        # Generate report
        report = validator.generate_report()
        
        logger.info("")
        logger.info("=" * 60)
        logger.info("VALIDATION REPORT")
        logger.info("=" * 60)
        logger.info(f"Total checks: {report['validation_summary']['total_checks']}")
        logger.info(f"Passed: {report['validation_summary']['passed']}")
        logger.info(f"Failed: {report['validation_summary']['failed']}")
        logger.info(f"Success rate: {report['validation_summary']['success_rate']:.1f}%")
        logger.info(f"Recommendation: {report['recommendation']}")
        
        if report['issues']:
            logger.info("")
            logger.info("Issues found:")
            for issue in report['issues']:
                logger.info(f"  - {issue}")
        
        # Write report to file
        with open('/opt/sutazaiapp/scripts/database-migration/validation_report.json', 'w') as f:
            json.dump(report, f, indent=2)
            logger.info("")
            logger.info("Full report saved to: validation_report.json")
            
    finally:
        validator.close_postgres()

if __name__ == '__main__':
    main()