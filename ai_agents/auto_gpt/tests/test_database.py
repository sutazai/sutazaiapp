#!/usr/bin/env python3.11
"""Tests for the database module."""

import pytest
from typing import Dict, Any, List
from pathlib import Path

    Database,
    DatabaseConfig,
    DatabaseError,
    DatabaseManager,
    SQLiteDatabase,
    PostgreSQLDatabase,
    MySQLDatabase,
    DatabaseConnection,
    DatabaseTransaction,
)


@pytest.fixture
def db_config():
    """Create test database configuration."""
    return DatabaseConfig(
        type="sqlite",
        host="localhost",
        port=5432,
        database="test_db",
        username="test_user",
        password="test_password",
    )


@pytest.fixture
def db_manager():
    """Create a test database manager."""
    return DatabaseManager()


@pytest.fixture
def sqlite_db(tmp_path):
    """Create a test SQLite database."""
    db_path = tmp_path / "test.db"
    return SQLiteDatabase(db_path)


@pytest.fixture
def postgres_db():
    """Create a test PostgreSQL database."""
    return PostgreSQLDatabase(
        host="localhost",
        port=5432,
        database="test_db",
        username="test_user",
        password="test_password",
    )


@pytest.fixture
def mysql_db():
    """Create a test MySQL database."""
    return MySQLDatabase(
        host="localhost",
        port=3306,
        database="test_db",
        username="test_user",
        password="test_password",
    )


def test_database_config():
    """Test database configuration."""
    # Test config creation
    config = DatabaseConfig(
        type="sqlite",
        host="localhost",
        port=5432,
        database="test_db",
        username="test_user",
        password="test_password",
    )
    assert config.type == "sqlite"
    assert config.host == "localhost"
    assert config.port == 5432
    assert config.database == "test_db"
    assert config.username == "test_user"
    assert config.password == "test_password"

    # Test config validation
    assert config.validate() is True

    # Test invalid config
    invalid_config = DatabaseConfig(
        type="invalid",
        host="localhost",
        port=5432,
        database="test_db",
    )
    assert invalid_config.validate() is False


def test_database_connection():
    """Test database connection handling."""
    # Test connection creation
    connection = DatabaseConnection(
        host="localhost",
        port=5432,
        database="test_db",
        username="test_user",
        password="test_password",
    )
    assert connection.is_connected() is False

    # Test connection opening
    connection.open()
    assert connection.is_connected() is True

    # Test connection closing
    connection.close()
    assert connection.is_connected() is False

    # Test connection reuse
    connection.open()
    assert connection.is_connected() is True
    connection.close()


def test_database_transaction():
    """Test database transaction handling."""
    # Test transaction creation
    transaction = DatabaseTransaction()
    assert transaction.is_active() is False

    # Test transaction start
    transaction.start()
    assert transaction.is_active() is True

    # Test transaction commit
    transaction.commit()
    assert transaction.is_active() is False

    # Test transaction rollback
    transaction.start()
    transaction.rollback()
    assert transaction.is_active() is False


def test_sqlite_database(sqlite_db):
    """Test SQLite database functionality."""
    # Test table creation
    sqlite_db.execute("""
        CREATE TABLE test (
            id INTEGER PRIMARY KEY,
            name TEXT,
            value INTEGER
        )
    """)

    # Test data insertion
    sqlite_db.execute(
        "INSERT INTO test (name, value) VALUES (?, ?)",
        ("test", 42),
    )

    # Test data retrieval
    result = sqlite_db.query("SELECT * FROM test WHERE name = ?", ("test",))
    assert len(result) == 1
    assert result[0]["name"] == "test"
    assert result[0]["value"] == 42

    # Test data update
    sqlite_db.execute(
        "UPDATE test SET value = ? WHERE name = ?",
        (43, "test"),
    )
    result = sqlite_db.query("SELECT * FROM test WHERE name = ?", ("test",))
    assert result[0]["value"] == 43

    # Test data deletion
    sqlite_db.execute("DELETE FROM test WHERE name = ?", ("test",))
    result = sqlite_db.query("SELECT * FROM test WHERE name = ?", ("test",))
    assert len(result) == 0


def test_postgres_database(postgres_db):
    """Test PostgreSQL database functionality."""
    # Test table creation
    postgres_db.execute("""
        CREATE TABLE test (
            id SERIAL PRIMARY KEY,
            name VARCHAR(100),
            value INTEGER
        )
    """)

    # Test data insertion
    postgres_db.execute(
        "INSERT INTO test (name, value) VALUES (%s, %s)",
        ("test", 42),
    )

    # Test data retrieval
    result = postgres_db.query("SELECT * FROM test WHERE name = %s", ("test",))
    assert len(result) == 1
    assert result[0]["name"] == "test"
    assert result[0]["value"] == 42

    # Test data update
    postgres_db.execute(
        "UPDATE test SET value = %s WHERE name = %s",
        (43, "test"),
    )
    result = postgres_db.query("SELECT * FROM test WHERE name = %s", ("test",))
    assert result[0]["value"] == 43

    # Test data deletion
    postgres_db.execute("DELETE FROM test WHERE name = %s", ("test",))
    result = postgres_db.query("SELECT * FROM test WHERE name = %s", ("test",))
    assert len(result) == 0


def test_mysql_database(mysql_db):
    """Test MySQL database functionality."""
    # Test table creation
    mysql_db.execute("""
        CREATE TABLE test (
            id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(100),
            value INT
        )
    """)

    # Test data insertion
    mysql_db.execute(
        "INSERT INTO test (name, value) VALUES (%s, %s)",
        ("test", 42),
    )

    # Test data retrieval
    result = mysql_db.query("SELECT * FROM test WHERE name = %s", ("test",))
    assert len(result) == 1
    assert result[0]["name"] == "test"
    assert result[0]["value"] == 42

    # Test data update
    mysql_db.execute(
        "UPDATE test SET value = %s WHERE name = %s",
        (43, "test"),
    )
    result = mysql_db.query("SELECT * FROM test WHERE name = %s", ("test",))
    assert result[0]["value"] == 43

    # Test data deletion
    mysql_db.execute("DELETE FROM test WHERE name = %s", ("test",))
    result = mysql_db.query("SELECT * FROM test WHERE name = %s", ("test",))
    assert len(result) == 0


def test_database_manager(db_manager):
    """Test database manager functionality."""
    # Test database registration
    sqlite_db = SQLiteDatabase("test.db")
    db_manager.register_database("sqlite", sqlite_db)
    assert "sqlite" in db_manager.databases

    # Test database operations through manager
    db_manager.execute("sqlite", "CREATE TABLE test (id INTEGER PRIMARY KEY)")
    db_manager.execute("sqlite", "INSERT INTO test (id) VALUES (1)")
    result = db_manager.query("sqlite", "SELECT * FROM test")
    assert len(result) == 1
    assert result[0]["id"] == 1

    # Test database unregistration
    db_manager.unregister_database("sqlite")
    assert "sqlite" not in db_manager.databases


def test_database_error_handling(db_manager):
    """Test database error handling."""
    # Test non-existent database
    with pytest.raises(DatabaseError):
        db_manager.execute("non_existent", "SELECT 1")

    # Test invalid SQL
    sqlite_db = SQLiteDatabase("test.db")
    db_manager.register_database("sqlite", sqlite_db)
    with pytest.raises(DatabaseError):
        db_manager.execute("sqlite", "INVALID SQL")

    # Test connection error
    with pytest.raises(DatabaseError):
        db_manager.connect("sqlite", "invalid_connection")

    # Test transaction error
    with pytest.raises(DatabaseError):
        db_manager.commit("sqlite", "invalid_transaction")


def test_database_migration():
    """Test database migration functionality."""
    # Test migration creation
    migration = DatabaseMigration(
        version="1.0.0",
        description="Initial migration",
        up_sql="CREATE TABLE test (id INTEGER PRIMARY KEY)",
        down_sql="DROP TABLE test",
    )
    assert migration.version == "1.0.0"
    assert migration.description == "Initial migration"

    # Test migration execution
    sqlite_db = SQLiteDatabase("test.db")
    migration.execute(sqlite_db)
    result = sqlite_db.query("SELECT name FROM sqlite_master WHERE type='table'")
    assert len(result) == 1
    assert result[0]["name"] == "test"

    # Test migration rollback
    migration.rollback(sqlite_db)
    result = sqlite_db.query("SELECT name FROM sqlite_master WHERE type='table'")
    assert len(result) == 0
