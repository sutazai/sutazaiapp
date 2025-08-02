#!/usr/bin/env python3
"""
PostgreSQL database creation script.

This script creates the database and pgvector extension if they don't exist.
"""

import sys
import os
import logging
import argparse
from urllib.parse import urlparse

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# Adjust path assuming script is run from root or scripts/ directory
# This might need tweaking depending on execution context
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))) # Add project root
from backend.core.config import get_settings

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("db_create")


def create_database():
    """Create PostgreSQL database if it doesn't exist."""
    settings = get_settings()

    # PostgreSQL connection URL without the database name
    base_connection_url = f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}"

    logger.info(
        f"Connecting to PostgreSQL server at {settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}"
    )

    try:
        # Connect to PostgreSQL server
        conn = psycopg2.connect(f"{base_connection_url}/postgres")
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)

        with conn.cursor() as cursor:
            # Check if database exists
            cursor.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s", (settings.POSTGRES_DB,)
            )
            database_exists = cursor.fetchone()

            if not database_exists:
                logger.info(f"Creating database '{settings.POSTGRES_DB}'")
                cursor.execute(f"CREATE DATABASE {settings.POSTGRES_DB}")
                logger.info(f"Database '{settings.POSTGRES_DB}' created successfully")
            else:
                logger.info(f"Database '{settings.POSTGRES_DB}' already exists")

        conn.close()

        # Now connect to the created database to install pgvector extension
        # Ensure the connection string uses the correct DB name
        db_connection_url = settings.SQLALCHEMY_DATABASE_URL
        if settings.POSTGRES_DB not in db_connection_url:
             # Attempt to reconstruct URL if needed (this is fragile)
             parsed_url = urlparse(db_connection_url)
             db_connection_url = f"{parsed_url.scheme}://{parsed_url.netloc}/{settings.POSTGRES_DB}"
             logger.warning(f"Adjusted connection URL to: {db_connection_url}")

        conn = psycopg2.connect(db_connection_url)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)

        with conn.cursor() as cursor:
            logger.info("Creating pgvector extension")
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            logger.info("pgvector extension installed successfully")

        conn.close()

        logger.info("Database initialization completed successfully")
        return True

    except Exception as e:
        logger.error(f"Error creating database: {str(e)}", exc_info=True) # Add exc_info for better debugging
        return False


def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(
        description="Create PostgreSQL database for SutazAI"
    )
    parser.parse_args()  # Parse args but don't assign

    success = create_database()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main()) 