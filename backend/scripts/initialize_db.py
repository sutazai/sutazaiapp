#!/usr/bin/env python3
"""
Database initialization script.

This script creates all database tables and sets up pgvector extension.
"""

import sys
import os
import logging
import argparse
from sqlalchemy import text
from backend.core.database import engine, Base

# Adjust Python path to include the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(str(project_root))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("db_init")


def initialize_database(drop_all: bool = False):
    """Initialize the database by creating tables and the pgvector extension."""
    logger.info("Initializing database")

    # Create pgvector extension if it doesn't exist
    try:
        with engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()
            logger.info("pgvector extension created or already exists")
    except Exception as e:
        logger.error(f"Error creating pgvector extension: {e}")
        raise

    # Drop all tables if specified
    if drop_all:
        logger.warning("Dropping all tables")
        try:
            Base.metadata.drop_all(engine)
            logger.info("All tables dropped")
        except Exception as e:
            logger.error(f"Error dropping tables: {e}")
            raise

    # Create all tables
    try:
        Base.metadata.create_all(engine)
        logger.info("All tables created successfully")
    except Exception as e:
        logger.error(f"Error creating tables: {e}")
        raise


def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(description="Initialize the database for SutazAI")
    parser.add_argument(
        "--drop-all", action="store_true", help="Drop all tables before creating them"
    )
    args = parser.parse_args()

    initialize_database(drop_all=args.drop_all)

    return 0


if __name__ == "__main__":
    sys.exit(main())
