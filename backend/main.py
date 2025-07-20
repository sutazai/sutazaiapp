#!/usr/-bin/env python3
"""
SutazAI Backend Main Entry Point
"""

import asyncio
import logging

from backend.api.main import app
from backend.core.component_initializer import initialize_components
from backend.utils.logging_setup import get_api_logger

logger = get_api_logger()

async def main():
    """
    Main function to initialize components and start the backend server.
    """
    logger.info("Initializing SutazAI backend...")
    await initialize_components(app)
    logger.info("SutazAI backend initialized.")

if __name__ == "__main__":
    import uvicorn
    from backend.config.settings import get_settings

    settings = get_settings()

    # Run the main async function to initialize components
    asyncio.run(main())

    logger.info(f"Starting Uvicorn server on {settings.SERVER_HOST}:{settings.SERVER_PORT}")
    uvicorn.run(
        app,
        host=settings.SERVER_HOST,
        port=settings.SERVER_PORT,
        log_level=settings.LOG_LEVEL.lower(),
    )