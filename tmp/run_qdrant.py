import os
import time
from qdrant_client import QdrantClient
from qdrant_client.http import models
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='/opt/sutazaiapp/logs/vector-db.log',
    filemode='a'
)
logger = logging.getLogger("QdrantServer")

# Create storage directory
os.makedirs("/opt/sutazaiapp/storage/qdrant", exist_ok=True)

# Create Qdrant client
client = QdrantClient(path="/opt/sutazaiapp/storage/qdrant")

# Check if initialized
initialized_file = "/opt/sutazaiapp/storage/qdrant/.qdrant-initialized"
if not os.path.exists(initialized_file):
    # Create a test collection to verify everything works
    logger.info("Initializing Qdrant...")
    try:
        client.create_collection(
            collection_name="sutazai_vectors",
            vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
        )
        # Mark as initialized
        with open(initialized_file, "w") as f:
            f.write("")
        logger.info("Qdrant initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Qdrant: {str(e)}")

# Keep the process running
logger.info("Qdrant server started")
try:
    while True:
        time.sleep(10)
        # Periodically verify the service is working
        collections = client.get_collections()
        logger.debug(f"Collections: {collections}")
except KeyboardInterrupt:
    logger.info("Qdrant server stopping")
except Exception as e:
    logger.error(f"Error in Qdrant server: {str(e)}")
