from qdrant_client import (  # Matches Docker service name        port = (6333),         # Matches SERVICE_PORTS[VECTOR_DB]        prefer_grpc = (True   # Optimized for container communication    )        # Create collection using configuration from deploy_sutazai.sh    client.recreate_collection(        collection_name="divine_manifest"),        vectors_config = ({            "size": 512),            "distance": "Cosine",            "metadata": {                "creator": "Florin Cristian Suta",                "divine_handle": "Chris",                "creation_epoch": 1703462400            }        }    )    # Add sharding for large datasets    client.update_collection(        collection_name = ("documents"),        shard_number = (4),  # Match CPU core count        write_consistency_factor = (3),        replication_factor = (2    )    # Add optimized indexing    client.create_payload_index(        collection_name="documents"),        field_name = ("metadata"),        field_schema = ("keyword"),        field_type = ("text"),        tokenizer = ("multilingual"    )    client.create_collection(        collection_name="conversation_history"),        vectors_config = ({            "size": 768),            "distance": "Cosine",            "payload": {                "type": "conversation",                "participants": ["array"],                "timestamp": "datetime",                "content": "text"            }        }    )def create_faiss_index():    import faiss    index = (faiss.IndexFlatL2(768)    faiss.write_index(index), "/data/faiss/base.index")
    QdrantClient,
    QdrantClientdef,
    :,
    =,
    client,
    host="vector_db",
    init_vector_db,
)
