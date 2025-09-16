#!/usr/bin/env python3
"""
Test script to verify ChromaDB v2 API connectivity
"""

import chromadb
from chromadb.config import Settings
import sys

def test_chromadb_v2():
    """Test ChromaDB v2 API connectivity"""
    
    # Test configurations for v2 API
    configs = [
        # Configuration 1: Using HttpClient with default path
        {
            "name": "HttpClient with default settings",
            "client": lambda: chromadb.HttpClient(
                host="sutazai-chromadb",
                port=8000
            )
        },
        # Configuration 2: Using HttpClient with explicit v2 path
        {
            "name": "HttpClient with v2 path",
            "client": lambda: chromadb.HttpClient(
                host="sutazai-chromadb",
                port=8000,
                settings=Settings(
                    chroma_api_impl="chromadb.api.segment.SegmentAPI",
                    chroma_server_host="sutazai-chromadb",
                    chroma_server_http_port=8000,
                    chroma_server_ssl_enabled=False
                )
            )
        },
        # Configuration 3: Using Client factory with HTTP settings
        {
            "name": "Client factory with HTTP settings",
            "client": lambda: chromadb.Client(
                Settings(
                    chroma_api_impl="chromadb.api.segment.SegmentAPI",
                    chroma_server_host="sutazai-chromadb",
                    chroma_server_http_port=8000,
                    chroma_server_ssl_enabled=False,
                    chroma_client_auth_provider=None,
                    chroma_client_auth_credentials=None
                )
            )
        }
    ]
    
    for config in configs:
        print(f"\n✅ Testing: {config['name']}")
        print("-" * 50)
        
        try:
            # Create client
            client = config['client']()
            print(f"  Client created successfully")
            
            # Test heartbeat (v2 API check)
            try:
                result = client.heartbeat()
                print(f"  Heartbeat successful: {result}")
            except Exception as e:
                print(f"  Heartbeat failed: {e}")
            
            # Test list collections (another v2 API check)
            try:
                collections = client.list_collections()
                print(f"  Collections listed: {len(collections)} collections found")
            except Exception as e:
                print(f"  List collections failed: {e}")
                
            # Test create/get collection
            try:
                collection = client.get_or_create_collection(name="test_v2_api")
                print(f"  Collection 'test_v2_api' accessed successfully")
                
                # Try to add some data
                collection.add(
                    documents=["Test document for v2 API"],
                    ids=["test_id_1"]
                )
                print(f"  Document added successfully")
                
                # Query the collection
                results = collection.query(
                    query_texts=["test"],
                    n_results=1
                )
                print(f"  Query executed successfully")
                
            except Exception as e:
                print(f"  Collection operations failed: {e}")
                
        except Exception as e:
            print(f"  ❌ Client creation failed: {e}")
            continue
    
    print("\n" + "=" * 50)
    print("ChromaDB v2 API Testing Complete")

if __name__ == "__main__":
    test_chromadb_v2()