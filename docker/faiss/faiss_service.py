import faiss
import numpy as np
from flask import Flask, request, jsonify
import pickle
import os

app = Flask(__name__)

# Global index storage
indexes = {}
data_path = os.environ.get('FAISS_DATA_PATH', '/data/faiss_indexes')
os.makedirs(data_path, exist_ok=True)

@app.route('/health')
def health():
    return jsonify({"status": "healthy"})

@app.route('/create_index', methods=['POST'])
def create_index():
    data = request.json
    index_name = data['name']
    dimension = data['dimension']
    index_type = data.get('type', 'IVFFlat')
    
    if index_type == 'IVFFlat':
        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, 100)
    else:
        index = faiss.IndexFlatL2(dimension)
    
    indexes[index_name] = index
    return jsonify({"status": "created", "name": index_name})

@app.route('/add_vectors', methods=['POST'])
def add_vectors():
    data = request.json
    index_name = data['index']
    vectors = np.array(data['vectors'], dtype=np.float32)
    
    if index_name not in indexes:
        return jsonify({"error": "Index not found"}), 404
    
    index = indexes[index_name]
    if not index.is_trained:
        index.train(vectors)
    
    index.add(vectors)
    return jsonify({"status": "added", "total": index.ntotal})

@app.route('/search', methods=['POST'])
def search():
    data = request.json
    index_name = data['index']
    query_vector = np.array(data['query'], dtype=np.float32).reshape(1, -1)
    k = data.get('k', 10)
    
    if index_name not in indexes:
        return jsonify({"error": "Index not found"}), 404
    
    index = indexes[index_name]
    distances, indices = index.search(query_vector, k)
    
    return jsonify({
        "distances": distances[0].tolist(),
        "indices": indices[0].tolist()
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
