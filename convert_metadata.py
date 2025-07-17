#!/usr/bin/env python3
"""
Convert metadata from pickle to JSON format
"""
import pickle
import json
import os

def convert_metadata():
    """Convert pickle metadata to JSON format"""
    
    # Load pickle metadata
    with open('./vector_store_faiss/metadata.pkl', 'rb') as f:
        data = pickle.load(f)
    
    # Convert to the expected format
    metadata_dict = {}
    id_counter = 0
    
    documents = data['documents']
    metadatas = data['metadatas']
    ids = data['ids']
    
    for i, (doc, meta, doc_id) in enumerate(zip(documents, metadatas, ids)):
        metadata_dict[str(i)] = {
            'text': doc,
            'source': meta.get('source', ''),
            'page_number': meta.get('page_number', 0),
            'chunk_index': meta.get('chunk_index', 0),
            'doc_id': doc_id,
            **meta  # Include all original metadata
        }
        id_counter = i + 1
    
    # Save in expected JSON format
    json_data = {
        'metadata': metadata_dict,
        'id_counter': id_counter
    }
    
    with open('./vector_store_faiss/enhanced_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Converted {len(metadata_dict)} metadata entries to JSON format")
    print(f"📁 Saved to: ./vector_store_faiss/enhanced_metadata.json")

if __name__ == "__main__":
    convert_metadata()
