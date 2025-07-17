#!/usr/bin/env python3
"""
Migrate from ChromaDB to FAISS for Streamlit Cloud compatibility
This script converts the existing ChromaDB vector store to FAISS format
"""

import os
import json
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

def migrate_chromadb_to_faiss():
    """Convert ChromaDB data to FAISS format"""
    print("🔄 Starting ChromaDB to FAISS migration...")
    
    # Load embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embedding_dim = model.get_sentence_embedding_dimension()
    print(f"📏 Embedding dimension: {embedding_dim}")
    
    # Try to extract data from ChromaDB
    try:
        import chromadb
        from chromadb.config import Settings
        
        # Initialize ChromaDB client
        chroma_client = chromadb.PersistentClient(path="./vector_store")
        collection = chroma_client.get_collection("pdf_chunks")
        
        # Get all data from ChromaDB
        print("📊 Extracting data from ChromaDB...")
        results = collection.get(include=['documents', 'metadatas', 'embeddings'])
        
        documents = results['documents']
        metadatas = results['metadatas'] 
        embeddings = results['embeddings']
        ids = results['ids']
        
        print(f"✅ Extracted {len(documents)} documents from ChromaDB")
        
    except Exception as e:
        print(f"❌ Error reading ChromaDB: {e}")
        print("🔧 Attempting to regenerate embeddings from documents...")
        
        # If ChromaDB fails, try to find document texts and regenerate
        # This is a fallback - you might need to adjust based on your data
        return False
    
    # Create FAISS index
    print("🏗️ Creating FAISS index...")
    faiss_index = faiss.IndexFlatIP(embedding_dim)  # Inner product for cosine similarity
    
    # Convert embeddings to numpy array and normalize
    embeddings_array = np.array(embeddings, dtype=np.float32)
    
    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings_array)
    
    # Add to FAISS index
    faiss_index.add(embeddings_array)
    
    # Create metadata storage
    metadata_store = {
        'documents': documents,
        'metadatas': metadatas,
        'ids': ids
    }
    
    # Create FAISS directory
    faiss_dir = "./vector_store_faiss"
    os.makedirs(faiss_dir, exist_ok=True)
    
    # Save FAISS index
    faiss_index_path = os.path.join(faiss_dir, "pdf_chunks.index")
    faiss.write_index(faiss_index, faiss_index_path)
    
    # Save metadata
    metadata_path = os.path.join(faiss_dir, "metadata.pkl")
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata_store, f)
    
    # Save configuration
    config_data = {
        'embedding_model': 'all-MiniLM-L6-v2',
        'embedding_dim': embedding_dim,
        'total_docs': len(documents),
        'created_at': str(np.datetime64('now'))
    }
    
    config_path = os.path.join(faiss_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=2)
    
    print(f"✅ Migration complete!")
    print(f"📁 FAISS vector store saved to: {faiss_dir}")
    print(f"📊 Migrated {len(documents)} documents")
    print(f"🗂️ Index size: {faiss_index.ntotal} vectors")
    
    return True

if __name__ == "__main__":
    success = migrate_chromadb_to_faiss()
    if success:
        print("\n🎉 Migration successful! Update your .env to use FAISS:")
        print("VECTOR_DB_TYPE=faiss")
        print("VECTOR_DB_PATH=./vector_store_faiss")
    else:
        print("\n❌ Migration failed. Check the logs above.")
