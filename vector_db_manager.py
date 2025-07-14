"""
Vector Database Manager - Handles embedding generation and vector storage

This module provides:
- Multiple vector database backends (ChromaDB, FAISS, Qdrant)
- Efficient embedding generation with sentence transformers
- Metadata storage and retrieval
- Batch processing for large datasets
"""

import os
import logging
import time
import numpy as np
from typing import List, Dict, Optional, Any
from sentence_transformers import SentenceTransformer
import json

# Vector Databases
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

class VectorDBManager:
    def __init__(self, config: Dict):
        try:
            self.config = config
            self.db_type = config.get('vector_db_type', 'chromadb').lower()
            self.db_path = config.get('vector_db_path', './vector_store')
            self.collection_name = config.get('collection_name', 'pdf_documents')
            self.embedding_model_name = config.get('embedding_model', 'all-MiniLM-L6-v2')
            
            # Validate configuration
            if not self.collection_name:
                raise ValueError("collection_name cannot be empty")
            
            # Setup logging
            self.logger = logging.getLogger(__name__)
            
            # Initialize embedding model with error handling
            self.embedding_model = None
            self.embedding_dim = None
            self._initialize_embedding_model()
            
            # Initialize vector database
            self.db_client = None
            self.collection = None
            self._initialize_database()
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize VectorDBManager: {str(e)}")
            raise
    
    def _initialize_embedding_model(self):
        """Initialize embedding model with proper error handling"""
        try:
            self.logger.info(f"🤖 Loading embedding model: {self.embedding_model_name}")
            
            # Try to load the model with timeout
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError("Model loading timeout")
            
            # Set timeout for model loading (60 seconds)
            if hasattr(signal, 'SIGALRM'):  # Unix only
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(60)
            
            try:
                self.embedding_model = SentenceTransformer(self.embedding_model_name)
                self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
                
                if hasattr(signal, 'SIGALRM'):
                    signal.alarm(0)  # Cancel timeout
                
                self.logger.info(f"✅ Model loaded successfully. Embedding dimension: {self.embedding_dim}")
                
                # Test the model
                test_embedding = self.embedding_model.encode(["test"], convert_to_numpy=True)
                if test_embedding.shape[1] != self.embedding_dim:
                    raise ValueError("Model dimension mismatch")
                    
            except TimeoutError:
                raise Exception(f"Model loading timeout for {self.embedding_model_name}")
            except Exception as e:
                if hasattr(signal, 'SIGALRM'):
                    signal.alarm(0)
                raise e
                
        except Exception as e:
            self.logger.error(f"❌ Failed to load embedding model: {str(e)}")
            self.logger.info("💡 Trying fallback model: all-MiniLM-L6-v2")
            
            try:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
                self.embedding_model_name = 'all-MiniLM-L6-v2'
                self.logger.info(f"✅ Fallback model loaded. Dimension: {self.embedding_dim}")
            except Exception as fallback_error:
                raise Exception(f"Failed to load both primary and fallback models: {str(fallback_error)}")
    
    def _initialize_database(self):
        """Initialize the chosen vector database"""
        os.makedirs(self.db_path, exist_ok=True)
        
        if self.db_type == 'chromadb':
            self._initialize_chromadb()
        elif self.db_type == 'faiss':
            self._initialize_faiss()
        else:
            raise ValueError(f"Unsupported vector database type: {self.db_type}")
    
    def _initialize_chromadb(self):
        """Initialize ChromaDB"""
        if not CHROMADB_AVAILABLE:
            raise ImportError("ChromaDB not available. Install with: pip install chromadb")
        
        self.logger.info("🗄️ Initializing ChromaDB")
        
        # Create ChromaDB client
        self.db_client = chromadb.PersistentClient(
            path=self.db_path,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        try:
            self.collection = self.db_client.get_collection(
                name=self.collection_name,
                embedding_function=None  # We'll handle embeddings ourselves
            )
            self.logger.info(f"📚 Loaded existing collection: {self.collection_name}")
        except:
            self.collection = self.db_client.create_collection(
                name=self.collection_name,
                embedding_function=None,
                metadata={"hnsw:space": "cosine"}
            )
            self.logger.info(f"📚 Created new collection: {self.collection_name}")
    
    def _initialize_faiss(self):
        """Initialize FAISS index"""
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS not available. Install with: pip install faiss-cpu")
        
        self.logger.info("🗄️ Initializing FAISS")
        
        index_path = os.path.join(self.db_path, f"{self.collection_name}.faiss")
        metadata_path = os.path.join(self.db_path, f"{self.collection_name}_metadata.json")
        
        if os.path.exists(index_path):
            # Load existing index
            self.db_client = faiss.read_index(index_path)
            with open(metadata_path, 'r') as f:
                self.metadata_store = json.load(f)
            self.logger.info(f"📚 Loaded existing FAISS index: {self.collection_name}")
        else:
            # Create new index
            self.db_client = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity
            self.metadata_store = {}
            self.logger.info(f"📚 Created new FAISS index: {self.collection_name}")
        
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.next_id = len(self.metadata_store)
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for a list of texts with error handling"""
        if not texts:
            return np.array([])
        
        if not self.embedding_model:
            raise RuntimeError("Embedding model not initialized")
        
        # Validate and clean texts
        valid_texts = []
        for text in texts:
            if isinstance(text, str) and text.strip():
                # Limit text length to prevent memory issues
                cleaned_text = text.strip()[:10000]  # Limit to 10k characters
                valid_texts.append(cleaned_text)
            else:
                valid_texts.append("empty")  # Placeholder for empty texts
        
        if not valid_texts:
            return np.array([])
        
        self.logger.info(f"🔄 Generating embeddings for {len(valid_texts)} texts")
        start_time = time.time()
        
        try:
            # Process in batches to manage memory
            all_embeddings = []
            
            for i in range(0, len(valid_texts), batch_size):
                batch_texts = valid_texts[i:i + batch_size]
                
                try:
                    batch_embeddings = self.embedding_model.encode(
                        batch_texts,
                        convert_to_numpy=True,
                        show_progress_bar=False,
                        batch_size=min(batch_size, 16),  # Limit batch size
                        normalize_embeddings=True  # Normalize for better similarity
                    )
                    
                    # Validate embeddings
                    if batch_embeddings.shape[0] != len(batch_texts):
                        raise ValueError(f"Embedding count mismatch: expected {len(batch_texts)}, got {batch_embeddings.shape[0]}")
                    
                    if batch_embeddings.shape[1] != self.embedding_dim:
                        raise ValueError(f"Embedding dimension mismatch: expected {self.embedding_dim}, got {batch_embeddings.shape[1]}")
                    
                    all_embeddings.append(batch_embeddings)
                    
                except Exception as e:
                    self.logger.error(f"❌ Error processing batch {i//batch_size + 1}: {str(e)}")
                    # Create zero embeddings as fallback
                    fallback_embeddings = np.zeros((len(batch_texts), self.embedding_dim))
                    all_embeddings.append(fallback_embeddings)
            
            if not all_embeddings:
                raise RuntimeError("No embeddings generated")
            
            embeddings = np.vstack(all_embeddings)
            
            processing_time = time.time() - start_time
            self.logger.info(f"✅ Generated {len(embeddings)} embeddings in {processing_time:.1f}s")
            
            return embeddings
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate embeddings: {str(e)}")
            # Return zero embeddings as last resort
            return np.zeros((len(valid_texts), self.embedding_dim))
    
    def add_documents_chromadb(self, chunks: List[Dict], batch_size: int = 100):
        """Add documents to ChromaDB"""
        if not chunks:
            return
        
        self.logger.info(f"📥 Adding {len(chunks)} documents to ChromaDB")
        
        # Process in batches
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            # Prepare data
            texts = [chunk['text'] for chunk in batch]
            ids = [f"{chunk['filename']}_{chunk['chunk_index']}" for chunk in batch]
            
            # Generate embeddings
            embeddings = self.generate_embeddings(texts)
            
            # Prepare metadata (remove 'text' as it's stored separately)
            metadatas = []
            for chunk in batch:
                metadata = {k: v for k, v in chunk.items() if k != 'text'}
                # ChromaDB requires string values for metadata
                metadata = {k: str(v) for k, v in metadata.items()}
                metadatas.append(metadata)
            
            # Add to collection
            self.collection.add(
                embeddings=embeddings.tolist(),
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            self.logger.info(f"✅ Added batch {i//batch_size + 1}: {len(batch)} documents")
    
    def add_documents_faiss(self, chunks: List[Dict], batch_size: int = 100):
        """Add documents to FAISS index"""
        if not chunks:
            return
        
        self.logger.info(f"📥 Adding {len(chunks)} documents to FAISS")
        
        # Process in batches
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            # Prepare data
            texts = [chunk['text'] for chunk in batch]
            
            # Generate embeddings
            embeddings = self.generate_embeddings(texts)
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            
            # Add to FAISS index
            self.db_client.add(embeddings)
            
            # Store metadata
            for j, chunk in enumerate(batch):
                doc_id = self.next_id + j
                self.metadata_store[str(doc_id)] = chunk
            
            self.next_id += len(batch)
            self.logger.info(f"✅ Added batch {i//batch_size + 1}: {len(batch)} documents")
        
        # Save index and metadata
        self.save_faiss_index()
    
    def save_faiss_index(self):
        """Save FAISS index and metadata to disk"""
        faiss.write_index(self.db_client, self.index_path)
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata_store, f, indent=2, default=str)
        self.logger.info("💾 FAISS index and metadata saved")
    
    def add_documents(self, chunks: List[Dict], batch_size: int = 100):
        """Add documents to the vector database"""
        if not chunks:
            self.logger.warning("⚠️ No chunks to add")
            return
        
        if self.db_type == 'chromadb':
            self.add_documents_chromadb(chunks, batch_size)
        elif self.db_type == 'faiss':
            self.add_documents_faiss(chunks, batch_size)
    
    def search_chromadb(self, query: str, n_results: int = 5) -> List[Dict]:
        """Search ChromaDB for similar documents"""
        query_embedding = self.generate_embeddings([query])
        
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=n_results,
            include=['documents', 'metadatas', 'distances']
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results['documents'][0])):
            formatted_results.append({
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i],
                'similarity': 1 - results['distances'][0][i]  # Convert distance to similarity
            })
        
        return formatted_results
    
    def search_faiss(self, query: str, n_results: int = 5) -> List[Dict]:
        """Search FAISS index for similar documents"""
        query_embedding = self.generate_embeddings([query])
        faiss.normalize_L2(query_embedding)
        
        similarities, indices = self.db_client.search(query_embedding, n_results)
        
        # Format results
        formatted_results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            if str(idx) in self.metadata_store:
                result = self.metadata_store[str(idx)].copy()
                result['similarity'] = float(similarities[0][i])
                formatted_results.append(result)
        
        return formatted_results
    
    def search(self, query: str, n_results: int = 5) -> List[Dict]:
        """Search for similar documents"""
        if self.db_type == 'chromadb':
            return self.search_chromadb(query, n_results)
        elif self.db_type == 'faiss':
            return self.search_faiss(query, n_results)
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the vector database collection"""
        if self.db_type == 'chromadb':
            try:
                count = self.collection.count()
                return {
                    'total_documents': count,
                    'embedding_dimension': self.embedding_dim,
                    'database_type': 'ChromaDB'
                }
            except:
                return {'total_documents': 0, 'embedding_dimension': self.embedding_dim, 'database_type': 'ChromaDB'}
        
        elif self.db_type == 'faiss':
            return {
                'total_documents': self.db_client.ntotal,
                'embedding_dimension': self.embedding_dim,
                'database_type': 'FAISS'
            }
    
    def reset_collection(self):
        """Reset/clear the collection"""
        if self.db_type == 'chromadb':
            self.db_client.delete_collection(self.collection_name)
            self.collection = self.db_client.create_collection(
                name=self.collection_name,
                embedding_function=None,
                metadata={"hnsw:space": "cosine"}
            )
            self.logger.info("🗑️ ChromaDB collection reset")
        
        elif self.db_type == 'faiss':
            self.db_client = faiss.IndexFlatIP(self.embedding_dim)
            self.metadata_store = {}
            self.next_id = 0
            self.logger.info("🗑️ FAISS index reset")
