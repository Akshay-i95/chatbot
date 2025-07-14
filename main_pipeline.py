"""
Main Pipeline - Orchestrates the complete PDF processing and vectorization workflow

This script:
1. Connects to Azure Blob Storage
2. Streams and processes PDF files in intelligent batches
3. Generates embeddings and stores in vector database
4. Provides search and query capabilities
"""

import os
import sys
import time
import logging
from typing import List, Dict
from dotenv import load_dotenv

# Import our custom modules
from azure_blob_test import create_blob_service_client, test_connection, load_config
from pdf_processor import PDFStreamProcessor
from vector_db_manager import VectorDBManager

def setup_logging():
    """Setup comprehensive logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('pdf_vectorization.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def get_pdf_files(container_client, folder_path: str) -> List[str]:
    """Get list of PDF files from the container"""
    pdf_files = []
    
    try:
        blob_list = container_client.list_blobs(name_starts_with=folder_path)
        
        for blob in blob_list:
            if blob.name.lower().endswith('.pdf') and not blob.name.endswith('/'):
                pdf_files.append(blob.name)
        
        return sorted(pdf_files)
        
    except Exception as e:
        logging.error(f"❌ Error listing PDF files: {str(e)}")
        return []

def main():
    """Main pipeline execution with comprehensive error handling"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 Starting PDF Vectorization Pipeline")
    
    # Track overall progress
    start_time = time.time()
    total_chunks = 0
    processed_files = 0
    failed_files = 0
    
    try:
        # Load and validate configuration
        config = load_config()
        
        # Validate required configuration
        required_fields = ['account_name', 'container_name']
        missing_fields = [field for field in required_fields if not config.get(field)]
        
        if missing_fields:
            logger.error(f"❌ Missing required configuration: {', '.join(missing_fields)}")
            logger.info("💡 Please check your .env file and ensure all required fields are set")
            return False
        
        logger.info(f"🏢 Account: {config['account_name']}")
        logger.info(f"📦 Container: {config['container_name']}")
        logger.info(f"📁 Folder: {config['folder_path'] or '(root)'}")
        
        # Connect to Azure Blob Storage with retry logic
        logger.info("🔗 Connecting to Azure Blob Storage...")
        max_connection_retries = 3
        container_client = None
        
        for attempt in range(max_connection_retries):
            try:
                blob_service_client = create_blob_service_client(config)
                container_client = test_connection(blob_service_client, config['container_name'])
                if container_client:
                    break
                else:
                    raise Exception("Connection test failed")
            except Exception as e:
                if attempt < max_connection_retries - 1:
                    logger.warning(f"⚠️ Connection attempt {attempt + 1} failed: {str(e)}, retrying...")
                    time.sleep(2)
                else:
                    logger.error(f"❌ Failed to connect after {max_connection_retries} attempts")
                    return False
        
        if not container_client:
            logger.error("❌ Failed to establish Azure Blob Storage connection")
            return False
        
        # Get list of PDF files
        logger.info("📋 Getting list of PDF files...")
        pdf_files = get_pdf_files(container_client, config['folder_path'])
        
        if not pdf_files:
            logger.warning("⚠️ No PDF files found in the specified path")
            logger.info("💡 Check your AZURE_BLOB_FOLDER_PATH configuration")
            return False
        
        logger.info(f"📄 Found {len(pdf_files)} PDF files to process")
        
        # Validate processor configuration
        processor_config = {
            'max_workers': min(int(config.get('max_workers', 8)), 16),  # Limit workers
            'chunk_size': max(100, int(config.get('chunk_size', 1000))),
            'chunk_overlap': max(0, int(config.get('chunk_overlap', 200))),
            'batch_size': max(1, min(int(config.get('batch_size', 50)), 100)),  # Limit batch size
            'max_memory_mb': max(512, int(config.get('max_memory_mb', 2048))),
            'min_chunk_length': max(50, int(config.get('min_chunk_length', 100))),
            'max_chunk_length': max(500, int(config.get('max_chunk_length', 2000)))
        }
        
        # Initialize PDF processor with error handling
        try:
            pdf_processor = PDFStreamProcessor(processor_config)
            logger.info("✅ PDF processor initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize PDF processor: {str(e)}")
            return False
        
        # Initialize Vector Database with error handling
        vector_config = {
            'vector_db_type': config.get('vector_db_type', 'chromadb'),
            'vector_db_path': config.get('vector_db_path', './vector_store'),
            'collection_name': config.get('collection_name', 'pdf_documents'),
            'embedding_model': config.get('embedding_model', 'all-MiniLM-L6-v2')
        }
        
        try:
            vector_db = VectorDBManager(vector_config)
            logger.info("✅ Vector database initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize vector database: {str(e)}")
            logger.info("💡 Try using 'chromadb' as vector_db_type or check internet connection for model download")
            return False
        
        # Process PDFs and build vector database
        logger.info("🔄 Starting PDF processing and vectorization...")
        
        try:
            batch_count = 0
            for batch_chunks in pdf_processor.process_all_pdfs(container_client, pdf_files):
                batch_count += 1
                
                if batch_chunks:
                    try:
                        # Add chunks to vector database
                        vector_db.add_documents(batch_chunks)
                        total_chunks += len(batch_chunks)
                        processed_files += len(set(chunk['filename'] for chunk in batch_chunks))
                        
                        logger.info(f"💾 Batch {batch_count}: {len(batch_chunks)} chunks added. Total: {total_chunks}")
                        
                    except Exception as e:
                        logger.error(f"❌ Error adding batch {batch_count} to database: {str(e)}")
                        failed_files += processor_config['batch_size']
                        continue
                else:
                    logger.warning(f"⚠️ Batch {batch_count} produced no chunks")
                    failed_files += processor_config['batch_size']
        
        except KeyboardInterrupt:
            logger.info("⚠️ Processing interrupted by user")
            return False
        except Exception as e:
            logger.error(f"❌ Error during PDF processing: {str(e)}")
            return False
        
        total_time = time.time() - start_time
        
        # Get final statistics
        try:
            db_stats = vector_db.get_collection_stats()
        except Exception as e:
            logger.warning(f"⚠️ Could not get database stats: {str(e)}")
            db_stats = {}
        
        # Report results
        logger.info("🎉 Pipeline completed!")
        logger.info(f"📊 Final Statistics:")
        logger.info(f"  📄 PDFs found: {len(pdf_files)}")
        logger.info(f"  ✅ PDFs processed: {processed_files}")
        logger.info(f"  ❌ PDFs failed: {failed_files}")
        logger.info(f"  📝 Total chunks: {total_chunks}")
        logger.info(f"  �️ Database type: {db_stats.get('database_type', 'Unknown')}")
        logger.info(f"  💾 Documents in DB: {db_stats.get('total_documents', 'Unknown')}")
        logger.info(f"  ⏱️ Total time: {total_time:.1f}s")
        
        if processed_files > 0:
            logger.info(f"  🚀 Processing speed: {processed_files/total_time:.2f} PDFs/second")
        
        # Test search functionality if we have documents
        if total_chunks > 0:
            logger.info("\n🔍 Testing search functionality...")
            test_queries = ["education", "policy", "curriculum", "assessment"]
            
            for query in test_queries[:2]:  # Limit to 2 test queries
                try:
                    logger.info(f"🔎 Testing search: '{query}'")
                    results = vector_db.search(query, n_results=2)
                    
                    if results:
                        logger.info(f"  ✅ Found {len(results)} results")
                        for i, result in enumerate(results, 1):
                            similarity = result.get('similarity', 0)
                            filename = result.get('filename', 'Unknown')
                            logger.info(f"    {i}. {filename} (similarity: {similarity:.3f})")
                    else:
                        logger.info(f"  ⚠️ No results found for '{query}'")
                        
                except Exception as e:
                    logger.warning(f"⚠️ Search test failed for '{query}': {str(e)}")
        
        success_rate = (processed_files / len(pdf_files)) * 100 if pdf_files else 0
        logger.info(f"\n✅ Pipeline completed with {success_rate:.1f}% success rate")
        
        if success_rate < 50:
            logger.warning("⚠️ Low success rate detected. Check logs for errors.")
            return False
        
        logger.info("💡 Use 'python interactive_search.py' to search your documents")
        return True
        
    except KeyboardInterrupt:
        logger.info("⚠️ Pipeline interrupted by user")
        return False
    except Exception as e:
        logger.error(f"❌ Pipeline failed with unexpected error: {str(e)}")
        logger.exception("Full error details:")
        return False
    finally:
        final_time = time.time() - start_time
        logger.info(f"🏁 Pipeline finished after {final_time:.1f}s")

if __name__ == "__main__":
    main()
