"""
Enhanced Main Pipeline - Phase 2 & 3 Implementation
Integrates OCR processing, chunk-level storage, and AI chatbot functionality

This pipeline provides:
- Enhanced PDF processing with OCR and validation
- Chunk-level vector storage optimized for AI retrieval
- Interactive chatbot interface with precise context
- Comprehensive monitoring and statistics
"""

import os
import sys
import logging
import time
import json
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import components
from pdf_processor import EnhancedPDFProcessor
from vector_db import EnhancedVectorDBManager
from chatbot import AIChhatbotInterface

# Original components for Azure connection
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import AzureError

# Environment and configuration
from dotenv import load_dotenv

class EnhancedPipeline:
    def __init__(self, config_file: str = '.env'):
        """Initialize the enhanced pipeline with all components"""
        try:
            # Load configuration
            load_dotenv(config_file)
            self.config = self._load_configuration()
            
            # Setup logging
            self._setup_logging()
            self.logger = logging.getLogger(__name__)
            
            # Initialize components
            self.pdf_processor = None
            self.vector_db = None
            self.chatbot = None
            self.blob_service_client = None
            self.container_client = None
            
            # Pipeline statistics
            self.pipeline_stats = {
                'session_start': datetime.now().isoformat(),
                'total_files_processed': 0,
                'successful_extractions': 0,
                'failed_extractions': 0,
                'total_chunks_created': 0,
                'ocr_extractions': 0,
                'processing_time': 0,
                'storage_time': 0,
                'chatbot_queries': 0
            }
            
            self.logger.info("🚀 Enhanced Pipeline initialized successfully")
            
        except Exception as e:
            print(f"❌ Failed to initialize pipeline: {str(e)}")
            raise
    
    def _load_configuration(self) -> Dict:
        """Load and validate configuration"""
        config = {
            # Azure Blob Storage
            'azure_storage_account_name': os.getenv('AZURE_STORAGE_ACCOUNT_NAME'),
            'azure_storage_container_name': os.getenv('AZURE_STORAGE_CONTAINER_NAME'),
            'azure_storage_connection_string': os.getenv('AZURE_STORAGE_CONNECTION_STRING'),
            'azure_blob_folder_path': os.getenv('AZURE_BLOB_FOLDER_PATH', ''),
            
            # Processing Configuration
            'max_workers': int(os.getenv('MAX_WORKERS', 4)),  # Reduced for OCR
            'chunk_size': int(os.getenv('CHUNK_SIZE', 1000)),
            'chunk_overlap': int(os.getenv('CHUNK_OVERLAP', 200)),
            'batch_size': int(os.getenv('BATCH_SIZE', 20)),  # Reduced for enhanced processing
            'max_memory_mb': int(os.getenv('MAX_MEMORY_MB', 2048)),
            'min_chunk_length': int(os.getenv('MIN_CHUNK_LENGTH', 100)),
            'max_chunk_length': int(os.getenv('MAX_CHUNK_LENGTH', 2000)),
            
            # Enhanced Processing (Phase 2)
            'enable_ocr': os.getenv('ENABLE_OCR', 'true').lower() == 'true',
            'ocr_language': os.getenv('OCR_LANGUAGE', 'eng'),
            'ocr_dpi': int(os.getenv('OCR_DPI', 300)),
            'image_to_text': os.getenv('IMAGE_TO_TEXT', 'true').lower() == 'true',
            'enable_repair': os.getenv('ENABLE_REPAIR', 'true').lower() == 'true',
            'max_file_size_mb': int(os.getenv('MAX_FILE_SIZE_MB', 100)),
            
            # Vector Database Configuration
            'vector_db_type': os.getenv('VECTOR_DB_TYPE', 'chromadb'),
            'vector_db_path': os.getenv('VECTOR_DB_PATH', './vector_store'),
            'collection_name': os.getenv('COLLECTION_NAME', 'pdf_chunks'),
            'embedding_model': os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2'),
            
            # AI Chatbot Configuration (Phase 3)
            'max_context_chunks': int(os.getenv('MAX_CONTEXT_CHUNKS', 5)),
            'max_context_length': int(os.getenv('MAX_CONTEXT_LENGTH', 4000)),
            'min_similarity_threshold': float(os.getenv('MIN_SIMILARITY_THRESHOLD', 0.6)),
            'enable_citations': os.getenv('ENABLE_CITATIONS', 'true').lower() == 'true',
            'enable_context_expansion': os.getenv('ENABLE_CONTEXT_EXPANSION', 'true').lower() == 'true',
            'llm_model': os.getenv('LLM_MODEL', 'gpt-3.5-turbo'),
            'max_response_tokens': int(os.getenv('MAX_RESPONSE_TOKENS', 1000)),
            'temperature': float(os.getenv('TEMPERATURE', 0.7)),
            
            # File Processing
            'filter_pdf_files': os.getenv('FILTER_PDF_FILES', 'true').lower() == 'true',
            'show_file_summary': os.getenv('SHOW_FILE_SUMMARY', 'true').lower() == 'true'
        }
        
        # Validate required configuration
        required_fields = ['azure_storage_account_name', 'azure_storage_container_name']
        for field in required_fields:
            if not config.get(field):
                raise ValueError(f"Required configuration field missing: {field}")
        
        return config
    
    def _setup_logging(self):
        """Setup enhanced logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('enhanced_pipeline.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def initialize_components(self) -> bool:
        """Initialize all pipeline components"""
        try:
            self.logger.info("🔧 Initializing pipeline components...")
            
            # 1. Initialize enhanced PDF processor
            self.logger.info("📄 Initializing Enhanced PDF Processor...")
            self.pdf_processor = EnhancedPDFProcessor(self.config)
            
            # 2. Initialize enhanced vector database
            self.logger.info("🗄️ Initializing Enhanced Vector Database...")
            self.vector_db = EnhancedVectorDBManager(self.config)
            
            # 3. Initialize Azure Blob Storage
            self.logger.info("☁️ Connecting to Azure Blob Storage...")
            if not self._initialize_azure_storage():
                return False
            
            # 4. Initialize AI Chatbot
            self.logger.info("🤖 Initializing AI Chatbot Interface...")
            self.chatbot = AIChhatbotInterface(self.vector_db, self.config)
            
            self.logger.info("✅ All components initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Component initialization failed: {str(e)}")
            return False
    
    def _initialize_azure_storage(self) -> bool:
        """Initialize Azure Blob Storage connection"""
        try:
            # Try connection string first
            connection_string = self.config.get('azure_storage_connection_string')
            if connection_string:
                self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
            else:
                # Fallback to account name (would need additional authentication)
                account_name = self.config['azure_storage_account_name']
                self.logger.warning("⚠️ No connection string provided, authentication may fail")
                return False
            
            # Get container client
            container_name = self.config['azure_storage_container_name']
            self.container_client = self.blob_service_client.get_container_client(container_name)
            
            # Test connection
            _ = self.container_client.get_container_properties()
            
            self.logger.info(f"✅ Connected to Azure container: {container_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Azure storage initialization failed: {str(e)}")
            return False
    
    def run_enhanced_processing(self) -> bool:
        """Run the enhanced PDF processing pipeline"""
        try:
            if not self.initialize_components():
                return False
            
            self.logger.info("🚀 Starting Enhanced PDF Processing Pipeline")
            start_time = time.time()
            
            # Step 1: Get list of PDF files
            pdf_files = self._get_pdf_files_list()
            if not pdf_files:
                self.logger.warning("⚠️ No PDF files found to process")
                return False
            
            self.logger.info(f"📁 Found {len(pdf_files)} PDF files to process")
            
            # Step 2: Process PDFs with enhanced capabilities
            all_chunks = self._process_pdfs_enhanced(pdf_files)
            
            if not all_chunks:
                self.logger.error("❌ No chunks were successfully created")
                return False
            
            # Step 3: Store chunks in vector database
            storage_success = self._store_chunks_in_vector_db(all_chunks)
            
            if not storage_success:
                self.logger.error("❌ Failed to store chunks in vector database")
                return False
            
            # Step 4: Create search index for optimization
            self.vector_db.create_search_index()
            
            # Step 5: Generate processing report
            self._generate_processing_report()
            
            total_time = time.time() - start_time
            self.pipeline_stats['processing_time'] = total_time
            
            self.logger.info(f"✅ Enhanced processing completed in {total_time:.2f}s")
            self.logger.info(f"📊 Success Rate: {self._calculate_success_rate():.1f}%")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Enhanced processing failed: {str(e)}")
            return False
    
    def _get_pdf_files_list(self) -> List[str]:
        """Get list of PDF files from Azure Blob Storage"""
        try:
            pdf_files = []
            folder_path = self.config.get('azure_blob_folder_path', '')
            
            # List blobs in container
            blobs = self.container_client.list_blobs(name_starts_with=folder_path)
            
            for blob in blobs:
                if self.config.get('filter_pdf_files', True):
                    if blob.name.lower().endswith('.pdf'):
                        pdf_files.append(blob.name)
                else:
                    pdf_files.append(blob.name)
            
            return pdf_files
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get PDF files list: {str(e)}")
            return []
    
    def _process_pdfs_enhanced(self, pdf_files: List[str]) -> List[Dict]:
        """Process PDFs using enhanced processor with OCR and validation"""
        try:
            all_chunks = []
            
            # Process in batches
            batch_size = self.config.get('batch_size', 20)
            total_batches = (len(pdf_files) + batch_size - 1) // batch_size
            
            for i in range(0, len(pdf_files), batch_size):
                batch = pdf_files[i:i + batch_size]
                batch_num = (i // batch_size) + 1
                
                self.logger.info(f"📦 Processing batch {batch_num}/{total_batches} ({len(batch)} files)")
                
                batch_chunks = self.pdf_processor.process_pdf_batch_enhanced(
                    self.container_client, batch
                )
                
                all_chunks.extend(batch_chunks)
                
                self.logger.info(f"✅ Batch {batch_num} completed: {len(batch_chunks)} chunks")
            
            # Update pipeline statistics
            processor_stats = self.pdf_processor.get_processing_stats()
            self.pipeline_stats.update(processor_stats)
            
            return all_chunks
            
        except Exception as e:
            self.logger.error(f"❌ Enhanced PDF processing failed: {str(e)}")
            return []
    
    def _store_chunks_in_vector_db(self, chunks: List[Dict]) -> bool:
        """Store processed chunks in vector database"""
        try:
            start_time = time.time()
            
            self.logger.info(f"🗄️ Storing {len(chunks)} chunks in vector database...")
            
            # Store in batches for better performance
            batch_size = 100
            successful_batches = 0
            
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                
                if self.vector_db.store_chunks_batch(batch):
                    successful_batches += 1
                else:
                    self.logger.warning(f"⚠️ Failed to store batch {i//batch_size + 1}")
            
            storage_time = time.time() - start_time
            self.pipeline_stats['storage_time'] = storage_time
            self.pipeline_stats['total_chunks_created'] = len(chunks)
            
            self.logger.info(f"✅ Stored {len(chunks)} chunks in {storage_time:.2f}s")
            
            return successful_batches > 0
            
        except Exception as e:
            self.logger.error(f"❌ Vector database storage failed: {str(e)}")
            return False
    
    def start_interactive_chatbot(self):
        """Start interactive chatbot interface"""
        try:
            if not self.chatbot:
                self.logger.error("❌ Chatbot not initialized")
                return
            
            print("\\n" + "="*60)
            print("🤖 AI CHATBOT - Enhanced with Chunk-Level Retrieval")
            print("="*60)
            print("Ask questions about the processed documents.")
            print("Type 'quit', 'exit', or 'bye' to end the session.")
            print("Type 'stats' to see processing statistics.")
            print("Type 'reset' to clear conversation history.")
            print("-"*60)
            
            while True:
                try:
                    user_input = input("\\n🙋 You: ").strip()
                    
                    if not user_input:
                        continue
                    
                    if user_input.lower() in ['quit', 'exit', 'bye']:
                        print("\\n👋 Goodbye! Thanks for using the AI Chatbot.")
                        break
                    
                    elif user_input.lower() == 'stats':
                        self._display_chatbot_stats()
                        continue
                    
                    elif user_input.lower() == 'reset':
                        self.chatbot.reset_conversation()
                        print("\\n🔄 Conversation history cleared.")
                        continue
                    
                    # Process the query
                    print("\\n🤖 AI: Searching for relevant information...")
                    
                    response = self.chatbot.process_query(user_input)
                    
                    # Display response
                    self._display_chatbot_response(response)
                    
                    # Update pipeline stats
                    self.pipeline_stats['chatbot_queries'] += 1
                    
                except KeyboardInterrupt:
                    print("\\n\\n👋 Session interrupted. Goodbye!")
                    break
                except Exception as e:
                    print(f"\\n❌ Error processing query: {str(e)}")
                    continue
            
        except Exception as e:
            self.logger.error(f"❌ Interactive chatbot failed: {str(e)}")
    
    def _display_chatbot_response(self, response: Dict):
        """Display formatted chatbot response"""
        print("\\n" + "-"*60)
        print("🤖 AI Response:")
        print("-"*60)
        print(response['response'])
        
        if response.get('sources'):
            print("\\n📚 Sources:")
            for i, source in enumerate(response['sources'][:3], 1):
                print(f"  {i}. {source['filename']} (Relevance: {source['relevance_score']:.2f})")
        
        print(f"\\n📊 Query Info: {response['chunks_used']} chunks used, "
              f"Confidence: {response.get('confidence', 0):.2f}, "
              f"Response time: {response['response_time']:.2f}s")
    
    def _display_chatbot_stats(self):
        """Display chatbot and pipeline statistics"""
        print("\\n" + "="*50)
        print("📊 SYSTEM STATISTICS")
        print("="*50)
        
        # Pipeline stats
        print("\\n📄 Document Processing:")
        print(f"  • Total files processed: {self.pipeline_stats['total_files_processed']}")
        print(f"  • Successful extractions: {self.pipeline_stats['successful_extractions']}")
        print(f"  • Success rate: {self._calculate_success_rate():.1f}%")
        print(f"  • OCR extractions: {self.pipeline_stats.get('ocr_extractions', 0)}")
        print(f"  • Total chunks created: {self.pipeline_stats['total_chunks_created']}")
        
        # Vector database stats
        if self.vector_db:
            db_stats = self.vector_db.get_collection_stats()
            print(f"\\n🗄️ Vector Database:")
            print(f"  • Database type: {db_stats.get('database_type', 'unknown')}")
            print(f"  • Total chunks stored: {db_stats.get('total_chunks', 0)}")
            print(f"  • Embedding model: {db_stats.get('embedding_model', 'unknown')}")
            print(f"  • Queries processed: {db_stats.get('queries_processed', 0)}")
            print(f"  • Avg query time: {db_stats.get('average_query_time_ms', 0):.1f}ms")
        
        # Chatbot stats
        if self.chatbot:
            chat_stats = self.chatbot.get_session_stats()
            print(f"\\n🤖 Chatbot Session:")
            print(f"  • Queries processed: {chat_stats.get('queries_processed', 0)}")
            print(f"  • Conversation turns: {chat_stats.get('conversation_turns', 0)}")
            print(f"  • Avg response time: {chat_stats.get('average_response_time', 0):.2f}s")
            print(f"  • Session duration: {chat_stats.get('session_duration_minutes', 0):.1f} minutes")
    
    def _calculate_success_rate(self) -> float:
        """Calculate processing success rate"""
        total = self.pipeline_stats.get('total_files_processed', 0)
        successful = self.pipeline_stats.get('successful_extractions', 0)
        return (successful / max(1, total)) * 100
    
    def _generate_processing_report(self):
        """Generate comprehensive processing report"""
        try:
            report = {
                'session_info': {
                    'start_time': self.pipeline_stats['session_start'],
                    'end_time': datetime.now().isoformat(),
                    'total_processing_time': self.pipeline_stats['processing_time'],
                    'storage_time': self.pipeline_stats['storage_time']
                },
                'processing_stats': self.pipeline_stats,
                'vector_db_stats': self.vector_db.get_collection_stats() if self.vector_db else {},
                'configuration': {
                    'ocr_enabled': self.config.get('enable_ocr', False),
                    'repair_enabled': self.config.get('enable_repair', False),
                    'vector_db_type': self.config.get('vector_db_type', 'unknown'),
                    'embedding_model': self.config.get('embedding_model', 'unknown')
                }
            }
            
            # Save report to file
            report_file = f"processing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"📊 Processing report saved to: {report_file}")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to generate processing report: {str(e)}")
    
    def query_documents(self, query: str) -> Dict:
        """Query documents programmatically (API-style interface)"""
        try:
            if not self.chatbot:
                raise ValueError("Chatbot not initialized")
            
            response = self.chatbot.process_query(query)
            self.pipeline_stats['chatbot_queries'] += 1
            
            return response
            
        except Exception as e:
            self.logger.error(f"❌ Document query failed: {str(e)}")
            return {
                'query': query,
                'response': f"Query failed: {str(e)}",
                'sources': [],
                'error': str(e)
            }

def main():
    """Main function to run the enhanced pipeline"""
    try:
        print("🚀 Starting Enhanced PDF Processing Pipeline")
        print("=" * 60)
        
        # Initialize pipeline
        pipeline = EnhancedPipeline()
        
        # Run processing
        success = pipeline.run_enhanced_processing()
        
        if success:
            print("\\n✅ Processing completed successfully!")
            print("\\n🤖 Starting interactive chatbot...")
            pipeline.start_interactive_chatbot()
        else:
            print("\\n❌ Processing failed. Check logs for details.")
            return 1
        
        return 0
        
    except Exception as e:
        print(f"\\n❌ Pipeline execution failed: {str(e)}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
