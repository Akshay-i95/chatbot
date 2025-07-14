"""
Interactive Query Interface - Search and explore your vectorized PDF collection

This script provides an interactive interface to:
- Search through your PDF collection using natural language queries
- Browse documents by similarity
- Explore metadata and document statistics
"""

import os
import sys
from dotenv import load_dotenv
from vector_db_manager import VectorDBManager

def print_banner():
    """Print welcome banner"""
    print("=" * 60)
    print("📚 PDF VECTOR SEARCH INTERFACE")
    print("=" * 60)
    print("🔍 Search your PDF collection using natural language")
    print("💡 Type 'help' for commands, 'quit' to exit")
    print("=" * 60)

def print_help():
    """Print help information"""
    print("\n📖 Available Commands:")
    print("  search <query>     - Search for documents")
    print("  stats             - Show database statistics")
    print("  help              - Show this help")
    print("  quit/exit         - Exit the interface")
    print("\n💡 Examples:")
    print("  search education policy")
    print("  search teaching methods")
    print("  search student assessment")

def format_search_results(results, query):
    """Format and display search results"""
    if not results:
        print(f"❌ No results found for query: '{query}'")
        return
    
    print(f"\n🔍 Search Results for: '{query}'")
    print("-" * 50)
    
    for i, result in enumerate(results, 1):
        # Extract metadata
        filename = result.get('filename', 'Unknown')
        chunk_index = result.get('chunk_index', 0)
        similarity = result.get('similarity', 0)
        file_pages = result.get('file_pages', 'Unknown')
        chunk_length = result.get('chunk_length', 0)
        
        # Get text content
        text = result.get('text', '')
        
        # Create preview (first 200 characters)
        preview = text[:200] + "..." if len(text) > 200 else text
        preview = preview.replace('\n', ' ').strip()
        
        print(f"\n📄 Result {i}")
        print(f"   📁 File: {filename}")
        print(f"   📊 Similarity: {similarity:.3f}")
        print(f"   📖 Chunk: {chunk_index} | Pages: {file_pages} | Length: {chunk_length} chars")
        print(f"   📝 Preview: {preview}")
        
        if i < len(results):
            print("-" * 30)

def show_stats(vector_db):
    """Show database statistics"""
    stats = vector_db.get_collection_stats()
    
    print("\n📊 Vector Database Statistics:")
    print("-" * 30)
    print(f"🗄️  Database Type: {stats.get('database_type', 'Unknown')}")
    print(f"📚 Total Documents: {stats.get('total_documents', 0)}")
    print(f"📏 Embedding Dimension: {stats.get('embedding_dimension', 0)}")
    print(f"🤖 Model: {vector_db.embedding_model_name}")

def main():
    """Main interactive interface with comprehensive error handling"""
    try:
        # Load configuration
        load_dotenv()
        
        # Initialize vector database
        config = {
            'vector_db_type': os.getenv('VECTOR_DB_TYPE', 'chromadb'),
            'vector_db_path': os.getenv('VECTOR_DB_PATH', './vector_store'),
            'collection_name': os.getenv('COLLECTION_NAME', 'pdf_documents'),
            'embedding_model': os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
        }
        
        # Validate configuration
        if not os.path.exists(config['vector_db_path']):
            print("❌ Vector database not found.")
            print(f"💡 Expected location: {config['vector_db_path']}")
            print("💡 Run 'python main_pipeline.py' first to process your PDFs.")
            return
        
        print("🔄 Initializing vector database...")
        try:
            vector_db = VectorDBManager(config)
        except Exception as e:
            print(f"❌ Failed to initialize vector database: {str(e)}")
            print("💡 Try running 'python validate_environment.py' to check your setup")
            return
        
        # Check if database has data
        try:
            stats = vector_db.get_collection_stats()
            if stats.get('total_documents', 0) == 0:
                print("⚠️ Vector database appears to be empty.")
                print("💡 Run 'python main_pipeline.py' first to process your PDFs.")
                return
        except Exception as e:
            print(f"⚠️ Could not get database statistics: {str(e)}")
            print("💡 Database may be corrupted. Try rebuilding with 'python main_pipeline.py'")
            return
        
        print_banner()
        show_stats(vector_db)
        
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        while True:
            try:
                # Get user input
                user_input = input("\n🔍 Query > ").strip()
                
                if not user_input:
                    continue
                
                # Reset error counter on successful input
                consecutive_errors = 0
                
                # Handle commands
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                elif user_input.lower() == 'help':
                    print_help()
                
                elif user_input.lower() == 'stats':
                    show_stats(vector_db)
                
                elif user_input.lower().startswith('search '):
                    query = user_input[7:].strip()
                    if query:
                        print(f"🔄 Searching for: '{query}'...")
                        try:
                            results = vector_db.search(query, n_results=5)
                            format_search_results(results, query)
                        except Exception as e:
                            print(f"❌ Search failed: {str(e)}")
                            print("💡 Try a different query or check the database status")
                    else:
                        print("❌ Please provide a search query. Example: search education policy")
                
                else:
                    # Treat as direct search query
                    query = user_input
                    if len(query) > 100:
                        print("⚠️ Query is very long. Truncating to 100 characters.")
                        query = query[:100]
                    
                    print(f"🔄 Searching for: '{query}'...")
                    try:
                        results = vector_db.search(query, n_results=5)
                        format_search_results(results, query)
                    except Exception as e:
                        print(f"❌ Search failed: {str(e)}")
                        print("💡 Try a simpler query or check the database status")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except EOFError:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                consecutive_errors += 1
                print(f"❌ Error: {str(e)}")
                
                if consecutive_errors >= max_consecutive_errors:
                    print(f"🚨 Too many consecutive errors ({consecutive_errors}). Exiting.")
                    print("💡 Check your environment with 'python validate_environment.py'")
                    break
                else:
                    print("💡 Type 'help' for commands or 'quit' to exit")
    
    except Exception as e:
        print(f"❌ Critical error: {str(e)}")
        print("💡 Run 'python validate_environment.py' to check your setup")
        print("💡 Make sure you've run 'python main_pipeline.py' first")

if __name__ == "__main__":
    main()
