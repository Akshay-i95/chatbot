"""
Streamlit Web Application - Phase 4 Deployment
Professional AI Chatbot Interface with OpenRouter Integration
"""

import streamlit as st
import os
import sys
import logging
import time
import json
from datetime import datetime
from typing import Dict, List, Optional
import traceback

# Configure page
st.set_page_config(
    page_title="Edify AI Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('streamlit_app.log'),
        logging.StreamHandler()
    ]
)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Custom CSS for professional styling
def load_custom_css():
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    
    /* Light mode styles */
    .user-message {
        background-color: #f0f2f6;
        border-left-color: #667eea;
        color: #262730;
    }
    
    .ai-message {
        background-color: #e8f4fd;
        border-left-color: #00a0dc;
        color: #262730;
    }
    
    .source-info {
        background-color: #f8f9fa;
        padding: 0.5rem;
        border-radius: 5px;
        font-size: 0.9rem;
        margin-top: 1rem;
        color: #262730;
    }
    
    .metrics-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: #262730;
    }
    
    .sidebar-info {
        background-color: #e9ecef;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        color: #262730;
    }
    
    /* Dark mode styles */
    @media (prefers-color-scheme: dark) {
        .user-message {
            background-color: #2d3748;
            border-left-color: #667eea;
            color: #ffffff !important;
        }
        
        .ai-message {
            background-color: #1a365d;
            border-left-color: #00a0dc;
            color: #ffffff !important;
        }
        
        .source-info {
            background-color: #2d3748;
            color: #e2e8f0 !important;
        }
        
        .metrics-container {
            background-color: #2d3748;
            color: #e2e8f0 !important;
        }
        
        .sidebar-info {
            background-color: #2d3748;
            color: #e2e8f0 !important;
        }
    }
    
    /* Streamlit dark theme detection */
    [data-theme="dark"] .user-message {
        background-color: #2d3748 !important;
        border-left-color: #667eea !important;
        color: #ffffff !important;
    }
    
    [data-theme="dark"] .ai-message {
        background-color: #1a365d !important;
        border-left-color: #00a0dc !important;
        color: #ffffff !important;
    }
    
    [data-theme="dark"] .source-info {
        background-color: #2d3748 !important;
        color: #e2e8f0 !important;
    }
    
    [data-theme="dark"] .metrics-container {
        background-color: #2d3748 !important;
        color: #e2e8f0 !important;
    }
    
    [data-theme="dark"] .sidebar-info {
        background-color: #2d3748 !important;
        color: #e2e8f0 !important;
    }
    
    .status-success {
        color: #28a745;
        font-weight: bold;
    }
    
    .status-warning {
        color: #ffc107;
        font-weight: bold;
    }
    
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

def initialize_session_state():
    """Initialize Streamlit session state"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = None
    
    if 'vector_db' not in st.session_state:
        st.session_state.vector_db = None
    
    if 'llm_service' not in st.session_state:
        st.session_state.llm_service = None
    
    if 'system_status' not in st.session_state:
        st.session_state.system_status = "initializing"
    
    if 'stats' not in st.session_state:
        st.session_state.stats = {
            'total_queries': 0,
            'successful_responses': 0,
            'avg_response_time': 0,
            'session_start': datetime.now()
        }

@st.cache_resource
def load_system_components():
    """Load and cache system components"""
    try:
        st.info("🔄 Loading system components...")
        
        # Configuration
        config = {
            'vector_db_path': './vector_store',
            'collection_name': 'pdf_chunks',
            'embedding_model': 'all-MiniLM-L6-v2',
            'max_context_chunks': 5,
            'min_similarity_threshold': 0.35,
            'enable_citations': True,
            'enable_context_expansion': True,
            'max_context_length': 4000
        }
        
        # Initialize Vector Database
        from vector_db import EnhancedVectorDBManager
        vector_db = EnhancedVectorDBManager(config)
        
        # Initialize AI Chatbot
        from chatbot import AIChhatbotInterface
        chatbot = AIChhatbotInterface(vector_db, config)
        
        # Initialize LLM Service
        from llm_service import LLMService
        llm_service = LLMService(config)
        
        st.success("✅ System components loaded successfully!")
        
        return {
            'vector_db': vector_db,
            'chatbot': chatbot,
            'llm_service': llm_service,
            'config': config,
            'status': 'ready'
        }
        
    except Exception as e:
        st.error(f"❌ Failed to load system components: {str(e)}")
        st.error("Please check your configuration and try again.")
        return {
            'vector_db': None,
            'chatbot': None,
            'llm_service': None,
            'config': {},
            'status': 'error',
            'error': str(e)
        }

def display_header():
    """Display application header"""
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Edify AI Chatbot</h1>
        <p>Advanced Document Analysis with AI-Powered Responses</p>
        <p><em>Ask questions about your educational documents and get intelligent answers</em></p>
    </div>
    
    <script>
    // Dynamic theme detection and CSS injection
    function updateThemeCSS() {
        const isDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
        const streamlitIsDark = document.querySelector('[data-testid="stAppViewContainer"]')?.style.backgroundColor === 'rgb(14, 17, 23)' || 
                               document.querySelector('.stApp')?.classList.contains('dark') ||
                               document.body.classList.contains('dark');
        
        const body = document.body;
        if (isDark || streamlitIsDark || body.style.backgroundColor === 'rgb(14, 17, 23)') {
            body.setAttribute('data-theme', 'dark');
        } else {
            body.setAttribute('data-theme', 'light');
        }
    }
    
    // Run immediately and on changes
    updateThemeCSS();
    if (window.matchMedia) {
        window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', updateThemeCSS);
    }
    
    // Also check periodically in case Streamlit changes theme
    setInterval(updateThemeCSS, 1000);
    </script>
    """, unsafe_allow_html=True)

def display_system_status(components):
    """Display system status in sidebar"""
    with st.sidebar:
        st.header("📊 System Status")
        
        if components['status'] == 'ready':
            st.markdown('<p class="status-success">🟢 System Ready</p>', unsafe_allow_html=True)
            
            # Vector DB Status
            try:
                db_info = components['vector_db'].get_database_info()
                st.markdown(f"""
                <div class="sidebar-info">
                    <strong>📚 Vector Database</strong><br>
                    Documents: {db_info.get('total_documents', 0)}<br>
                    Chunks: {db_info.get('total_chunks', 0)}<br>
                    Status: Ready
                </div>
                """, unsafe_allow_html=True)
            except:
                st.warning("⚠️ Vector DB info unavailable")
            
            # LLM Status
            api_key = os.getenv('OPENROUTER_API_KEY')
            if api_key and api_key != 'your_openrouter_api_key_here':
                st.markdown('<p class="status-success">🟢 OpenRouter API Connected</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p class="status-warning">🟡 Using Fallback Responses</p>', unsafe_allow_html=True)
                st.info("💡 Add OPENROUTER_API_KEY to .env for enhanced AI responses")
        
        elif components['status'] == 'error':
            st.markdown('<p class="status-error">🔴 System Error</p>', unsafe_allow_html=True)
            st.error(components.get('error', 'Unknown error'))
        
        else:
            st.markdown('<p class="status-warning">🟡 Initializing...</p>', unsafe_allow_html=True)

def display_session_stats():
    """Display session statistics"""
    with st.sidebar:
        st.header("📈 Session Stats")
        
        stats = st.session_state.stats
        session_duration = (datetime.now() - stats['session_start']).total_seconds() / 60
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Queries", stats['total_queries'])
            st.metric("Success Rate", f"{(stats['successful_responses'] / max(stats['total_queries'], 1) * 100):.1f}%")
        
        with col2:
            st.metric("Avg Response", f"{stats['avg_response_time']:.2f}s")
            st.metric("Session Time", f"{session_duration:.1f}m")

def display_conversation_controls():
    """Display conversation control buttons"""
    with st.sidebar:
        st.header("🎛️ Controls")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Clear Chat", help="Clear conversation history"):
                st.session_state.messages = []
                if st.session_state.chatbot:
                    st.session_state.chatbot.reset_conversation()
                st.rerun()
        
        with col2:
            if st.button("📊 Reset Stats", help="Reset session statistics"):
                st.session_state.stats = {
                    'total_queries': 0,
                    'successful_responses': 0,
                    'avg_response_time': 0,
                    'session_start': datetime.now()
                }
                st.rerun()

def display_chat_message(message):
    """Display a chat message with proper styling"""
    role = message["role"]
    content = message["content"]
    
    if role == "user":
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>👤 You:</strong><br>
            {content}
        </div>
        """, unsafe_allow_html=True)
    
    else:  # AI message
        # Parse AI response data
        if isinstance(content, dict):
            response_text = content.get('response', str(content))
            sources = content.get('sources', [])
            chunks_used = content.get('chunks_used', 0)
            confidence = content.get('confidence', 0)
            response_time = content.get('response_time', 0)
        else:
            response_text = str(content)
            sources = []
            chunks_used = 0
            confidence = 0
            response_time = 0
        
        # Clean response text from any HTML tags before display
        import re
        clean_response_text = re.sub(r'<[^>]+>', '', response_text)  # Remove HTML tags
        clean_response_text = re.sub(r'</?\w+[^>]*>', '', clean_response_text)  # Extra cleaning
        clean_response_text = clean_response_text.replace('&nbsp;', ' ').replace('&amp;', '&')
        clean_response_text = ' '.join(clean_response_text.split())  # Normalize whitespace
        
        # Display main response
        st.markdown(f"""
        <div class="chat-message ai-message">
            <strong>🤖 AI Assistant:</strong><br>
            {clean_response_text}
        </div>
        """, unsafe_allow_html=True)
        
        # Display source information
        if sources or chunks_used > 0:
            with st.expander("📚 Source Information", expanded=False):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Chunks Used", chunks_used)
                with col2:
                    st.metric("Confidence", f"{confidence:.2f}")
                with col3:
                    st.metric("Response Time", f"{response_time:.2f}s")
                
                if sources:
                    st.subheader("📄 Source Documents")
                    for i, source in enumerate(sources, 1):
                        st.write(f"**{i}. {source.get('filename', 'Unknown')}**")
                        st.write(f"   - Relevance: {source.get('relevance_score', 0):.3f}")
                        st.write(f"   - Method: {source.get('extraction_method', 'Unknown')}")

def process_user_query(user_input: str, components: Dict):
    """Process user query and return response"""
    try:
        start_time = time.time()
        
        # Update stats
        st.session_state.stats['total_queries'] += 1
        
        # Get response from chatbot
        with st.spinner("🔍 Analyzing documents and generating response..."):
            if components['chatbot']:
                # Get chunk-level response
                chatbot_response = components['chatbot'].process_query(user_input)
                
                # Enhanced response with LLM if available
                if components['llm_service'] and chatbot_response.get('chunks_used', 0) > 0:
                    # Extract context from chunks for LLM
                    context = ""
                    if 'sources' in chatbot_response:
                        # First, try to get actual chunk content from the chatbot
                        # We need to get the actual chunks with text content
                        try:
                            # Get the actual chunks with content from the chatbot's last retrieval
                            vector_db = components.get('vector_db')
                            if vector_db:
                                # Re-search to get chunks with content
                                search_results = vector_db.search_similar_chunks(user_input, top_k=3)
                                if search_results:
                                    context_parts = []
                                    for i, result in enumerate(search_results[:3]):
                                        filename = result.get('metadata', {}).get('filename', 'Unknown')
                                        text_content = result.get('text', '')
                                        if text_content.strip():
                                            # Limit text to first 600 characters for context
                                            context_parts.append(f"[Source: {filename}]\n{text_content[:600]}")
                                    context = "\n\n".join(context_parts)
                        except Exception as e:
                            # No fallback - use empty context
                            context = ""
                    
                    # Only use LLM if we have actual context from documents
                    if context:
                        # Get enhanced response from LLM
                        conversation_history = [
                            msg for msg in st.session_state.messages[-4:] 
                            if msg["role"] == "user"
                        ]
                        
                        try:
                            llm_result = components['llm_service'].generate_response(
                                user_input, 
                                context, 
                                conversation_history
                            )
                            
                            # Only replace response if LLM actually provides a meaningful response
                            if llm_result and llm_result.get('response') and len(llm_result.get('response', '').strip()) > 10:
                                chatbot_response['response'] = llm_result['response']
                                chatbot_response['llm_enhanced'] = True
                                chatbot_response['model_used'] = llm_result.get('model_used', 'unknown')
                            else:
                                # Keep original chatbot response if LLM fails
                                chatbot_response['llm_enhanced'] = False
                                chatbot_response['model_used'] = 'chatbot_fallback'
                                
                        except Exception as e:
                            # If LLM fails, keep original chatbot response
                            chatbot_response['llm_enhanced'] = False
                            chatbot_response['model_used'] = 'chatbot_fallback'
                            chatbot_response['llm_error'] = str(e)
                
                # Update response time
                total_time = time.time() - start_time
                chatbot_response['response_time'] = total_time
                
                # Update session stats
                if chatbot_response.get('chunks_used', 0) > 0:
                    st.session_state.stats['successful_responses'] += 1
                
                # Update average response time
                current_avg = st.session_state.stats['avg_response_time']
                query_count = st.session_state.stats['total_queries']
                new_avg = (current_avg * (query_count - 1) + total_time) / query_count
                st.session_state.stats['avg_response_time'] = new_avg
                
                # Final HTML cleaning of the response to ensure no HTML tags leak through
                import re
                if 'response' in chatbot_response:
                    response_text = chatbot_response['response']
                    # Comprehensive HTML cleaning
                    response_text = re.sub(r'<[^>]+>', '', response_text)  # Remove HTML tags
                    response_text = re.sub(r'</?\w+[^>]*>', '', response_text)  # Extra cleaning
                    response_text = response_text.replace('&nbsp;', ' ').replace('&amp;', '&')  # HTML entities
                    response_text = response_text.replace('\n', ' ').replace('\r', ' ')  # Line breaks
                    response_text = ' '.join(response_text.split())  # Normalize whitespace
                    chatbot_response['response'] = response_text.strip()
                
                return chatbot_response
            
            else:
                return {
                    'response': "System not properly initialized. Please refresh the page.",
                    'sources': [],
                    'chunks_used': 0,
                    'confidence': 0,
                    'response_time': time.time() - start_time
                }
                
    except Exception as e:
        logging.error(f"Query processing error: {str(e)}")
        return {
            'response': f"I encountered an error processing your query: {str(e)}",
            'sources': [],
            'chunks_used': 0,
            'confidence': 0,
            'response_time': time.time() - start_time,
            'error': str(e)
        }

def display_example_queries():
    """Display example queries for user guidance"""
    with st.sidebar:
        st.header("💡 Example Questions")
        
        examples = [
            "What is formative assessment?",
            "How does summative assessment work?",
            "What are different types of evaluation?",
            "Tell me about assessment strategies",
            "How to implement effective testing?"
        ]
        
        st.markdown("Click any example to try it:")
        
        for example in examples:
            if st.button(f"💬 {example}", key=f"example_{hash(example)}"):
                st.session_state.example_query = example
                st.rerun()

def main():
    """Main Streamlit application"""
    try:
        # Load custom CSS
        load_custom_css()
        
        # Initialize session state
        initialize_session_state()
        
        # Load system components
        components = load_system_components()
        
        # Store components in session state
        st.session_state.vector_db = components.get('vector_db')
        st.session_state.chatbot = components.get('chatbot')
        st.session_state.llm_service = components.get('llm_service')
        st.session_state.system_status = components.get('status')
        
        # Display header
        display_header()
        
        # Create main layout
        col1, col2 = st.columns([3, 1])
        
        with col2:
            # Sidebar content
            display_system_status(components)
            display_session_stats()
            display_conversation_controls()
            display_example_queries()
        
        with col1:
            # Main chat interface
            st.header("💬 Chat Interface")
            
            if components['status'] != 'ready':
                st.warning("⚠️ System not ready. Please wait for initialization to complete.")
                st.stop()
            
            # Display conversation history
            for message in st.session_state.messages:
                display_chat_message(message)
            
            # Handle example query
            if hasattr(st.session_state, 'example_query'):
                user_input = st.session_state.example_query
                del st.session_state.example_query
            else:
                # Chat input
                user_input = st.chat_input("Ask a question about your documents...")
            
            # Process user input
            if user_input:
                # Add user message to history
                st.session_state.messages.append({"role": "user", "content": user_input})
                
                # Display user message immediately
                display_chat_message({"role": "user", "content": user_input})
                
                # Process query and get response
                response = process_user_query(user_input, components)
                
                # Add AI response to history
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                # Display AI response
                display_chat_message({"role": "assistant", "content": response})
                
                # Rerun to update the interface
                st.rerun()
            
            # Display helpful information for new users
            if not st.session_state.messages:
                st.info("""
                👋 **Welcome to Edify AI Chatbot!**
                
                This intelligent assistant can answer questions about your educational documents using advanced AI.
                
                **Features:**
                - 🔍 Document search with chunk-level precision
                - 🤖 AI-powered responses with source attribution
                - 📚 Citation and confidence scoring
                - 💬 Conversation memory and context
                
                **Tips:**
                - Ask specific questions about assessment, evaluation, or educational methods
                - Try the example questions in the sidebar
                - Use natural language - no special formatting needed
                
                **Get started** by typing a question below or clicking an example! 👇
                """)
    
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.error("Please refresh the page and try again.")
        logging.error(f"Streamlit app error: {str(e)}")
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    main()
