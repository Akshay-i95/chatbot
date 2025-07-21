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

# Force reload environment variables for Streamlit
def reload_env():
    """Force reload environment variables"""
    load_dotenv(override=True)
    return True

# Call this to ensure fresh environment loading
reload_env()

# Custom CSS for modern Gemini-like styling
def load_custom_css():
    st.markdown("""
    <style>
    /* Main container styling */
    .main-header {
        background: linear-gradient(135deg, #4285f4 0%, #34a853 50%, #ea4335 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    
    /* Modern chat message styling */
    .chat-message {
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
    }
    
    .chat-message:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.12);
    }
    
    /* User message - Gemini-like blue */
    .user-message {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-left: 4px solid #2196f3;
        color: #1565c0;
    }
    
    /* AI message - Gemini-like clean white */
    .ai-message {
        background: linear-gradient(135deg, #fafafa 0%, #f5f5f5 100%);
        border-left: 4px solid #4caf50;
        color: #2e7d32;
    }
    
    /* Dark mode AI message */
    [data-theme="dark"] .ai-message {
        background: linear-gradient(135deg, #1e1e1e 0%, #2d2d2d 100%);
        color: #81c784 !important;
    }
    
    [data-theme="dark"] .user-message {
        background: linear-gradient(135deg, #1a237e 0%, #283593 100%);
        color: #90caf9 !important;
    }
    
    /* Reasoning section - Gemini-inspired */
    .reasoning-section {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border: 1px solid #dee2e6;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #ff9800;
    }
    
    [data-theme="dark"] .reasoning-section {
        background: linear-gradient(135deg, #2d3748 0%, #1a202c 100%);
        border-color: #4a5568;
        color: #e2e8f0 !important;
    }
    
    /* Source information styling */
    .source-container {
        background: linear-gradient(135deg, #f0f4f8 0%, #e2e8f0 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #6366f1;
    }
    
    [data-theme="dark"] .source-container {
        background: linear-gradient(135deg, #374151 0%, #1f2937 100%);
        color: #f3f4f6 !important;
    }
    
    /* Modern metrics */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        border: 1px solid #e2e8f0;
        text-align: center;
    }
    
    [data-theme="dark"] .metric-card {
        background: linear-gradient(135deg, #2d3748 0%, #1a202c 100%);
        border-color: #4a5568;
        color: #e2e8f0 !important;
    }
    
    /* Download button styling */
    .download-button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        cursor: pointer;
        transition: all 0.3s ease;
        font-weight: 500;
    }
    
    .download-button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
    }
    
    /* Quality indicators */
    .quality-high { color: #059669; font-weight: 600; }
    .quality-medium { color: #d97706; font-weight: 600; }
    .quality-low { color: #dc2626; font-weight: 600; }
    
    /* Sidebar styling */
    .sidebar-section {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 4px solid #8b5cf6;
    }
    
    [data-theme="dark"] .sidebar-section {
        background: linear-gradient(135deg, #2d3748 0%, #1a202c 100%);
        color: #e2e8f0 !important;
    }
    
    /* Animation for thinking */
    .thinking-animation {
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 0.6; }
        50% { opacity: 1; }
        100% { opacity: 0.6; }
    }
    
    /* Modern scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%);
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
            'total_response_time': 0,
            'avg_response_time': 0,
            'session_start': datetime.now()
        }

@st.cache_resource
def load_system_components():
    """Load and cache system components"""
    try:
        st.info("🔄 Loading system components...")
        
        # Configuration - Read from environment variables
        config = {
            'vector_db_type': os.getenv('VECTOR_DB_TYPE', 'faiss'),
            'vector_db_path': os.getenv('VECTOR_DB_PATH', './vector_store_faiss'),
            'collection_name': os.getenv('COLLECTION_NAME', 'pdf_chunks'),
            'embedding_model': os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2'),
            'max_context_chunks': int(os.getenv('MAX_CONTEXT_CHUNKS', '2')),
            'min_similarity_threshold': float(os.getenv('MIN_SIMILARITY_THRESHOLD', '0.65')),
            'enable_citations': os.getenv('ENABLE_CITATIONS', 'true').lower() == 'true',
            'enable_context_expansion': os.getenv('ENABLE_CONTEXT_EXPANSION', 'false').lower() == 'true',
            'max_context_length': int(os.getenv('MAX_CONTEXT_LENGTH', '2000')),
            # Azure configuration for PDF downloads
            'azure_connection_string': os.getenv('AZURE_STORAGE_CONNECTION_STRING'),
            'azure_account_name': os.getenv('AZURE_STORAGE_ACCOUNT_NAME'),
            'azure_account_key': os.getenv('AZURE_STORAGE_ACCOUNT_KEY'),
            'azure_container_name': os.getenv('AZURE_STORAGE_CONTAINER_NAME'),
            'azure_folder_path': os.getenv('AZURE_BLOB_FOLDER_PATH')
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
    """Display modern Gemini-inspired application header"""
    st.markdown("""
    <div class="main-header">
        <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 1rem;">
            <div style="width: 48px; height: 48px; border-radius: 50%; background: linear-gradient(135deg, #ffffff 0%, #f0f0f0 100%); display: flex; align-items: center; justify-content: center; margin-right: 1rem; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                <span style="font-size: 24px;">🤖</span>
            </div>
            <h1 style="margin: 0; font-size: 2.5rem; font-weight: 300; background: linear-gradient(135deg, #ffffff 0%, #e8f4fd 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                Edify AI
            </h1>
        </div>
        <p style="font-size: 1.2rem; font-weight: 300; margin-bottom: 0.5rem; opacity: 0.95;">
            Advanced Educational Assistant
        </p>
        <p style="font-size: 1rem; opacity: 0.8; margin: 0;">
            💡 Ask questions about your documents • 🧠 Get intelligent reasoning • 📚 Access source materials
        </p>
    </div>
    """, unsafe_allow_html=True)

def display_system_status(components):
    """Display modern system status in sidebar"""
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-section">
            <h3 style="margin-bottom: 1rem; color: #374151;">⚡ System Status</h3>
        </div>
        """, unsafe_allow_html=True)
        
        if components['status'] == 'ready':
            st.success("🟢 **System Ready**")
            
            # Vector DB Status
            try:
                db_info = components['vector_db'].get_database_info()
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("📚 Documents", db_info.get('total_documents', 0))
                with col2:
                    st.metric("🧩 Chunks", db_info.get('total_chunks', 0))
            except:
                st.warning("⚠️ Vector DB info unavailable")
            
            # AI Service Status
            api_key = os.getenv('OPENROUTER_API_KEY')
            gemini_key = os.getenv('GEMINI_API_KEY')
            
            if (api_key and api_key != 'your_openrouter_api_key_here') or (gemini_key and gemini_key != 'your_gemini_api_key_here'):
                st.success("🧠 **Advanced AI Active**")
                if gemini_key and gemini_key != 'your_gemini_api_key_here':
                    st.caption("🌟 Gemini AI Primary")
                if api_key and api_key != 'your_openrouter_api_key_here':
                    st.caption("🌐 OpenRouter Fallback")
            else:
                st.info("🔧 **Local Processing Mode**")
                st.caption("Add API keys for enhanced AI")
        
        elif components['status'] == 'error':
            st.error("🔴 **System Error**")
            st.caption(components.get('error', 'Unknown error'))
        
        else:
            st.warning("🟡 **Initializing...**")
            st.caption("Please wait")

def display_session_stats():
    """Display modern session statistics"""
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-section">
            <h3 style="margin-bottom: 1rem; color: #374151;">� Session Stats</h3>
        </div>
        """, unsafe_allow_html=True)
        
        stats = st.session_state.stats
        session_duration = (datetime.now() - stats['session_start']).total_seconds() / 60
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("💬 Queries", stats['total_queries'])
            st.metric("⏱️ Duration", f"{session_duration:.1f}m")
        with col2:
            st.metric("✅ Success", stats['successful_responses'])
            st.metric("⚡ Avg Time", f"{stats['avg_response_time']:.1f}s")

def display_conversation_controls():
    """Display conversation control buttons"""
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-section">
            <h3 style="margin-bottom: 1rem; color: #374151;">🎛️ Controls</h3>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Clear Chat", help="Clear conversation history"):
                st.session_state.messages = []
                if st.session_state.chatbot:
                    st.session_state.chatbot.reset_conversation()
                st.rerun()
        
        with col2:
            if st.button("📊 Reset", help="Reset session statistics", use_container_width=True):
                st.session_state.stats = {
                    'total_queries': 0,
                    'successful_responses': 0,
                    'avg_response_time': 0,
                    'session_start': datetime.now()
                }
                st.rerun()
        
        # Full width reload button
        if st.button("�🔄 Reload System", help="Clear cache and reload components", use_container_width=True):
            # Clear all caches
            st.cache_resource.clear()
            
            # Force reload environment
            reload_env()
            
            # Reset session state
            for key in ['vector_db', 'chatbot', 'llm_service', 'system_status']:
                if key in st.session_state:
                    del st.session_state[key]
            
            st.success("System reloaded! Azure service should now work.")
            st.rerun()

def display_azure_service_status(components):
    """Display Azure download service status (removed from sidebar, now only available in source info)"""
    # Azure service status is now only shown in chat responses through source information
    # This keeps the sidebar clean and modern like Gemini
    pass

def display_chat_message(message):
    """Display a chat message with modern Gemini-like styling"""
    role = message["role"]
    content = message["content"]
    
    if role == "user":
        st.markdown(f"""
        <div class="chat-message user-message">
            <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                <div style="width: 32px; height: 32px; border-radius: 50%; background: linear-gradient(135deg, #2196f3 0%, #1976d2 100%); display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; margin-right: 0.75rem;">
                    👤
                </div>
                <strong style="color: #1565c0;">You</strong>
            </div>
            <div style="margin-left: 2.5rem; font-size: 1rem; line-height: 1.6;">
                {content}
            </div>
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
            reasoning = content.get('reasoning', '')
            model_used = content.get('model_used', 'unknown')
            reasoning_quality = content.get('reasoning_quality', 'medium')
        else:
            response_text = str(content)
            sources = []
            chunks_used = 0
            confidence = 0
            response_time = 0
            reasoning = ''
            model_used = 'unknown'
            reasoning_quality = 'medium'
        
        # Clean response text
        import re
        clean_response_text = re.sub(r'<[^>]+>', '', response_text)
        clean_response_text = clean_response_text.replace('&nbsp;', ' ').replace('&amp;', '&')
        clean_response_text = ' '.join(clean_response_text.split())
        
        # Display main AI response
        st.markdown(f"""
        <div class="chat-message ai-message">
            <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                <div style="width: 32px; height: 32px; border-radius: 50%; background: linear-gradient(135deg, #4caf50 0%, #388e3c 100%); display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; margin-right: 0.75rem;">
                    🤖
                </div>
                <strong style="color: #2e7d32;">Edify AI</strong>
                <div style="margin-left: auto; font-size: 0.8rem; color: #666;">
                    <span class="quality-{reasoning_quality}">{model_used}</span>
                </div>
            </div>
            <div style="margin-left: 2.5rem; font-size: 1rem; line-height: 1.7;">
                {clean_response_text}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Display advanced reasoning if available
        if reasoning:
            with st.expander("🧠 **Advanced AI Reasoning Process**", expanded=False):
                quality_icons = {
                    'high': '🟢 High Quality',
                    'medium': '🟡 Medium Quality', 
                    'low': '🔴 Basic Quality'
                }
                quality_display = quality_icons.get(reasoning_quality, '🟡 Medium Quality')
                
                st.markdown(f"""
                <div class="reasoning-section">
                    <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                        <div style="font-size: 1.1rem; font-weight: 600; color: #f57c00;">
                            ⚡ How I Analyzed Your Question
                        </div>
                        <div style="margin-left: auto; font-size: 0.85rem;">
                            {quality_display}
                        </div>
                    </div>
                    <div style="background: rgba(255,152,0,0.1); padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
                        <strong>🔍 Model:</strong> {model_used}<br>
                        <strong>⏱️ Processing:</strong> {response_time:.2f}s<br>
                        <strong>📊 Context:</strong> {chunks_used} relevant chunks
                    </div>
                    <div style="font-size: 0.95rem; line-height: 1.6; color: #424242;">
                        {reasoning}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Display source information with download functionality
        if sources or chunks_used > 0:
            with st.expander(f"📚 **Source Evidence** ({len(sources)} documents)", expanded=False):
                # Performance metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size: 1.2rem; font-weight: 600; color: #6366f1;">{chunks_used}</div>
                        <div style="font-size: 0.8rem; color: #64748b;">Chunks Used</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size: 1.2rem; font-weight: 600; color: #10b981;">{confidence:.2f}</div>
                        <div style="font-size: 0.8rem; color: #64748b;">Confidence</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size: 1.2rem; font-weight: 600; color: #f59e0b;">{response_time:.2f}s</div>
                        <div style="font-size: 0.8rem; color: #64748b;">Response Time</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size: 1.2rem; font-weight: 600; color: #ef4444;">{len(sources)}</div>
                        <div style="font-size: 0.8rem; color: #64748b;">Sources</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Source documents with download functionality
                if sources:
                    st.markdown("### 📄 Source Documents")
                    timestamp = int(time.time() * 1000)
                    
                    for i, source in enumerate(sources, 1):
                        st.markdown(f"""
                        <div class="source-container">
                            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.75rem;">
                                <div style="font-weight: 600; color: #374151; font-size: 1rem;">
                                    📋 {i}. {source.get('filename', 'Unknown Document')}
                                </div>
                                <div style="font-size: 0.85rem; color: #6b7280;">
                                    Relevance: {source.get('relevance_score', 0):.3f}
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Document details
                        col_info, col_download = st.columns([3, 1])
                        
                        with col_info:
                            st.markdown(f"""
                            <div style="font-size: 0.9rem; color: #6b7280; margin-left: 1rem;">
                                📊 Method: {source.get('extraction_method', 'Unknown')}<br>
                                {f"📏 Size: {source.get('file_size_mb', 0):.1f} MB<br>" if source.get('file_size_mb') else ''}
                                {f"📄 Pages: {source.get('total_pages', 0)}<br>" if source.get('total_pages') else ''}
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col_download:
                            # Download functionality - only show if available
                            if source.get('download_available') and source.get('download_url'):
                                st.success("✅ Available")
                                
                                unique_key = f"download_{timestamp}_{i}_{hash(source.get('filename', 'unknown'))}"
                                if st.button("📥", key=unique_key, help="Download PDF", use_container_width=True):
                                    st.success("🔗 **Download Generated!**")
                                    st.code(source['download_url'], language=None)
                                    st.warning("⚠️ **Expires in 2 hours**")
                                
                                # Direct download link
                                st.markdown(f"""
                                <a href="{source['download_url']}" target="_blank" style="text-decoration: none;">
                                    <div style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; padding: 0.4rem; border-radius: 6px; text-align: center; font-size: 0.8rem; font-weight: 500;">
                                        🔗 Download
                                    </div>
                                </a>
                                """, unsafe_allow_html=True)
                                
                            elif source.get('filename', 'Unknown') != 'Unknown':
                                st.warning("⚠️ Not Available")
                                st.caption("File not in storage")
                            else:
                                st.info("ℹ️ Reference Only")
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        if i < len(sources):
                            st.divider()

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
                                # Pass through reasoning and other LLM data
                                chatbot_response['reasoning'] = llm_result.get('reasoning', '')
                                chatbot_response['reasoning_quality'] = llm_result.get('reasoning_quality', 'medium')
                                chatbot_response['llm_response_time'] = llm_result.get('response_time', 0)
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
    """Display modern example queries for user guidance"""
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-section">
            <h3 style="margin-bottom: 1rem; color: #374151;">💡 Try These Questions</h3>
            <p style="font-size: 0.9rem; color: #6b7280; margin-bottom: 1rem;">
                Click any example to get started with our AI assistant
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        examples = [
            {
                "question": "What is formative assessment?",
                "icon": "📊",
                "category": "Assessment"
            },
            {
                "question": "How does summative assessment work?",
                "icon": "✅",
                "category": "Evaluation"
            },
            {
                "question": "What are different types of evaluation?",
                "icon": "📝",
                "category": "Methods"
            },
            {
                "question": "Tell me about assessment strategies",
                "icon": "🎯",
                "category": "Strategy"
            },
            {
                "question": "How to implement effective testing?",
                "icon": "⚡",
                "category": "Implementation"
            }
        ]
        
        for i, example in enumerate(examples):
            unique_key = f"example_{i}_{hash(example['question'])}"
            if st.button(
                f"{example['icon']} {example['question']}", 
                key=unique_key,
                help=f"Category: {example['category']}",
                use_container_width=True
            ):
                st.session_state.example_query = example['question']
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
            display_azure_service_status(components)
            display_example_queries()
            
            # Add cache clearing button for debugging
            st.sidebar.markdown("---")
            if st.sidebar.button("🔄 Clear Cache & Reload", help="Clear Streamlit cache and reload components"):
                st.cache_resource.clear()
                st.rerun()
        
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
