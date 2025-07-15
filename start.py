"""
Quick Start Script for Phase 4 Deployment
Run this to start the complete Edify AI Chatbot system
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def main():
    """Quick start for Edify AI Chatbot"""
    try:
        print("=" * 60)
        print("🚀 EDIFY AI CHATBOT - PHASE 4 DEPLOYMENT")
        print("=" * 60)
        print("Starting professional AI chatbot with Streamlit interface...")
        print()
        
        # Check if we're in the right directory
        required_files = [
            'app.py',
            'llm_service.py',
            'chatbot.py',
            'vector_db.py',
            'vector_store'
        ]
        
        missing_files = []
        for file in required_files:
            if not os.path.exists(file):
                missing_files.append(file)
        
        if missing_files:
            print("❌ Missing required files:")
            for file in missing_files:
                print(f"   - {file}")
            print("\nPlease ensure all files are in the current directory.")
            return 1
        
        print("✅ All required files found")
        print()
        
        # Check environment variables
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.getenv('OPENROUTER_API_KEY')
        if not api_key or api_key == 'your_openrouter_api_key_here':
            print("⚠️  OPENROUTER_API_KEY not configured")
            print("   The system will use fallback responses")
            print("   For enhanced AI responses, add your API key to .env file")
        else:
            print("✅ OpenRouter API key configured")
        
        print()
        
        # Start Streamlit app
        print("🌐 Starting Streamlit web application...")
        print("📂 App will be available at: http://localhost:8501")
        print()
        print("🔥 FEATURES READY:")
        print("   - Professional web interface")
        print("   - AI-powered responses with OpenRouter integration")
        print("   - Document search with chunk-level precision")
        print("   - Source attribution and confidence scoring")
        print("   - Conversation memory and session analytics")
        print()
        print("Press Ctrl+C to stop the application")
        print("-" * 60)
        
        # Run Streamlit
        try:
            subprocess.run([
                sys.executable, "-m", "streamlit", "run", 
                "app.py",
                "--server.port=8501",
                "--server.address=localhost",
                "--server.headless=false"
            ], check=True)
        except KeyboardInterrupt:
            print("\n\n🛑 Application stopped by user")
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Failed to start Streamlit: {e}")
            print("Make sure Streamlit is installed: pip install streamlit")
            return 1
        
        print("\n👋 Thank you for using Edify AI Chatbot!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Startup failed: {str(e)}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
