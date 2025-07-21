#!/usr/bin/env python3
"""Quick test of reasoning functionality"""

import os
import sys
from dotenv import load_dotenv

# Add current directory to path
sys.path.append('/workspaces/chatbot')

# Load environment variables
load_dotenv()

from llm_service import LLMService

def test_reasoning():
    print("🔄 Testing LLM Service Reasoning...")
    
    # Initialize LLM service
    config = {
        'openrouter_api_key': os.getenv('OPENROUTER_API_KEY'),
        'gemini_api_key': os.getenv('GEMINI_API_KEY')
    }
    
    llm = LLMService(config)
    
    # Test query
    query = "What is formative assessment?"
    context = "Formative assessment is assessment for learning that provides information to plan the next stage in learning. It helps teachers understand how students are progressing."
    
    print(f"Query: {query}")
    print(f"Context: {context[:100]}...")
    print("\n" + "="*50)
    
    # Generate response
    result = llm.generate_response(query, context)
    
    print("📋 Result Structure:")
    print(f"Type: {type(result)}")
    print(f"Keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
    print("\n" + "="*50)
    
    if isinstance(result, dict):
        print("✅ Response:")
        print(result.get('response', 'No response found'))
        print("\n🧠 Reasoning:")
        print(result.get('reasoning', 'No reasoning found'))
        print(f"\n🤖 Model: {result.get('model_used', 'Unknown')}")
        print(f"📊 Quality: {result.get('reasoning_quality', 'Unknown')}")
    else:
        print("❌ Result is not a dictionary!")
        print(f"Result: {result}")

if __name__ == "__main__":
    test_reasoning()
