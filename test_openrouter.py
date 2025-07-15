"""
Quick test for OpenRouter API to diagnose the issue
"""
import os
import requests
from dotenv import load_dotenv

load_dotenv()

def test_openrouter_api():
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        print("❌ No API key found")
        return
    
    print(f"✅ API Key loaded: {api_key[:20]}...")
    
    # Test API call
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://edify-chatbot.local",
        "X-Title": "Edify Educational Chatbot"
    }
    
    payload = {
        "model": "google/gemma-2-9b-it:free",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is formative assessment? Give a brief answer."}
        ],
        "max_tokens": 200,
        "temperature": 0.7
    }
    
    try:
        print("🔄 Testing OpenRouter API...")
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        print(f"📊 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                content = result['choices'][0]['message']['content']
                print(f"✅ API Response: {content[:100]}...")
                return True
            else:
                print(f"❌ No choices in response: {result}")
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Exception: {str(e)}")
    
    return False

if __name__ == "__main__":
    test_openrouter_api()
