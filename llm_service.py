"""
LLM Service with OpenRouter Integration
Provides high-quality responses with reasoning and fallback support
"""

import os
import json
import logging
import requests
from typing import Dict, List, Optional
from datetime import datetime
import time

class LLMService:
    def __init__(self, config: Dict):
        """Initialize LLM service with OpenRouter"""
        self.config = config
        self.api_key = os.getenv('OPENROUTER_API_KEY')
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        
        # Best free models with high request limits (updated order)
        self.models = {
            'primary': 'google/gemma-2-9b-it:free',  # Fast and reliable
            'fallback': 'microsoft/phi-3-mini-4k-instruct:free',  # Good fallback
            'fast': 'qwen/qwen-2-7b-instruct:free'  # Alternative option
        }
        
        self.logger = logging.getLogger(__name__)
        
        if not self.api_key or self.api_key == 'your_openrouter_api_key_here':
            self.logger.warning("⚠️ OPENROUTER_API_KEY not configured. Using fallback responses.")
            self.api_available = False
        else:
            self.api_available = True
            self.logger.info("✅ OpenRouter API configured successfully")
    
    def generate_response(self, query: str, context: str, conversation_history: List = None) -> Dict:
        """Generate enhanced response with reasoning"""
        try:
            start_time = time.time()
            
            if self.api_available:
                # Try OpenRouter API
                response = self._call_openrouter_api(query, context, conversation_history)
                if response:
                    return {
                        'response': response,
                        'model_used': 'openrouter',
                        'reasoning_quality': 'high',
                        'response_time': time.time() - start_time,
                        'timestamp': datetime.now().isoformat()
                    }
            
            # Fallback to local generation
            response = self._generate_fallback_response(query, context)
            
            return {
                'response': response,
                'model_used': 'fallback',
                'reasoning_quality': 'basic',
                'response_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"❌ LLM generation failed: {str(e)}")
            return {
                'response': self._generate_error_response(query, str(e)),
                'model_used': 'error_fallback',
                'error': str(e),
                'response_time': time.time() - start_time
            }
    
    def _call_openrouter_api(self, query: str, context: str, conversation_history: List = None) -> Optional[str]:
        """Call OpenRouter API with enhanced prompts"""
        try:
            # Create enhanced system prompt
            system_prompt = self._create_system_prompt()
            
            # Create user prompt with context
            user_prompt = self._create_user_prompt(query, context, conversation_history)
            
            # Try primary model first
            for model_type in ['primary', 'fallback', 'fast']:
                try:
                    model_name = self.models[model_type]
                    
                    headers = {
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://edify-chatbot.local",  # Optional
                        "X-Title": "Edify Educational Chatbot"  # Optional
                    }
                    
                    payload = {
                        "model": model_name,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        "max_tokens": 1000,
                        "temperature": 0.7,
                        "top_p": 0.9
                    }
                    
                    response = requests.post(
                        self.base_url,
                        headers=headers,
                        json=payload,
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        if 'choices' in result and len(result['choices']) > 0:
                            content = result['choices'][0]['message']['content']
                            self.logger.info(f"✅ OpenRouter response from {model_name}: {content[:100]}...")
                            return content
                        else:
                            self.logger.warning(f"⚠️ No choices in OpenRouter response from {model_name}: {result}")
                    else:
                        self.logger.warning(f"⚠️ OpenRouter API error {response.status_code} with {model_name}: {response.text}")
                        if model_type == 'fast':  # Last attempt
                            self.logger.error(f"❌ All OpenRouter models failed")
                            
                except requests.exceptions.RequestException as e:
                    self.logger.warning(f"⚠️ Network error with {model_type}: {str(e)}")
                    continue
                except Exception as e:
                    self.logger.warning(f"⚠️ Error with {model_type}: {str(e)}")
                    continue
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ OpenRouter API call failed: {str(e)}")
            return None
    
    def _create_system_prompt(self) -> str:
        """Create enhanced system prompt for educational content"""
        return """You are a friendly AI assistant that gives short, conversational answers about educational topics.

Keep responses:
- Very short (2-3 sentences max)
- Conversational and natural
- Based only on the provided context
- Like you're explaining from memory

Don't use headers, bullet points, or formal structure. Just answer naturally."""

    def _create_user_prompt(self, query: str, context: str, conversation_history: List = None) -> str:
        """Create short, conversational prompt"""
        prompt_parts = []
        
        # Add conversation context if available
        if conversation_history:
            recent_queries = [item.get('query', '') for item in conversation_history[-2:]]
            if recent_queries:
                prompt_parts.append(f"Previous conversation: {' | '.join(recent_queries)}")
        
        # Add main context
        prompt_parts.extend([
            "CONTEXT:",
            context,
            "",
            f"QUESTION: {query}",
            "",
            "Give a short, conversational answer (2-3 sentences max) based on the context. Sound natural like you're explaining from memory."
        ])
        
        return "\n".join(prompt_parts)
    
    def _generate_fallback_response(self, query: str, context: str) -> str:
        """Generate short fallback response when API is unavailable"""
        try:
            # Extract key information from context
            context_lines = context.split('\n')
            sources = []
            content_parts = []
            
            current_source = "Unknown"
            
            for line in context_lines:
                line = line.strip()
                if line.startswith('[Source:'):
                    # Extract new source
                    current_source = line.replace('[Source:', '').replace(']', '').strip()
                    if current_source not in sources:
                        sources.append(current_source)
                elif line and not line.startswith('---') and len(line) > 20:
                    # Take meaningful content
                    content_parts.append(line[:150])  # First 150 chars
                    if len(content_parts) >= 2:  # Only need 2 content parts
                        break
            
            # Generate short, conversational response
            if content_parts:
                response = f"Based on the documents, {content_parts[0].lower()}"
                if len(content_parts) > 1:
                    response += f" {content_parts[1]}"
                return response
            else:
                return "I found some relevant information in the documents but couldn't extract a clear answer to your question."
            
        except Exception as e:
            self.logger.error(f"❌ Fallback response generation failed: {str(e)}")
            return "I found some information but couldn't process it properly."
    
    def _generate_error_response(self, query: str, error: str) -> str:
        """Generate response when all methods fail"""
        return f"""I apologize, but I encountered an error while processing your query: "{query}"

Error details: {error}

Please try:
1. Rephrasing your question
2. Using different keywords
3. Asking about a more specific topic
4. Checking if the documents contain information about this topic

If the problem persists, there may be a technical issue that needs attention."""

def main():
    """Test LLM Service"""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        config = {}
        llm = LLMService(config)
        
        test_query = "What is formative assessment?"
        test_context = """[Source: assessment_guide.pdf, Section 1]
Formative assessment is an ongoing process that provides feedback to both students and teachers during the learning process. It helps identify areas for improvement and guides instructional decisions.

[Source: educational_methods.pdf, Section 3]
Unlike summative assessment, formative assessment occurs during learning and is designed to improve student understanding rather than evaluate final performance."""
        
        print("Testing LLM Service...")
        result = llm.generate_response(test_query, test_context)
        
        print(f"Model Used: {result['model_used']}")
        print(f"Response Time: {result['response_time']:.2f}s")
        print(f"Response:\n{result['response']}")
        
    except Exception as e:
        print(f"Test failed: {str(e)}")

if __name__ == "__main__":
    main()
