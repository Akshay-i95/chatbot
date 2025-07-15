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
        return """You are an expert educational AI assistant with advanced knowledge in assessment, learning, and educational technology.

Your Role:
- Provide accurate, evidence-based answers about educational topics
- Focus on practical, actionable insights for educators
- Maintain academic rigor while being accessible
- Draw connections between educational theory and practice

Guidelines:
1. **Accuracy First**: Base answers strictly on the provided context
2. **Educational Focus**: Emphasize learning outcomes and student benefits
3. **Practical Application**: Include implementation strategies when relevant
4. **Source Attribution**: Reference specific documents and sections
5. **Balanced Perspective**: Present multiple viewpoints when appropriate
6. **Clear Structure**: Use headings, bullet points, and examples

Response Format:
- Start with a direct answer to the question
- Provide detailed explanation with context
- Include practical examples or applications
- End with key takeaways or next steps
- Add source references throughout

Quality Standards:
- Comprehensive yet concise responses
- Professional educational terminology
- Evidence-based recommendations
- Clear, actionable guidance"""

    def _create_user_prompt(self, query: str, context: str, conversation_history: List = None) -> str:
        """Create enhanced user prompt with context"""
        prompt_parts = []
        
        # Add conversation context if available
        if conversation_history:
            recent_queries = [item.get('query', '') for item in conversation_history[-2:]]
            if recent_queries:
                prompt_parts.append(f"Previous conversation context: {' | '.join(recent_queries)}")
        
        # Add main context
        prompt_parts.extend([
            "DOCUMENT CONTEXT:",
            context,
            "",
            f"QUESTION: {query}",
            "",
            "Please provide a comprehensive answer based on the document context above. Include:",
            "1. Direct answer to the question",
            "2. Detailed explanation with examples",
            "3. Practical applications or implications",
            "4. References to specific sources",
            "5. Key takeaways for educators"
        ])
        
        return "\n".join(prompt_parts)
    
    def _generate_fallback_response(self, query: str, context: str) -> str:
        """Generate enhanced fallback response when API is unavailable"""
        try:
            # Extract key information from context
            context_lines = context.split('\n')
            sources = []
            content_sections = []
            
            current_source = "Unknown"
            current_content = []
            
            for line in context_lines:
                line = line.strip()
                if line.startswith('[Source:'):
                    # Save previous section
                    if current_content:
                        content_sections.append({
                            'source': current_source,
                            'content': ' '.join(current_content)
                        })
                        current_content = []
                    
                    # Extract new source
                    current_source = line.replace('[Source:', '').replace(']', '').strip()
                    if current_source not in sources:
                        sources.append(current_source)
                        
                elif line and not line.startswith('---'):
                    current_content.append(line)
            
            # Save last section
            if current_content:
                content_sections.append({
                    'source': current_source,
                    'content': ' '.join(current_content)
                })
            
            # Generate structured response
            response_parts = []
            
            # Header
            response_parts.append(f"## Response to: {query}\n")
            
            if content_sections:
                response_parts.append("Based on the available educational documents, here's what I found:\n")
                
                # Main content
                response_parts.append("### Key Information\n")
                
                for i, section in enumerate(content_sections[:3], 1):
                    content = section['content']
                    if len(content) > 200:
                        content = content[:200] + "..."
                    
                    response_parts.append(f"**{i}. From {section['source']}:**")
                    response_parts.append(f"{content}\n")
                
                # Summary
                if len(content_sections) > 1:
                    response_parts.append("### Summary")
                    response_parts.append("The documents provide multiple perspectives on this topic. ")
                    response_parts.append("For comprehensive understanding, review all referenced sources.\n")
                
                # Sources
                response_parts.append("### Sources Referenced")
                for i, source in enumerate(sources, 1):
                    response_parts.append(f"{i}. {source}")
                
            else:
                response_parts.append("I found some relevant information, but couldn't extract clear details to answer your specific question. Please try rephrasing your query or asking about a different aspect of the topic.")
            
            return '\n'.join(response_parts)
            
        except Exception as e:
            self.logger.error(f"❌ Fallback response generation failed: {str(e)}")
            return self._generate_error_response(query, str(e))
    
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
