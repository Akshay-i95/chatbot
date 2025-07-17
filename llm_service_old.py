"""
LLM Service with Gemini AI Integration
Provides high-quality responses with reasoning and fallback support
"""

import os
import json
import logging
import requests
import re
from typing import Dict, List, Optional
from datetime import datetime
import time

class LLMService:
    def __init__(self, config: Dict):
        """Initialize LLM service with Gemini AI"""
        self.config = config
        
        # Primary: Gemini API
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        self.gemini_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        
        # Fallback: OpenRouter API  
        self.openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
        self.openrouter_url = "https://openrouter.ai/api/v1/chat/completions"
        
        # Fallback models for OpenRouter
        self.openrouter_models = {
            'primary': 'meta-llama/llama-3.1-8b-instruct:free',
            'fallback': 'mistralai/mistral-7b-instruct:free'
        }
        
        self.logger = logging.getLogger(__name__)
        
        # Check API availability
        if self.gemini_api_key and self.gemini_api_key != 'your_gemini_api_key_here':
            self.gemini_available = True
            self.logger.info("✅ Gemini AI API configured successfully")
        else:
            self.gemini_available = False
            self.logger.warning("⚠️ GEMINI_API_KEY not configured")
        
        if self.openrouter_api_key and self.openrouter_api_key != 'your_openrouter_api_key_here':
            self.openrouter_available = True
            self.logger.info("✅ OpenRouter API configured as fallback")
        else:
            self.openrouter_available = False
            self.logger.warning("⚠️ OPENROUTER_API_KEY not configured")
        
        if not self.gemini_available and not self.openrouter_available:
            self.logger.warning("⚠️ No API keys configured. Using intelligent fallback responses.")
    
    def generate_response(self, query: str, context: str, conversation_history: List = None) -> Dict:
        """Generate enhanced response with reasoning"""
        try:
            start_time = time.time()
            
            self.logger.info(f"🔄 Generating response for query: {query[:50]}...")
            self.logger.info(f"📄 Context length: {len(context)} characters")
            
            # Try Gemini API first (Primary)
            if self.gemini_available:
                self.logger.info("🌟 Trying Gemini AI API...")
                response = self._call_gemini_api(query, context, conversation_history)
                if response:
                    self.logger.info(f"✅ Gemini AI successful: {response[:100]}...")
                    return {
                        'response': response,
                        'model_used': 'gemini-2.0-flash',
                        'reasoning_quality': 'high',
                        'response_time': time.time() - start_time,
                        'timestamp': datetime.now().isoformat()
                    }
                else:
                    self.logger.warning("⚠️ Gemini API returned no response, trying fallback...")
            
            # Try OpenRouter API (Fallback)
            if self.openrouter_available:
                self.logger.info("🌐 Trying OpenRouter API...")
                response = self._call_openrouter_api(query, context, conversation_history)
                if response:
                    self.logger.info(f"✅ OpenRouter API successful: {response[:100]}...")
                    return {
                        'response': response,
                        'model_used': 'openrouter',
                        'reasoning_quality': 'high',
                        'response_time': time.time() - start_time,
                        'timestamp': datetime.now().isoformat()
                    }
                else:
                    self.logger.warning("⚠️ OpenRouter API returned no response, falling back...")
            
            # Use intelligent fallback if both APIs fail
            self.logger.info("🔄 Using intelligent fallback response generation...")
            fallback_response = self._generate_intelligent_fallback_response(query, context)
            
            return {
                'response': fallback_response,
                'model_used': 'intelligent_fallback',
                'reasoning_quality': 'medium',
                'response_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"❌ Response generation error: {str(e)}")
            return {
                'response': f"I encountered an error generating the response. Please try rephrasing your question.",
                'model_used': 'error_handler',
                'reasoning_quality': 'low',
                'response_time': time.time() - start_time,
                'error': str(e)
            }
    
    def _call_gemini_api(self, query: str, context: str, conversation_history: List = None) -> Optional[str]:
        """Call Gemini AI API"""
        try:
            # Build conversation context
            system_prompt = """You are an intelligent educational assistant. Based on the provided context from educational documents, answer the user's question clearly and accurately. 

Instructions:
1. Use ONLY the information provided in the context
2. Give clear, concise, and helpful answers
3. If the question is about holidays, lists, or schedules, organize the information properly
4. Be conversational but professional
5. If context doesn't contain the answer, say so honestly

Context from educational documents:
"""
            
            # Prepare the prompt
            full_prompt = f"""{system_prompt}
{context}

User Question: {query}

Answer based on the context above:"""
            
            # Prepare request payload for Gemini
            payload = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": full_prompt
                            }
                        ]
                    }
                ]
            }
            
            headers = {
                'Content-Type': 'application/json',
                'X-goog-api-key': self.gemini_api_key
            }
            
            # Make the API call
            response = requests.post(
                self.gemini_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract text from Gemini response structure
                if 'candidates' in data and len(data['candidates']) > 0:
                    candidate = data['candidates'][0]
                    if 'content' in candidate and 'parts' in candidate['content']:
                        text_parts = candidate['content']['parts']
                        if len(text_parts) > 0 and 'text' in text_parts[0]:
                            generated_text = text_parts[0]['text'].strip()
                            
                            # Validate response quality
                            if self._is_valid_response(generated_text, query):
                                return generated_text
                            else:
                                self.logger.warning("⚠️ Gemini response failed validation")
                                return None
                
                self.logger.warning("⚠️ Unexpected Gemini response structure")
                return None
            
            else:
                error_msg = response.text
                self.logger.warning(f"⚠️ Gemini API error {response.status_code}: {error_msg}")
                return None
                
        except requests.exceptions.Timeout:
            self.logger.warning("⚠️ Gemini API timeout")
            return None
        except Exception as e:
            self.logger.warning(f"⚠️ Gemini API error: {str(e)}")
            return None
                    self.logger.info(f"✅ OpenRouter API successful: {response[:100]}...")
                    return {
                        'response': response,
                        'model_used': 'openrouter',
                        'reasoning_quality': 'high',
                        'response_time': time.time() - start_time,
                        'timestamp': datetime.now().isoformat()
                    }
                else:
                    self.logger.warning("⚠️ OpenRouter API returned no response, falling back...")
            else:
                self.logger.info("❌ OpenRouter API not available, using fallback...")
            
            # Fallback to local generation
            self.logger.info("🔄 Using fallback response generation...")
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
                            
                            # Validate response quality
                            if self._is_valid_response(content, query):
                                self.logger.info(f"✅ OpenRouter response from {model_name}: {content[:100]}...")
                                return content
                            else:
                                self.logger.warning(f"⚠️ Invalid response from {model_name}, trying next model...")
                                continue
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
        return """You are an educational AI assistant. You MUST answer based ONLY on the provided document context.

CRITICAL RULES:
- Use ONLY the information provided in the context below
- If the context contains relevant information, use it to answer
- Give short, conversational answers (2-3 sentences max)
- Never say "I need more context" - the context is already provided
- Never ask follow-up questions - just answer based on what's given
- Sound natural and helpful, like explaining from memory"""

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
            "DOCUMENT CONTEXT (USE THIS TO ANSWER):",
            context,
            "",
            f"USER QUESTION: {query}",
            "",
            "Answer the question using ONLY the context above. Be conversational and brief (2-3 sentences)."
        ])
        
        return "\n".join(prompt_parts)
    
    def _generate_fallback_response(self, query: str, context: str) -> str:
        """Generate intelligent fallback response when API is unavailable"""
        try:
            self.logger.info("🔄 Using enhanced fallback response generation")
            
            query_lower = query.lower()
            
            # Special handling for holiday-related queries
            if any(word in query_lower for word in ['holiday', 'holidays', 'leave', 'vacation']):
                return self._generate_holiday_response(context, query)
            
            # Special handling for assessment-related queries
            elif any(word in query_lower for word in ['assessment', 'evaluation', 'test', 'exam']):
                return self._generate_assessment_response(context, query)
            
            # General response generation
            else:
                return self._generate_general_response(context, query)
                
        except Exception as e:
            self.logger.error(f"❌ Fallback response generation failed: {str(e)}")
            return "I found some information but couldn't process it properly."
    
    def _generate_holiday_response(self, context: str, query: str) -> str:
        """Generate specific response for holiday queries"""
        try:
            # Extract holiday information from context
            context_lines = context.split('\n')
            holidays = []
            
            for line in context_lines:
                line = line.strip()
                # Look for holiday patterns
                if any(word in line.upper() for word in ['HOLIDAY', 'VARALAKSHMI', 'ACADEMIC YEAR', 'STAFF']):
                    clean_line = re.sub(r'<[^>]+>', '', line)
                    clean_line = clean_line.replace('&nbsp;', ' ').replace('&amp;', '&')
                    if len(clean_line) > 10:
                        holidays.append(clean_line)
            
            if holidays:
                if 'which day' in query.lower() or 'what day' in query.lower():
                    return f"Based on the school documents, {holidays[0].lower()}. The holiday calendar includes various observances for all teaching and non-teaching staff."
                else:
                    response = "According to the school's academic calendar, staff holidays include: "
                    response += holidays[0] if holidays else "various scheduled holidays throughout the year"
                    response += ". All teaching and non-teaching staff are entitled to these holidays."
                    return response
            
            return "Based on the school documents, there are scheduled holidays for all school staff as per the academic calendar."
            
        except Exception as e:
            return "The school has scheduled holidays for all staff members as outlined in the academic calendar."
    
    def _generate_assessment_response(self, context: str, query: str) -> str:
        """Generate specific response for assessment queries"""
        try:
            context_lines = context.split('\n')
            assessment_info = []
            
            for line in context_lines:
                line = line.strip()
                if any(word in line.lower() for word in ['assessment', 'formative', 'summative', 'evaluation']):
                    clean_line = re.sub(r'<[^>]+>', '', line)
                    clean_line = clean_line.replace('&nbsp;', ' ').replace('&amp;', '&')
                    if len(clean_line) > 20:
                        assessment_info.append(clean_line)
            
            if assessment_info:
                # Take the most relevant content
                content = assessment_info[0][:300]
                if 'formative' in query.lower():
                    return f"Formative assessment is {content.lower()}. It's used during the learning process to provide ongoing feedback."
                elif 'summative' in query.lower():
                    return f"Summative assessment is {content.lower()}. It evaluates student learning at the end of an instructional unit."
                else:
                    return f"Assessment involves {content.lower()}. There are different types including formative and summative evaluation methods."
            
            return "Assessment is a systematic process of evaluating student learning and providing feedback to improve educational outcomes."
            
        except Exception as e:
            return "Assessment involves various methods to evaluate and improve student learning outcomes."
    
    def _generate_general_response(self, context: str, query: str) -> str:
        """Generate general response from context"""
        try:
            # Extract meaningful content
            context_lines = context.split('\n')
            content_parts = []
            
            for line in context_lines:
                line = line.strip()
                if line and not line.startswith('[Source:') and len(line) > 30:
                    clean_line = re.sub(r'<[^>]+>', '', line)
                    clean_line = clean_line.replace('&nbsp;', ' ').replace('&amp;', '&')
                    clean_line = ' '.join(clean_line.split())
                    if clean_line:
                        content_parts.append(clean_line[:200])
                        if len(content_parts) >= 2:
                            break
            
            if content_parts:
                response = f"Based on the documents, {content_parts[0].lower()}"
                if len(content_parts) > 1:
                    response += f" Additionally, {content_parts[1].lower()}"
                
                if not response.endswith('.'):
                    response += '.'
                
                return response
            
            return "I found relevant information in the documents but need more context to provide a specific answer."
            
        except Exception as e:
            return "I found some information but couldn't extract a clear answer."
    
    def _is_valid_response(self, response: str, query: str) -> bool:
        """Check if the LLM response is valid and relevant"""
        try:
            response_lower = response.lower()
            
            # Invalid response patterns
            invalid_patterns = [
                "please provide me with",
                "i need more information",
                "i need more context",
                "please provide more context",
                "can you help me write",
                "please complete this sentence",
                "what is the meaning of the word",
                "let me know, and i'll be happy to help",
                "more information to understand",
                "please provide the context"
            ]
            
            # Check if response contains invalid patterns
            for pattern in invalid_patterns:
                if pattern in response_lower:
                    return False
            
            # Check if response is too short (likely generic)
            if len(response.strip()) < 20:
                return False
            
            # Check if response seems to be about the query topic
            query_words = set(query.lower().split())
            response_words = set(response_lower.split())
            
            # If there's some overlap or response is substantial, consider it valid
            overlap = len(query_words.intersection(response_words))
            if overlap > 0 or len(response) > 50:
                return True
                
            return False
            
        except Exception:
            return True  # If validation fails, assume it's valid
    
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
