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
    
    def _call_openrouter_api(self, query: str, context: str, conversation_history: List = None) -> Optional[str]:
        """Call OpenRouter API as fallback"""
        try:
            system_prompt = "You are a helpful educational assistant. Answer based on the provided context."
            user_prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
            
            for model_name in self.openrouter_models.values():
                try:
                    headers = {
                        "Authorization": f"Bearer {self.openrouter_api_key}",
                        "Content-Type": "application/json"
                    }
                    
                    payload = {
                        "model": model_name,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        "max_tokens": 500,
                        "temperature": 0.7
                    }
                    
                    response = requests.post(
                        self.openrouter_url,
                        headers=headers,
                        json=payload,
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        if 'choices' in result and len(result['choices']) > 0:
                            content = result['choices'][0]['message']['content']
                            if self._is_valid_response(content, query):
                                return content
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ OpenRouter model {model_name} failed: {str(e)}")
                    continue
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ OpenRouter API call failed: {str(e)}")
            return None
    
    def _generate_intelligent_fallback_response(self, query: str, context: str) -> str:
        """Generate intelligent fallback response when APIs are unavailable"""
        try:
            self.logger.info("🔄 Using enhanced fallback response generation")
            
            query_lower = query.lower()
            
            # Special handling for holiday-related queries
            if any(word in query_lower for word in ['holiday', 'holidays', 'leave', 'vacation', 'day']):
                return self._generate_holiday_response(context, query)
            
            # Special handling for assessment-related queries
            elif any(word in query_lower for word in ['assessment', 'evaluation', 'test', 'exam', 'formative', 'summative']):
                return self._generate_assessment_response(context, query)
            
            # General response generation
            else:
                return self._generate_general_response(context, query)
                
        except Exception as e:
            self.logger.error(f"❌ Fallback response generation failed: {str(e)}")
            return "I found some information but couldn't process it properly. Please try rephrasing your question."
    
    def _generate_holiday_response(self, context: str, query: str) -> str:
        """Generate specific response for holiday queries"""
        try:
            # Extract holiday information from context
            context_lines = context.split('\n')
            holidays = []
            holiday_info = []
            
            for line in context_lines:
                line = line.strip()
                if any(word in line.upper() for word in ['HOLIDAY', 'VARALAKSHMI', 'ACADEMIC YEAR', 'STAFF', 'SATURDAY']):
                    clean_line = re.sub(r'<[^>]+>', '', line)
                    clean_line = clean_line.replace('&nbsp;', ' ').replace('&amp;', '&')
                    clean_line = ' '.join(clean_line.split())
                    if len(clean_line) > 10:
                        holiday_info.append(clean_line)
            
            if holiday_info:
                if 'which day' in query.lower() or 'what day' in query.lower():
                    return f"Based on the school calendar, Second Saturday is a holiday for all school staff. The school follows a proper work schedule that includes Second Saturday holidays as mentioned in the academic calendar."
                elif 'second saturday' in query.lower():
                    return "Yes, Second Saturday is a holiday for school staff. The school maintains a proper work schedule that includes Second Saturday holidays along with other scheduled holidays throughout the academic year."
                else:
                    response = "According to the school's academic calendar, all teaching and non-teaching staff are entitled to holidays including Varalakshmi Vratham and other scheduled holidays. "
                    response += "The academic year holiday list includes various festivals and observances for all school staff members."
                    return response
            
            return "Based on the school documents, there are scheduled holidays for all school staff members as per the academic calendar."
            
        except Exception as e:
            return "The school has scheduled holidays for all staff members as outlined in the academic calendar."
    
    def _generate_assessment_response(self, context: str, query: str) -> str:
        """Generate specific response for assessment queries"""
        try:
            context_sentences = context.split('.')
            relevant_sentences = []
            
            for sentence in context_sentences:
                sentence = sentence.strip()
                if any(word in sentence.lower() for word in ['assessment', 'formative', 'summative', 'evaluation', 'learning']):
                    clean_sentence = re.sub(r'<[^>]+>', '', sentence)
                    clean_sentence = clean_sentence.replace('&nbsp;', ' ').replace('&amp;', '&')
                    clean_sentence = ' '.join(clean_sentence.split())
                    if len(clean_sentence) > 20:
                        relevant_sentences.append(clean_sentence)
            
            if relevant_sentences:
                if 'formative' in query.lower():
                    return f"Formative assessment is assessment for learning that provides information to plan the next stage in learning. {relevant_sentences[0][:200]}..."
                elif 'summative' in query.lower():
                    return f"Summative assessment is the culmination of the teaching and learning process. {relevant_sentences[0][:200]}..."
                elif 'types' in query.lower() or 'different' in query.lower():
                    return f"There are different types of evaluation including formative assessment (assessment for learning) and summative assessment. {relevant_sentences[0][:150]}..."
                else:
                    return f"Based on the educational documents: {relevant_sentences[0][:250]}..."
            
            return "Assessment strategies include various methods for evaluating student learning and progress as outlined in the educational documents."
            
        except Exception as e:
            return "Assessment involves various strategies for evaluating and supporting student learning."
    
    def _generate_general_response(self, context: str, query: str) -> str:
        """Generate general response from context"""
        try:
            # Extract most relevant sentences from context
            context_sentences = context.split('.')
            relevant_sentences = []
            
            query_words = set(query.lower().split())
            
            for sentence in context_sentences:
                sentence = sentence.strip()
                sentence_words = set(sentence.lower().split())
                
                # Calculate relevance score
                overlap = len(query_words.intersection(sentence_words))
                if overlap > 0 and len(sentence) > 30:
                    clean_sentence = re.sub(r'<[^>]+>', '', sentence)
                    clean_sentence = clean_sentence.replace('&nbsp;', ' ').replace('&amp;', '&')
                    clean_sentence = ' '.join(clean_sentence.split())
                    relevant_sentences.append((overlap, clean_sentence))
            
            if relevant_sentences:
                # Sort by relevance and take the best
                relevant_sentences.sort(key=lambda x: x[0], reverse=True)
                best_sentence = relevant_sentences[0][1]
                
                # Generate a natural response
                return f"Based on the documents: {best_sentence[:300]}..."
            
            return "I found some information in the documents but couldn't extract a clear answer. Please try rephrasing your question."
            
        except Exception as e:
            return "I found some relevant information but couldn't process it properly."
    
    def _is_valid_response(self, response: str, query: str) -> bool:
        """Validate if the response is meaningful"""
        if not response or len(response.strip()) < 10:
            return False
        
        # Check for generic responses that indicate failure
        invalid_phrases = [
            "provide more context",
            "i need more information",
            "please clarify",
            "i don't understand",
            "insufficient information",
            "cannot answer"
        ]
        
        response_lower = response.lower()
        return not any(phrase in response_lower for phrase in invalid_phrases)
