import google.generativeai as genai
import os
from typing import Dict, Any, Optional
import time
import json
from dotenv import load_dotenv

class GeminiClient:
    """Client for interacting with Google's Gemini API"""
    
    def __init__(self, api_key: Optional[str] = None):
        load_dotenv()
        
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("Gemini API key not provided. Set GEMINI_API_KEY environment variable.")
        
        genai.configure(api_key=self.api_key)
        
        # Initialize the model
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1.0  # Minimum seconds between requests
    
    def _rate_limit(self):
        """Implement rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def generate_response(self, prompt: str, max_retries: int = 3) -> str:
        """Generate response from Gemini API with retry logic"""
        
        for attempt in range(max_retries):
            try:
                self._rate_limit()
                
                response = self.model.generate_content(prompt)
                
                if response.text:
                    return response.text.strip()
                else:
                    raise Exception("Empty response from Gemini API")
                    
            except Exception as e:
                if attempt == max_retries - 1:
                    raise Exception(f"Failed to get response after {max_retries} attempts: {str(e)}")
                
                # Exponential backoff
                wait_time = (2 ** attempt) * 2
                time.sleep(wait_time)
        
        return ""
    
    def process_masked_query(self, masked_query: str, context: str = "") -> str:
        """Process a masked query through Gemini"""
        
        # Construct the prompt
        system_prompt = """You are a helpful AI assistant. Please respond to the following query naturally and helpfully. 
        The query may contain masked or anonymized information, but please provide a complete and useful response."""
        
        if context:
            system_prompt += f"\n\nContext: This appears to be a {context} related query."
        
        full_prompt = f"{system_prompt}\n\nQuery: {masked_query}\n\nResponse:"
        
        return self.generate_response(full_prompt)
    
    def test_connection(self) -> bool:
        """Test the connection to Gemini API"""
        try:
            test_response = self.generate_response("Hello, this is a test. Please respond with 'Connection successful.'")
            return "successful" in test_response.lower()
        except:
            return False

class ResponseProcessor:
    """Process and restore PII in LLM responses"""
    
    def __init__(self):
        self.restoration_patterns = {
            'your phone number': 'phone_number',
            'your address': 'address',
            'your account': 'account_number',
            'your date of birth': 'date_of_birth',
            'your name': 'full_name',
            'your email': 'email_address'
        }
    
    def restore_pii_in_response(self, response: str, original_query: str, 
                               replacements: Dict[str, str]) -> str:
        """Restore original PII in response where appropriate"""
        
        restored_response = response
        
        # Extract original values from replacements
        original_values = {}
        fake_to_original = {}
        
        for pii_type, replacement_info in replacements.items():
            if ' -> ' in replacement_info:
                original, fake = replacement_info.split(' -> ', 1)
                original = original.strip()
                fake = fake.strip()
                original_values[pii_type] = original
                fake_to_original[fake] = original
        
        # Always restore original entities that were preserved in the query
        # (entities that were kept for functional purposes)
        preserved_entities = self._get_preserved_entities(original_query, replacements)
        
        for fake_value, original_value in fake_to_original.items():
            if fake_value in restored_response:
                # Check if this entity was preserved (not masked) in the original query
                if original_value in preserved_entities:
                    # Always restore preserved entities
                    restored_response = restored_response.replace(fake_value, original_value)
                else:
                    # For masked entities, restore based on context
                    if self._should_restore_in_response(fake_value, response):
                        restored_response = restored_response.replace(fake_value, original_value)
        
        # Replace generic references with personalized ones
        for pattern, pii_type in self.restoration_patterns.items():
            if pattern in restored_response.lower() and pii_type in original_values:
                if pii_type == 'phone_number':
                    restored_response = restored_response.replace(
                        pattern, f"your phone number ({original_values[pii_type]})"
                    )
                elif pii_type == 'address':
                    restored_response = restored_response.replace(
                        pattern, f"your address ({original_values[pii_type]})"
                    )
                elif pii_type == 'full_name':
                    restored_response = restored_response.replace(
                        pattern, f"you ({original_values[pii_type]})"
                    )
        
        # Make response more personal by replacing generic pronouns
        restored_response = self._personalize_response(restored_response, original_values)
        
        return restored_response
    
    def _get_preserved_entities(self, original_query: str, replacements: Dict[str, str]) -> set:
        """Get entities that were preserved (not masked) in the original query"""
        preserved = set()
        
        for pii_type, replacement_info in replacements.items():
            if ' -> ' in replacement_info:
                original, fake = replacement_info.split(' -> ', 1)
                original = original.strip()
                fake = fake.strip()
                
                # If original and fake are the same, it means it was preserved
                if original == fake:
                    preserved.add(original)
        
        return preserved
    
    def _should_restore_in_response(self, fake_value: str, response: str) -> bool:
        """Determine if PII should be restored in the response"""
        response_lower = response.lower()
        
        # Always restore in mathematical/computational contexts
        if any(word in response_lower for word in ['sum', 'total', 'calculation', 'result', 'answer', 'equals']):
            return True
        
        # Restore in location-based contexts
        if any(word in response_lower for word in ['nearest', 'location', 'distance', 'address', 'directions']):
            return True
        
        # Restore when response directly references the entity
        if fake_value.lower() in response_lower:
            return True
        
        return False
    
    def _personalize_response(self, response: str, original_values: Dict[str, str]) -> str:
        """Make response more personal and user-friendly"""
        personalized = response
        
        # Replace generic references with personal ones
        replacements = {
            'the person': 'you',
            'the individual': 'you', 
            'the user': 'you',
            'the customer': 'you',
            'this person': 'you',
            'that person': 'you'
        }
        
        for generic, personal in replacements.items():
            personalized = personalized.replace(generic, personal)
            personalized = personalized.replace(generic.title(), personal.title())
        
        return personalized
    
    def make_response_personal(self, response: str, original_query: str) -> str:
        """Make the response more personal and user-friendly"""
        
        # Replace generic references with personal ones
        personal_replacements = {
            'the person': 'you',
            'the individual': 'you',
            'the user': 'you',
            'the customer': 'you',
            'the patient': 'you',
            'the student': 'you'
        }
        
        personalized_response = response
        for generic, personal in personal_replacements.items():
            personalized_response = personalized_response.replace(generic, personal)
        
        return personalized_response