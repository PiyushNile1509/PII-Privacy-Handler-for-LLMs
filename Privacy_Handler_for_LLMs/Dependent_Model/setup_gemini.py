#!/usr/bin/env python3
"""
Setup Gemini API Key for PII Privacy Handler
"""

import os

def setup_gemini_api():
    """Setup Gemini API key"""
    
    print("🔑 Gemini API Setup")
    print("=" * 30)
    
    # Check if API key already exists
    existing_key = os.getenv('GEMINI_API_KEY')
    if existing_key and existing_key != 'your-api-key-here':
        print(f"✅ Gemini API key already configured: {existing_key[:10]}...")
        return existing_key
    
    print("To use Gemini API, you need an API key from Google AI Studio:")
    print("1. Go to https://makersuite.google.com/app/apikey")
    print("2. Create a new API key")
    print("3. Copy the key and paste it below")
    print()
    
    api_key = input("Enter your Gemini API key (or press Enter to use fallback): ").strip()
    
    if api_key:
        # Set environment variable for current session
        os.environ['GEMINI_API_KEY'] = api_key
        
        # Create .env file
        env_path = os.path.join(os.path.dirname(__file__), '.env')
        with open(env_path, 'w') as f:
            f.write(f"GEMINI_API_KEY={api_key}\n")
        
        print(f"✅ API key configured and saved to .env file")
        return api_key
    else:
        print("⚠️ No API key provided. Using fallback responses.")
        return None

def test_gemini_connection():
    """Test Gemini API connection"""
    
    try:
        from src.privacy_handler import PIIPrivacyHandler
        
        handler = PIIPrivacyHandler()
        
        # Test with a simple query
        test_query = "Hello, how are you?"
        response = handler.call_gemini_api(test_query)
        
        if "processed your query" not in response:  # Not fallback response
            print("✅ Gemini API connection successful!")
            print(f"Test response: {response[:100]}...")
            return True
        else:
            print("⚠️ Using fallback responses (API key not working or not provided)")
            return False
            
    except Exception as e:
        print(f"❌ Error testing Gemini API: {e}")
        return False

if __name__ == "__main__":
    # Setup API key
    api_key = setup_gemini_api()
    
    if api_key:
        print("\n🧪 Testing connection...")
        test_gemini_connection()
    
    print("\n🚀 Setup complete! You can now run:")
    print("streamlit run demo.py")