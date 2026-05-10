#!/usr/bin/env python3
"""
Quick start script for PII Privacy Handler
"""

import os
import sys

def check_requirements():
    """Check if all requirements are installed"""
    print("Checking requirements...")
    
    required_packages = [
        'tensorflow', 'numpy', 'pandas', 'sklearn', 
        'nltk', 'faker', 'requests', 'dotenv'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Please run: pip install -r requirements.txt")
        return False
    
    return True

def setup_environment():
    """Setup environment variables"""
    print("\nSetting up environment...")
    
    env_file = '.env'
    if not os.path.exists(env_file):
        print("Creating .env file from template...")
        with open('.env.example', 'r') as template:
            content = template.read()
        
        with open(env_file, 'w') as env:
            env.write(content)
        
        print("⚠️  Please edit .env file and add your Gemini API key!")
        return False
    
    print("✅ Environment file exists")
    return True

def check_dataset():
    """Check if dataset exists"""
    print("\nChecking dataset...")
    
    dataset_path = 'pii_100000_new.csv'
    if os.path.exists(dataset_path):
        print(f"✅ Dataset found: {dataset_path}")
        
        # Check dataset size
        import pandas as pd
        try:
            df = pd.read_csv(dataset_path)
            print(f"✅ Dataset loaded: {len(df)} rows")
            return True
        except Exception as e:
            print(f"❌ Error reading dataset: {e}")
            return False
    else:
        print(f"❌ Dataset not found: {dataset_path}")
        print("Please ensure the dataset file is in the root directory")
        return False

def run_quick_demo():
    """Run a quick demo"""
    print("\nRunning quick demo...")
    
    try:
        # Add src to path
        sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
        
        from src.privacy_handler import PIIPrivacyHandler
        
        # Initialize handler (without trained model for quick demo)
        handler = PIIPrivacyHandler()
        
        # Test queries
        test_queries = [
            "My name is John Smith and I live in New York.",
            "Calculate the sum of digits in phone number 555-1234.",
            "What's the weather like today?"
        ]
        
        print("\n" + "="*50)
        print("QUICK DEMO RESULTS")
        print("="*50)
        
        for i, query in enumerate(test_queries, 1):
            print(f"\nExample {i}: {query}")
            
            try:
                result = handler.process_query(query)
                
                if 'error' in result:
                    print(f"Error: {result['error']}")
                    continue
                
                print(f"Context: {result.get('context', 'unknown')}")
                print(f"Entities detected: {result.get('detected_entities', [])}")
                print(f"Entities masked: {result.get('entities_masked', [])}")
                print(f"Privacy preserved: {result.get('privacy_preserved', False)}")
                
                if result.get('final_response'):
                    print(f"Response: {result['final_response'][:100]}...")
                
            except Exception as e:
                print(f"Error processing query: {e}")
        
        print("\n" + "="*50)
        print("✅ Quick demo completed!")
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return False

def main():
    """Main quick start function"""
    print("🚀 PII Privacy Handler - Quick Start")
    print("="*50)
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Requirements check failed!")
        return
    
    # Setup environment
    env_ready = setup_environment()
    
    # Check dataset
    dataset_ready = check_dataset()
    
    # Run quick demo
    if run_quick_demo():
        print("\n🎉 Quick start completed successfully!")
        
        print("\nNext steps:")
        print("1. Edit .env file with your Gemini API key (if not done)")
        print("2. Train the model: python train_model.py")
        print("3. Run full demo: python demo.py")
        print("4. Run tests: python run_tests.py")
        
        if not env_ready:
            print("\n⚠️  Don't forget to add your Gemini API key to .env file!")
        
        if not dataset_ready:
            print("\n⚠️  Dataset not found - some features may not work!")
    
    else:
        print("\n❌ Quick start failed!")

if __name__ == "__main__":
    main()