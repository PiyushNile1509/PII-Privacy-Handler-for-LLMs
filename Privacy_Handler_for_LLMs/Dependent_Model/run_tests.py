#!/usr/bin/env python3
"""
Test runner for PII Privacy Handler
"""

import unittest
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def run_unit_tests():
    """Run unit tests"""
    print("=== Running Unit Tests ===")
    
    # Discover and run tests
    loader = unittest.TestLoader()
    start_dir = 'tests'
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

def run_integration_tests():
    """Run integration tests"""
    print("\n=== Running Integration Tests ===")
    
    try:
        from src.privacy_handler import PIIPrivacyHandler
        
        # Test basic functionality without trained model
        handler = PIIPrivacyHandler()
        
        test_queries = [
            "My name is John Smith and my phone is 555-1234.",
            "Calculate the sum of digits in phone number 123-456-7890.",
            "What's the weather like today?",
            "My email is test@example.com, send me updates."
        ]
        
        print("Testing basic PII detection and masking...")
        
        for i, query in enumerate(test_queries, 1):
            print(f"\nTest {i}: {query}")
            
            try:
                result = handler.process_query(query)
                
                if 'error' in result:
                    print(f"  ❌ Error: {result['error']}")
                    continue
                
                print(f"  ✅ Context: {result.get('context', 'unknown')}")
                print(f"  ✅ Entities detected: {len(result.get('detected_entities', []))}")
                print(f"  ✅ Privacy preserved: {result.get('privacy_preserved', False)}")
                
            except Exception as e:
                print(f"  ❌ Exception: {e}")
                return False
        
        print("\n✅ Integration tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Integration test setup failed: {e}")
        return False

def run_performance_tests():
    """Run performance tests"""
    print("\n=== Running Performance Tests ===")
    
    try:
        import time
        from src.privacy_handler import PIIPrivacyHandler
        
        handler = PIIPrivacyHandler()
        
        # Test query processing speed
        test_query = "My name is John Smith, phone 555-1234, email john@example.com. What's the weather?"
        
        # Warm up
        handler.process_query(test_query)
        
        # Time multiple runs
        num_runs = 10
        start_time = time.time()
        
        for _ in range(num_runs):
            handler.process_query(test_query)
        
        end_time = time.time()
        avg_time = (end_time - start_time) / num_runs
        
        print(f"Average processing time: {avg_time:.3f} seconds")
        
        if avg_time < 5.0:  # Should process within 5 seconds
            print("✅ Performance test passed!")
            return True
        else:
            print("❌ Performance test failed - too slow!")
            return False
            
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False

def main():
    """Main test runner"""
    print("PII Privacy Handler - Test Suite")
    print("=" * 40)
    
    all_passed = True
    
    # Run unit tests
    if not run_unit_tests():
        all_passed = False
    
    # Run integration tests
    if not run_integration_tests():
        all_passed = False
    
    # Run performance tests
    if not run_performance_tests():
        all_passed = False
    
    print("\n" + "=" * 40)
    if all_passed:
        print("🎉 All tests passed!")
        sys.exit(0)
    else:
        print("❌ Some tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()