#!/usr/bin/env python3
"""
Unit tests for PII Detector
"""

import unittest
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.pii_detector import PIIDetector, ContextAnalyzer

class TestPIIDetector(unittest.TestCase):
    
    def setUp(self):
        self.detector = PIIDetector()
    
    def test_detect_full_name(self):
        """Test full name detection"""
        text = "My name is John Smith and I live in New York."
        entities = self.detector.detect_pii_entities(text)
        
        name_entities = [e for e in entities if e['type'] == 'full_name']
        self.assertTrue(len(name_entities) > 0)
        self.assertEqual(name_entities[0]['value'], 'John Smith')
    
    def test_detect_email(self):
        """Test email detection"""
        text = "Contact me at john.doe@example.com for more information."
        entities = self.detector.detect_pii_entities(text)
        
        email_entities = [e for e in entities if e['type'] == 'email_address']
        self.assertTrue(len(email_entities) > 0)
        self.assertEqual(email_entities[0]['value'], 'john.doe@example.com')
    
    def test_detect_phone_number(self):
        """Test phone number detection"""
        text = "Call me at 555-123-4567 or (555) 123-4567."
        entities = self.detector.detect_pii_entities(text)
        
        phone_entities = [e for e in entities if e['type'] == 'phone_number']
        self.assertTrue(len(phone_entities) >= 1)
    
    def test_detect_ssn(self):
        """Test SSN detection"""
        text = "My SSN is XXX-XX-1234 for verification."
        entities = self.detector.detect_pii_entities(text)
        
        ssn_entities = [e for e in entities if e['type'] == 'ssn']
        self.assertTrue(len(ssn_entities) > 0)
        self.assertEqual(ssn_entities[0]['value'], 'XXX-XX-1234')
    
    def test_detect_credit_card(self):
        """Test credit card detection"""
        text = "My card number is 4532-1234-5678-9012."
        entities = self.detector.detect_pii_entities(text)
        
        cc_entities = [e for e in entities if e['type'] == 'credit_card_number']
        self.assertTrue(len(cc_entities) > 0)
    
    def test_mask_pii_entities(self):
        """Test PII masking"""
        text = "My name is John Smith and my email is john@example.com."
        entities_to_mask = ['full_name', 'email_address']
        
        masked_text, replacements = self.detector.mask_pii_entities(text, entities_to_mask)
        
        # Check that original values are not in masked text
        self.assertNotIn('John Smith', masked_text)
        self.assertNotIn('john@example.com', masked_text)
        
        # Check that replacements were made
        self.assertTrue(len(replacements) > 0)
    
    def test_generate_fake_replacement(self):
        """Test fake data generation"""
        fake_name = self.detector.generate_fake_replacement('full_name')
        self.assertIsInstance(fake_name, str)
        self.assertTrue(len(fake_name) > 0)
        
        fake_email = self.detector.generate_fake_replacement('email_address')
        self.assertIsInstance(fake_email, str)
        self.assertIn('@', fake_email)

class TestContextAnalyzer(unittest.TestCase):
    
    def setUp(self):
        self.analyzer = ContextAnalyzer()
    
    def test_analyze_mathematical_context(self):
        """Test mathematical context detection"""
        text = "Calculate the sum of digits in my phone number 555-1234."
        context = self.analyzer.analyze_context(text)
        self.assertEqual(context, 'mathematical')
    
    def test_analyze_location_context(self):
        """Test location-based context detection"""
        text = "What are the nearest hospitals to my address 123 Main St?"
        context = self.analyzer.analyze_context(text)
        self.assertEqual(context, 'location_based')
    
    def test_analyze_medical_context(self):
        """Test medical context detection"""
        text = "Patient John Doe needs a prescription refill."
        context = self.analyzer.analyze_context(text)
        self.assertEqual(context, 'medical')
    
    def test_analyze_financial_context(self):
        """Test financial context detection"""
        text = "Check my account balance for account number 123456789."
        context = self.analyzer.analyze_context(text)
        self.assertEqual(context, 'financial')
    
    def test_determine_masking_necessity_mathematical(self):
        """Test masking decisions for mathematical context"""
        text = "Calculate the sum of digits in my phone number 555-1234."
        detected_entities = ['phone_number', 'full_name']
        
        decisions = self.analyzer.determine_masking_necessity(text, detected_entities)
        
        # Phone number should be preserved for calculation
        self.assertFalse(decisions.get('phone_number', True))
        # Name should be masked as it's not needed
        self.assertTrue(decisions.get('full_name', False))
    
    def test_determine_masking_necessity_location(self):
        """Test masking decisions for location context"""
        text = "Find directions to my address 123 Main Street."
        detected_entities = ['address', 'full_name']
        
        decisions = self.analyzer.determine_masking_necessity(text, detected_entities)
        
        # Address should be preserved for location query
        self.assertFalse(decisions.get('address', True))
        # Name should be masked
        self.assertTrue(decisions.get('full_name', False))
    
    def test_determine_masking_necessity_general(self):
        """Test masking decisions for general context"""
        text = "What's the weather like today?"
        detected_entities = ['full_name', 'address']
        
        decisions = self.analyzer.determine_masking_necessity(text, detected_entities)
        
        # All PII should be masked for general queries
        for entity, should_mask in decisions.items():
            self.assertTrue(should_mask)

if __name__ == '__main__':
    unittest.main()