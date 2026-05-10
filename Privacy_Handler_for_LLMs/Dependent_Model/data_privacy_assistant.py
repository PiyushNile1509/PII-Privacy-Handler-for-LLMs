#!/usr/bin/env python3

import re
import json
import pickle
import os
from typing import Dict, List, Any

class DataPrivacyAssistant:
    def __init__(self):
        # Initialize PII patterns
        self.pii_patterns = {
            'NAME': [
                r'\b(?:my name is|I am|I\'m|called|This is)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
                r'\b(Dr\.|Prof\.|Mr\.|Ms\.|Mrs\.)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
                r'\(([A-Z][a-z]+\s+[A-Z][a-z]+)\)',
                r'\bfor\s+([A-Z][a-z]+\s+[A-Z][a-z]+)\b',
                r'\bunder\s+([A-Z][a-z]+\s+[A-Z][a-z]+)\b',
                r'\bbelongs to\s+([A-Z][a-z]+\s+[A-Z][a-z]+)\b',
                r'\bclient\s+([A-Z][a-z]+\s+[A-Z][a-z]+)\b',
                r'\buser\s+([A-Z][a-z]+)\b',
                r'\bpolicyholder:\s+([A-Z][a-z]+\s+[A-Z][a-z]+)\b'
            ],
            'PHONE': [
                r'\+91\s+(\d{10})',
                r'\+44\s+(\d{4}\s+\d{6})',
                r'\((\d{3})\)\s+(\d{3}-\d{4})',
                r'\b(\d{10})\b',
                r'phone\s+([+\d\s-]+)',
                r'contact\s+([+\d\s-]+)',
                r'number\s+([+\d\s-]+)'
            ],
            'EMAIL': [
                r'\b([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b',
                r'mailto:([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'
            ],
            'ACCOUNT': [
                r'\baccount\s+(\d{8,16})\b',
                r'\baccount is\s+(\d{8,16})\b'
            ],
            'CREDIT_CARD': [
                r'\b(\d{4}\s+\d{4}\s+\d{4}\s+\d{4})\b',
                r'\b(\d{4}-\d{4}-\d{4}-\d{4})\b'
            ],
            'SSN': [
                r'\b(\d{3}-\d{2}-\d{4})\b'
            ],
            'DOB': [
                r'\bborn\s+(\d{4}-\d{2}-\d{2})\b',
                r'\bDOB\s+(\d{2}/\d{2}/\d{4})\b'
            ],
            'AADHAAR': [
                r'\bAadhaar\s+(\d{4}\s+\d{4}\s+\d{4})\b'
            ],
            'PAN': [
                r'\bPAN\s+([A-Z]{5}\d{4}[A-Z])\b'
            ],
            'PASSPORT': [
                r'\bPassport\s+([A-Z]\d{7})\b',
                r'\bPassport\s+([A-Z]{2}\d{7})\b'
            ],
            'VOTER_ID': [
                r'\bVoter ID\s+([A-Z]{3}\d{6})\b'
            ],
            'DRIVING_LICENSE': [
                r'\bDriving licence\s+([A-Z]{2}\d{2}\s+\d{10})\b'
            ],
            'TAX_ID': [
                r'\bTax ID\s+([A-Z]{2}-\d{9})\b',
                r'\bTIN-([A-Z0-9]+)\b'
            ],
            'GSTIN': [
                r'\bGSTIN\s+([A-Z0-9]{15})\b'
            ],
            'IFSC': [
                r'\bIFSC\s+([A-Z]{4}\d{7})\b'
            ],
            'IMEI': [
                r'\bIMEI\s+(\d{15})\b'
            ],
            'MAC': [
                r'\bMAC\s+([0-9A-F]{2}:[0-9A-F]{2}:[0-9A-F]{2}:[0-9A-F]{2}:[0-9A-F]{2}:[0-9A-F]{2})\b'
            ],
            'IP': [
                r'\bIP\s+(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b'
            ],
            'UUID': [
                r'\bUUID\s+([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\b'
            ],
            'STUDENT_ID': [
                r'\bstudent ID\s+([A-Z]{3}-\d{6})\b',
                r'\bRoll no\s+(\d{6})\b'
            ],
            'EMPLOYEE_ID': [
                r'\bEmployee\s+([A-Z]{3}-\d{6})\b'
            ],
            'PATIENT_ID': [
                r'\bPatient ID\s+([A-Z]{3}-\d{6})\b',
                r'\bpatient code\s+([A-Z]-\d{6})\b'
            ],
            'MEDICAL_RECORD': [
                r'\bMRN\s+([A-Z]{3}-\d{6})\b'
            ],
            'INSURANCE_ID': [
                r'\bhealth insurance\s+([A-Z]{2}-\d{6})\b',
                r'\bPolicy\s+([A-Z]{3}-\d{6})\b',
                r'\bClaim\s+([A-Z]{3}-\d{6})\b',
                r'\bHealth policy\s+([A-Z]{2}-\d{6})\b',
                r'\bInsurance ID\s+([A-Z]-\d{6})\b',
                r'\bGroup\s+([A-Z]{3}-\d{6})\b'
            ],
            'ORDER_ID': [
                r'\bOrder\s+([A-Z]{3}-\d{6})\b',
                r'\bTracking\s+([A-Z]{3}-\d{6})\b',
                r'\border\s+([A-Z]{3}-\d{6})\b'
            ],
            'BOOKING_ID': [
                r'\bBooking\s+([A-Z]{2}-\d{6})\b',
                r'\bticket\s+([A-Z]{2}-\d{6})\b',
                r'\bFlight\s+([A-Z]{3}-\d{6})\b',
                r'\bHotel booking\s+([A-Z]{3}-\d{6})\b'
            ],
            'FREQUENT_FLYER': [
                r'\bFrequent flyer\s+([A-Z]{2}-\d{6})\b'
            ],
            'CUSTOMER_ID': [
                r'\bCustomer ID\s+([A-Z]{4}-\d{6})\b',
                r'\bCustomer\s+([A-Z]{3}-\d{6})\b'
            ],
            'SIM_ID': [
                r'\bSIM ID\s+([A-Z]{3}-\d{6})\b'
            ],
            'IMSI': [
                r'\bIMSI\s+(\d{15})\b'
            ],
            'BADGE_ID': [
                r'\bBadge\s+([A-Z]{4}-\d{6})\b'
            ],
            'DEPT_ID': [
                r'\bdept id\s+([A-Z]{4}-\d{4})\b'
            ],
            'PROJECT_ID': [
                r'\bproject\s+([A-Z]{3}-\d{6})\b'
            ],
            'LIBRARY_CARD': [
                r'\bLibrary card\s+([A-Z]{2}-\d{6})\b'
            ],
            'GIFT_VOUCHER': [
                r'\bGift voucher\s+([A-Z]{2}-\d{6})\b'
            ],
            'CASE_ID': [
                r'\bLegal case\s+([A-Z]{4}-\d{6})\b'
            ],
            'SOCIAL_HANDLE': [
                r'\bUsername\s+(@[a-zA-Z0-9_]+)\b',
                r'\bTwitter\s+(@[a-zA-Z0-9_]+)\b',
                r'\bAccount\s+([A-Z]{3}-\d{6})\b'
            ],
            'LINKEDIN': [
                r'\bLinkedIn\s+(linkedin\.com/in/[a-zA-Z0-9-]+)\b'
            ],
            'ADDRESS': [
                r'\baddress\s+(\d+\s+[A-Za-z\s,]+)\b'
            ]
        }
        
        # Load trained model if available
        self.trained_classifier = None
        try:
            if os.path.exists('models/simple_classifier.pkl'):
                with open('models/simple_classifier.pkl', 'rb') as f:
                    self.trained_classifier = pickle.load(f)
                print("Loaded trained PII classifier")
        except Exception as e:
            print(f"Could not load trained classifier: {e}")
    
    def detect_pii(self, text: str) -> List[Dict[str, Any]]:
        """Detect PII entities in text"""
        entities = []
        
        for pii_type, patterns in self.pii_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    groups = match.groups()
                    if groups:
                        # Handle multiple groups
                        for i, group in enumerate(groups):
                            if group:
                                start_idx = match.start(i+1)
                                end_idx = match.end(i+1)
                                entities.append({
                                    'type': pii_type,
                                    'text': group,
                                    'start': start_idx,
                                    'end': end_idx,
                                    'full_match': match.group()
                                })
                                break
                    else:
                        entities.append({
                            'type': pii_type,
                            'text': match.group(),
                            'start': match.start(),
                            'end': match.end(),
                            'full_match': match.group()
                        })
        
        return entities
    
    def is_computation_required(self, text: str, pii_entity: Dict) -> bool:
        """Check if PII is needed for computation using trained model or fallback rules"""
        
        # Use trained model if available
        if self.trained_classifier:
            try:
                prediction, confidence = self.trained_classifier.predict(text)
                return prediction == 1
            except Exception as e:
                print(f"Error using trained classifier: {e}")
        
        # Fallback to rule-based approach
        text_lower = text.lower()
        
        # Computational keywords
        comp_keywords = [
            'add', 'subtract', 'multiply', 'divide', 'sum', 'calculate', 'plus', 'minus',
            'extract', 'reverse', 'count', 'checksum', 'mod', 'parity', 'length',
            'digits', 'numeric', 'last', 'first', 'even', 'odd', 'compare', 'check',
            'domain', 'append', 'double', 'divide by', 'multiply by', 'print', 'display',
            'output', 'return', 'get', 'letters', 'characters', 'initials', 'username',
            'vowels', 'total', 'how many'
        ]
        
        # Check if any computational operation involves this PII
        for keyword in comp_keywords:
            if keyword in text_lower:
                # Check if the operation is related to this PII entity
                entity_context = text_lower[max(0, pii_entity['start']-50):pii_entity['end']+50]
                if keyword in entity_context:
                    return True
        
        # Special cases for conditional operations
        if any(word in text_lower for word in ['if', 'condition', 'when', 'compare']):
            return True
            
        return False
    
    def should_mask_name(self, text: str) -> bool:
        """Determine if names should be masked based on context"""
        # Names are generally masked unless specifically needed for identification
        return True
    
    def process_text(self, text: str) -> Dict[str, Any]:
        """Main processing function"""
        # Detect PII entities
        pii_entities = self.detect_pii(text)
        
        # Determine masking decisions
        processed_entities = []
        masked_text = text
        
        # Sort entities by position (reverse order to maintain indices)
        pii_entities.sort(key=lambda x: x['start'], reverse=True)
        
        for entity in pii_entities:
            # Check if PII is needed for computation
            needs_computation = self.is_computation_required(text, entity)
            
            # Special handling for names - usually mask unless needed
            if entity['type'] == 'NAME':
                should_mask = self.should_mask_name(text)
            else:
                should_mask = not needs_computation
            
            processed_entity = {
                'type': entity['type'],
                'text': entity['text'],
                'masked': should_mask,
                'reason': 'Required for computation' if needs_computation else 'Not needed for operation'
            }
            processed_entities.append(processed_entity)
            
            # Apply masking
            if should_mask:
                replacement = f"[MASKED_{entity['type']}]"
                masked_text = (
                    masked_text[:entity['start']] + 
                    replacement + 
                    masked_text[entity['end']:]
                )
        
        # Reverse the list to maintain original order
        processed_entities.reverse()
        
        return {
            'original_text': text,
            'masked_text': masked_text,
            'pii_entities': processed_entities,
            'privacy_preserved': len([e for e in processed_entities if e['masked']]) > 0
        }

def test_assistant():
    """Test the data privacy assistant with comprehensive test cases"""
    assistant = DataPrivacyAssistant()
    
    # Comprehensive test cases from all domains
    test_cases = [
        # Finance/Banking
        "My name is Rohan Mehta and my account is 5566778899. Add 2450 to my account and return the result.",
        "Transfer ₹3000 to account 0034556677 if last digit of phone +91 9876543210 is even.",
        "PAN ABCDE1234F — extract numeric part and multiply by 3.",
        "This is Sneha Gupta, credit card 4111 1111 1111 1111. Sum card digits and tell me checksum.",
        
        # Healthcare
        "Patient ID HRC-998877 belongs to Dr. Priya Iyer. Reverse the numeric part and report.",
        "This is Amit Patel, born 1989-07-10; calculate my current age.",
        "I'm Dr. K. Rao, patient code P-112233; if sum of digits is odd, return 'FLAG'.",
        
        # Education
        "I'm Sarah Johnson, student ID STU-445566 — add 500 and return numeric result.",
        "Roll no 220145 — multiply by 7 and give last 4 digits.",
        "Email john.kim@university.edu — extract domain length.",
        
        # Corporate/Employment
        "Employee EMP-112233 (Rajesh Kumar) gets ₹1000 for each '1' in his ID.",
        "Work email lisa.wong@company.com — extract company and append '-HR'.",
        
        # Government/ID
        "Aadhaar 1234 5678 9876 belongs to Meena Gupta. Multiply last 4 digits by 5.",
        "Passport M9876543 — return the last three digits multiplied by 2.",
        "SSN 213-45-6789 — check last four digits and return them.",
        
        # Technology
        "Device IMEI 356938035643809 — add first and last digit and return sum.",
        "MAC 00:1A:2B:3C:4D:5E — count hex pairs and return count.",
        "IP 192.168.1.10 — add octets and compare >500?",
        
        # Social/Online
        "Username @prasad_official and email prasad@mail.com — return username length.",
        "LinkedIn linkedin.com/in/jane-doe — extract 'jane-doe' and count hyphens.",
        "Email mailto:alex.smith@startup.io — return domain and its length."
    ]
    
    print("=== Comprehensive Data Privacy Assistant Test ===\n")
    
    for i, test_text in enumerate(test_cases, 1):
        print(f"Test Case {i}:")
        print(f"Input: {test_text}")
        
        result = assistant.process_text(test_text)
        
        print(f"Masked: {result['masked_text']}")
        print(f"Privacy Preserved: {result['privacy_preserved']}")
        
        if result['pii_entities']:
            print("PII Entities:")
            for entity in result['pii_entities']:
                status = "MASKED" if entity['masked'] else "KEPT"
                print(f"  - {entity['type']}: {entity['text']} [{status}] - {entity['reason']}")
        
        print("-" * 80)

if __name__ == "__main__":
    test_assistant()