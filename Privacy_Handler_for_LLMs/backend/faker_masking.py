"""
Faker-based PII Masking Utility
Provides realistic fake data for masking PII instead of generic placeholders
"""

import re
from typing import Dict, List, Tuple
from faker import Faker


class FakerMasking:
    """Utility class for masking PII with Faker-generated realistic data"""
    
    def __init__(self, seed: int = 42):
        self.fake = Faker()
        Faker.seed(seed)
        
        # PII detection patterns
        self.patterns = {
            'EMAIL': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'PHONE': r'\b\d{10}\b',
            'AADHAAR': r'\b\d{12}\b',
            'PAN': r'\b[A-Z]{5}\d{4}[A-Z]\b',
            'SSN': r'\b\d{3}-\d{2}-\d{4}\b',
            'CREDIT_CARD': r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',
            'ZIP': r'\b\d{5}(?:-\d{4})?\b',
            'ADDRESS': r'\b\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd)\b',
        }
    
    def generate_fake(self, pii_type: str) -> str:
        """Generate fake data by PII type"""
        pii_type = pii_type.upper()
        
        generators = {
            'NAME': lambda: self.fake.name(),
            'EMAIL': lambda: self.fake.email(),
            'PHONE': lambda: ''.join([str(self.fake.random_digit()) for _ in range(10)]),
            'AADHAAR': lambda: ''.join([str(self.fake.random_digit()) for _ in range(12)]),
            'PAN': lambda: ''.join([self.fake.random_uppercase_letter() for _ in range(5)]) + 
                          ''.join([str(self.fake.random_digit()) for _ in range(4)]) + 
                          self.fake.random_uppercase_letter(),
            'SSN': lambda: self.fake.ssn(),
            'CREDIT_CARD': lambda: self.fake.credit_card_number(),
            'ADDRESS': lambda: self.fake.address().replace('\n', ', '),
            'DATE': lambda: self.fake.date(),
            'DOB': lambda: self.fake.date_of_birth().strftime('%Y-%m-%d'),
            'COMPANY': lambda: self.fake.company(),
            'CITY': lambda: self.fake.city(),
            'COUNTRY': lambda: self.fake.country(),
            'ZIP': lambda: self.fake.zipcode(),
            'ZIPCODE': lambda: self.fake.zipcode(),
        }
        
        return generators.get(pii_type, lambda: f'FAKE_{pii_type}')()
    
    def mask_text(self, text: str) -> Tuple[str, Dict[str, str], List[str]]:
        """
        Mask PII in text with fake data
        
        Returns:
            - masked_text: Text with fake data
            - replacements: Dict mapping fake_value -> original_value
            - detected: List of detected PII types
        """
        masked = text
        replacements = {}
        detected = []
        
        # Detect and replace each PII type
        for pii_type, pattern in self.patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if match not in replacements.values():
                    fake_value = self.generate_fake(pii_type)
                    masked = masked.replace(match, fake_value, 1)
                    replacements[fake_value] = match
                    detected.append(pii_type)
        
        # Detect names (context-based)
        name_patterns = [
            r'my name is\s+(\w+(?:\s+\w+)*)',
            r'i am\s+(\w+(?:\s+\w+)*)',
            r'myself\s+(\w+(?:\s+\w+)*)',
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                original_name = match.group(1)
                if original_name not in replacements.values():
                    fake_name = self.generate_fake('NAME')
                    masked = masked.replace(original_name, fake_name, 1)
                    replacements[fake_name] = original_name
                    detected.append('NAME')
                break
        
        return masked, replacements, detected
    
    def unmask_text(self, masked_text: str, replacements: Dict[str, str]) -> str:
        """Restore original values from masked text"""
        unmasked = masked_text
        for fake_value, original_value in replacements.items():
            unmasked = unmasked.replace(fake_value, original_value)
        return unmasked
