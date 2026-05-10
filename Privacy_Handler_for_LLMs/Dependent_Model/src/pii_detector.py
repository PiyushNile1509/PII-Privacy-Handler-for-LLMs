import re
import numpy as np
from typing import Dict, List, Tuple, Any
from faker import Faker
import random
# Fallback name detection without spaCy
nlp = None

class PIIDetector:
    """PII Detection and Entity Recognition"""
    
    def __init__(self):
        self.fake = Faker()
        self.pii_patterns = self._build_pii_patterns()
        self.replacement_cache = {}
        
    def _build_pii_patterns(self) -> Dict[str, str]:
        """Build regex patterns for comprehensive PII detection"""
        patterns = {
            # Direct PII - High sensitivity
            'email_address': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone_number': r'(?:\+\d{1,3}[\s.-]?)?(?:\(?\d{3,4}\)?[\s.-]?)?\d{3,4}[\s.-]?\d{4,6}(?:x\d+)?',
            'date_of_birth': r'\b(?:born|birth|dob)\s+(?:on\s+)?(?:0[1-9]|[12][0-9]|3[01])[-/](?:0[1-9]|1[0-2])[-/](?:19|20)\d{2}\b',
            'health_id': r'\b(?:MRN|patient\s+id|medical\s+record)[-\s]*([A-Z]{2,4}[-]?\d{4,8})\b',
            'student_id': r'\b(?:student\s+id|stu)[-\s]*([A-Z]{2,4}[-]?\d{4,8})\b',
            'account_number': r'\b(?:account\s+number|acc)\s+(?:is\s+)?([0-9]{4}\s+[0-9]{4}\s+[0-9]{4}\s+[0-9]{4})\b',
            'aadhar_number': r'\b\d{4}\s+\d{4}\s+\d{4}\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            
            # Indirect PII - Contextual
            'age_indirect': r'\b(?:age|aged?|years?\s+old)\s+(\d{1,3})\b',
            'city': r'\b(?:from|in|at|live\s+in)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)(?:\s*,|\s*\.|$)',
            'job_title': r'\b(?:working\s+as|work\s+as|job|profession|occupation)\s+(?:a\s+)?([a-z\s]+?)(?:\s+in|\s+at|\.|,|$)',
            'employer': r'\b(?:work\s+at|employed\s+at|company)\s+([A-Z][a-zA-Z\s]+?)(?:\s+as|\.|,|$)',
            'education_field': r'\b(?:studying|major|degree\s+in)\s+([A-Z][a-zA-Z\s]+?)(?:\.|,|$)',
            
            # Financial Information
            'credit_card_number': r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3[0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b',
            'bank_account_number': r'\b\d{8,17}\b',
            'ssn': r'\b(?:XXX-XX-\d{4}|\d{3}-\d{2}-\d{4})\b',
            
            # Medical Information
            'medical_record_number': r'\b(?:MRN|MR)[-\s]?[A-Z0-9]{5,10}\b',
            'patient_id': r'\b(?:P|PT|PAT)[-\s]?\d{3,8}\b',
            
            'drivers_license': r'\b[A-Z]{1,2}\d{6,8}\b',
            'passport_number': r'\b[A-Z]{1,2}\d{6,9}\b',
            
            # Digital Identifiers
            'ip_address': r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
            'mac_address': r'\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b',
            
            # Location Information
            'address': r'\b\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd|Way|Place|Pl)\b',
            'zip_code': r'\b\d{5}(?:-\d{4})?\b',
            
            # Educational
            'student_id': r'\b(?:S|ST|STU)[-\s]?\d{4,8}\b',
            
            # Business
            'employee_id': r'\b(?:EMP|E)[-\s]?\d{3,8}\b',
            'account_number': r'\b(?:AC|ACC|ACCT)[-\s]?\d{6,12}\b',
        }
        return patterns
    
    def detect_pii_entities(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """Detect both direct and indirect PII entities"""
        direct_pii = []
        indirect_pii = []
        
        # Detect names using context-aware patterns
        name_entities = self._detect_names_contextually(text)
        direct_pii.extend(name_entities)
        
        # Use regex patterns for all PII types
        for pii_type, pattern in self.pii_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if not self._overlaps_with_existing(match.start(), match.end(), direct_pii + indirect_pii):
                    entity = {
                        'type': pii_type,
                        'value': match.group(1) if match.groups() else match.group(),
                        'start': match.start(1) if match.groups() else match.start(),
                        'end': match.end(1) if match.groups() else match.end(),
                        'confidence': 0.9
                    }
                    
                    # Categorize as direct or indirect PII
                    if pii_type in ['age_indirect', 'city', 'job_title', 'employer', 'education_field']:
                        indirect_pii.append(entity)
                    else:
                        direct_pii.append(entity)
        
        return {
            'direct_pii': direct_pii,
            'indirect_pii': indirect_pii,
            'all_entities': direct_pii + indirect_pii
        }
    
    def _detect_names_contextually(self, text: str) -> List[Dict[str, Any]]:
        """Detect person names using contextual patterns"""
        entities = []
        
        # Context patterns that indicate a name follows
        name_contexts = [
            r"(?:I'm|I am|my name is|name is|called|this is)\s+([A-Za-z]+\s+[A-Za-z]+)",
            r"(?:contact|call|ask|meet|see)\s+([A-Za-z]+\s+[A-Za-z]+)(?:\s+at|\s+for|\s*,|\s*\.|$)",
            r"(?:Dr\.|Mr\.|Mrs\.|Ms\.|Miss)\s+([A-Za-z]+(?:\s+[A-Za-z]+)*)",
        ]
        
        for pattern in name_contexts:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                name_candidate = match.group(1)
                if self._is_valid_person_name(name_candidate):
                    entities.append({
                        'type': 'full_name',
                        'value': name_candidate,
                        'start': match.start(1),
                        'end': match.end(1),
                        'confidence': 0.85
                    })
        
        return entities
    
    def _is_valid_person_name(self, text: str) -> bool:
        """Validate if text is likely a person name"""
        # Common non-name words
        non_names = {
            'and', 'or', 'the', 'is', 'are', 'was', 'were', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
            'can', 'must', 'shall', 'email', 'phone', 'number', 'address', 'aadhar',
            'aadhaar', 'my', 'your', 'his', 'her', 'their', 'our', 'account', 'system',
            'working', 'fine', 'correct', 'valid', 'accurate', 'complete', 'format',
            'balance', 'good', 'information', 'help', 'assist', 'check', 'please'
        }
        
        words = text.lower().split()
        # If any word is in non-names list, it's not a valid name
        if any(word in non_names for word in words):
            return False
            
        # Must have at least 2 words for full name
        if len(words) < 2:
            return False
            
        # Each word should be alphabetic
        original_words = text.split()
        for word in original_words:
            if not word or not word.isalpha():
                return False
                
        # Common name patterns
        if len(words) == 2:  # First Last
            return True
        elif len(words) == 3:  # First Middle Last or Title First Last
            return True
            
        return len(words) <= 4  # Allow up to 4 words for names
    
    def _overlaps_with_existing(self, start: int, end: int, entities: List[Dict]) -> bool:
        """Check if position overlaps with existing entities"""
        for entity in entities:
            if not (end <= entity['start'] or start >= entity['end']):
                return True
        return False
    
    def generate_fake_replacement(self, pii_type: str, original_value: str = None) -> str:
        """Generate fake replacement for PII"""
        
        # Use cached replacement if available
        cache_key = f"{pii_type}_{original_value}"
        if cache_key in self.replacement_cache:
            return self.replacement_cache[cache_key]
        
        fake_value = None
        
        if pii_type == 'full_name':
            fake_value = self.fake.name()
        elif pii_type == 'email_address':
            fake_value = self.fake.email()
        elif pii_type == 'phone_number':
            fake_value = self.fake.phone_number()
        elif pii_type == 'date_of_birth':
            fake_value = self.fake.date_of_birth().strftime('%m/%d/%Y')
        elif pii_type == 'age':
            fake_value = str(random.randint(18, 80))
        elif pii_type == 'credit_card_number':
            fake_value = self.fake.credit_card_number()
        elif pii_type == 'bank_account_number':
            fake_value = ''.join([str(random.randint(0, 9)) for _ in range(10)])
        elif pii_type == 'ssn':
            fake_value = f"XXX-XX-{random.randint(1000, 9999)}"
        elif pii_type == 'medical_record_number':
            fake_value = f"MRN-{random.randint(10000, 99999)}"
        elif pii_type == 'patient_id':
            fake_value = f"P-{random.randint(1000, 9999)}"
        elif pii_type == 'drivers_license':
            fake_value = f"DL{random.randint(100000, 999999)}"
        elif pii_type == 'passport_number':
            fake_value = f"P{random.randint(1000000, 9999999)}"
        elif pii_type == 'ip_address':
            fake_value = self.fake.ipv4()
        elif pii_type == 'mac_address':
            fake_value = self.fake.mac_address()
        elif pii_type == 'address':
            fake_value = self.fake.street_address()
        elif pii_type == 'zip_code':
            fake_value = self.fake.zipcode()
        elif pii_type == 'student_id':
            fake_value = f"S-{random.randint(10000, 99999)}"
        elif pii_type == 'employee_id':
            fake_value = f"EMP{random.randint(1000, 9999)}"
        elif pii_type == 'account_number':
            fake_value = ''.join([str(random.randint(0, 9)) for _ in range(8)])
        elif pii_type == 'aadhar_number':
            fake_value = f"{random.randint(1000, 9999)} {random.randint(1000, 9999)} {random.randint(1000, 9999)}"
        elif pii_type == 'health_id':
            fake_value = f"MRN-{random.randint(10000, 99999)}"
        elif pii_type == 'student_id':
            fake_value = f"STU-{random.randint(10000, 99999)}"
        elif pii_type == 'age_indirect':
            fake_value = str(random.randint(20, 65))
        elif pii_type == 'city':
            cities = ['Boston', 'Chicago', 'Seattle', 'Denver', 'Austin']
            fake_value = random.choice(cities)
        elif pii_type == 'job_title':
            jobs = ['consultant', 'analyst', 'specialist', 'coordinator', 'manager']
            fake_value = random.choice(jobs)
        elif pii_type == 'employer':
            companies = ['TechCorp', 'DataSystems', 'GlobalTech', 'InnovateCo', 'FutureTech']
            fake_value = random.choice(companies)
        elif pii_type == 'education_field':
            fields = ['Business Administration', 'Information Technology', 'Engineering', 'Data Science']
            fake_value = random.choice(fields)
        else:
            fake_value = f"[{pii_type.upper()}]"
        
        # Cache the replacement
        self.replacement_cache[cache_key] = fake_value
        return fake_value
    
    def mask_pii_entities(self, text: str, entities_to_mask: List[str], mask_indirect: bool = False) -> Tuple[str, Dict[str, str]]:
        """Mask specified PII entities in text with support for indirect PII"""
        masked_text = text
        replacements = {}
        
        # Detect all PII entities
        pii_results = self.detect_pii_entities(text)
        
        # Combine entities based on masking preferences
        entities_to_process = pii_results['direct_pii'].copy()
        if mask_indirect:
            entities_to_process.extend(pii_results['indirect_pii'])
        
        # Sort by position (reverse order to maintain indices)
        entities_to_process.sort(key=lambda x: x['start'], reverse=True)
        
        for entity in entities_to_process:
            if entity['type'] in entities_to_mask:
                original_value = entity['value']
                fake_value = self.generate_fake_replacement(entity['type'], original_value)
                
                # Replace in text
                masked_text = (
                    masked_text[:entity['start']] + 
                    fake_value + 
                    masked_text[entity['end']:]
                )
                
                # Store replacement with unique key
                key = f"{entity['type']}_{len(replacements)}"
                replacements[key] = f"{original_value} -> {fake_value}"
        
        return masked_text, replacements

class ContextAnalyzer:
    """Analyze context to determine which PII entities should be masked"""
    
    def __init__(self):
        self.context_keywords = {
            'mathematical': ['calculate', 'sum', 'add', 'multiply', 'divide', 'digits', 'number'],
            'location_based': ['nearest', 'location', 'address', 'directions', 'map', 'distance'],
            'medical': ['patient', 'doctor', 'medical', 'prescription', 'diagnosis', 'treatment'],
            'financial': ['account', 'balance', 'payment', 'transaction', 'bank', 'credit'],
            'educational': ['student', 'grade', 'course', 'school', 'university', 'transcript'],
            'general': ['weather', 'information', 'help', 'question', 'tell me']
        }
    
    def analyze_context(self, text: str) -> str:
        """Determine the context/domain of the query"""
        text_lower = text.lower()
        
        domain_scores = {}
        for domain, keywords in self.context_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                domain_scores[domain] = score
        
        if domain_scores:
            return max(domain_scores, key=domain_scores.get)
        return 'general'
    
    def determine_masking_necessity(self, text: str, pii_results: Dict) -> Dict[str, Dict[str, bool]]:
        """Determine which PII entities should be masked based on context"""
        context = self.analyze_context(text)
        text_lower = text.lower()
        
        direct_decisions = {}
        indirect_decisions = {}
        
        # Process direct PII
        for entity in pii_results['direct_pii']:
            entity_type = entity['type']
            should_mask = True  # Default to masking for privacy
            
            # Context-specific rules for direct PII
            if context == 'mathematical':
                if entity_type in ['phone_number', 'account_number']:
                    if any(word in text_lower for word in ['calculate', 'sum', 'add', 'digits']):
                        should_mask = False
            elif context == 'medical':
                if entity_type in ['health_id', 'date_of_birth']:
                    should_mask = False
            elif context == 'educational':
                if entity_type in ['student_id']:
                    should_mask = False
            elif context == 'financial':
                if entity_type in ['account_number']:
                    if any(word in text_lower for word in ['balance', 'transaction']):
                        should_mask = False
            
            direct_decisions[entity_type] = should_mask
        
        # Process indirect PII - generally preserve for context
        for entity in pii_results['indirect_pii']:
            entity_type = entity['type']
            # Indirect PII is usually preserved unless explicitly sensitive
            should_mask = entity_type in ['age_indirect']  # Only mask age by default
            indirect_decisions[entity_type] = should_mask
        
        return {
            'direct_pii': direct_decisions,
            'indirect_pii': indirect_decisions
        }