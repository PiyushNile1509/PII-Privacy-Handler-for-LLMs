#!/usr/bin/env python3
"""
🔒 PII Detection, Classification & Masking System - Final Year Project
Complete implementation with contextual dependency analysis
"""

import os
import sys
import json
import pickle
import re
import logging
from typing import Dict, List, Any, Tuple, Optional
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

# Try to import optional dependencies
try:
    import pandas as pd
except ImportError:
    pd = None
    
try:
    import numpy as np
except ImportError:
    np = None

try:
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
except ImportError:
    # Fallback implementations
    def accuracy_score(y_true, y_pred):
        correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
        return correct / len(y_true) if y_true else 0
    
    def precision_recall_fscore_support(y_true, y_pred, average='weighted'):
        # Simple fallback
        acc = accuracy_score(y_true, y_pred)
        return acc, acc, acc, None

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.pii_detector import PIIDetector, ContextAnalyzer
    from src.data_processor import DataProcessor
    from src.privacy_handler import PIIPrivacyHandler
except ImportError:
    print("Warning: Some modules not found, using fallback implementations")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensivePIIModel:
    """
    🧠 Complete PII Detection, Classification & Contextual Masking System
    Handles all 16+ PII domains with dependency analysis
    """
    
    def __init__(self):
        self.pii_patterns = self._build_comprehensive_patterns()
        self.computation_keywords = self._build_computation_keywords()
        self.dependency_patterns = self._build_dependency_patterns()
        self.feature_weights = {}
        self.training_stats = {}
        self.model_ready = False
        
        # Initialize fallback components if imports fail
        try:
            self.pii_detector = PIIDetector()
            self.context_analyzer = ContextAnalyzer()
        except:
            self.pii_detector = None
            self.context_analyzer = None
            logger.warning("Using fallback PII detection")
    
    def _build_comprehensive_patterns(self) -> Dict[str, List[str]]:
        """🧩 Build comprehensive PII patterns for all 16+ domains"""
        return {
            # General Identity - Enhanced patterns for better name detection
            'NAME': [
                # Standard name patterns
                r'\b(?:my name is|I am|I\'m|called|This is)\s+([A-Za-z]+(?:\s+[A-Za-z]+)*)',
                r'\b(?:myself|hello)\s+([A-Za-z]+(?:\s+[A-Za-z]+)*)',
                r'\b(Dr\.|Prof\.|Mr\.|Ms\.|Mrs\.)\s+([A-Za-z]+(?:\s+[A-Za-z]+)*)',
                
                # Context-based name patterns
                r'\bpatient\s+([A-Za-z]+(?:\s+[A-Za-z]+)*)',
                r'\buser\s+([A-Za-z]+(?:\s+[A-Za-z]+)*)',
                r'\bcustomer\s+([A-Za-z]+(?:\s+[A-Za-z]+)*)',
                r'\bemployee\s+([A-Za-z]+(?:\s+[A-Za-z]+)*)',
                r'\bstudent\s+([A-Za-z]+(?:\s+[A-Za-z]+)*)',
                
                # Standalone name patterns (common Indian names)
                r'\b(Piyush|Rahul|Priya|Amit|Sneha|Rajesh|Pooja|Vikram|Anita|Suresh|Kavya|Arjun|Deepika|Ravi|Meera|Kiran|Sanjay|Nisha|Arun|Divya|Manoj|Shreya|Vishal|Asha|Rohit|Sunita|Ajay|Rekha|Nitin|Swati|Sachin|Geeta|Akash|Shilpa|Ramesh|Neha|Vinod|Preeti|Ashok|Smita|Sunil|Ritu|Prakash|Seema|Anil|Jyoti|Mukesh|Vandana|Rajeev|Kavita|Mahesh|Sushma|Naveen|Rashmi|Dinesh|Shweta|Yogesh|Pallavi|Hemant|Archana|Subhash|Madhuri|Santosh|Usha|Mohan|Sarita|Kishore|Anupama|Jagdish|Sudha|Naresh|Bharti|Ganesh|Lata|Sudhir|Manju|Brijesh|Kalpana|Devendra|Sunanda|Narayan|Pushpa|Govind|Shanti|Hari|Kamala|Shyam|Radha|Krishna|Sita|Rama|Gita|Monika|Sonal|Payal|Richa|Simran|Tanya|Varun|Yash|Zara)\b',
                
                # Generic name patterns for any alphabetic sequence in name contexts
                r'\bname\s+(?:is\s+)?([A-Za-z]+(?:\s+[A-Za-z]+)*)',
                r'\bcalled\s+([A-Za-z]+(?:\s+[A-Za-z]+)*)',
                r'\bhi\s+(?:i\'m|i am)\s+([A-Za-z]+)',
                r'\bhello\s+(?:i\'m|i am)\s+([A-Za-z]+)',
                
                # Catch common name introductions
                r'^([A-Za-z]+)\s+(?:here|speaking)\b',
                r'\bthis is\s+([A-Za-z]+)\b'
            ],
            
            # Contact Information
            'PHONE': [
                r'\+91\s*(\d{10})',
                r'\+44\s*(\d{4}\s*\d{6})',
                r'\((\d{3})\)\s*(\d{3}-\d{4})',
                r'\b(\d{10})\b',
                r'phone\s*(?:number)?\s*(?:is)?\s*([+\d\s-]{10,15})',
                r'contact\s*(?:number)?\s*(?:is)?\s*([+\d\s-]{10,15})',
                r'mobile\s*(?:number)?\s*(?:is)?\s*([+\d\s-]{10,15})'
            ],
            
            'EMAIL': [
                r'\b([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b',
                r'email\s*(?:is|id)?\s*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})',
                r'mailto:([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'
            ],
            
            # Government IDs
            'AADHAAR': [
                r'\bAadhaar\s*(?:number)?\s*(?:is)?\s*(\d{4}\s*\d{4}\s*\d{4})',
                r'\bUID\s*(?:number)?\s*(?:is)?\s*(\d{4}\s*\d{4}\s*\d{4})',
                r'\b(\d{4}\s*\d{4}\s*\d{4})\b'
            ],
            
            'PAN': [
                r'\bPAN\s*(?:number)?\s*(?:is)?\s*([A-Z]{5}\d{4}[A-Z])',
                r'\b([A-Z]{5}\d{4}[A-Z])\b'
            ],
            
            'PASSPORT': [
                r'\bPassport\s*(?:number)?\s*(?:is)?\s*([A-Z]\d{7})',
                r'\bPassport\s*(?:number)?\s*(?:is)?\s*([A-Z]{2}\d{7})',
                r'\b([A-Z]\d{7})\b',
                r'\b([A-Z]{2}\d{7})\b'
            ],
            
            'SSN': [
                r'\bSSN\s*(?:number)?\s*(?:is)?\s*(\d{3}-\d{2}-\d{4})',
                r'\bSocial Security\s*(?:number)?\s*(?:is)?\s*(\d{3}-\d{2}-\d{4})',
                r'\b(\d{3}-\d{2}-\d{4})\b'
            ],
            
            # Financial
            'ACCOUNT_NUMBER': [
                r'\baccount\s*(?:number)?\s*(?:is)?\s*(\d{8,16})',
                r'\bbank account\s*(?:number)?\s*(?:is)?\s*(\d{8,16})',
                r'\baccount\s*:\s*(\d{8,16})'
            ],
            
            'CREDIT_CARD': [
                r'\b(\d{4}\s*\d{4}\s*\d{4}\s*\d{4})\b',
                r'\b(\d{4}-\d{4}-\d{4}-\d{4})\b',
                r'card\s*(?:number)?\s*(?:is)?\s*(\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4})'
            ],
            
            'IFSC': [
                r'\bIFSC\s*(?:code)?\s*(?:is)?\s*([A-Z]{4}0[A-Z0-9]{6})',
                r'\b([A-Z]{4}0[A-Z0-9]{6})\b'
            ],
            
            # Corporate/Employment
            'EMPLOYEE_ID': [
                r'\bEmployee\s*(?:ID)?\s*(?:is)?\s*([A-Z]{2,4}-?\d{4,8})',
                r'\bEmp\s*(?:ID)?\s*(?:is)?\s*([A-Z]{2,4}-?\d{4,8})',
                r'\bstaff\s*(?:ID)?\s*(?:is)?\s*([A-Z]{2,4}-?\d{4,8})'
            ],
            
            # Education
            'STUDENT_ID': [
                r'\bstudent\s*(?:ID)?\s*(?:is)?\s*([A-Z]{2,4}-?\d{4,8})',
                r'\bRoll\s*(?:no|number)?\s*(?:is)?\s*(\d{4,10})',
                r'\benrollment\s*(?:number)?\s*(?:is)?\s*([A-Z]{2,4}\d{4,8})'
            ],
            
            # Healthcare
            'PATIENT_ID': [
                r'\bPatient\s*(?:ID)?\s*(?:is)?\s*([A-Z]{2,4}-?\d{4,8})',
                r'\bMRN\s*(?:number)?\s*(?:is)?\s*(MRN-?\d{4,8})',
                r'\bpatient\s*(?:code)?\s*(?:is)?\s*([A-Z]-?\d{4,8})'
            ],
            
            # Address
            'ADDRESS': [
                r'\baddress\s*(?:is)?\s*(.{10,100})',
                r'\blive\s*(?:at|in)\s*(.{5,50})',
                r'\bresiding\s*(?:at|in)\s*(.{5,50})'
            ],
            
            'CITY': [
                r'\bcity\s*(?:is)?\s*([A-Z][a-z]+)',
                r'\bfrom\s*([A-Z][a-z]+)',
                r'\bin\s*([A-Z][a-z]+)',
                r'\b(Mumbai|Delhi|Bangalore|Chennai|Kolkata|Hyderabad|Pune|Ahmedabad)\b'
            ],
            
            # Dates
            'DOB': [
                r'\bborn\s*(?:on)?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{4})',
                r'\bDOB\s*(?:is)?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{4})',
                r'\bdate of birth\s*(?:is)?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{4})'
            ],
            
            # Miscellaneous
            'IP_ADDRESS': [
                r'\bIP\s*(?:address)?\s*(?:is)?\s*(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})',
                r'\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b'
            ],
            
            'MAC_ADDRESS': [
                r'\bMAC\s*(?:address)?\s*(?:is)?\s*([0-9A-F]{2}:[0-9A-F]{2}:[0-9A-F]{2}:[0-9A-F]{2}:[0-9A-F]{2}:[0-9A-F]{2})',
                r'\b([0-9A-F]{2}:[0-9A-F]{2}:[0-9A-F]{2}:[0-9A-F]{2}:[0-9A-F]{2}:[0-9A-F]{2})\b'
            ]
        }
    
    def _build_computation_keywords(self) -> List[str]:
        """🧮 Build comprehensive computation keywords"""
        return [
            # Mathematical operations
            'count', 'calculate', 'add', 'sum', 'multiply', 'divide', 'subtract',
            'plus', 'minus', 'times', 'total', 'average', 'mean', 'median',
            
            # String operations
            'extract', 'find', 'get', 'retrieve', 'obtain', 'fetch',
            'letters', 'characters', 'digits', 'numbers', 'length', 'size',
            'first', 'last', 'initial', 'initials', 'reverse', 'uppercase', 'lowercase',
            
            # Validation operations
            'check', 'verify', 'validate', 'confirm', 'test', 'examine',
            'compare', 'match', 'equal', 'same', 'different',
            
            # Output operations
            'print', 'display', 'show', 'output', 'return', 'give', 'tell',
            'list', 'enumerate', 'format', 'convert',
            
            # Logical operations
            'if', 'whether', 'condition', 'when', 'while', 'until',
            'and', 'or', 'not', 'but', 'however',
            
            # Domain-specific
            'domain', 'username', 'extension', 'prefix', 'suffix',
            'vowels', 'consonants', 'alphanumeric', 'numeric', 'alphabetic',
            'checksum', 'hash', 'encode', 'decode', 'encrypt', 'decrypt'
        ]
    
    def _build_dependency_patterns(self) -> List[str]:
        """🔗 Build patterns that indicate PII dependency"""
        return [
            r'count\s+(?:letters|characters|digits)\s+in',
            r'(?:add|sum|calculate)\s+(?:all\s+)?digits',
            r'extract\s+(?:domain|username)\s+from',
            r'find\s+(?:initials|first|last)',
            r'check\s+if\s+.*\s+(?:has|contains|is)',
            r'verify\s+(?:if|whether)',
            r'validate\s+(?:if|whether)',
            r'get\s+(?:last|first)\s+\d+',
            r'reverse\s+(?:the\s+)?(?:string|text|name)',
            r'length\s+of',
            r'how\s+many\s+(?:letters|digits|characters)',
            r'total\s+(?:letters|digits|characters)',
            r'(?:uppercase|lowercase)\s+(?:the\s+)?(?:name|text)',
            r'format\s+(?:the\s+)?(?:number|phone|date)'
        ]
    
    def detect_pii_entities(self, text: str) -> List[Dict[str, Any]]:
        """🔍 Detect all PII entities in text"""
        entities = []
        
        # Use existing detector if available
        if self.pii_detector:
            try:
                pii_results = self.pii_detector.detect_pii_entities(text)
                all_entities = pii_results.get('direct_pii', []) + pii_results.get('indirect_pii', [])
                
                for entity in all_entities:
                    entities.append({
                        'type': entity.get('type', 'UNKNOWN'),
                        'text': entity.get('value', ''),
                        'start': entity.get('start', 0),
                        'end': entity.get('end', 0),
                        'confidence': entity.get('confidence', 0.8),
                        'source': 'existing_detector'
                    })
            except Exception as e:
                logger.warning(f"Existing detector failed: {e}")
        
        # Use pattern-based detection
        for pii_type, patterns in self.pii_patterns.items():
            for pattern in patterns:
                try:
                    matches = re.finditer(pattern, text, re.IGNORECASE)
                    for match in matches:
                        groups = match.groups()
                        if groups:
                            for i, group in enumerate(groups):
                                if group and group.strip():
                                    start_idx = match.start(i+1)
                                    end_idx = match.end(i+1)
                                    
                                    # Check for overlaps
                                    overlap = any(
                                        (e['start'] <= start_idx < e['end']) or 
                                        (e['start'] < end_idx <= e['end']) or
                                        (start_idx <= e['start'] < end_idx)
                                        for e in entities
                                    )
                                    
                                    if not overlap:
                                        entities.append({
                                            'type': pii_type,
                                            'text': group.strip(),
                                            'start': start_idx,
                                            'end': end_idx,
                                            'confidence': 0.9,
                                            'source': 'pattern_based'
                                        })
                                    break
                except Exception as e:
                    logger.warning(f"Pattern matching failed for {pii_type}: {e}")
        
        # Sort by start position
        entities.sort(key=lambda x: x['start'])
        return entities
    
    def analyze_dependency(self, text: str, entity: Dict[str, Any]) -> Tuple[bool, str]:
        """🧠 Analyze if PII entity is required for computation/logic"""
        text_lower = text.lower()
        entity_text = entity['text'].lower()
        entity_type = entity['type']
        
        # Default: mask all PII unless specifically needed
        dependency_found = False
        dependency_reason = "Sensitive information, not required for task"
        
        # Check for direct computation keywords
        computation_indicators = 0
        for keyword in self.computation_keywords:
            if keyword in text_lower:
                computation_indicators += 1
        
        # Check for dependency patterns
        for pattern in self.dependency_patterns:
            if re.search(pattern, text_lower):
                dependency_found = True
                dependency_reason = f"Required for computational task: {pattern}"
                break
        
        # Specific dependency checks only if computation context exists
        if not dependency_found and computation_indicators > 0:
            
            # Mathematical operations on numeric entities
            if any(op in text_lower for op in ['add', 'sum', 'calculate', 'multiply', 'divide', 'subtract', 'plus', 'minus']):
                if entity_type in ['PHONE', 'ACCOUNT_NUMBER', 'CREDIT_CARD', 'AADHAAR'] and any(target in text_lower for target in ['digit', 'number']):
                    dependency_found = True
                    dependency_reason = "Required for mathematical operation on digits"
            
            # String operations on text entities
            if any(op in text_lower for op in ['count', 'length', 'letters', 'characters', 'vowels', 'consonants']):
                if entity_type in ['NAME', 'EMAIL', 'ADDRESS'] and (entity_text in text_lower or 'name' in text_lower):
                    dependency_found = True
                    dependency_reason = "Required for string analysis operation"
            
            # Validation operations
            if any(op in text_lower for op in ['check', 'verify', 'validate', 'confirm']):
                if any(target in text_lower for target in ['format', 'valid', 'correct', 'length', 'digit', 'character']):
                    dependency_found = True
                    dependency_reason = "Required for validation operation"
            
            # Extraction operations
            if any(op in text_lower for op in ['extract', 'get', 'find', 'retrieve']):
                if 'domain' in text_lower and entity_type == 'EMAIL':
                    dependency_found = True
                    dependency_reason = "Required for domain extraction"
                elif any(target in text_lower for target in ['initial', 'first', 'last']) and entity_type == 'NAME':
                    dependency_found = True
                    dependency_reason = "Required for name component extraction"
            
            # Format/conversion operations
            if any(op in text_lower for op in ['format', 'convert', 'transform', 'reverse', 'uppercase', 'lowercase']):
                if entity_type in ['NAME', 'EMAIL', 'PHONE', 'ADDRESS']:
                    dependency_found = True
                    dependency_reason = "Required for format/conversion operation"
        
        # Special case: Simple introductions should always be masked
        simple_intro_patterns = [
            r'^\s*(?:myself|hello|hi|hey)\s+[a-zA-Z]+\s*$',
            r'^\s*(?:i am|i\'m|my name is)\s+[a-zA-Z]+\s*$',
            r'^\s*[a-zA-Z]+\s+(?:here|speaking)\s*$'
        ]
        
        for pattern in simple_intro_patterns:
            if re.match(pattern, text.strip(), re.IGNORECASE):
                dependency_found = False
                dependency_reason = "Simple introduction - name should be masked for privacy"
                break
        
        return dependency_found, dependency_reason
    
    def mask_text(self, text: str, entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """🎭 Generate masked text with dependency analysis"""
        masked_text = text
        pii_entities = []
        offset = 0
        
        for entity in entities:
            is_dependent, reason = self.analyze_dependency(text, entity)
            
            if is_dependent:
                # Keep original text
                pii_entities.append({
                    'entity': entity['text'],
                    'type': entity['type'],
                    'masked': False,
                    'reason': reason,
                    'start': entity['start'],
                    'end': entity['end'],
                    'confidence': entity['confidence']
                })
            else:
                # Mask the entity
                placeholder = f"[{entity['type']}]"
                start_pos = entity['start'] + offset
                end_pos = entity['end'] + offset
                
                masked_text = masked_text[:start_pos] + placeholder + masked_text[end_pos:]
                offset += len(placeholder) - (entity['end'] - entity['start'])
                
                pii_entities.append({
                    'entity': entity['text'],
                    'type': entity['type'],
                    'masked': True,
                    'reason': "Sensitive information, not required for task",
                    'start': entity['start'],
                    'end': entity['end'],
                    'confidence': entity['confidence']
                })
        
        return {
            'original_text': text,
            'masked_text': masked_text,
            'pii_entities': pii_entities,
            'total_entities': len(entities),
            'masked_entities': len([e for e in pii_entities if e['masked']]),
            'preserved_entities': len([e for e in pii_entities if not e['masked']])
        }
    
    def extract_features(self, text: str) -> Dict[str, float]:
        """🔧 Extract comprehensive features for training"""
        text_lower = text.lower()
        features = {}
        
        # Computation keyword features
        for keyword in self.computation_keywords:
            features[f'has_{keyword}'] = 1.0 if keyword in text_lower else 0.0
        
        # Dependency pattern features
        for i, pattern in enumerate(self.dependency_patterns):
            features[f'dependency_pattern_{i}'] = 1.0 if re.search(pattern, text_lower) else 0.0
        
        # PII type features
        entities = self.detect_pii_entities(text)
        for pii_type in self.pii_patterns.keys():
            features[f'has_{pii_type.lower()}'] = 1.0 if any(e['type'] == pii_type for e in entities) else 0.0
        
        # Structural features
        features['text_length'] = len(text) / 100.0
        features['word_count'] = len(text.split()) / 20.0
        features['sentence_count'] = len([s for s in text.split('.') if s.strip()]) / 5.0
        features['has_question'] = 1.0 if '?' in text else 0.0
        features['has_numbers'] = 1.0 if any(c.isdigit() for c in text) else 0.0
        features['has_punctuation'] = 1.0 if any(c in '.,!?;:' for c in text) else 0.0
        
        # Linguistic features
        features['question_words'] = 1.0 if any(w in text_lower for w in ['what', 'how', 'when', 'where', 'why', 'which']) else 0.0
        features['imperative_words'] = 1.0 if any(w in text_lower for w in ['calculate', 'find', 'get', 'show', 'tell']) else 0.0
        features['math_operations'] = 1.0 if any(op in text_lower for op in ['+', '-', '*', '/', 'add', 'subtract']) else 0.0
        
        return features
    
    def load_training_data(self) -> List[Dict]:
        """📚 Load training data from all available sources"""
        all_data = []
        base_dir = os.path.dirname(__file__)
        
        # Load from main CSV (if pandas available)
        if pd is not None:
            csv_path = os.path.join(base_dir, 'pii_100000_new.csv')
            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path)
                    logger.info(f"Loading from main CSV: {len(df)} rows")
                    
                    for _, row in df.iterrows():
                        query = str(row.get('query', ''))
                        reason = str(row.get('reason_for_not_replaced', ''))
                        domain = str(row.get('domain', 'unknown'))
                        
                        if len(query.strip()) > 5:  # Valid query
                            # Determine label based on reason
                            label = 1 if any(keyword in reason.lower() for keyword in [
                                'required', 'needed', 'necessary', 'operation', 'calculation',
                                'task', 'counting', 'mathematical', 'computation', 'used'
                            ]) else 0
                            
                            all_data.append({
                                'text': query,
                                'label': label,
                                'source': 'main_csv',
                                'domain': domain,
                                'reason': reason
                            })
                    
                    logger.info(f"Loaded {len(all_data)} samples from main CSV")
                except Exception as e:
                    logger.error(f"Error loading main CSV: {e}")
        
        # Load from comprehensive CSV (if pandas available)
        if pd is not None:
            comp_csv_path = os.path.join(base_dir, 'models', 'comprehensive_training_data.csv')
            if os.path.exists(comp_csv_path):
                try:
                    df = pd.read_csv(comp_csv_path)
                    comp_data = []
                    
                    for _, row in df.iterrows():
                        text = str(row.get('text', ''))
                        if len(text.strip()) > 5:
                            comp_data.append({
                                'text': text,
                                'label': int(row.get('label', 0)),
                                'source': 'comprehensive_csv',
                                'entity_type': str(row.get('entity_type', 'unknown')),
                                'reason': str(row.get('reason', ''))
                            })
                    
                    all_data.extend(comp_data)
                    logger.info(f"Loaded {len(comp_data)} samples from comprehensive CSV")
                except Exception as e:
                    logger.error(f"Error loading comprehensive CSV: {e}")
        
        # Generate synthetic training data
        synthetic_data = self._generate_synthetic_data()
        all_data.extend(synthetic_data)
        logger.info(f"Generated {len(synthetic_data)} synthetic samples")
        
        logger.info(f"Total training data: {len(all_data)} samples")
        return all_data
    
    def _generate_synthetic_data(self) -> List[Dict]:
        """🎯 Generate synthetic training data"""
        synthetic_data = []
        
        # Computation examples (label = 1)
        computation_examples = [
            ("My name is John Smith, count the letters in my name", 1),
            ("I am Alice, calculate sum of digits in my phone 1234567890", 1),
            ("Extract domain from email user@example.com", 1),
            ("Find initials of Sarah Johnson", 1),
            ("Count vowels in name Michael Brown", 1),
            ("Get last 4 digits of account 123456789", 1),
            ("Reverse the string Hello World", 1),
            ("Calculate checksum for PAN ABCDE1234F", 1),
            ("Add all digits from Aadhaar 1234 5678 9012", 1),
            ("Verify if phone 9876543210 has 10 digits", 1),
            ("Check if email test@domain.com is valid format", 1),
            ("Find length of password abc123xyz", 1),
            ("Convert name JOHN to lowercase", 1),
            ("Extract username from email john.doe@company.com", 1),
            ("Count characters in address 123 Main Street", 1)
        ]
        
        # Non-computation examples (label = 0) - Enhanced with more intro patterns
        non_computation_examples = [
            ("My name is John Smith", 0),
            ("I am Alice Johnson", 0),
            ("Myself Piyush", 0),
            ("Hello Piyush", 0),
            ("Hi I'm Piyush", 0),
            ("This is Piyush", 0),
            ("Piyush here", 0),
            ("Piyush speaking", 0),
            ("Contact me at user@example.com", 0),
            ("Patient Sarah needs appointment", 0),
            ("Employee Mike works here", 0),
            ("Student Bob enrolled in course", 0),
            ("Customer Jane placed an order", 0),
            ("Hello, I'm David from Mumbai", 0),
            ("This is Mary speaking", 0),
            ("User account: test@domain.com", 0),
            ("My phone number is 9876543210", 0),
            ("I live at 123 Main Street", 0),
            ("Born on 15/08/1990", 0),
            ("PAN number is ABCDE1234F", 0),
            ("Aadhaar number is 1234 5678 9012", 0),
            ("I am Rahul", 0),
            ("Myself Priya", 0),
            ("Hello Amit", 0),
            ("Hi Sneha", 0),
            ("This is Rajesh", 0),
            ("Pooja here", 0),
            ("Vikram speaking", 0)
        ]
        
        # Add all examples
        for text, label in computation_examples + non_computation_examples:
            synthetic_data.append({
                'text': text,
                'label': label,
                'source': 'synthetic',
                'type': 'computation' if label == 1 else 'non_computation'
            })
        
        return synthetic_data
    
    def train(self) -> Dict[str, Any]:
        """🚀 Train the comprehensive PII model"""
        logger.info("🚀 Starting comprehensive PII model training...")
        
        # Load training data
        training_data = self.load_training_data()
        
        if len(training_data) < 10:
            logger.error("Insufficient training data!")
            return {'success': False, 'error': 'Insufficient training data'}
        
        # Extract features and labels
        X = []
        y = []
        
        logger.info("🔧 Extracting features...")
        for sample in training_data:
            features = self.extract_features(sample['text'])
            X.append(features)
            y.append(sample['label'])
        
        # Calculate feature weights
        logger.info("⚖️ Calculating feature weights...")
        feature_counts = {'computation': Counter(), 'no_computation': Counter()}
        
        for i, sample in enumerate(training_data):
            features = X[i]
            category = 'computation' if y[i] == 1 else 'no_computation'
            
            for feature, value in features.items():
                if value > 0:
                    feature_counts[category][feature] += 1
        
        # Calculate weights using information gain
        total_computation = sum(y)
        total_no_computation = len(y) - total_computation
        
        all_features = set()
        for features in X:
            all_features.update(features.keys())
        
        for feature in all_features:
            comp_freq = feature_counts['computation'][feature] / max(total_computation, 1)
            no_comp_freq = feature_counts['no_computation'][feature] / max(total_no_computation, 1)
            
            # Information gain based weight
            if comp_freq + no_comp_freq > 0:
                weight = comp_freq - no_comp_freq
                # Add entropy bonus for discriminative features
                if comp_freq > 0 and no_comp_freq > 0:
                    entropy = -(comp_freq * np.log2(comp_freq + 1e-8) + no_comp_freq * np.log2(no_comp_freq + 1e-8))
                    weight *= (1 + entropy)
                
                self.feature_weights[feature] = weight
        
        # Store training statistics
        self.training_stats = {
            'total_samples': len(training_data),
            'computation_samples': total_computation,
            'no_computation_samples': total_no_computation,
            'feature_count': len(self.feature_weights),
            'data_sources': Counter([s.get('source', 'unknown') for s in training_data]),
            'domains': Counter([s.get('domain', 'unknown') for s in training_data]),
            'top_computation_features': sorted(
                [(f, w) for f, w in self.feature_weights.items() if w > 0], 
                key=lambda x: x[1], reverse=True
            )[:10],
            'top_masking_features': sorted(
                [(f, w) for f, w in self.feature_weights.items() if w < 0], 
                key=lambda x: x[1]
            )[:10]
        }
        
        self.model_ready = True
        
        # Evaluate on training data
        logger.info("📊 Evaluating model...")
        predictions = []
        confidences = []
        
        for sample in training_data:
            pred, conf, _ = self.predict(sample['text'])
            predictions.append(pred)
            confidences.append(conf)
        
        accuracy = accuracy_score(y, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(y, predictions, average='weighted')
        
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        evaluation_results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'avg_confidence': avg_conf
        }
        
        logger.info(f"✅ Training completed!")
        logger.info(f"📈 Accuracy: {accuracy:.3f}")
        logger.info(f"📈 F1-Score: {f1:.3f}")
        logger.info(f"📊 Data sources: {dict(self.training_stats['data_sources'])}")
        
        # Save model
        self._save_model()
        
        return {
            'success': True,
            'training_stats': self.training_stats,
            'evaluation': evaluation_results,
            'model_path': self._get_model_path()
        }
    
    def predict(self, text: str) -> Tuple[int, float, Dict[str, Any]]:
        """🔮 Predict if PII should be masked or preserved"""
        if not self.model_ready:
            logger.warning("Model not trained yet!")
            return 0, 0.5, {'error': 'Model not trained'}
        
        features = self.extract_features(text)
        
        # Calculate weighted score
        score = 0.0
        feature_contributions = {}
        
        for feature, value in features.items():
            if feature in self.feature_weights and value > 0:
                contribution = self.feature_weights[feature] * value
                score += contribution
                feature_contributions[feature] = contribution
        
        # Adaptive threshold
        threshold = 0.1
        prediction = 1 if score > threshold else 0
        confidence = min(abs(score) * 2, 1.0)  # Scale confidence
        
        # Detect PII entities
        entities = self.detect_pii_entities(text)
        
        analysis = {
            'score': score,
            'threshold': threshold,
            'prediction': prediction,
            'confidence': confidence,
            'pii_entities_detected': len(entities),
            'pii_types': [e['type'] for e in entities],
            'top_features': sorted(feature_contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:5],
            'entities': entities
        }
        
        return prediction, confidence, analysis
    
    def process_text(self, text: str) -> Dict[str, Any]:
        """🎯 Complete PII processing pipeline"""
        # Detect PII entities
        entities = self.detect_pii_entities(text)
        
        # Generate masked output
        result = self.mask_text(text, entities)
        
        # Add prediction analysis
        prediction, confidence, analysis = self.predict(text)
        result['dependency_analysis'] = {
            'requires_computation': prediction == 1,
            'confidence': confidence,
            'analysis': analysis
        }
        
        return result
    
    def _save_model(self):
        """💾 Save the trained model"""
        model_data = {
            'feature_weights': self.feature_weights,
            'training_stats': self.training_stats,
            'pii_patterns': self.pii_patterns,
            'computation_keywords': self.computation_keywords,
            'dependency_patterns': self.dependency_patterns,
            'model_ready': self.model_ready
        }
        
        model_path = self._get_model_path()
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"💾 Model saved to {model_path}")
    
    def _get_model_path(self) -> str:
        """Get model save path"""
        base_dir = os.path.dirname(__file__)
        return os.path.join(base_dir, 'models', 'final_pii_model.pkl')
    
    def load_model(self) -> bool:
        """📂 Load pre-trained model"""
        model_path = self._get_model_path()
        
        if not os.path.exists(model_path):
            logger.warning("No saved model found")
            return False
        
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.feature_weights = model_data.get('feature_weights', {})
            self.training_stats = model_data.get('training_stats', {})
            self.model_ready = model_data.get('model_ready', False)
            
            # Update patterns if saved
            if 'pii_patterns' in model_data:
                self.pii_patterns.update(model_data['pii_patterns'])
            if 'computation_keywords' in model_data:
                self.computation_keywords.extend(model_data['computation_keywords'])
            if 'dependency_patterns' in model_data:
                self.dependency_patterns.extend(model_data['dependency_patterns'])
            
            logger.info("📂 Model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

def main():
    """🎯 Main function for training and testing"""
    logger.info("🔒 PII Detection, Classification & Masking System - Final Year Project")
    logger.info("=" * 70)
    
    # Initialize model
    model = ComprehensivePIIModel()
    
    # Try to load existing model
    if not model.load_model():
        logger.info("🚀 Training new model...")
        training_result = model.train()
        
        if not training_result['success']:
            logger.error("❌ Training failed!")
            return
        
        logger.info("✅ Training completed successfully!")
    else:
        logger.info("📂 Loaded existing model")
    
    # Test cases for demonstration
    test_cases = [
        "My name is Prasad and my email is prasad@gmail.com",
        "Count letters in name Prasad",
        "I am John Smith, calculate sum of digits in my phone 9876543210",
        "Patient Sarah Johnson needs appointment",
        "Extract domain from email user@example.com",
        "My PAN number is ABCDE1234F",
        "Verify if PAN ABCDE1234F has 10 characters",
        "Add 100 to my account number 123456789",
        "I live in Mumbai and work at TCS",
        "Find initials of Dr. Rajesh Kumar"
    ]
    
    logger.info("\n🧪 Testing the model:")
    logger.info("=" * 70)
    
    for i, test_text in enumerate(test_cases, 1):
        logger.info(f"\n📝 Test Case {i}: {test_text}")
        
        # Process the text
        result = model.process_text(test_text)
        
        logger.info(f"🎭 Masked Text: {result['masked_text']}")
        logger.info(f"📊 Entities Found: {result['total_entities']}")
        logger.info(f"🔒 Masked: {result['masked_entities']}, 🔓 Preserved: {result['preserved_entities']}")
        
        if result['pii_entities']:
            for entity in result['pii_entities']:
                status = "🔒 MASKED" if entity['masked'] else "🔓 PRESERVED"
                logger.info(f"   {status}: {entity['type']} = '{entity['entity']}' - {entity['reason']}")
        
        logger.info("-" * 50)
    
    # Print model statistics
    if model.training_stats:
        logger.info(f"\n📈 Model Statistics:")
        logger.info(f"   Total Training Samples: {model.training_stats['total_samples']}")
        logger.info(f"   Computation Samples: {model.training_stats['computation_samples']}")
        logger.info(f"   Non-Computation Samples: {model.training_stats['no_computation_samples']}")
        logger.info(f"   Features: {model.training_stats['feature_count']}")
        logger.info(f"   Data Sources: {dict(model.training_stats['data_sources'])}")
    
    logger.info("\n🎉 Final Year Project Demo Completed!")
    logger.info("💡 This model can handle all 16+ PII domains with contextual dependency analysis")

if __name__ == "__main__":
    main()