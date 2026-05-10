#!/usr/bin/env python3
"""
ULTIMATE Comprehensive Training Script for PII Privacy Handler
Uses ALL files, ALL data sources, and ALL existing code components
"""

import os
import sys
import pandas as pd
import numpy as np
import json
import pickle
import re
import ast
from typing import Dict, List, Any, Tuple
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
import logging

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import ALL existing components
from src.data_processor import DataProcessor
from src.pii_detector import PIIDetector, ContextAnalyzer
from src.privacy_handler import PIIPrivacyHandler
from src.pytorch_model import PIIProcessor
from src.gemini_client import GeminiClient, ResponseProcessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UltimateUnifiedPIIModel:
    """Ultimate unified PII model combining ALL approaches and data sources"""
    
    def __init__(self):
        # Initialize ALL existing components
        self.data_processor = DataProcessor()
        self.pii_detector = PIIDetector()
        self.context_analyzer = ContextAnalyzer()
        self.pytorch_processor = PIIProcessor()
        self.gemini_client = None
        self.response_processor = ResponseProcessor()
        
        try:
            self.gemini_client = GeminiClient()
        except:
            logger.warning("Gemini client not available")
        
        # Ultimate computation keywords from ALL sources
        self.computation_keywords = [
            'count', 'calculate', 'add', 'sum', 'multiply', 'divide', 'subtract',
            'extract', 'print', 'display', 'reverse', 'find', 'output', 'return',
            'get', 'letters', 'characters', 'digits', 'initials', 'domain',
            'username', 'numeric', 'last', 'first', 'before', 'after', 'vowels',
            'length', 'total', 'how many', 'checksum', 'mod', 'parity', 'even', 'odd',
            'plus', 'minus', 'compare', 'check', 'append', 'double', 'hex', 'pairs',
            'octets', 'hyphens', 'underscore'
        ]
        
        # Ultimate PII patterns combining ALL sources
        self.ultimate_pii_patterns = self._build_ultimate_patterns()
        
        self.feature_weights = {}
        self.training_stats = {}
        
    def _build_ultimate_patterns(self):
        """Build ultimate PII patterns from all sources"""
        patterns = {
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
                r'\b(\d{3}-\d{2}-\d{4})\b',
                r'\bXXX-XX-(\d{4})\b'
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
            'IMEI': [
                r'\bIMEI\s+(\d{15})\b'
            ],
            'MAC': [
                r'\bMAC\s+([0-9A-F]{2}:[0-9A-F]{2}:[0-9A-F]{2}:[0-9A-F]{2}:[0-9A-F]{2}:[0-9A-F]{2})\b'
            ],
            'IP': [
                r'\bIP\s+(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b'
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
            ]
        }
        return patterns
    
    def extract_ultimate_features(self, text: str) -> Dict[str, float]:
        """Extract ultimate features from text using ALL approaches"""
        text_lower = text.lower()
        features = {}
        
        # Keyword features from ALL sources
        for keyword in self.computation_keywords:
            features[f'has_{keyword}'] = 1.0 if keyword in text_lower else 0.0
        
        # Punctuation and structure features
        features['has_comma'] = 1.0 if ',' in text else 0.0
        features['has_question'] = 1.0 if '?' in text else 0.0
        features['has_period'] = 1.0 if '.' in text else 0.0
        features['has_colon'] = 1.0 if ':' in text else 0.0
        features['has_dash'] = 1.0 if '—' in text or '-' in text else 0.0
        
        # Length and complexity features
        features['text_length'] = len(text) / 100.0
        features['word_count'] = len(text.split()) / 20.0
        features['avg_word_length'] = np.mean([len(word) for word in text.split()]) / 10.0
        features['sentence_count'] = len([s for s in text.split('.') if s.strip()]) / 5.0
        
        # PII presence features using ultimate patterns
        for pii_type, patterns in self.ultimate_pii_patterns.items():
            features[f'has_{pii_type.lower()}'] = 0.0
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    features[f'has_{pii_type.lower()}'] = 1.0
                    break
        
        # Context features using existing analyzer
        context = self.context_analyzer.analyze_context(text)
        for domain in ['mathematical', 'location_based', 'medical', 'financial', 'educational', 'general']:
            features[f'context_{domain}'] = 1.0 if context == domain else 0.0
        
        # Advanced linguistic features
        features['has_numbers'] = 1.0 if any(char.isdigit() for char in text) else 0.0
        features['has_uppercase'] = 1.0 if any(char.isupper() for char in text) else 0.0
        features['question_words'] = 1.0 if any(word in text_lower for word in ['what', 'how', 'when', 'where', 'why', 'which']) else 0.0
        features['imperative_words'] = 1.0 if any(word in text_lower for word in ['calculate', 'find', 'get', 'show', 'tell', 'give']) else 0.0
        
        # Mathematical operation features
        features['math_operations'] = 1.0 if any(op in text_lower for op in ['+', '-', '*', '/', 'add', 'subtract', 'multiply', 'divide']) else 0.0
        features['digit_operations'] = 1.0 if any(word in text_lower for word in ['digits', 'digit', 'sum', 'total']) else 0.0
        
        return features