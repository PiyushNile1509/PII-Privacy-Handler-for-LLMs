#!/usr/bin/env python3
"""
Final Unified Training Script for PII Privacy Handler
Combines ALL data sources and existing code to create one comprehensive model
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

# Import existing components
from src.data_processor import DataProcessor
from src.pii_detector import PIIDetector, ContextAnalyzer
from src.privacy_handler import PIIPrivacyHandler

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinalUnifiedPIIModel:
    """Final unified PII model combining all approaches"""
    
    def __init__(self):
        # Initialize all existing components
        self.data_processor = DataProcessor()
        self.pii_detector = PIIDetector()
        self.context_analyzer = ContextAnalyzer()
        
        # Enhanced computation keywords from all sources
        self.computation_keywords = [
            'count', 'calculate', 'add', 'sum', 'multiply', 'divide', 'subtract',
            'extract', 'print', 'display', 'reverse', 'find', 'output', 'return',
            'get', 'letters', 'characters', 'digits', 'initials', 'domain',
            'username', 'numeric', 'last', 'first', 'before', 'after', 'vowels',
            'length', 'total', 'how many', 'checksum', 'mod', 'parity', 'even', 'odd',
            'plus', 'minus', 'compare', 'check', 'append', 'double'
        ]
        
        # Enhanced PII patterns combining all sources
        self.enhanced_pii_patterns = {
            'NAME': [
                r'\\b(?:my name is|I am|I\\'m|called|This is)\\s+([A-Z][a-z]+(?:\\s+[A-Z][a-z]+)*)',
                r'\\b(Dr\\.|Prof\\.|Mr\\.|Ms\\.|Mrs\\.)\\s+([A-Z][a-z]+(?:\\s+[A-Z][a-z]+)*)',
                r'\\(([A-Z][a-z]+\\s+[A-Z][a-z]+)\\)',
                r'\\bfor\\s+([A-Z][a-z]+\\s+[A-Z][a-z]+)\\b',
                r'\\bunder\\s+([A-Z][a-z]+\\s+[A-Z][a-z]+)\\b',
                r'\\bbelongs to\\s+([A-Z][a-z]+\\s+[A-Z][a-z]+)\\b',
                r'\\bclient\\s+([A-Z][a-z]+\\s+[A-Z][a-z]+)\\b',
                r'\\buser\\s+([A-Z][a-z]+)\\b',
                r'\\bpolicyholder:\\s+([A-Z][a-z]+\\s+[A-Z][a-z]+)\\b'
            ],
            'PHONE': [
                r'\\+91\\s+(\\d{10})',
                r'\\+44\\s+(\\d{4}\\s+\\d{6})',
                r'\\((\\d{3})\\)\\s+(\\d{3}-\\d{4})',
                r'\\b(\\d{10})\\b',
                r'phone\\s+([+\\d\\s-]+)',
                r'number\\s+([+\\d\\s-]+)'
            ],
            'EMAIL': [
                r'\\b([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,})\\b',
                r'mailto:([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,})'
            ],
            'ACCOUNT': [
                r'\\baccount\\s+(\\d{8,16})\\b',
                r'\\baccount is\\s+(\\d{8,16})\\b'
            ],
            'CREDIT_CARD': [
                r'\\b(\\d{4}\\s+\\d{4}\\s+\\d{4}\\s+\\d{4})\\b',
                r'\\b(\\d{4}-\\d{4}-\\d{4}-\\d{4})\\b'
            ],
            'SSN': [
                r'\\b(\\d{3}-\\d{2}-\\d{4})\\b'
            ],
            'DOB': [
                r'\\bborn\\s+(\\d{4}-\\d{2}-\\d{2})\\b',
                r'\\bDOB\\s+(\\d{2}/\\d{2}/\\d{4})\\b'
            ],
            'AADHAAR': [
                r'\\bAadhaar\\s+(\\d{4}\\s+\\d{4}\\s+\\d{4})\\b'
            ],
            'PAN': [
                r'\\bPAN\\s+([A-Z]{5}\\d{4}[A-Z])\\b'
            ]
        }
        
        self.feature_weights = {}
        self.training_stats = {}
        
    def extract_comprehensive_features(self, text: str) -> Dict[str, float]:
        """Extract comprehensive features from text using all approaches"""
        text_lower = text.lower()
        features = {}
        
        # Keyword features
        for keyword in self.computation_keywords:
            features[f'has_{keyword}'] = 1.0 if keyword in text_lower else 0.0
        
        # Punctuation and structure features
        features['has_comma'] = 1.0 if ',' in text else 0.0
        features['has_question'] = 1.0 if '?' in text else 0.0
        features['has_period'] = 1.0 if '.' in text else 0.0
        features['has_colon'] = 1.0 if ':' in text else 0.0
        
        # Length features
        features['text_length'] = len(text) / 100.0  # Normalized
        features['word_count'] = len(text.split()) / 20.0  # Normalized
        features['avg_word_length'] = np.mean([len(word) for word in text.split()]) / 10.0
        
        # PII presence features using enhanced patterns
        for pii_type, patterns in self.enhanced_pii_patterns.items():
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
        
        return features
    
    def detect_all_pii_entities(self, text: str) -> List[Dict[str, Any]]:
        """Detect PII entities using all available methods"""
        entities = []
        
        # Use existing PII detector
        pii_results = self.pii_detector.detect_pii_entities(text)
        
        # Combine direct and indirect PII
        all_entities = pii_results['direct_pii'] + pii_results['indirect_pii']
        
        # Convert to standard format
        for entity in all_entities:
            entities.append({
                'type': entity['type'],
                'text': entity['value'],
                'start': entity['start'],
                'end': entity['end'],
                'confidence': entity.get('confidence', 0.8)
            })
        
        return entities
    
    def train(self, training_data: List[Dict]):
        """Train the unified model using all available techniques"""
        logger.info(f"Training unified model on {len(training_data)} samples")
        
        # Calculate feature importance using multiple methods
        feature_counts = {'computation': Counter(), 'no_computation': Counter()}
        
        for sample in training_data:
            features = self.extract_comprehensive_features(sample['text'])
            category = 'computation' if sample['label'] == 1 else 'no_computation'
            
            for feature, value in features.items():
                if value > 0:
                    feature_counts[category][feature] += 1
        
        # Calculate weights based on frequency difference and information gain
        total_computation = len([s for s in training_data if s['label'] == 1])
        total_no_computation = len([s for s in training_data if s['label'] == 0])
        
        for feature in set().union(*[self.extract_comprehensive_features(s['text']).keys() for s in training_data]):
            comp_freq = feature_counts['computation'][feature] / max(total_computation, 1)
            no_comp_freq = feature_counts['no_computation'][feature] / max(total_no_computation, 1)
            
            # Enhanced weight calculation with information gain
            if comp_freq + no_comp_freq > 0:
                # Information gain based weight
                p_comp = total_computation / len(training_data)
                p_no_comp = total_no_computation / len(training_data)
                
                # Calculate entropy
                if comp_freq > 0 and no_comp_freq > 0:
                    entropy = -(comp_freq * np.log2(comp_freq + 1e-8) + no_comp_freq * np.log2(no_comp_freq + 1e-8))
                    weight = (comp_freq - no_comp_freq) * (1 + entropy)
                else:
                    weight = comp_freq - no_comp_freq
                
                self.feature_weights[feature] = weight
        
        # Store training statistics
        self.training_stats = {
            'total_samples': len(training_data),
            'computation_samples': total_computation,
            'no_computation_samples': total_no_computation,
            'feature_count': len(self.feature_weights),
            'top_computation_features': sorted(self.feature_weights.items(), key=lambda x: x[1], reverse=True)[:10],
            'top_masking_features': sorted(self.feature_weights.items(), key=lambda x: x[1])[:10]
        }
        
        logger.info(f"Training completed with {len(self.feature_weights)} features")
        logger.info(f"Top computation indicators: {[f[0] for f in self.training_stats['top_computation_features'][:5]]}")
    
    def predict(self, text: str) -> Tuple[int, float, Dict[str, Any]]:
        """Predict if computation is required with detailed analysis"""
        features = self.extract_comprehensive_features(text)
        
        # Calculate weighted score
        score = 0.0
        feature_contributions = {}
        
        for feature, value in features.items():
            if feature in self.feature_weights and value > 0:
                contribution = self.feature_weights[feature] * value
                score += contribution
                feature_contributions[feature] = contribution
        
        # Adaptive threshold based on feature strength
        threshold = 0.1
        prediction = 1 if score > threshold else 0
        confidence = min(abs(score), 1.0)  # Cap confidence at 1.0
        
        # Additional analysis using existing components
        context = self.context_analyzer.analyze_context(text)
        pii_entities = self.detect_all_pii_entities(text)
        
        analysis = {
            'score': score,
            'threshold': threshold,
            'context': context,
            'pii_entities_detected': len(pii_entities),
            'pii_types': [e['type'] for e in pii_entities],
            'top_features': sorted(feature_contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        }
        
        return prediction, confidence, analysis
    
    def evaluate(self, test_data: List[Dict]) -> Dict[str, Any]:
        """Comprehensive evaluation of the model"""
        predictions = []
        true_labels = []
        detailed_results = []
        
        for sample in test_data:
            prediction, confidence, analysis = self.predict(sample['text'])
            predictions.append(prediction)
            true_labels.append(sample['label'])
            
            detailed_results.append({
                'text': sample['text'],
                'true_label': sample['label'],
                'predicted_label': prediction,
                'confidence': confidence,
                'correct': prediction == sample['label'],
                'analysis': analysis
            })
        
        # Calculate comprehensive metrics
        accuracy = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions, average='weighted')
        
        # Confusion matrix components
        tp = sum(1 for t, p in zip(true_labels, predictions) if t == 1 and p == 1)
        tn = sum(1 for t, p in zip(true_labels, predictions) if t == 0 and p == 0)
        fp = sum(1 for t, p in zip(true_labels, predictions) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(true_labels, predictions) if t == 1 and p == 0)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'confusion_matrix': {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn},
            'correct': sum(1 for r in detailed_results if r['correct']),
            'total': len(test_data),
            'classification_report': classification_report(true_labels, predictions),
            'detailed_results': detailed_results
        }

def load_all_training_data():
    """Load training data from ALL available sources"""
    
    all_training_data = []
    
    # 1. Load from main CSV dataset
    csv_data = load_main_csv_data()
    if csv_data:
        all_training_data.extend(csv_data)
        logger.info(f"Loaded {len(csv_data)} samples from main CSV dataset")
    
    # 2. Load from comprehensive training data CSV
    comprehensive_csv_data = load_comprehensive_csv_data()
    if comprehensive_csv_data:
        all_training_data.extend(comprehensive_csv_data)
        logger.info(f"Loaded {len(comprehensive_csv_data)} samples from comprehensive CSV")
    
    # 3. Load from existing simple classifier training data
    simple_classifier_data = load_simple_classifier_data()
    if simple_classifier_data:
        all_training_data.extend(simple_classifier_data)
        logger.info(f"Loaded {len(simple_classifier_data)} samples from simple classifier")
    
    # 4. Generate comprehensive test cases from specifications
    comprehensive_cases = generate_all_comprehensive_cases()
    all_training_data.extend(comprehensive_cases)
    logger.info(f"Generated {len(comprehensive_cases)} comprehensive test cases")
    
    # 5. Generate domain-specific cases
    domain_cases = generate_all_domain_cases()
    all_training_data.extend(domain_cases)
    logger.info(f"Generated {len(domain_cases)} domain-specific cases")
    
    # Remove duplicates based on text
    seen_texts = set()
    unique_data = []
    for item in all_training_data:
        if item['text'] not in seen_texts:
            seen_texts.add(item['text'])
            unique_data.append(item)
    
    logger.info(f"Total unique training samples: {len(unique_data)} (removed {len(all_training_data) - len(unique_data)} duplicates)")
    return unique_data

def load_main_csv_data():
    """Load and process main CSV dataset"""
    try:
        df = pd.read_csv('pii_100000_new.csv')
        csv_data = []
        
        # Sample a larger subset for comprehensive training
        df_sample = df.sample(n=min(10000, len(df)), random_state=42)
        
        for _, row in df_sample.iterrows():
            try:
                query = str(row['query'])
                replaced_pii = str(row.get('replaced_pii', ''))
                reason = str(row.get('reason_for_not_replaced', ''))
                
                # Enhanced logic for determining labels
                # If there's a reason for not replacing, it means computation was required
                label = 1 if (reason and reason != 'nan' and len(reason.strip()) > 0) else 0
                
                # Also check if PII was actually replaced
                if not label and replaced_pii and replaced_pii != 'nan' and len(replaced_pii.strip()) > 0:
                    label = 1
                
                csv_data.append({
                    'text': query,
                    'label': label,
                    'source': 'main_csv_dataset',
                    'domain': row.get('domain', 'general'),
                    'reason': reason
                })
            except:
                continue
        
        return csv_data
    except FileNotFoundError:
        logger.warning("Main CSV file not found, skipping")
        return []

def load_comprehensive_csv_data():
    """Load comprehensive training data CSV"""
    try:
        df = pd.read_csv('models/comprehensive_training_data.csv')
        csv_data = []
        
        for _, row in df.iterrows():
            try:
                csv_data.append({
                    'text': str(row['text']),
                    'label': int(row['label']),
                    'source': 'comprehensive_csv',
                    'entity_type': row.get('entity_type', 'unknown'),
                    'reason': row.get('reason', '')
                })
            except:
                continue
        
        return csv_data
    except FileNotFoundError:
        logger.warning("Comprehensive CSV file not found, skipping")
        return []

def load_simple_classifier_data():
    """Load data from simple classifier if available"""
    try:
        with open('models/simple_classifier.pkl', 'rb') as f:
            classifier = pickle.load(f)
            if hasattr(classifier, 'training_data') and classifier.training_data:
                return classifier.training_data
    except:
        logger.warning("Simple classifier data not found, skipping")
    return []

def generate_all_comprehensive_cases():
    """Generate all comprehensive test cases from detailed specifications"""
    
    # All the detailed test cases from the specifications
    comprehensive_cases = [
        # Names
        ("My name is Prasad.", 0, "NAME", "Not required for logic, purely identity info."),
        ("I am Prasad.", 0, "NAME", "Name not used in any operation."),
        ("Myself Prasad.", 0, "NAME", "Pure introduction, mask name."),
        ("My name is Prasad, count total characters in my name.", 1, "NAME", "Name is required for the task (counting characters)."),
        ("Name: Prasad Reddy, calculate how many letters are in it.", 1, "NAME", "Needed for computation."),
        ("The person's name is Prasad, print the initials.", 1, "NAME", "Name required to extract initials."),
        ("Hi, I'm Prasad, but just display my name reversed.", 1, "NAME", "Name used for operation."),
        
        # Phone numbers
        ("My phone number is 9876543210.", 0, "PHONE", "Just identity info."),
        ("Contact me at +91 9876543210.", 0, "PHONE", "Not needed for logic."),
        ("My phone number is 9876543210, add 1234 to it.", 1, "PHONE", "Required for arithmetic operation."),
        ("Number 9876543210, find sum of its digits.", 1, "PHONE", "Needed for digit calculation."),
        ("Phone 9876543210, output last 3 digits only.", 1, "PHONE", "Dependent for substring extraction."),
        
        # Email addresses
        ("My email is prasad@gmail.com.", 0, "EMAIL", "No computation, personal info."),
        ("Contact: prasad@gmail.com.", 0, "EMAIL", "Pure identity."),
        ("Email is prasad@gmail.com, print the domain part.", 1, "EMAIL", "Domain extraction requires full email."),
        ("My email is prasad@gmail.com, count characters before @.", 1, "EMAIL", "Needed for computation."),
        ("Send mail to prasad@gmail.com.", 0, "EMAIL", "No logic needed."),
        
        # Addresses
        ("I live at 12 MG Road, Mumbai.", 0, "ADDRESS", "Sensitive location, not needed."),
        ("Address: 12 MG Road, Mumbai.", 0, "ADDRESS", "No logic."),
        ("My address is 12 MG Road, Mumbai, count total characters.", 1, "ADDRESS", "Needed for string operation."),
        ("I live at 22 Park Avenue, get the PIN code only.", 1, "ADDRESS", "Required to extract subfield (PIN)."),
        
        # Government IDs
        ("My Aadhaar number is 5678 1234 9876.", 0, "AADHAAR", "Identity only."),
        ("Aadhaar: 5678 1234 9876, find sum of last 4 digits.", 1, "AADHAAR", "Used in numeric computation."),
        ("PAN: ABCDE1234F", 0, "PAN", "No logic required."),
        ("PAN: ABCDE1234F, extract only numeric digits.", 1, "PAN", "Dependent for extraction."),
        
        # Bank details
        ("My account number is 1234567890.", 0, "ACCOUNT", "Just sensitive info."),
        ("Account 1234567890, sum all digits.", 1, "ACCOUNT", "Dependent for computation."),
        ("IFSC code HDFC0001234.", 0, "IFSC", "Not required."),
        ("IFSC HDFC0001234, extract numeric part.", 1, "IFSC", "Needed for logic."),
        
        # Age/DOB
        ("I was born on 14/02/1990.", 0, "DOB", "Not used."),
        ("My DOB is 14/02/1990, calculate my age.", 1, "DOB", "DOB needed to derive age."),
        ("I'm 30 years old.", 0, "AGE", "Static statement."),
        ("I'm 30 years old, add 5 to get my retirement age.", 1, "AGE", "Age used in arithmetic."),
        
        # Employee/Work
        ("My employee ID is EMP-778899.", 0, "EMPLOYEE_ID", "Sensitive info."),
        ("ID EMP-778899, add all digits.", 1, "EMPLOYEE_ID", "Dependent on numeric part."),
        ("I work at Infosys.", 0, "COMPANY", "Identity only."),
        ("I work at Infosys, count total letters.", 1, "COMPANY", "Name needed for operation."),
        
        # Healthcare
        ("Patient ID P-112233.", 0, "PATIENT_ID", "Not needed."),
        ("Patient P-112233, multiply digits by 2.", 1, "PATIENT_ID", "Required for logic."),
        ("Hospital: Apollo Health.", 0, "HOSPITAL", "Personal info."),
        ("Hospital name Apollo Health, count characters.", 1, "HOSPITAL", "Required for operation."),
        
        # Mixed examples
        ("My name is Prasad and my phone is 9876543210. Add all digits in my phone number.", 1, "MIXED", "Phone used for math, name not needed."),
        ("I'm Aisha, born on 14/02/1990, calculate my age.", 1, "MIXED", "DOB used to compute age, name masked."),
        ("Email me at prasad@gmail.com and count characters before @.", 1, "MIXED", "Logic depends on email."),
        ("My Aadhaar is 5678 1234 9876, name is Meena. Return the sum of last 4 Aadhaar digits.", 1, "MIXED", "Aadhaar for math, name is independent."),
    ]
    
    training_data = []
    for text, label, entity_type, reason in comprehensive_cases:
        training_data.append({
            'text': text,
            'label': label,
            'entity_type': entity_type,
            'reason': reason,
            'source': 'comprehensive_specification'
        })
    
    return training_data

def generate_all_domain_cases():
    """Generate comprehensive domain-specific test cases"""
    
    domain_cases = [
        # Finance/Banking
        ("My name is Rohan Mehta and my account is 5566778899. Add 2450 to my account.", 1, "FINANCE"),
        ("Transfer ₹3000 to account 0034556677 if last digit of phone +91 9876543210 is even.", 1, "FINANCE"),
        ("PAN ABCDE1234F — extract numeric part and multiply by 3.", 1, "FINANCE"),
        ("This is Sneha Gupta, credit card 4111 1111 1111 1111. Sum card digits.", 1, "FINANCE"),
        
        # Healthcare
        ("Patient ID HRC-998877 belongs to Dr. Priya Iyer. Reverse the numeric part.", 1, "HEALTHCARE"),
        ("This is Amit Patel, born 1989-07-10; calculate my current age.", 1, "HEALTHCARE"),
        ("I'm Dr. K. Rao, patient code P-112233; if sum of digits is odd, return 'FLAG'.", 1, "HEALTHCARE"),
        
        # Education
        ("I'm Sarah Johnson, student ID STU-445566 — add 500 and return result.", 1, "EDUCATION"),
        ("Roll no 220145 — multiply by 7 and give last 4 digits.", 1, "EDUCATION"),
        ("Email john.kim@university.edu — extract domain length.", 1, "EDUCATION"),
        
        # Corporate/Employment
        ("Employee EMP-112233 (Rajesh Kumar) gets ₹1000 for each '1' in his ID.", 1, "CORPORATE"),
        ("Work email lisa.wong@company.com — extract company and append '-HR'.", 1, "CORPORATE"),
        
        # Government/ID
        ("Aadhaar 1234 5678 9876 belongs to Meena Gupta. Multiply last 4 digits by 5.", 1, "GOVERNMENT"),
        ("Passport M9876543 — return the last three digits multiplied by 2.", 1, "GOVERNMENT"),
        ("SSN 213-45-6789 — check last four digits and return them.", 1, "GOVERNMENT"),
        
        # Technology
        ("Device IMEI 356938035643809 — add first and last digit.", 1, "TECHNOLOGY"),
        ("MAC 00:1A:2B:3C:4D:5E — count hex pairs.", 1, "TECHNOLOGY"),
        ("IP 192.168.1.10 — add octets and compare >500?", 1, "TECHNOLOGY"),
        
        # Social/Online
        ("Username @prasad_official and email prasad@mail.com — return username length.", 1, "SOCIAL"),
        ("LinkedIn linkedin.com/in/jane-doe — extract 'jane-doe' and count hyphens.", 1, "SOCIAL"),
        ("Email mailto:alex.smith@startup.io — return domain and its length.", 1, "SOCIAL"),
        
        # General cases (no computation)
        ("Hi, my name is John Smith. How are you?", 0, "GENERAL"),
        ("My email is contact@company.com for inquiries.", 0, "GENERAL"),
        ("Call me at 555-1234 if you need help.", 0, "GENERAL"),
        ("I live in New York and work as a consultant.", 0, "GENERAL"),
        ("My account number is 1234567890 for reference.", 0, "GENERAL"),
    ]
    
    training_data = []
    for text, label, domain in domain_cases:
        training_data.append({
            'text': text,
            'label': label,
            'domain': domain,
            'source': 'domain_specific_cases'
        })
    
    return training_data

def train_final_unified_model():
    """Train the final unified model with ALL available data and code"""
    
    logger.info("=== Training Final Unified PII Privacy Model ===")
    logger.info("Combining ALL data sources and existing code components")
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    
    # Load ALL training data
    all_training_data = load_all_training_data()
    
    if not all_training_data:
        logger.error("No training data available!")
        return None
    
    # Analyze data distribution
    label_dist = Counter([d['label'] for d in all_training_data])
    source_dist = Counter([d.get('source', 'unknown') for d in all_training_data])
    
    logger.info(f"Data distribution by label: {dict(label_dist)}")
    logger.info(f"Data distribution by source: {dict(source_dist)}")
    
    # Split data with stratification
    train_data, test_data = train_test_split(
        all_training_data, 
        test_size=0.2, 
        random_state=42, 
        stratify=[d['label'] for d in all_training_data]
    )
    
    logger.info(f"Training samples: {len(train_data)}")
    logger.info(f"Test samples: {len(test_data)}")
    
    # Initialize and train the final unified model
    model = FinalUnifiedPIIModel()
    model.train(train_data)
    
    # Comprehensive evaluation
    logger.info("\n=== Comprehensive Evaluation ===")
    results = model.evaluate(test_data)
    
    logger.info(f"Test Accuracy: {results['accuracy']:.2%} ({results['correct']}/{results['total']})")
    logger.info(f"F1 Score: {results['f1_score']:.3f}")
    logger.info(f"Precision: {results['precision']:.3f}")
    logger.info(f"Recall: {results['recall']:.3f}")
    logger.info(f"\nConfusion Matrix:")
    logger.info(f"  TP: {results['confusion_matrix']['tp']}, TN: {results['confusion_matrix']['tn']}")
    logger.info(f"  FP: {results['confusion_matrix']['fp']}, FN: {results['confusion_matrix']['fn']}")
    
    # Test on specific examples
    logger.info("\n=== Testing on Sample Cases ===")
    test_cases = [
        "My name is Prasad.",  # Should be 0 (mask)
        "My name is Prasad, count total characters in my name.",  # Should be 1 (keep)
        "My phone number is 9876543210.",  # Should be 0 (mask)
        "Phone 9876543210, output last 3 digits only.",  # Should be 1 (keep)
        "Email is prasad@gmail.com, print the domain part.",  # Should be 1 (keep)
        "I live at 12 MG Road, Mumbai.",  # Should be 0 (mask)
        "My address is 12 MG Road, Mumbai, count total characters.",  # Should be 1 (keep)
        "My name is Rohan Mehta and my account is 5566778899. Add 2450 to my account.",  # Should be 1 (keep)
        "Device IMEI 356938035643809 — add first and last digit.",  # Should be 1 (keep)
        "Hi, my name is John Smith. How are you?",  # Should be 0 (mask)
    ]
    
    for text in test_cases:
        prediction, confidence, analysis = model.predict(text)
        decision = "KEEP (computation required)" if prediction == 1 else "MASK (no computation)"
        logger.info(f"Text: {text}")
        logger.info(f"Decision: {decision} (confidence: {confidence:.3f})")
        logger.info(f"Context: {analysis['context']}, PII entities: {len(analysis['pii_entities_detected'])}")
        logger.info("-" * 80)
    
    # Save the final unified model
    logger.info("\n=== Saving Final Unified Model ===")
    with open('models/final_unified_pii_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Save comprehensive training statistics
    final_stats = {
        'model_version': 'final_unified_v1.0',
        'training_date': pd.Timestamp.now().isoformat(),
        'total_training_samples': len(all_training_data),
        'train_samples': len(train_data),
        'test_samples': len(test_data),
        'test_accuracy': results['accuracy'],
        'test_f1_score': results['f1_score'],
        'test_precision': results['precision'],
        'test_recall': results['recall'],
        'confusion_matrix': results['confusion_matrix'],
        'data_sources': dict(source_dist),
        'label_distribution': dict(label_dist),
        'feature_count': len(model.feature_weights),
        'training_stats': model.training_stats,
        'components_used': [
            'DataProcessor', 'PIIDetector', 'ContextAnalyzer', 
            'Enhanced feature extraction', 'Information gain weighting',
            'Multi-source data integration'
        ]
    }
    
    with open('models/final_unified_training_stats.json', 'w') as f:
        json.dump(final_stats, f, indent=2, default=str)
    
    # Save all training data for reference
    training_df = pd.DataFrame(all_training_data)
    training_df.to_csv('models/final_unified_training_data.csv', index=False)
    
    logger.info(f"Final unified model saved to: models/final_unified_pii_model.pkl")
    logger.info(f"Training stats saved to: models/final_unified_training_stats.json")
    logger.info(f"Training data saved to: models/final_unified_training_data.csv")
    
    return model, final_stats

class FinalUnifiedPIIPrivacyHandler:
    """Final unified PII Privacy Handler using the comprehensive trained model"""
    
    def __init__(self, model_path='models/final_unified_pii_model.pkl'):
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            logger.info("Loaded final unified PII model")
        except FileNotFoundError:
            logger.error(f"Model file {model_path} not found. Please train the model first.")
            self.model = None
    
    def process_query(self, text: str) -> Dict[str, Any]:
        """Process a query using the final unified model"""
        if not self.model:
            return {'error': 'Model not loaded'}
        
        # Get comprehensive prediction
        computation_required, confidence, analysis = self.model.predict(text)
        
        # Detect PII entities
        pii_entities = self.model.detect_all_pii_entities(text)
        
        # Apply intelligent masking logic
        masked_text = text
        processed_entities = []
        
        # Sort entities by position (reverse order to maintain indices)
        pii_entities.sort(key=lambda x: x['start'], reverse=True)
        
        for entity in pii_entities:
            # Enhanced masking logic
            if entity['type'] == 'NAME':
                # Names are always masked for privacy unless specifically needed for computation
                should_mask = not (computation_required == 1 and any(keyword in text.lower() for keyword in ['count', 'characters', 'letters', 'initials', 'reverse']))
                reason = "Names masked for privacy protection" if should_mask else "Name required for computation"
            else:
                # Other PII: mask if computation not required
                should_mask = computation_required == 0
                reason = "Required for computation" if computation_required == 1 else "Not needed for operation"
            
            processed_entity = {
                'type': entity['type'],
                'text': entity['text'],
                'masked': should_mask,
                'reason': reason,
                'confidence': confidence
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
            'computation_required': computation_required == 1,
            'confidence': confidence,
            'context': analysis['context'],
            'analysis': analysis,
            'privacy_preserved': len([e for e in processed_entities if e['masked']]) > 0,
            'model_version': 'final_unified_v1.0'
        }

def test_final_unified_model():
    """Test the final unified model with comprehensive examples"""
    
    handler = FinalUnifiedPIIPrivacyHandler()
    
    if not handler.model:
        logger.error("Cannot test - model not loaded")
        return
    
    # Comprehensive test cases covering all domains
    test_cases = [
        # Finance
        "My name is Rohan Mehta and my account is 5566778899. Add 2450 to my account and return the result.",
        "PAN ABCDE1234F — extract numeric part and multiply by 3.",
        
        # Healthcare
        "Patient ID HRC-998877 belongs to Dr. Priya Iyer. Reverse the numeric part and report.",
        "This is Amit Patel, born 1989-07-10; calculate my current age.",
        
        # Education
        "I'm Sarah Johnson, student ID STU-445566 — add 500 and return numeric result.",
        "Roll no 220145 — multiply by 7 and give last 4 digits.",
        
        # Technology
        "Device IMEI 356938035643809 — add first and last digit and return sum.",
        "IP 192.168.1.10 — add octets and compare >500?",
        
        # Social/Online
        "Username @prasad_official and email prasad@mail.com — return username length.",
        "Email mailto:alex.smith@startup.io — return domain and its length.",
        
        # General (no computation)
        "My name is John Smith. How's the weather today?",
        "Contact me at john@email.com for more information.",
        "I live in New York and work as a consultant.",
    ]
    
    logger.info("\n=== Testing Final Unified Model ===")
    
    correct_predictions = 0
    total_predictions = len(test_cases)
    
    for i, test_case in enumerate(test_cases, 1):
        logger.info(f"\nTest Case {i}:")
        logger.info(f"Input: {test_case}")
        
        result = handler.process_query(test_case)
        
        logger.info(f"Masked: {result['masked_text']}")
        logger.info(f"Computation Required: {result['computation_required']}")
        logger.info(f"Context: {result['context']}")
        logger.info(f"Confidence: {result['confidence']:.3f}")
        logger.info(f"Privacy Preserved: {result['privacy_preserved']}")
        
        if result['pii_entities']:
            logger.info("PII Entities:")
            for entity in result['pii_entities']:
                status = "MASKED" if entity['masked'] else "KEPT"
                logger.info(f"  - {entity['type']}: {entity['text']} [{status}] - {entity['reason']}")
        
        logger.info("-" * 80)

if __name__ == "__main__":
    try:
        # Train the final unified model
        model, stats = train_final_unified_model()
        
        if model:
            # Test the final unified model
            test_final_unified_model()
            
            logger.info("\n=== Final Training Summary ===")
            logger.info(f"Model Version: {stats['model_version']}")
            logger.info(f"Total samples: {stats['total_training_samples']}")
            logger.info(f"Test accuracy: {stats['test_accuracy']:.2%}")
            logger.info(f"Test F1 score: {stats['test_f1_score']:.3f}")
            logger.info(f"Data sources: {stats['data_sources']}")
            logger.info(f"Components used: {stats['components_used']}")
            
            logger.info("\n🎉 Final unified model training completed successfully!")
            logger.info("This model combines ALL available data sources and code components.")
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        import traceback
        traceback.print_exc()