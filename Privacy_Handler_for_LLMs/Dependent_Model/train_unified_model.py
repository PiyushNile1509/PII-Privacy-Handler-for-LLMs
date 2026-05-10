#!/usr/bin/env python3
"""
Unified Training Script for PII Privacy Handler
Combines all data sources and creates one comprehensive model
"""

import os
import sys
import pandas as pd
import numpy as np
import json
import pickle
import re
from typing import Dict, List, Any, Tuple
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UnifiedPIIClassifier:
    """Unified PII classifier combining all data sources"""
    
    def __init__(self):
        self.computation_keywords = [
            'count', 'calculate', 'add', 'sum', 'multiply', 'divide', 'subtract',
            'extract', 'print', 'display', 'reverse', 'find', 'output', 'return',
            'get', 'letters', 'characters', 'digits', 'initials', 'domain',
            'username', 'numeric', 'last', 'first', 'before', 'after', 'vowels',
            'length', 'total', 'how many', 'checksum', 'mod', 'parity', 'even', 'odd'
        ]
        
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
            ]
        }
        
        self.feature_weights = {}
        self.training_data = []
        
    def extract_features(self, text: str) -> Dict[str, float]:
        """Extract comprehensive features from text"""
        text_lower = text.lower()
        features = {}
        
        # Keyword features
        for keyword in self.computation_keywords:
            features[f'has_{keyword}'] = 1.0 if keyword in text_lower else 0.0
        
        # Punctuation features
        features['has_comma'] = 1.0 if ',' in text else 0.0
        features['has_question'] = 1.0 if '?' in text else 0.0
        features['has_period'] = 1.0 if '.' in text else 0.0
        
        # Length features
        features['text_length'] = len(text) / 100.0  # Normalized
        features['word_count'] = len(text.split()) / 20.0  # Normalized
        
        # PII presence features
        for pii_type, patterns in self.pii_patterns.items():
            features[f'has_{pii_type.lower()}'] = 0.0
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    features[f'has_{pii_type.lower()}'] = 1.0
                    break
        
        return features
    
    def detect_pii_entities(self, text: str) -> List[Dict[str, Any]]:
        """Detect PII entities in text"""
        entities = []
        
        for pii_type, patterns in self.pii_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    groups = match.groups()
                    if groups:
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
    
    def train(self, training_data: List[Dict]):
        """Train the unified classifier"""
        self.training_data = training_data
        
        # Calculate feature importance
        feature_counts = {'computation': Counter(), 'no_computation': Counter()}
        
        for sample in training_data:
            features = self.extract_features(sample['text'])
            category = 'computation' if sample['label'] == 1 else 'no_computation'
            
            for feature, value in features.items():
                if value > 0:
                    feature_counts[category][feature] += 1
        
        # Calculate weights based on frequency difference
        total_computation = len([s for s in training_data if s['label'] == 1])
        total_no_computation = len([s for s in training_data if s['label'] == 0])
        
        for feature in set().union(*[features.keys() for features in [self.extract_features(s['text']) for s in training_data]]):
            comp_freq = feature_counts['computation'][feature] / max(total_computation, 1)
            no_comp_freq = feature_counts['no_computation'][feature] / max(total_no_computation, 1)
            
            # Weight is the difference in frequencies
            self.feature_weights[feature] = comp_freq - no_comp_freq
        
        logger.info(f"Trained on {len(training_data)} samples")
        logger.info(f"Top computation indicators: {sorted(self.feature_weights.items(), key=lambda x: x[1], reverse=True)[:10]}")
    
    def predict(self, text: str) -> Tuple[int, float]:
        """Predict if computation is required"""
        features = self.extract_features(text)
        
        score = 0.0
        for feature, value in features.items():
            if feature in self.feature_weights:
                score += self.feature_weights[feature] * value
        
        # Adaptive threshold based on feature strength
        threshold = 0.1
        prediction = 1 if score > threshold else 0
        confidence = abs(score)
        
        return prediction, confidence
    
    def evaluate(self, test_data: List[Dict]) -> Dict[str, float]:
        """Evaluate the classifier"""
        predictions = []
        true_labels = []
        
        for sample in test_data:
            prediction, _ = self.predict(sample['text'])
            predictions.append(prediction)
            true_labels.append(sample['label'])
        
        accuracy = accuracy_score(true_labels, predictions)
        
        return {
            'accuracy': accuracy,
            'correct': sum(1 for p, t in zip(predictions, true_labels) if p == t),
            'total': len(test_data),
            'classification_report': classification_report(true_labels, predictions)
        }

def load_comprehensive_training_data():
    """Load training data from all available sources"""
    
    training_data = []
    
    # 1. Load from CSV dataset if available
    csv_data = load_csv_data()
    if csv_data:
        training_data.extend(csv_data)
        logger.info(f"Loaded {len(csv_data)} samples from CSV")
    
    # 2. Load comprehensive test cases
    comprehensive_cases = generate_comprehensive_test_cases()
    training_data.extend(comprehensive_cases)
    logger.info(f"Added {len(comprehensive_cases)} comprehensive test cases")
    
    # 3. Load domain-specific cases
    domain_cases = generate_domain_specific_cases()
    training_data.extend(domain_cases)
    logger.info(f"Added {len(domain_cases)} domain-specific cases")
    
    logger.info(f"Total training samples: {len(training_data)}")
    return training_data

def load_csv_data():
    """Load and process CSV data"""
    try:
        df = pd.read_csv('pii_100000_new.csv')
        csv_data = []
        
        # Sample a subset for faster training
        df_sample = df.sample(n=min(5000, len(df)), random_state=42)
        
        for _, row in df_sample.iterrows():
            try:
                query = str(row['query'])
                replaced_pii = str(row.get('replaced_pii', ''))
                
                # If PII was replaced, it means computation was required (label=1)
                # If no PII was replaced, it means masking was appropriate (label=0)
                label = 1 if replaced_pii and replaced_pii != 'nan' and len(replaced_pii.strip()) > 0 else 0
                
                csv_data.append({
                    'text': query,
                    'label': label,
                    'source': 'csv_dataset',
                    'domain': row.get('domain', 'general')
                })
            except:
                continue
        
        return csv_data
    except FileNotFoundError:
        logger.warning("CSV file not found, skipping CSV data")
        return []

def generate_comprehensive_test_cases():
    """Generate comprehensive test cases from detailed specifications"""
    
    test_cases = [
        # Names - should be masked unless computation required
        ("My name is Prasad.", 0, "NAME", "Not required for logic, purely identity info."),
        ("My name is Prasad, count total characters in my name.", 1, "NAME", "Name is required for the task (counting characters)."),
        ("Hi, I'm Prasad, but just display my name reversed.", 1, "NAME", "Name used for operation."),
        
        # Phone numbers
        ("My phone number is 9876543210.", 0, "PHONE", "Just identity info."),
        ("Phone 9876543210, output last 3 digits only.", 1, "PHONE", "Dependent for substring extraction."),
        ("Number 9876543210, find sum of its digits.", 1, "PHONE", "Needed for digit calculation."),
        
        # Email addresses
        ("My email is prasad@gmail.com.", 0, "EMAIL", "No computation, personal info."),
        ("Email is prasad@gmail.com, print the domain part.", 1, "EMAIL", "Domain extraction requires full email."),
        ("My email is prasad@gmail.com, count characters before @.", 1, "EMAIL", "Needed for computation."),
        
        # Addresses
        ("I live at 12 MG Road, Mumbai.", 0, "ADDRESS", "Sensitive location, not needed."),
        ("My address is 12 MG Road, Mumbai, count total characters.", 1, "ADDRESS", "Needed for string operation."),
        
        # Government IDs
        ("My Aadhaar number is 5678 1234 9876.", 0, "AADHAAR", "Identity only."),
        ("Aadhaar: 5678 1234 9876, find sum of last 4 digits.", 1, "AADHAAR", "Used in numeric computation."),
        ("PAN: ABCDE1234F", 0, "PAN", "No logic required."),
        ("PAN: ABCDE1234F, extract only numeric digits.", 1, "PAN", "Dependent for extraction."),
        
        # Bank details
        ("My account number is 1234567890.", 0, "ACCOUNT", "Just sensitive info."),
        ("Account 1234567890, sum all digits.", 1, "ACCOUNT", "Dependent for computation."),
        
        # Age/DOB
        ("I was born on 14/02/1990.", 0, "DOB", "Not used."),
        ("My DOB is 14/02/1990, calculate my age.", 1, "DOB", "DOB needed to derive age."),
        
        # Mixed examples
        ("My name is Prasad and my phone is 9876543210. Add all digits in my phone number.", 1, "MIXED", "Phone used for math, name not needed."),
        ("I'm Aisha, born on 14/02/1990, calculate my age.", 1, "MIXED", "DOB used to compute age, name masked."),
    ]
    
    training_data = []
    for text, label, entity_type, reason in test_cases:
        training_data.append({
            'text': text,
            'label': label,
            'entity_type': entity_type,
            'reason': reason,
            'source': 'comprehensive_cases'
        })
    
    return training_data

def generate_domain_specific_cases():
    """Generate domain-specific test cases"""
    
    domain_cases = [
        # Finance
        ("My name is Rohan Mehta and my account is 5566778899. Add 2450 to my account.", 1, "FINANCE"),
        ("PAN ABCDE1234F — extract numeric part and multiply by 3.", 1, "FINANCE"),
        ("This is Sneha Gupta, credit card 4111 1111 1111 1111. Sum card digits.", 1, "FINANCE"),
        
        # Healthcare
        ("Patient ID HRC-998877 belongs to Dr. Priya Iyer. Reverse the numeric part.", 1, "HEALTHCARE"),
        ("This is Amit Patel, born 1989-07-10; calculate my current age.", 1, "HEALTHCARE"),
        
        # Education
        ("I'm Sarah Johnson, student ID STU-445566 — add 500 and return result.", 1, "EDUCATION"),
        ("Roll no 220145 — multiply by 7 and give last 4 digits.", 1, "EDUCATION"),
        ("Email john.kim@university.edu — extract domain length.", 1, "EDUCATION"),
        
        # Technology
        ("Device IMEI 356938035643809 — add first and last digit.", 1, "TECHNOLOGY"),
        ("MAC 00:1A:2B:3C:4D:5E — count hex pairs.", 1, "TECHNOLOGY"),
        ("IP 192.168.1.10 — add octets and compare >500?", 1, "TECHNOLOGY"),
        
        # General cases (no computation)
        ("Hi, my name is John Smith. How are you?", 0, "GENERAL"),
        ("My email is contact@company.com for inquiries.", 0, "GENERAL"),
        ("Call me at 555-1234 if you need help.", 0, "GENERAL"),
    ]
    
    training_data = []
    for text, label, domain in domain_cases:
        training_data.append({
            'text': text,
            'label': label,
            'domain': domain,
            'source': 'domain_specific'
        })
    
    return training_data

def train_unified_model():
    """Train the unified model with all data sources"""
    
    logger.info("=== Training Unified PII Privacy Model ===")
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    
    # Load comprehensive training data
    training_data = load_comprehensive_training_data()
    
    if not training_data:
        logger.error("No training data available!")
        return None
    
    # Split data
    train_data, test_data = train_test_split(training_data, test_size=0.2, random_state=42, stratify=[d['label'] for d in training_data])
    
    logger.info(f"Training samples: {len(train_data)}")
    logger.info(f"Test samples: {len(test_data)}")
    
    # Initialize and train classifier
    classifier = UnifiedPIIClassifier()
    classifier.train(train_data)
    
    # Evaluate on test set
    logger.info("\n=== Evaluation Results ===")
    results = classifier.evaluate(test_data)
    logger.info(f"Test Accuracy: {results['accuracy']:.2%} ({results['correct']}/{results['total']})")
    logger.info(f"\nClassification Report:\n{results['classification_report']}")
    
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
    ]
    
    for text in test_cases:
        prediction, confidence = classifier.predict(text)
        decision = "KEEP (computation required)" if prediction == 1 else "MASK (no computation)"
        logger.info(f"Text: {text}")
        logger.info(f"Decision: {decision} (confidence: {confidence:.3f})")
        logger.info("-" * 60)
    
    # Save the unified model
    logger.info("\n=== Saving Unified Model ===")
    with open('models/unified_pii_classifier.pkl', 'wb') as f:
        pickle.dump(classifier, f)
    
    # Save training statistics
    stats = {
        'total_training_samples': len(training_data),
        'train_samples': len(train_data),
        'test_samples': len(test_data),
        'test_accuracy': results['accuracy'],
        'data_sources': {
            'csv_dataset': len([d for d in training_data if d.get('source') == 'csv_dataset']),
            'comprehensive_cases': len([d for d in training_data if d.get('source') == 'comprehensive_cases']),
            'domain_specific': len([d for d in training_data if d.get('source') == 'domain_specific'])
        },
        'label_distribution': {
            'computation_required': len([d for d in training_data if d['label'] == 1]),
            'mask_recommended': len([d for d in training_data if d['label'] == 0])
        }
    }
    
    with open('models/unified_training_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Save training data for reference
    training_df = pd.DataFrame(training_data)
    training_df.to_csv('models/unified_training_data.csv', index=False)
    
    logger.info(f"Unified model saved to: models/unified_pii_classifier.pkl")
    logger.info(f"Training stats saved to: models/unified_training_stats.json")
    logger.info(f"Training data saved to: models/unified_training_data.csv")
    
    return classifier, stats

class UnifiedPIIPrivacyHandler:
    """Unified PII Privacy Handler using the trained model"""
    
    def __init__(self, model_path='models/unified_pii_classifier.pkl'):
        try:
            with open(model_path, 'rb') as f:
                self.classifier = pickle.load(f)
            logger.info("Loaded unified PII classifier")
        except FileNotFoundError:
            logger.error(f"Model file {model_path} not found. Please train the model first.")
            self.classifier = None
    
    def process_query(self, text: str) -> Dict[str, Any]:
        """Process a query and return masking decisions"""
        if not self.classifier:
            return {'error': 'Model not loaded'}
        
        # Detect PII entities
        pii_entities = self.classifier.detect_pii_entities(text)
        
        # Get computation requirement prediction
        computation_required, confidence = self.classifier.predict(text)
        
        # Apply masking logic
        masked_text = text
        processed_entities = []
        
        # Sort entities by position (reverse order to maintain indices)
        pii_entities.sort(key=lambda x: x['start'], reverse=True)
        
        for entity in pii_entities:
            # Names are always masked for privacy
            if entity['type'] == 'NAME':
                should_mask = True
                reason = "Names are masked for privacy protection"
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
            'privacy_preserved': len([e for e in processed_entities if e['masked']]) > 0
        }

def test_unified_model():
    """Test the unified model with comprehensive examples"""
    
    handler = UnifiedPIIPrivacyHandler()
    
    if not handler.classifier:
        logger.error("Cannot test - model not loaded")
        return
    
    # Comprehensive test cases
    test_cases = [
        "My name is Rohan Mehta and my account is 5566778899. Add 2450 to my account and return the result.",
        "PAN ABCDE1234F — extract numeric part and multiply by 3.",
        "Patient ID HRC-998877 belongs to Dr. Priya Iyer. Reverse the numeric part and report.",
        "I'm Sarah Johnson, student ID STU-445566 — add 500 and return numeric result.",
        "Device IMEI 356938035643809 — add first and last digit and return sum.",
        "Username @prasad_official and email prasad@mail.com — return username length.",
        "My name is John Smith. How's the weather today?",
        "Contact me at john@email.com for more information.",
    ]
    
    logger.info("\n=== Testing Unified Model ===")
    
    for i, test_case in enumerate(test_cases, 1):
        logger.info(f"\nTest Case {i}:")
        logger.info(f"Input: {test_case}")
        
        result = handler.process_query(test_case)
        
        logger.info(f"Masked: {result['masked_text']}")
        logger.info(f"Computation Required: {result['computation_required']}")
        logger.info(f"Privacy Preserved: {result['privacy_preserved']}")
        
        if result['pii_entities']:
            logger.info("PII Entities:")
            for entity in result['pii_entities']:
                status = "MASKED" if entity['masked'] else "KEPT"
                logger.info(f"  - {entity['type']}: {entity['text']} [{status}] - {entity['reason']}")
        
        logger.info("-" * 80)

if __name__ == "__main__":
    try:
        # Train the unified model
        classifier, stats = train_unified_model()
        
        if classifier:
            # Test the unified model
            test_unified_model()
            
            logger.info("\n=== Training Summary ===")
            logger.info(f"Total samples: {stats['total_training_samples']}")
            logger.info(f"Test accuracy: {stats['test_accuracy']:.2%}")
            logger.info(f"Data sources: {stats['data_sources']}")
            logger.info(f"Label distribution: {stats['label_distribution']}")
            
            logger.info("\n🎉 Unified model training completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        import traceback
        traceback.print_exc()