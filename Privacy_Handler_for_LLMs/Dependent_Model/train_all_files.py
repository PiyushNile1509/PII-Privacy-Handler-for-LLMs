#!/usr/bin/env python3
"""
ULTIMATE Comprehensive Training Script - Uses ALL Files Including CSV
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
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_processor import DataProcessor
from src.pii_detector import PIIDetector, ContextAnalyzer
from src.privacy_handler import PIIPrivacyHandler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComprehensiveDataLoader:
    """Load training data from ALL available sources"""
    
    def __init__(self, base_dir: str = None):
        self.base_dir = base_dir or os.path.dirname(__file__)
        
    def load_all_training_data(self) -> List[Dict]:
        """Load training data from ALL files including CSV files"""
        all_data = []
        
        # 1. Load from main CSV file
        csv_data = self._load_main_csv()
        all_data.extend(csv_data)
        logger.info(f"Loaded {len(csv_data)} samples from main CSV")
        
        # 2. Load from comprehensive training CSV
        comp_data = self._load_comprehensive_csv()
        all_data.extend(comp_data)
        logger.info(f"Loaded {len(comp_data)} samples from comprehensive CSV")
        
        # 3. Load from existing training files
        existing_data = self._load_existing_training_files()
        all_data.extend(existing_data)
        logger.info(f"Loaded {len(existing_data)} samples from existing files")
        
        # 4. Generate synthetic data from patterns
        synthetic_data = self._generate_synthetic_data()
        all_data.extend(synthetic_data)
        logger.info(f"Generated {len(synthetic_data)} synthetic samples")
        
        logger.info(f"Total training data: {len(all_data)} samples")
        return all_data
    
    def _load_main_csv(self) -> List[Dict]:
        """Load from pii_100000_new.csv"""
        csv_path = os.path.join(self.base_dir, 'pii_100000_new.csv')
        if not os.path.exists(csv_path):
            return []
        
        try:
            df = pd.read_csv(csv_path)
            data = []
            
            for _, row in df.iterrows():
                query = str(row.get('query', ''))
                reason = str(row.get('reason_for_not_replaced', ''))
                domain = str(row.get('domain', 'unknown'))
                
                # Determine label based on reason
                label = 1 if any(keyword in reason.lower() for keyword in [
                    'required', 'needed', 'necessary', 'operation', 'calculation',
                    'task', 'counting', 'mathematical', 'computation'
                ]) else 0
                
                data.append({
                    'text': query,
                    'label': label,
                    'source': 'main_csv',
                    'domain': domain,
                    'reason': reason
                })
            
            return data
        except Exception as e:
            logger.error(f"Error loading main CSV: {e}")
            return []
    
    def _load_comprehensive_csv(self) -> List[Dict]:
        """Load from models/comprehensive_training_data.csv"""
        csv_path = os.path.join(self.base_dir, 'models', 'comprehensive_training_data.csv')
        if not os.path.exists(csv_path):
            return []
        
        try:
            df = pd.read_csv(csv_path)
            data = []
            
            for _, row in df.iterrows():
                data.append({
                    'text': str(row.get('text', '')),
                    'label': int(row.get('label', 0)),
                    'source': 'comprehensive_csv',
                    'entity_type': str(row.get('entity_type', 'unknown')),
                    'reason': str(row.get('reason', ''))
                })
            
            return data
        except Exception as e:
            logger.error(f"Error loading comprehensive CSV: {e}")
            return []
    
    def _load_existing_training_files(self) -> List[Dict]:
        """Load from existing training result files"""
        data = []
        
        # Load from training_results.json if exists
        results_path = os.path.join(self.base_dir, 'models', 'training_results.json')
        if os.path.exists(results_path):
            try:
                with open(results_path, 'r') as f:
                    results = json.load(f)
                    if 'training_data' in results:
                        for item in results['training_data']:
                            data.append({
                                'text': item.get('text', ''),
                                'label': item.get('label', 0),
                                'source': 'training_results',
                                'confidence': item.get('confidence', 0.5)
                            })
            except Exception as e:
                logger.error(f"Error loading training results: {e}")
        
        return data
    
    def _generate_synthetic_data(self) -> List[Dict]:
        """Generate synthetic training data from patterns"""
        synthetic_data = []
        
        # Computation examples (label = 1)
        computation_templates = [
            "My name is {name}, count the letters in my name",
            "I am {name}, calculate the sum of digits in my phone {phone}",
            "User {name} wants to add all digits from {number}",
            "Extract the domain from email {email}",
            "Find the initials of {name}",
            "Count vowels in name {name}",
            "Get the last 4 digits of {number}",
            "Calculate checksum for {number}",
            "Reverse the string {name}",
            "Find length of {text}"
        ]
        
        # Non-computation examples (label = 0)
        non_computation_templates = [
            "My name is {name}",
            "I am {name}",
            "Contact {name} at {email}",
            "User {name} has phone {phone}",
            "Patient {name} needs appointment",
            "Employee {name} works here",
            "Student {name} enrolled",
            "Customer {name} ordered",
            "Hello, I'm {name}",
            "This is {name}"
        ]
        
        # Sample data for templates
        names = ["John Smith", "Alice Johnson", "Bob Wilson", "Sarah Davis", "Mike Brown"]
        phones = ["1234567890", "9876543210", "5555551234", "4445556789"]
        emails = ["user@example.com", "test@domain.org", "sample@mail.net"]
        numbers = ["12345", "67890", "11111", "99999"]
        
        # Generate computation examples
        for template in computation_templates:
            for i in range(5):  # 5 variations per template
                text = template.format(
                    name=names[i % len(names)],
                    phone=phones[i % len(phones)],
                    email=emails[i % len(emails)],
                    number=numbers[i % len(numbers)],
                    text="sample text"
                )
                synthetic_data.append({
                    'text': text,
                    'label': 1,
                    'source': 'synthetic',
                    'type': 'computation'
                })
        
        # Generate non-computation examples
        for template in non_computation_templates:
            for i in range(3):  # 3 variations per template
                text = template.format(
                    name=names[i % len(names)],
                    phone=phones[i % len(phones)],
                    email=emails[i % len(emails)]
                )
                synthetic_data.append({
                    'text': text,
                    'label': 0,
                    'source': 'synthetic',
                    'type': 'non_computation'
                })
        
        return synthetic_data

class UltimateUnifiedModel:
    """Ultimate model using ALL available data sources"""
    
    def __init__(self):
        self.data_loader = ComprehensiveDataLoader()
        self.pii_detector = PIIDetector()
        self.context_analyzer = ContextAnalyzer()
        self.feature_weights = {}
        self.training_stats = {}
        
        # Enhanced computation keywords
        self.computation_keywords = [
            'count', 'calculate', 'add', 'sum', 'multiply', 'divide', 'subtract',
            'extract', 'print', 'display', 'reverse', 'find', 'output', 'return',
            'get', 'letters', 'characters', 'digits', 'initials', 'domain',
            'username', 'numeric', 'last', 'first', 'before', 'after', 'vowels',
            'length', 'total', 'how many', 'checksum', 'mod', 'parity', 'even', 'odd'
        ]
    
    def extract_features(self, text: str) -> Dict[str, float]:
        """Extract comprehensive features"""
        text_lower = text.lower()
        features = {}
        
        # Keyword features
        for keyword in self.computation_keywords:
            features[f'has_{keyword}'] = 1.0 if keyword in text_lower else 0.0
        
        # Structure features
        features['has_question'] = 1.0 if '?' in text else 0.0
        features['has_numbers'] = 1.0 if any(c.isdigit() for c in text) else 0.0
        features['word_count'] = len(text.split()) / 20.0
        features['text_length'] = len(text) / 100.0
        
        # PII detection features
        pii_results = self.pii_detector.detect_pii_entities(text)
        features['has_pii'] = 1.0 if pii_results['direct_pii'] or pii_results['indirect_pii'] else 0.0
        features['pii_count'] = len(pii_results['direct_pii'] + pii_results['indirect_pii']) / 5.0
        
        # Context features
        context = self.context_analyzer.analyze_context(text)
        for ctx in ['mathematical', 'general', 'medical', 'financial']:
            features[f'context_{ctx}'] = 1.0 if context == ctx else 0.0
        
        return features
    
    def train(self):
        """Train using ALL available data"""
        # Load ALL training data
        training_data = self.data_loader.load_all_training_data()
        
        if not training_data:
            logger.error("No training data loaded!")
            return
        
        logger.info(f"Training on {len(training_data)} total samples")
        
        # Calculate feature weights
        feature_counts = {'computation': Counter(), 'no_computation': Counter()}
        
        for sample in training_data:
            features = self.extract_features(sample['text'])
            category = 'computation' if sample['label'] == 1 else 'no_computation'
            
            for feature, value in features.items():
                if value > 0:
                    feature_counts[category][feature] += 1
        
        # Calculate weights
        total_computation = len([s for s in training_data if s['label'] == 1])
        total_no_computation = len([s for s in training_data if s['label'] == 0])
        
        for feature in set().union(*[self.extract_features(s['text']).keys() for s in training_data]):
            comp_freq = feature_counts['computation'][feature] / max(total_computation, 1)
            no_comp_freq = feature_counts['no_computation'][feature] / max(total_no_computation, 1)
            
            weight = comp_freq - no_comp_freq
            self.feature_weights[feature] = weight
        
        # Store training statistics
        self.training_stats = {
            'total_samples': len(training_data),
            'computation_samples': total_computation,
            'no_computation_samples': total_no_computation,
            'data_sources': Counter([s.get('source', 'unknown') for s in training_data]),
            'domains': Counter([s.get('domain', 'unknown') for s in training_data]),
            'top_features': sorted(self.feature_weights.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
        }
        
        logger.info(f"Training completed!")
        logger.info(f"Data sources: {dict(self.training_stats['data_sources'])}")
        logger.info(f"Top features: {[f[0] for f in self.training_stats['top_features'][:5]]}")
        
        # Save model
        self._save_model()
    
    def predict(self, text: str) -> Tuple[int, float, Dict]:
        """Predict with comprehensive analysis"""
        features = self.extract_features(text)
        
        score = sum(self.feature_weights.get(f, 0) * v for f, v in features.items() if v > 0)
        prediction = 1 if score > 0.1 else 0
        confidence = min(abs(score), 1.0)
        
        analysis = {
            'score': score,
            'prediction': prediction,
            'confidence': confidence,
            'active_features': [f for f, v in features.items() if v > 0]
        }
        
        return prediction, confidence, analysis
    
    def _save_model(self):
        """Save the trained model"""
        model_data = {
            'feature_weights': self.feature_weights,
            'training_stats': self.training_stats,
            'computation_keywords': self.computation_keywords
        }
        
        model_path = os.path.join(self.data_loader.base_dir, 'models', 'ultimate_unified_model.pkl')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {model_path}")

def main():
    """Main training function"""
    logger.info("Starting ULTIMATE comprehensive training using ALL files...")
    
    model = UltimateUnifiedModel()
    model.train()
    
    # Test the model
    test_cases = [
        "My name is John Smith, count the letters in my name",
        "I am Alice Johnson",
        "Calculate sum of digits in phone 1234567890",
        "Contact me at john@example.com"
    ]
    
    logger.info("\nTesting trained model:")
    for test_text in test_cases:
        pred, conf, analysis = model.predict(test_text)
        logger.info(f"Text: {test_text}")
        logger.info(f"Prediction: {pred}, Confidence: {conf:.3f}")
        logger.info(f"Score: {analysis['score']:.3f}\n")

if __name__ == "__main__":
    main()