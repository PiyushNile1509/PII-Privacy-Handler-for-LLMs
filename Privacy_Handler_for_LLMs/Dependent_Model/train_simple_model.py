#!/usr/bin/env python3

import pandas as pd
import numpy as np
import json
import re
from typing import Dict, List, Any, Tuple
import pickle
from collections import Counter

def generate_comprehensive_training_data():
    """Generate comprehensive training data based on the test cases"""
    
    training_data = []
    
    # 1. PERSON NAME test cases
    name_cases = [
        ("My name is Prasad.", 0, "NAME", "Not required for logic, purely identity info."),
        ("I am Prasad.", 0, "NAME", "Name not used in any operation."),
        ("Myself Prasad.", 0, "NAME", "Pure introduction, mask name."),
        ("My name is Prasad, count total characters in my name.", 1, "NAME", "Name is required for the task (counting characters)."),
        ("Name: Prasad Reddy, calculate how many letters are in it.", 1, "NAME", "Needed for computation."),
        ("The person's name is Prasad, print the initials.", 1, "NAME", "Name required to extract initials."),
        ("Hi, I'm Prasad, but just display my name reversed.", 1, "NAME", "Name used for operation."),
    ]
    
    # 2. PHONE NUMBER test cases
    phone_cases = [
        ("My phone number is 9876543210.", 0, "PHONE", "Just identity info."),
        ("Contact me at +91 9876543210.", 0, "PHONE", "Not needed for logic."),
        ("My phone number is 9876543210, add 1234 to it.", 1, "PHONE", "Required for arithmetic operation."),
        ("Number 9876543210, find sum of its digits.", 1, "PHONE", "Needed for digit calculation."),
        ("Phone 9876543210, output last 3 digits only.", 1, "PHONE", "Dependent for substring extraction."),
    ]
    
    # 3. EMAIL ADDRESS test cases
    email_cases = [
        ("My email is prasad@gmail.com.", 0, "EMAIL", "No computation, personal info."),
        ("Contact: prasad@gmail.com.", 0, "EMAIL", "Pure identity."),
        ("Email is prasad@gmail.com, print the domain part.", 1, "EMAIL", "Domain extraction requires full email."),
        ("My email is prasad@gmail.com, count characters before @.", 1, "EMAIL", "Needed for computation."),
        ("Send mail to prasad@gmail.com.", 0, "EMAIL", "No logic needed."),
    ]
    
    # 4. ADDRESS test cases
    address_cases = [
        ("I live at 12 MG Road, Mumbai.", 0, "ADDRESS", "Sensitive location, not needed."),
        ("Address: 12 MG Road, Mumbai.", 0, "ADDRESS", "No logic."),
        ("My address is 12 MG Road, Mumbai, count total characters.", 1, "ADDRESS", "Needed for string operation."),
        ("I live at 22 Park Avenue, get the PIN code only.", 1, "ADDRESS", "Required to extract subfield (PIN)."),
    ]
    
    # 5. GOVERNMENT ID test cases
    gov_id_cases = [
        ("My Aadhaar number is 5678 1234 9876.", 0, "AADHAAR", "Identity only."),
        ("Aadhaar: 5678 1234 9876, find sum of last 4 digits.", 1, "AADHAAR", "Used in numeric computation."),
        ("PAN: ABCDE1234F", 0, "PAN", "No logic required."),
        ("PAN: ABCDE1234F, extract only numeric digits.", 1, "PAN", "Dependent for extraction."),
    ]
    
    # 6. BANK DETAILS test cases
    bank_cases = [
        ("My account number is 1234567890.", 0, "ACCOUNT", "Just sensitive info."),
        ("Account 1234567890, sum all digits.", 1, "ACCOUNT", "Dependent for computation."),
        ("IFSC code HDFC0001234.", 0, "IFSC", "Not required."),
        ("IFSC HDFC0001234, extract numeric part.", 1, "IFSC", "Needed for logic."),
    ]
    
    # 7. AGE / DOB test cases
    age_dob_cases = [
        ("I was born on 14/02/1990.", 0, "DOB", "Not used."),
        ("My DOB is 14/02/1990, calculate my age.", 1, "DOB", "DOB needed to derive age."),
        ("I'm 30 years old.", 0, "AGE", "Static statement."),
        ("I'm 30 years old, add 5 to get my retirement age.", 1, "AGE", "Age used in arithmetic."),
    ]
    
    # 8. EMPLOYEE / WORK test cases
    work_cases = [
        ("My employee ID is EMP-778899.", 0, "EMPLOYEE_ID", "Sensitive info."),
        ("ID EMP-778899, add all digits.", 1, "EMPLOYEE_ID", "Dependent on numeric part."),
        ("I work at Infosys.", 0, "COMPANY", "Identity only."),
        ("I work at Infosys, count total letters.", 1, "COMPANY", "Name needed for operation."),
    ]
    
    # 9. HEALTHCARE test cases
    health_cases = [
        ("Patient ID P-112233.", 0, "PATIENT_ID", "Not needed."),
        ("Patient P-112233, multiply digits by 2.", 1, "PATIENT_ID", "Required for logic."),
        ("Hospital: Apollo Health.", 0, "HOSPITAL", "Personal info."),
        ("Hospital name Apollo Health, count characters.", 1, "HOSPITAL", "Required for operation."),
    ]
    
    # 10. PROPERTY test cases
    property_cases = [
        ("Property ID PROP-445566.", 0, "PROPERTY_ID", "Identification info."),
        ("PROP-445566, add last 2 digits.", 1, "PROPERTY_ID", "Needed for arithmetic."),
        ("22 Sky Towers, Pune.", 0, "ADDRESS", "Pure PII."),
    ]
    
    # 11. TAX / BUSINESS test cases
    tax_cases = [
        ("GSTIN 27ABCDE1123F1Z8.", 0, "GSTIN", "Identity."),
        ("GSTIN 27ABCDE1123F1Z8, extract numbers only.", 1, "GSTIN", "Logic required."),
        ("PAN ABCDE1234F.", 0, "PAN", "Not needed."),
        ("PAN ABCDE1234F, calculate number of digits.", 1, "PAN", "Dependent."),
    ]
    
    # 12. SOCIAL MEDIA test cases
    social_cases = [
        ("Instagram: @prasad_official.", 0, "SOCIAL_HANDLE", "Identity only."),
        ("Handle @prasad_official, count characters before underscore.", 1, "SOCIAL_HANDLE", "Used in string logic."),
        ("prasad@work.com", 0, "EMAIL", "Personal."),
        ("prasad@work.com, extract username.", 1, "EMAIL", "Logic dependent."),
    ]
    
    # 13. EDUCATION test cases
    education_cases = [
        ("Roll no. STU-112233.", 0, "STUDENT_ID", "Identity."),
        ("STU-112233, sum numeric digits.", 1, "STUDENT_ID", "Needed for logic."),
        ("School: DPS Pune.", 0, "SCHOOL", "Identity info."),
        ("School DPS Pune, count vowels.", 1, "SCHOOL", "Logic uses name."),
    ]
    
    # 14. LOCATION test cases
    location_cases = [
        ("I live in Mumbai.", 0, "CITY", "PII, not needed."),
        ("City Mumbai, count letters.", 1, "CITY", "Needed for logic."),
        ("I'm from India.", 0, "COUNTRY", "General but still PII."),
        ("Country India, return first letter.", 1, "COUNTRY", "Dependent for logic."),
    ]
    
    # 15. MIXED EXAMPLES
    mixed_cases = [
        ("My name is Prasad and my phone is 9876543210. Add all digits in my phone number.", 1, "MIXED", "Phone used for math, name not needed."),
        ("I'm Aisha, born on 14/02/1990, calculate my age.", 1, "MIXED", "DOB used to compute age, name masked."),
        ("Email me at prasad@gmail.com and count characters before @.", 1, "MIXED", "Logic depends on email."),
        ("My Aadhaar is 5678 1234 9876, name is Meena. Return the sum of last 4 Aadhaar digits.", 1, "MIXED", "Aadhaar for math, name is independent."),
    ]
    
    # Combine all test cases
    all_cases = (name_cases + phone_cases + email_cases + address_cases + 
                gov_id_cases + bank_cases + age_dob_cases + work_cases + 
                health_cases + property_cases + tax_cases + social_cases + 
                education_cases + location_cases + mixed_cases)
    
    # Convert to training format
    for text, label, entity_type, reason in all_cases:
        training_data.append({
            'text': text,
            'label': label,  # 1 if computation required, 0 if should mask
            'entity_type': entity_type,
            'reason': reason
        })
    
    return training_data

class SimpleRuleBasedPIIClassifier:
    """Simple rule-based classifier for PII computation requirements"""
    
    def __init__(self):
        self.computation_keywords = [
            'count', 'calculate', 'add', 'sum', 'multiply', 'divide', 'subtract',
            'extract', 'print', 'display', 'reverse', 'find', 'output', 'return',
            'get', 'letters', 'characters', 'digits', 'initials', 'domain',
            'username', 'numeric', 'last', 'first', 'before', 'after', 'vowels',
            'length', 'total', 'how many'
        ]
        
        self.training_data = None
        self.feature_weights = {}
    
    def extract_features(self, text: str) -> Dict[str, float]:
        """Extract features from text"""
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
        
        return features
    
    def train(self, training_data: List[Dict]):
        """Train the classifier using simple feature weighting"""
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
        
        for feature in self.computation_keywords:
            feature_key = f'has_{feature}'
            comp_freq = feature_counts['computation'][feature_key] / max(total_computation, 1)
            no_comp_freq = feature_counts['no_computation'][feature_key] / max(total_no_computation, 1)
            
            # Weight is the difference in frequencies
            self.feature_weights[feature_key] = comp_freq - no_comp_freq
        
        print(f"Trained on {len(training_data)} samples")
        print(f"Top computation indicators: {sorted(self.feature_weights.items(), key=lambda x: x[1], reverse=True)[:5]}")
    
    def predict(self, text: str) -> Tuple[int, float]:
        """Predict if computation is required"""
        features = self.extract_features(text)
        
        score = 0.0
        for feature, value in features.items():
            if feature in self.feature_weights:
                score += self.feature_weights[feature] * value
        
        # Simple threshold
        prediction = 1 if score > 0.1 else 0
        confidence = abs(score)
        
        return prediction, confidence
    
    def evaluate(self, test_data: List[Dict]) -> Dict[str, float]:
        """Evaluate the classifier"""
        correct = 0
        total = len(test_data)
        
        for sample in test_data:
            prediction, _ = self.predict(sample['text'])
            if prediction == sample['label']:
                correct += 1
        
        accuracy = correct / total
        return {'accuracy': accuracy, 'correct': correct, 'total': total}

def train_and_evaluate():
    """Train and evaluate the simple classifier"""
    print("Generating comprehensive training data...")
    training_data = generate_comprehensive_training_data()
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(training_data)
    print(f"Generated {len(df)} training samples")
    print(f"Label distribution: {df['label'].value_counts().to_dict()}")
    
    # Split data (simple split)
    np.random.seed(42)
    indices = np.random.permutation(len(training_data))
    split_idx = int(0.8 * len(training_data))
    
    train_data = [training_data[i] for i in indices[:split_idx]]
    test_data = [training_data[i] for i in indices[split_idx:]]
    
    # Train classifier
    print("\\nTraining simple rule-based classifier...")
    classifier = SimpleRuleBasedPIIClassifier()
    classifier.train(train_data)
    
    # Evaluate
    print("\\nEvaluating on test set...")
    results = classifier.evaluate(test_data)
    print(f"Test Accuracy: {results['accuracy']:.2%} ({results['correct']}/{results['total']})")
    
    # Test on specific examples
    print("\\n=== Testing on Sample Cases ===")
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
        print(f"Text: {text}")
        print(f"Decision: {decision} (confidence: {confidence:.3f})")
        print("-" * 60)
    
    # Save the model and training data
    print("\\nSaving model and data...")
    with open('models/simple_classifier.pkl', 'wb') as f:
        pickle.dump(classifier, f)
    
    df.to_csv('models/comprehensive_training_data.csv', index=False)
    
    return classifier, df

if __name__ == "__main__":
    # Create models directory if it doesn't exist
    import os
    os.makedirs('models', exist_ok=True)
    
    # Train and evaluate
    classifier, training_df = train_and_evaluate()
    
    print(f"\\nTraining completed with {len(training_df)} samples")
    print("Model saved to: models/simple_classifier.pkl")
    print("Training data saved to: models/comprehensive_training_data.csv")