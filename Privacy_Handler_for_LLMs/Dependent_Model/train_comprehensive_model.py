#!/usr/bin/env python3

import pandas as pd
import numpy as np
import json
import re
from typing import Dict, List, Any, Tuple
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

class PIIDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_length=128):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize and convert to indices
        tokens = text.lower().split()[:self.max_length]
        indices = [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]
        
        # Pad sequence
        if len(indices) < self.max_length:
            indices.extend([self.vocab['<PAD>']] * (self.max_length - len(indices)))
        
        return torch.tensor(indices), torch.tensor(label, dtype=torch.float)

class PIIClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_classes=1):
        super(PIIClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        # Use the last output
        last_output = lstm_out[:, -1, :]
        dropped = self.dropout(last_output)
        output = self.classifier(dropped)
        return self.sigmoid(output)

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
    
    # Combine all test cases
    all_cases = (name_cases + phone_cases + email_cases + address_cases + 
                gov_id_cases + bank_cases + age_dob_cases + work_cases + 
                health_cases + property_cases + tax_cases + social_cases + 
                education_cases + location_cases)
    
    # Convert to training format
    for text, label, entity_type, reason in all_cases:
        training_data.append({
            'text': text,
            'label': label,  # 1 if computation required, 0 if should mask
            'entity_type': entity_type,
            'reason': reason
        })
    
    return training_data

def build_vocabulary(texts):
    """Build vocabulary from texts"""
    vocab = {'<PAD>': 0, '<UNK>': 1}
    word_freq = {}
    
    for text in texts:
        for word in text.lower().split():
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Add words with frequency > 1
    for word, freq in word_freq.items():
        if freq > 1 and word not in vocab:
            vocab[word] = len(vocab)
    
    return vocab

def train_model():
    """Train the PII classification model"""
    print("Generating training data...")
    training_data = generate_comprehensive_training_data()
    
    # Convert to DataFrame
    df = pd.DataFrame(training_data)
    print(f"Generated {len(df)} training samples")
    
    # Prepare data
    texts = df['text'].tolist()
    labels = df['label'].tolist()
    
    # Build vocabulary
    print("Building vocabulary...")
    vocab = build_vocabulary(texts)
    print(f"Vocabulary size: {len(vocab)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Create datasets
    train_dataset = PIIDataset(X_train, y_train, vocab)
    test_dataset = PIIDataset(X_test, y_test, vocab)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Initialize model
    model = PIIClassifier(vocab_size=len(vocab))
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    print("Training model...")
    model.train()
    for epoch in range(20):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data).squeeze()
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f'Epoch {epoch+1}/20, Loss: {total_loss/len(train_loader):.4f}')
    
    # Evaluation
    print("Evaluating model...")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data).squeeze()
            predicted = (output > 0.5).float()
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    
    # Save model and vocabulary
    print("Saving model...")
    torch.save(model.state_dict(), 'models/pii_classifier.pth')
    
    with open('models/vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)
    
    # Save training data for reference
    df.to_csv('models/training_data.csv', index=False)
    
    print("Training completed!")
    return model, vocab, df

class TrainedPIIClassifier:
    """Wrapper class for the trained model"""
    
    def __init__(self, model_path='models/pii_classifier.pth', vocab_path='models/vocab.pkl'):
        # Load vocabulary
        with open(vocab_path, 'rb') as f:
            self.vocab = pickle.load(f)
        
        # Load model
        self.model = PIIClassifier(vocab_size=len(self.vocab))
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
    
    def predict_computation_required(self, text: str) -> float:
        """Predict if computation is required for the given text"""
        # Tokenize
        tokens = text.lower().split()[:128]
        indices = [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]
        
        # Pad
        if len(indices) < 128:
            indices.extend([self.vocab['<PAD>']] * (128 - len(indices)))
        
        # Predict
        with torch.no_grad():
            input_tensor = torch.tensor([indices])
            output = self.model(input_tensor)
            return output.item()

def test_trained_model():
    """Test the trained model with sample cases"""
    try:
        classifier = TrainedPIIClassifier()
        
        test_cases = [
            "My name is Prasad.",  # Should be 0 (mask)
            "My name is Prasad, count total characters in my name.",  # Should be 1 (keep)
            "My phone number is 9876543210.",  # Should be 0 (mask)
            "Phone 9876543210, output last 3 digits only.",  # Should be 1 (keep)
            "Email is prasad@gmail.com, print the domain part.",  # Should be 1 (keep)
        ]
        
        print("\n=== Testing Trained Model ===")
        for text in test_cases:
            prob = classifier.predict_computation_required(text)
            decision = "KEEP" if prob > 0.5 else "MASK"
            print(f"Text: {text}")
            print(f"Probability: {prob:.3f} -> {decision}")
            print("-" * 50)
            
    except FileNotFoundError:
        print("Model not found. Please train the model first.")

if __name__ == "__main__":
    # Train the model
    model, vocab, training_df = train_model()
    
    # Test the trained model
    test_trained_model()
    
    print(f"\nTraining completed with {len(training_df)} samples")
    print("Model saved to: models/pii_classifier.pth")
    print("Vocabulary saved to: models/vocab.pkl")
    print("Training data saved to: models/training_data.csv")