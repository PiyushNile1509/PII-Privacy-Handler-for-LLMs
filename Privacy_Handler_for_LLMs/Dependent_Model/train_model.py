#!/usr/bin/env python3
"""
PyTorch-based training script for PII Privacy Handler Model
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel, get_scheduler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import json
import ast
from tqdm import tqdm
import logging
from dotenv import load_dotenv

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.pytorch_model import PIIModel, PIIProcessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")
if torch.cuda.is_available():
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

class PIIDataset(Dataset):
    def __init__(self, texts, labels, context_labels, domain_labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.context_labels = context_labels
        self.domain_labels = domain_labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        labels = self.labels[idx]
        context_labels = self.context_labels[idx] if self.context_labels else labels
        domain_label = self.domain_labels[idx] if self.domain_labels else 7  # default to 'general'
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
            return_offsets_mapping=True
        )
        
        aligned_labels = self.align_labels_with_tokens(
            text, labels, encoding['offset_mapping'].squeeze().tolist()
        )
        aligned_context_labels = self.align_labels_with_tokens(
            text, context_labels, encoding['offset_mapping'].squeeze().tolist()
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(aligned_labels, dtype=torch.long),
            'context_labels': torch.tensor(aligned_context_labels, dtype=torch.long),
            'domain_labels': torch.tensor(domain_label, dtype=torch.long)
        }
    
    def align_labels_with_tokens(self, text, labels, offset_mapping):
        aligned_labels = []
        for token_start, token_end in offset_mapping:
            if token_start == 0 and token_end == 0:
                aligned_labels.append(0)
            else:
                token_labels = [labels[i] for i in range(token_start, min(token_end, len(labels)))]
                if token_labels:
                    aligned_labels.append(max(set(token_labels), key=token_labels.count))
                else:
                    aligned_labels.append(0)
        return aligned_labels

# PIIModel and PIIProcessor are now imported from src.pytorch_model
        super().__init__()
        self.num_labels = num_labels
        self.transformer = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        
        # Enhanced architecture for generalized PII detection
        hidden_size = self.transformer.config.hidden_size
        
        # Multi-task heads
        self.pii_classifier = nn.Linear(hidden_size, num_labels)  # PII detection
        self.context_classifier = nn.Linear(hidden_size, num_labels)  # Context decisions
        self.domain_classifier = nn.Linear(hidden_size, 8)  # Domain classification
        
        # Feature enhancement layers
        self.feature_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU()
        )
        
    def forward(self, input_ids, attention_mask=None, labels=None, context_labels=None, domain_labels=None):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        pooled_output = outputs[1] if len(outputs) > 1 else sequence_output.mean(dim=1)
        
        # Apply dropout
        sequence_output = self.dropout(sequence_output)
        pooled_output = self.dropout(pooled_output)
        
        # Enhanced features
        enhanced_features = self.feature_layer(pooled_output)
        
        # Multi-task predictions
        pii_logits = self.pii_classifier(sequence_output)
        context_logits = self.context_classifier(sequence_output)
        domain_logits = self.domain_classifier(pooled_output)
        
        total_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            pii_loss = loss_fct(pii_logits.view(-1, self.num_labels), labels.view(-1))
            
            context_loss = 0
            if context_labels is not None:
                context_loss = loss_fct(context_logits.view(-1, self.num_labels), context_labels.view(-1))
            
            domain_loss = 0
            if domain_labels is not None:
                domain_loss_fct = nn.CrossEntropyLoss()
                domain_loss = domain_loss_fct(domain_logits, domain_labels)
            
            total_loss = pii_loss + 0.5 * context_loss + 0.3 * domain_loss
        
        return {
            'loss': total_loss,
            'pii_logits': pii_logits,
            'context_logits': context_logits,
            'domain_logits': domain_logits
        }

class PIIProcessor:
    def __init__(self):
        # Enhanced PII labels for generalized detection
        self.labels = [
            'O', 'B-PERSON', 'I-PERSON', 'B-EMAIL', 'I-EMAIL', 
            'B-PHONE', 'I-PHONE', 'B-ADDRESS', 'I-ADDRESS',
            'B-DATE', 'I-DATE', 'B-SSN', 'I-SSN', 'B-CREDIT', 'I-CREDIT',
            'B-ACCOUNT', 'I-ACCOUNT', 'B-ID', 'I-ID', 'B-MEDICAL', 'I-MEDICAL'
        ]
        self.label_to_id = {label: i for i, label in enumerate(self.labels)}
        self.id_to_label = {i: label for i, label in enumerate(self.labels)}
        self.num_labels = len(self.labels)
        
        # Domain mapping
        self.domains = ['medical', 'financial', 'educational', 'legal', 'government', 'business', 'personal', 'general']
        self.domain_to_id = {domain: i for i, domain in enumerate(self.domains)}
    
    def safe_eval(self, x):
        if pd.isna(x) or x == '':
            return []
        try:
            if isinstance(x, str):
                x = x.strip()
                if x.startswith('[') and x.endswith(']'):
                    return ast.literal_eval(x)
            return x if isinstance(x, list) else []
        except:
            return []
    
    def create_bio_labels(self, text, pii_entities):
        labels = ['O'] * len(text)
        
        for entity in pii_entities:
            if isinstance(entity, dict) and 'start' in entity and 'end' in entity:
                start, end = entity['start'], entity['end']
                entity_type = entity.get('type', 'PERSON').upper()
                
                # Map entity types to BIO labels
                if entity_type in ['FULL_NAME', 'NAME']:
                    bio_type = 'PERSON'
                elif entity_type in ['EMAIL_ADDRESS', 'EMAIL']:
                    bio_type = 'EMAIL'
                elif entity_type in ['PHONE_NUMBER', 'PHONE']:
                    bio_type = 'PHONE'
                elif entity_type in ['DATE_OF_BIRTH', 'DATE']:
                    bio_type = 'DATE'
                elif entity_type in ['SSN', 'SOCIAL_SECURITY']:
                    bio_type = 'SSN'
                elif entity_type in ['CREDIT_CARD', 'CREDIT']:
                    bio_type = 'CREDIT'
                elif entity_type in ['ACCOUNT_NUMBER', 'BANK_ACCOUNT']:
                    bio_type = 'ACCOUNT'
                elif entity_type in ['MEDICAL_RECORD', 'PATIENT_ID']:
                    bio_type = 'MEDICAL'
                else:
                    bio_type = 'ID'
                
                if start < len(labels) and end <= len(text):
                    labels[start] = f'B-{bio_type}'
                    for i in range(start + 1, min(end, len(labels))):
                        labels[i] = f'I-{bio_type}'
        
        return [self.label_to_id.get(label, 0) for label in labels]
    
    def get_domain_label(self, text, domain_hint=None):
        """Determine domain based on text content"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['doctor', 'patient', 'medical', 'hospital', 'prescription']):
            return self.domain_to_id['medical']
        elif any(word in text_lower for word in ['bank', 'account', 'credit', 'payment', 'financial']):
            return self.domain_to_id['financial']
        elif any(word in text_lower for word in ['student', 'school', 'university', 'grade', 'education']):
            return self.domain_to_id['educational']
        elif any(word in text_lower for word in ['government', 'ssn', 'passport', 'license']):
            return self.domain_to_id['government']
        elif any(word in text_lower for word in ['employee', 'company', 'business', 'work']):
            return self.domain_to_id['business']
        else:
            return self.domain_to_id['general']

def load_and_preprocess_data(file_path):
    """Load and preprocess the PII dataset with enhanced features"""
    logger.info(f"Loading data from {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Successfully loaded {len(df):,} samples")
    except FileNotFoundError:
        logger.error(f"File {file_path} not found!")
        return None, None, None, None, None
    
    processor = PIIProcessor()
    texts = []
    labels = []
    context_labels = []
    domain_labels = []
    
    logger.info("Processing samples...")
    for idx, row in tqdm(df.iterrows(), total=min(len(df), 10000), desc="Processing"):
        if idx >= 10000:  # Limit for faster training
            break
            
        try:
            text = str(row['query'])
            pii_entities = processor.safe_eval(row['pii_entities'])
            bio_labels = processor.create_bio_labels(text, pii_entities)
            
            # Create context labels (simplified - same as bio_labels for now)
            context_bio_labels = bio_labels.copy()
            
            # Get domain label
            domain_hint = row.get('domain', None)
            domain_label = processor.get_domain_label(text, domain_hint)
            
            if len(bio_labels) > 0 and len(text.strip()) > 0:
                texts.append(text)
                labels.append(bio_labels)
                context_labels.append(context_bio_labels)
                domain_labels.append(domain_label)
                
        except Exception as e:
            continue
    
    logger.info(f"Processed {len(texts):,} valid samples")
    return texts, labels, context_labels, domain_labels, processor

def evaluate_model(model, dataloader, device):
    """Evaluate the model"""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs['pii_logits'], dim=2)
            
            predictions = predictions.view(-1).cpu().numpy()
            labels = labels.view(-1).cpu().numpy()
            
            mask = labels != -100
            all_predictions.extend(predictions[mask])
            all_labels.extend(labels[mask])
    
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    model.train()
    return f1

def create_directories():
    """Create necessary directories"""
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('plots', exist_ok=True)

def train_model():
    """Main PyTorch training function"""
    
    logger.info("=== PII Privacy Handler Model Training ===")
    
    # Load environment variables
    load_dotenv()
    
    # Create directories
    create_directories()
    
    # Load data
    texts, labels, context_labels, domain_labels, processor = load_and_preprocess_data('pii_100000_new.csv')
    if texts is None:
        return None, None, None
    
    # Split data
    train_texts, val_texts, train_labels, val_labels, train_context, val_context, train_domains, val_domains = train_test_split(
        texts, labels, context_labels, domain_labels, test_size=0.1, random_state=42
    )
    
    logger.info(f"Training samples: {len(train_texts):,}")
    logger.info(f"Validation samples: {len(val_texts):,}")
    
    # Initialize tokenizer and model
    model_name = 'distilbert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    # Create datasets
    train_dataset = PIIDataset(train_texts, train_labels, train_context, train_domains, tokenizer)
    val_dataset = PIIDataset(val_texts, val_labels, val_context, val_domains, tokenizer)
    
    # Create data loaders
    batch_size = 8 if device.type == 'cuda' else 4
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    model = PIIModel(model_name, processor.num_labels)
    model.transformer.resize_token_embeddings(len(tokenizer))
    model.to(device)
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=2e-5)
    epochs = 3
    total_steps = len(train_loader) * epochs
    scheduler = get_scheduler("linear", optimizer, 0, total_steps)
    
    logger.info(f"Starting training for {epochs} epochs...")
    logger.info(f"Batch size: {batch_size}")
    
    # Training loop
    best_f1 = 0
    for epoch in range(epochs):
        logger.info(f"Epoch {epoch + 1}/{epochs}")
        
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")
        
        for batch in progress_bar:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            context_labels = batch['context_labels'].to(device)
            domain_labels = batch['domain_labels'].to(device)
            
            outputs = model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                labels=labels,
                context_labels=context_labels,
                domain_labels=domain_labels
            )
            loss = outputs['loss']
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(train_loader)
        logger.info(f"Average training loss: {avg_loss:.4f}")
        
        # Validation
        val_f1 = evaluate_model(model, val_loader, device)
        logger.info(f"Validation F1: {val_f1:.4f}")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            # Save best model
            torch.save({
                'model_state_dict': model.state_dict(),
                'tokenizer': tokenizer,
                'processor': processor,
                'f1_score': val_f1
            }, 'models/best_pii_model_500.pth')
            logger.info(f"Saved best model (F1: {val_f1:.4f})")
    
    # Save training results
    results = {
        'best_f1_score': float(best_f1),
        'total_epochs': epochs,
        'training_samples': len(train_texts),
        'validation_samples': len(val_texts)
    }
    
    with open('models/training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Training completed! Best F1: {best_f1:.4f}")
    logger.info(f"Model saved to: models/best_pii_model_500.pth")
    logger.info(f"Results saved to: models/training_results.json")
    
    return model, tokenizer, processor

def plot_training_history(results):
    """Plot training results"""
    logger.info("Training visualization not implemented for PyTorch version")
    logger.info(f"Best F1 Score: {results.get('best_f1_score', 'N/A')}")

def evaluate_model_performance():
    """Evaluate trained PyTorch model on test cases"""
    
    try:
        # Load trained model
        checkpoint = torch.load('models/best_pii_model_500.pth', map_location=device, weights_only=False)
        model_state = checkpoint['model_state_dict']
        tokenizer = checkpoint['tokenizer']
        processor = checkpoint['processor']
        
        # Reconstruct model
        model = PIIModel('distilbert-base-uncased', processor.num_labels)
        model.load_state_dict(model_state)
        model.to(device)
        model.eval()
        
        logger.info("Model loaded successfully for evaluation")
        
        # Test cases
        test_cases = [
            "Hi, my name is Prasad Zade and I live in Pune. How's the weather today?",
            "My name is Prasad Zade, my phone number is 7897897456. Can you add all digits?",
            "I am John Smith, my email is john@email.com"
        ]
        
        logger.info("\n=== Model Performance Evaluation ===")
        
        for i, test_text in enumerate(test_cases, 1):
            logger.info(f"\nTest Case {i}: {test_text}")
            
            # Simple tokenization for testing
            encoding = tokenizer(
                test_text,
                truncation=True,
                padding='max_length',
                max_length=512,
                return_tensors='pt'
            )
            
            with torch.no_grad():
                input_ids = encoding['input_ids'].to(device)
                attention_mask = encoding['attention_mask'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs['pii_logits'], dim=2)
                
            logger.info(f"Predictions shape: {predictions.shape}")
            logger.info("Model inference completed")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")

if __name__ == "__main__":
    try:
        # Train the model
        model, tokenizer, processor = train_model()
        
        if model is not None:
            # Evaluate performance
            evaluate_model_performance()
            
            logger.info("\n=== Training and Evaluation Complete ===")
            logger.info("🎉 PyTorch training completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        import traceback
        traceback.print_exc()