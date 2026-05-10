#!/usr/bin/env python3
"""
PyTorch-based training script for PII Privacy Handler Model
"""

import os
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

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PIIDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        labels = self.labels[idx]
        
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
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(aligned_labels, dtype=torch.long)
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

class PIIModel(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.num_labels = num_labels
        self.transformer = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.transformer.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return {'loss': loss, 'logits': logits}

class PIIProcessor:
    def __init__(self):
        self.labels = [
            'O', 'B-PERSON', 'I-PERSON', 'B-EMAIL', 'I-EMAIL', 
            'B-PHONE', 'I-PHONE', 'B-ADDRESS', 'I-ADDRESS'
        ]
        self.label_to_id = {label: i for i, label in enumerate(self.labels)}
        self.id_to_label = {i: label for i, label in enumerate(self.labels)}
        self.num_labels = len(self.labels)
    
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
                entity_type = entity.get('type', 'PERSON')
                
                if start < len(labels) and end <= len(text):
                    labels[start] = f'B-{entity_type}'
                    for i in range(start + 1, min(end, len(labels))):
                        labels[i] = f'I-{entity_type}'
        
        return [self.label_to_id.get(label, 0) for label in labels]

def load_and_preprocess_data(file_path):
    logger.info(f"Loading data from {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Successfully loaded {len(df):,} samples")
    except FileNotFoundError:
        logger.error(f"File {file_path} not found!")
        return None, None, None
    
    processor = PIIProcessor()
    texts = []
    labels = []
    
    logger.info("Processing samples...")
    for idx, row in tqdm(df.iterrows(), total=min(len(df), 10000), desc="Processing"):
        if idx >= 10000:  # Limit for faster training
            break
            
        try:
            text = str(row['query'])
            pii_entities = processor.safe_eval(row['pii_entities'])
            bio_labels = processor.create_bio_labels(text, pii_entities)
            
            if len(bio_labels) > 0 and len(text.strip()) > 0:
                texts.append(text)
                labels.append(bio_labels)
                
        except Exception as e:
            continue
    
    logger.info(f"Processed {len(texts):,} valid samples")
    return texts, labels, processor

def train_model():
    # Check GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Load data
    texts, labels, processor = load_and_preprocess_data('pii_100000_new.csv')
    if texts is None:
        return
    
    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.1, random_state=42
    )
    
    logger.info(f"Training samples: {len(train_texts):,}")
    logger.info(f"Validation samples: {len(val_texts):,}")
    
    # Initialize tokenizer and model
    model_name = 'distilbert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    # Create datasets
    train_dataset = PIIDataset(train_texts, train_labels, tokenizer)
    val_dataset = PIIDataset(val_texts, val_labels, tokenizer)
    
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
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
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
            os.makedirs('models', exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'tokenizer': tokenizer,
                'processor': processor,
                'f1_score': val_f1
            }, 'models/best_pii_model.pth')
            logger.info(f"Saved best model (F1: {val_f1:.4f})")
    
    logger.info(f"Training completed! Best F1: {best_f1:.4f}")
    return model, tokenizer, processor

def evaluate_model(model, dataloader, device):
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs['logits'], dim=2)
            
            predictions = predictions.view(-1).cpu().numpy()
            labels = labels.view(-1).cpu().numpy()
            
            mask = labels != -100
            all_predictions.extend(predictions[mask])
            all_labels.extend(labels[mask])
    
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    model.train()
    return f1

if __name__ == "__main__":
    try:
        model, tokenizer, processor = train_model()
        logger.info("🎉 Training completed successfully!")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()