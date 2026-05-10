#!/usr/bin/env python3
"""
PyTorch model classes for PII Privacy Handler
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import ast
import pandas as pd

class PIIModel(nn.Module):
    def __init__(self, model_name, num_labels):
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