import pandas as pd
import numpy as np
import re
import json
from typing import Dict, List, Tuple, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import ast

class DataProcessor:
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.vocab = {}
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.max_length = 128
        
    def load_data(self, csv_path: str) -> pd.DataFrame:
        """Load and preprocess the PII dataset"""
        df = pd.read_csv(csv_path)
        
        # Clean the data
        df = df.dropna(subset=['query', 'masked_output'])
        df['pii_entities'] = df['pii_entities'].apply(self._parse_entities)
        df['replaced_pii'] = df['replaced_pii'].apply(self._parse_replaced_pii)
        
        return df
    
    def _parse_entities(self, entities_str: str) -> List[str]:
        """Parse PII entities from string format"""
        try:
            if pd.isna(entities_str):
                return []
            # Handle different formats in the CSV
            entities_str = str(entities_str).strip()
            if entities_str.startswith('[') and entities_str.endswith(']'):
                return ast.literal_eval(entities_str)
            else:
                # Split by comma and clean
                return [e.strip().strip("'\"") for e in entities_str.split(',') if e.strip()]
        except:
            return []
    
    def _parse_replaced_pii(self, replaced_str: str) -> Dict[str, str]:
        """Parse replaced PII mapping from string format"""
        try:
            if pd.isna(replaced_str):
                return {}
            replaced_str = str(replaced_str).strip()
            if replaced_str.startswith('{') and replaced_str.endswith('}'):
                return ast.literal_eval(replaced_str)
            else:
                # Parse key-value pairs
                result = {}
                pairs = replaced_str.split(',')
                for pair in pairs:
                    if ':' in pair:
                        key, value = pair.split(':', 1)
                        result[key.strip().strip("'")] = value.strip().strip("'")
                return result
        except:
            return {}
    
    def build_vocabulary(self, texts: List[str]) -> Dict[str, int]:
        """Build vocabulary from text data"""
        word_freq = {}
        for text in texts:
            words = self._tokenize(text)
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Keep words with frequency > 1
        vocab_words = [word for word, freq in word_freq.items() if freq > 1]
        vocab_words = ['<PAD>', '<UNK>', '<START>', '<END>'] + sorted(vocab_words)
        
        self.word_to_idx = {word: idx for idx, word in enumerate(vocab_words)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        self.vocab = self.word_to_idx
        
        return self.vocab
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        text = text.lower()
        # Keep alphanumeric, spaces, and common punctuation
        text = re.sub(r'[^\w\s\-@.]', ' ', text)
        return text.split()
    
    def text_to_sequence(self, text: str) -> List[int]:
        """Convert text to sequence of token indices"""
        words = self._tokenize(text)
        sequence = [self.word_to_idx.get(word, self.word_to_idx['<UNK>']) for word in words]
        
        # Pad or truncate to max_length
        if len(sequence) > self.max_length:
            sequence = sequence[:self.max_length]
        else:
            sequence.extend([self.word_to_idx['<PAD>']] * (self.max_length - len(sequence)))
        
        return sequence
    
    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for training"""
        # Build vocabulary from all text
        all_texts = df['query'].tolist() + df['masked_output'].tolist()
        self.build_vocabulary(all_texts)
        
        # Convert texts to sequences
        X = np.array([self.text_to_sequence(text) for text in df['query']])
        
        # Create labels for PII detection (multi-label)
        pii_types = set()
        pii_counts = {}
        for entities in df['pii_entities']:
            for entity in entities:
                pii_counts[entity] = pii_counts.get(entity, 0) + 1
                pii_types.add(entity)
        
        # Filter out rare PII types to prevent memory issues (keep only top 50)
        pii_types = sorted(pii_counts.items(), key=lambda x: x[1], reverse=True)[:50]
        pii_types = [pii for pii, count in pii_types if count >= 5]  # At least 5 occurrences
        self.pii_label_encoder = {pii: idx for idx, pii in enumerate(pii_types)}
        
        # Multi-label encoding for PII entities
        y_pii = np.zeros((len(df), len(pii_types)))
        for i, entities in enumerate(df['pii_entities']):
            for entity in entities:
                if entity in self.pii_label_encoder:
                    y_pii[i, self.pii_label_encoder[entity]] = 1
        
        # Create labels for masking decisions (binary for each detected PII)
        y_mask = np.zeros((len(df), len(pii_types)))
        for i, (entities, replaced) in enumerate(zip(df['pii_entities'], df['replaced_pii'])):
            for entity in entities:
                if entity in self.pii_label_encoder:
                    # If entity was replaced, mask=0 (don't mask), else mask=1 (mask it)
                    entity_idx = self.pii_label_encoder[entity]
                    y_mask[i, entity_idx] = 1 if entity not in replaced else 0
        
        return X, y_pii, y_mask
    
    def split_data(self, X: np.ndarray, y_pii: np.ndarray, y_mask: np.ndarray, 
                   test_size: float = 0.1) -> Tuple:
        """Split data into train and test sets"""
        return train_test_split(X, y_pii, y_mask, test_size=test_size, random_state=42)
    
    def save_vocab(self, path: str):
        """Save vocabulary to file"""
        with open(path, 'w') as f:
            json.dump({
                'word_to_idx': self.word_to_idx,
                'idx_to_word': self.idx_to_word,
                'pii_label_encoder': self.pii_label_encoder,
                'max_length': self.max_length
            }, f, indent=2)
    
    def load_vocab(self, path: str):
        """Load vocabulary from file"""
        with open(path, 'r') as f:
            data = json.load(f)
            self.word_to_idx = data['word_to_idx']
            self.idx_to_word = {int(k): v for k, v in data['idx_to_word'].items()}
            self.pii_label_encoder = data['pii_label_encoder']
            self.max_length = data['max_length']
            self.vocab = self.word_to_idx