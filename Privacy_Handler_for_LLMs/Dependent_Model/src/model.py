import os
# Force CPU-only to avoid GPU compilation issues
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Tuple, Dict, Any
from tqdm import tqdm

# Disable XLA compilation
tf.config.optimizer.set_jit(False)

print("Using CPU for training (GPU disabled to avoid compilation issues)")

class PIIPrivacyModel:
    def __init__(self, vocab_size: int, embedding_dim: int = 300, 
                 lstm_units: int = 256, num_pii_classes: int = 25):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.num_pii_classes = num_pii_classes  # Reduced for generalized detection
        self.model = None
        
        # PII class mapping for generalized detection
        self.pii_classes = [
            'full_name', 'email_address', 'phone_number', 'date_of_birth', 'age',
            'credit_card_number', 'bank_account_number', 'ssn', 'aadhar_number',
            'medical_record_number', 'patient_id', 'drivers_license', 'passport_number',
            'ip_address', 'mac_address', 'address', 'zip_code', 'student_id',
            'employee_id', 'account_number', 'username', 'password', 'nationality',
            'gender', 'biometric_id'
        ]
        
    def build_model(self, max_length: int = 128) -> keras.Model:
        """Build the multi-task neural network model"""
        
        # Input layer
        input_layer = layers.Input(shape=(max_length,), name='input_text')
        
        # Embedding layer (trainable)
        embedding = layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_dim,
            input_length=max_length,
            mask_zero=True,
            name='embedding'
        )(input_layer)
        
        # Use Dense layers instead of LSTM to avoid GPU compilation issues
        # Global average pooling to get fixed-size representation
        pooled = layers.GlobalAveragePooling1D(name='global_avg_pool')(embedding)
        
        # Enhanced feature extraction for generalized PII detection
        dense_seq = layers.Dense(self.lstm_units * 2, activation='relu', name='dense_seq')(pooled)
        dense_seq = layers.BatchNormalization()(dense_seq)
        dense_seq = layers.Dropout(0.3)(dense_seq)
        
        # Multi-scale feature extraction
        dense1 = layers.Dense(512, activation='relu', name='dense_1')(dense_seq)
        dense1 = layers.BatchNormalization()(dense1)
        dense1 = layers.Dropout(0.4)(dense1)
        
        dense2 = layers.Dense(256, activation='relu', name='dense_2')(dense1)
        dense2 = layers.BatchNormalization()(dense2)
        dense2 = layers.Dropout(0.3)(dense2)
        
        # Context-aware feature layer
        context_features = layers.Dense(128, activation='relu', name='context_features')(dense2)
        context_features = layers.Dropout(0.2)(context_features)
        
        # PII Detection Head (multi-label classification)
        pii_detection = layers.Dense(
            self.num_pii_classes, 
            activation='sigmoid', 
            name='pii_detection'
        )(context_features)
        
        # Context Decision Head (binary classification for each PII type)
        context_decision = layers.Dense(
            self.num_pii_classes, 
            activation='sigmoid', 
            name='context_decision'
        )(context_features)
        
        # Domain Classification Head (additional context understanding)
        domain_classification = layers.Dense(
            8,  # medical, financial, educational, general, etc.
            activation='softmax',
            name='domain_classification'
        )(context_features)
        
        # Create model with enhanced outputs
        model = keras.Model(
            inputs=input_layer,
            outputs=[pii_detection, context_decision, domain_classification],
            name='pii_privacy_model'
        )
        
        self.model = model
        return model
    
    def compile_model(self, learning_rate: float = 0.001):
        """Compile the model with appropriate loss functions"""
        if self.model is None:
            raise ValueError("Model must be built before compilation")
        
        # Custom weighted loss for imbalanced data
        def weighted_binary_crossentropy(y_true, y_pred):
            # Calculate class weights dynamically
            pos_weight = tf.reduce_sum(1 - y_true) / (tf.reduce_sum(y_true) + 1e-8)
            loss = tf.nn.weighted_cross_entropy_with_logits(
                labels=y_true, logits=y_pred, pos_weight=pos_weight
            )
            return tf.reduce_mean(loss)
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss={
                'pii_detection': 'binary_crossentropy',
                'context_decision': 'binary_crossentropy',
                'domain_classification': 'categorical_crossentropy'
            },
            loss_weights={
                'pii_detection': 1.0,
                'context_decision': 1.5,  # Higher weight for context decisions
                'domain_classification': 0.5  # Lower weight for domain classification
            },
            metrics={
                'pii_detection': ['accuracy', 'precision', 'recall'],
                'context_decision': ['accuracy', 'precision', 'recall'],
                'domain_classification': ['accuracy']
            }
        )
    
    def train(self, X_train: np.ndarray, y_pii_train: np.ndarray, y_mask_train: np.ndarray,
              y_domain_train: np.ndarray, X_val: np.ndarray, y_pii_val: np.ndarray, 
              y_mask_val: np.ndarray, y_domain_val: np.ndarray,
              epochs: int = 50, batch_size: int = 32) -> keras.callbacks.History:
        """Train the model"""
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=10, restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6
            ),
            keras.callbacks.ModelCheckpoint(
                'models/best_model.h5', save_best_only=True, monitor='val_loss'
            )
        ]
        
        # Train the model with progress bar
        print(f"Starting training for {epochs} epochs...")
        history = self.model.fit(
            X_train,
            {
                'pii_detection': y_pii_train, 
                'context_decision': y_mask_train,
                'domain_classification': y_domain_train
            },
            validation_data=(
                X_val, 
                {
                    'pii_detection': y_pii_val, 
                    'context_decision': y_mask_val,
                    'domain_classification': y_domain_val
                }
            ),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model must be built and trained before prediction")
        
        predictions = self.model.predict(X)
        pii_predictions = predictions[0]
        context_predictions = predictions[1]
        domain_predictions = predictions[2]
        
        return pii_predictions, context_predictions, domain_predictions
    
    def predict_pii_entities(self, text_sequence: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
        """Predict PII entities and masking decisions for a text sequence"""
        pii_pred, context_pred, domain_pred = self.predict(text_sequence.reshape(1, -1))
        
        # Get predictions above threshold
        detected_pii = []
        masking_decisions = {}
        
        for i, class_name in enumerate(self.pii_classes):
            if pii_pred[0][i] > threshold:
                detected_pii.append(class_name)
                masking_decisions[class_name] = context_pred[0][i] > 0.5
        
        # Get domain prediction
        domain_names = ['medical', 'financial', 'educational', 'legal', 'government', 'business', 'personal', 'general']
        predicted_domain = domain_names[np.argmax(domain_pred[0])]
        
        return {
            'detected_pii': detected_pii,
            'masking_decisions': masking_decisions,
            'predicted_domain': predicted_domain,
            'confidence_scores': {
                'pii': pii_pred[0].tolist(),
                'context': context_pred[0].tolist(),
                'domain': domain_pred[0].tolist()
            }
        }
    
    def save_model(self, path: str):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save")
        self.model.save(path)
    
    def load_model(self, path: str):
        """Load a trained model"""
        self.model = keras.models.load_model(path)
    
    def get_model_summary(self):
        """Get model architecture summary"""
        if self.model is None:
            raise ValueError("Model must be built first")
        return self.model.summary()

class CustomEmbedding:
    """Custom word embedding trainer"""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 300):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embeddings = None
    
    def train_embeddings(self, sequences: list, window_size: int = 5, 
                        epochs: int = 10) -> np.ndarray:
        """Train word embeddings using skip-gram approach"""
        
        # Initialize embeddings randomly
        self.embeddings = np.random.uniform(
            -0.1, 0.1, (self.vocab_size, self.embedding_dim)
        )
        
        # Simple skip-gram training
        learning_rate = 0.01
        
        print(f"Training embeddings for {epochs} epochs...")
        for epoch in tqdm(range(epochs), desc="Training embeddings"):
            for sequence in tqdm(sequences, desc=f"Epoch {epoch+1}", leave=False):
                for i, target_word in enumerate(sequence):
                    # Get context words
                    start = max(0, i - window_size)
                    end = min(len(sequence), i + window_size + 1)
                    
                    for j in range(start, end):
                        if i != j and sequence[j] != 0:  # Skip padding
                            context_word = sequence[j]
                            
                            # Simple update rule (simplified skip-gram)
                            target_vec = self.embeddings[target_word]
                            context_vec = self.embeddings[context_word]
                            
                            # Compute similarity
                            similarity = np.dot(target_vec, context_vec)
                            
                            # Update embeddings
                            self.embeddings[target_word] += learning_rate * context_vec
                            self.embeddings[context_word] += learning_rate * target_vec
        
        return self.embeddings
    
    def save_embeddings(self, path: str):
        """Save trained embeddings"""
        if self.embeddings is not None:
            np.save(path, self.embeddings)
    
    def load_embeddings(self, path: str):
        """Load pre-trained embeddings"""
        self.embeddings = np.load(path)