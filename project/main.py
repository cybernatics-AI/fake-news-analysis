import os
import re
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout, Conv1D, GlobalMaxPooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

# Advanced Preprocessing
def preprocess_text(text):
    """
    Enhanced text preprocessing to clean and standardize input text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Configuration
MAX_SEQUENCE_LENGTH = 150
EMBEDDING_DIM = 300
BATCH_SIZE = 64
EPOCHS = 20

# Load and Preprocess Data
data = pd.read_csv("fake_news_dataset.csv")

# Enhanced Feature Engineering
data['processed_statement'] = data['statement'].apply(preprocess_text)
data['text_length'] = data['processed_statement'].apply(len)
credibility_features = ['barely_true_counts', 'false_counts', 'half_true_counts', 'mostly_true_counts', 'pants_on_fire_counts']
data['credibility_score'] = data[credibility_features].sum(axis=1)

# Tokenization
tokenizer = Tokenizer(oov_token='<OOV>')
tokenizer.fit_on_texts(data['processed_statement'])
vocab_size = len(tokenizer.word_index) + 1

# Sequence Preparation
sequences = tokenizer.texts_to_sequences(data['processed_statement'])
X = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, truncating='post', padding='post')

# Label Encoding
le = LabelEncoder()
y = le.fit_transform(data['label'])

# Train-Test Split with Stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    stratify=y, 
    random_state=42
)

# Handle Class Imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Learning Rate Schedule
initial_learning_rate = 0.001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=100,
    decay_rate=0.96,
    staircase=True
)

# Advanced Model Architecture
def create_model():
    model = Sequential([
        # Embedding Layer
        Embedding(
            input_dim=vocab_size, 
            output_dim=EMBEDDING_DIM, 
            input_length=MAX_SEQUENCE_LENGTH,
            embeddings_regularizer=l2(1e-4)
        ),
        
        # Convolutional Layer
        Conv1D(
            filters=128, 
            kernel_size=3, 
            activation='relu',
            kernel_regularizer=l2(1e-4)
        ),
        
        # Bidirectional LSTM Layers
        Bidirectional(LSTM(
            256, 
            return_sequences=True,  # Keep this to maintain 3D input for next layer
            activation='tanh',
            kernel_regularizer=l2(1e-4)
        )),
        
        # GlobalMaxPooling to reduce dimensionality
        GlobalMaxPooling1D(),
        
        # Dense Layers with Dropout
        Dense(512, activation='relu', kernel_regularizer=l2(1e-4)),
        Dropout(0.5),
        
        Dense(256, activation='relu', kernel_regularizer=l2(1e-4)),
        Dropout(0.4),
        
        # Output Layer
        Dense(len(np.unique(y)), activation='softmax')
    ])
    
    # Adaptive Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
    model.compile(
        optimizer=optimizer, 
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy']
    )
    
    return model

# Callbacks
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.2,
    patience=3, 
    min_lr=0.00001
)

# Compute Class Weights
class_weights = dict(enumerate(
    compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
))

# Model Training
model = create_model()
history = model.fit(
    X_train, y_train, 
    validation_data=(X_test, y_test), 
    epochs=EPOCHS, 
    batch_size=BATCH_SIZE, 
    class_weight=class_weights,
    callbacks=[reduce_lr]
)

# Model Evaluation
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy}")

# Optional: Prediction Function
def predict_fake_news(text):
    processed_text = preprocess_text(text)
    sequence = tokenizer.texts_to_sequences([processed_text])
    padded_sequence = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH, truncating='post', padding='post')
    prediction = model.predict(padded_sequence)
    predicted_class = le.inverse_transform([np.argmax(prediction)])[0]
    return predicted_class, np.max(prediction)