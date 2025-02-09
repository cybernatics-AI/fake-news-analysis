# Fake News Detection Project

## Overview
This project implements a sophisticated deep learning model for detecting fake news using a hybrid CNN-BiLSTM architecture. The system combines advanced text preprocessing, feature engineering, and state-of-the-art deep learning techniques to accurately classify news articles as real or fake.

## Features
- Advanced text preprocessing and cleaning
- Hybrid CNN-BiLSTM neural network architecture
- Class imbalance handling using SMOTE
- Adaptive learning rate scheduling
- RESTful API endpoint for real-time predictions
- Comprehensive regularization techniques to prevent overfitting

## Technical Requirements

### Dependencies
```
torch==2.0.0
torchtext==0.15.1
scikit-learn==1.2.2
numpy==1.24.3
pandas==1.5.3
matplotlib==3.7.1
seaborn==0.12.2
fastapi==0.95.0
uvicorn==0.23.1
tensorflow==2.15.0
keras==2.15.0
huggingface_hub==0.26.2
datasets==3.1.0
googletrans==4.0.0-rc1
nltk==3.9.1
deep-translator
joblib==1.2.0 
imbalanced-learn 
gensim
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd fake-news-detection
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Model Architecture

The model implements a sophisticated architecture consisting of:
- Embedding layer with L2 regularization
- 1D Convolutional layer (128 filters)
- Bidirectional LSTM layer (256 units)
- Global Max Pooling
- Dense layers with dropout for robust feature learning
- Softmax output layer for classification

### Key Features:
- Sequence length: 150 tokens
- Embedding dimension: 300
- Batch size: 64
- Training epochs: 20
- Adaptive learning rate with exponential decay
- L2 regularization throughout the network

## Usage

### Training the Model

To train the model on your dataset:

```python
python train.py
```

The training script includes:
- Data preprocessing and cleaning
- Feature engineering
- SMOTE for handling class imbalance
- Model training with early stopping and learning rate reduction

### Running the API

Start the FastAPI server:

```bash
uvicorn app:app --reload
```

The API will be available at `http://localhost:8000`

### Making Predictions

Send POST requests to the `/predict/` endpoint:

```python
import requests

url = "http://localhost:8000/predict/"
data = {"text": "Your news article text here"}
response = requests.post(url, json=data)
prediction = response.json()
```

## Data Preprocessing

The system implements comprehensive text preprocessing:
- Lowercase conversion
- Special character and number removal
- Whitespace normalization
- Text length feature extraction
- Credibility score calculation based on multiple factors

## Model Training Features

- Stratified train-test split
- SMOTE for class balance
- Adaptive learning rate scheduling
- Class weight balancing
- L2 regularization
- Dropout layers for preventing overfitting

## API Endpoints

### POST /predict/
- Input: JSON with "text" field containing the news article
- Output: JSON with prediction ("real" or "fake")
- Maximum sequence length: 200 tokens

## Performance Monitoring

The model includes:
- Validation accuracy tracking
- Learning rate reduction on plateau
- Class-weighted evaluation metrics
- Real-time prediction probability scores

## Future Improvements

1. Implement multi-language support
2. Add model explainability features
3. Incorporate attention mechanisms
4. Add ensemble methods
5. Implement real-time model updating
