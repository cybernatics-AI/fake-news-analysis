from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# FastAPI app instance
app = FastAPI()

# Load the model, tokenizer, and label encoder
model = load_model("model.h5")
tokenizer = joblib.load("tokenizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

MAX_SEQUENCE_LENGTH = 200
MAX_WORDS = 10000

class TextInput(BaseModel):
    text: str

@app.post("/predict/")
def predict(input_data: TextInput):
    """
    Predict whether the input text is fake or real.
    """
    # Preprocess the text
    text = input_data.text.lower()
    tokens = text.split()
    tokens = [token for token in tokens if token not in tokenizer.word_index]

    sequence = tokenizer.texts_to_sequences([tokens])
    padded_sequence = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

    # Predict using the loaded model
    prediction = model.predict(padded_sequence)
    predicted_class = np.argmax(prediction, axis=1)
    
    # Decode the predicted class
    class_label = label_encoder.inverse_transform(predicted_class)
    
    return {"prediction": class_label[0]}

# To run the app, use the command:
# uvicorn app:app --reload
