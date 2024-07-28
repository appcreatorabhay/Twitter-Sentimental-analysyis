import streamlit as st
import numpy as np
import re
import pickle
import tensorflow as tf

# Load the model, tokenizer, and label encoder
model = tf.keras.models.load_model('sentiment_model.h5')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Text cleaning function
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    return text

# Streamlit UI
st.set_page_config(page_title="Sentiment Analysis with LSTM", layout="wide")
st.title("Sentiment Analysis with LSTM")

# Text input for sentiment analysis
user_input = st.text_area("Enter text for sentiment analysis:", "")

if user_input:
    # Clean the user input
    cleaned_input = clean_text(user_input)

    # Tokenize and pad the user input
    input_seq = tokenizer.texts_to_sequences([cleaned_input])
    input_seq = tf.keras.preprocessing.sequence.pad_sequences(input_seq, padding='pre', maxlen=500)  # Adjust maxlen to match training data

    # Predict sentiment
    prediction = model.predict(input_seq)
    sentiment_class = np.argmax(prediction, axis=1)[0]
    sentiment_label = label_encoder.inverse_transform([sentiment_class])[0]

    st.write(f"Predicted Sentiment: {sentiment_label}")
