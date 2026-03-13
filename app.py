import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st





word_index =imdb.get_word_index()
reverse_word_index={value:key for key,value in word_index.items()}

model=load_model("simple_rnn_imdb.h5")



def preprocess_text(text):
    words = text.lower().split()
    
    encoded_review = [
        (word_index[word] + 3) if word in word_index and word_index[word] < 10000 else 2
        for word in words
    ]
    
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    
    return padded_review

def predict_sentiment(review):
    preprocessed_input=preprocess_text(review)
    prediction=model.predict(preprocessed_input)
    sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"
    if prediction[0][0] > 0.54 :
        sentiment="Positive"
    elif prediction[0][0]<0.48:
        sentiment="Negative"
    else :
        sentiment="Neutral"
    return sentiment,prediction[0][0]



st.title("IMBD MOview Review Sentiment Analysis")

st.write("Enter a movie review to classify whether it is Negative or Positive:")
user_input = st.text_area("Movie Review")

if st.button("Classify"):
    
    sentiment,prediction=predict_sentiment(user_input)
    st.write(f"Sentiment:{sentiment}")
    st.write(f"Prediction score:{prediction}")

else:
    st.write("Please Enter Movie review!")