
import joblib
import re

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from final import get_heartfelt_reply , preprocess

# Load models and vectorizer
vectorizer = joblib.load("tfidf_vectorizer.pkl")
nb_model = joblib.load("nb_model.pkl")



##streamlit UI
import streamlit as st
import pandas as pd
import numpy as np
import datetime

##title of the application
st.title("❤️ My Mood Journal")

#choosing a select box
options = ["😀Happy" , "😡Angry" , "😭Sad" , "Something else"]
choice = st.selectbox("Choose your mood : ", options)

if choice == "😀Happy":
    st.write("Wow! Good to know , tell me more about it")
elif choice == "😡Angry" :
    st.write("I can undertand , feel free to express your anger here ")
elif choice == "😭Sad":
    st.write("Not all days can be good , would you like to talk about it?")
elif choice== "Something else":
    st.write("Talk to me about anything , I will never judge ")  

## Display a simple text
st.write("Feel Free to Share")
st.write("⏲️" , datetime.datetime.now().strftime("%D %m %Y ,  %H : %M :%S"))

journal_text = st.text_area("What's on your mind today?", height=300, placeholder="Start typing here...")


if st.button("Analyze My Mood"):
    if journal_text.strip():
        cleaned_input = preprocess(journal_text)
        x_new = vectorizer.transform([cleaned_input])
        predicted_emotion = nb_model.predict(x_new)[0]

        st.markdown(f"**Predicted Emotion:** `{predicted_emotion}` 🎭")

        Your_AI_Friend = get_heartfelt_reply(journal_text, predicted_emotion)
        st.markdown("**🤗 Your_AI_Friend:**")
        st.write(Your_AI_Friend)     
