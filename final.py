import joblib
import re
import nltk
import os
from dotenv import load_dotenv
import google.generativeai as genai

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from openai import OpenAI


load_dotenv()
GEMINI_API_KEY = os.getenv("api_key")

genai.configure(api_key=GEMINI_API_KEY)

vectorizer = joblib.load("tfidf_vectorizer.pkl")
nb_model = joblib.load("nb_model.pkl")

def preprocess(text):
    text = re.sub("[^a-zA-Z]", " ", text)  # Remove non-alphabet characters
    text = text.lower()                    # Convert to lowercase
    tokens = word_tokenize(text)           # Tokenize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in set(stopwords.words("english"))]
    return " ".join(tokens)

model = genai.GenerativeModel('gemini-1.5-pro')

def get_heartfelt_reply(user_text, predicted_emotion):
    prompt = (
        f"My best friend just said: \"{user_text}\" and is feeling really {predicted_emotion}. "
        "Respond like a caring and emotionally supportive best friend. Be warm, friendly, encouraging, and use sweet emojis 💖🥺🌈✨."
    )
    response = model.generate_content(prompt)
    return response.text








## main execution
if __name__ == "__main__":
    test_input = "I failed my exam and feel really disappointed."
    test_emotion = "sadness"
    print(get_heartfelt_reply(test_input, test_emotion))


    user_input = input("What's on your mind today? ")

    # Preprocess the input and predict
    cleaned_input = preprocess(user_input)
    x_new = vectorizer.transform([cleaned_input])
    predicted_emotion = nb_model.predict(x_new)[0]
    print("\nPredicted Emotion:", predicted_emotion)

    # Get and display the friendly analysis
    analysis = get_heartfelt_reply(user_input, predicted_emotion)
    print("\n🤗 Friendly Analysis:")
    print(analysis)