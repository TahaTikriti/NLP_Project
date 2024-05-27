import os
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import json

# Load intents from the JSON file
def load_intents(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data['intents']

# Define the path to your JSON file
intents_file_path = './intents.json' 

# Handling SSL certificate verification for NLTK download
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context

nltk.data.path.append(os.path.abspath('nltk_data'))
nltk.download('punkt')

# Load the intents from the JSON file
bot_contents = load_intents(intents_file_path)

# Create the vectorizer and classifier
vectorizer = TfidfVectorizer()
clfr = LogisticRegression(random_state=0, max_iter=1000)

# Preprocess the data
tags = []
patterns = []
for bot_content in bot_contents:
    for pattern in bot_content['patterns']:
        tags.append(bot_content['tag'])
        patterns.append(pattern)

# Training the model
x = vectorizer.fit_transform(patterns)
y = tags
clfr.fit(x, y)

# A python function to chat with the chatbot
def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clfr.predict(input_text)[0]
    for bot_content in bot_contents:
        if bot_content['tag'] == tag:
            response = random.choice(bot_content['responses'])
            return response
    return "Sorry, I didn't understand that."

# Deploy the chatbot using Python with streamlit
counter = 0

def main():
    global counter
    st.title("Chatbot")
    st.write("Welcome to the Greating_chatbot. Please type a message and press Enter to start the conversation.")

    counter += 1
    user_input = st.text_input("You:", key=f"user_input_{counter}")

    if user_input:
        response = chatbot(user_input)
        st.text_area("Chatbot:", value=response, height=100, max_chars=None, key=f"chatbot_response_{counter}")

        if response.lower() in ['goodbye', 'bye']:
            st.write("Thank you for chatting with me. Have a great day!")
            st.write("Said ካሕሳይ")
            st.stop()

if __name__ == '__main__':
    main()
