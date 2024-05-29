import os
import json
import nltk
import ssl
import random
import streamlit as st
import re
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from nltk.corpus import stopwords

# Ensure required NLTK resources are downloaded
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Load intents from the JSON file
def load_intents(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data['intents']

# Text preprocessing function
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Remove punctuation and non-word characters
    text = re.sub(r'[\W_]+', ' ', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Tokenize text
    tokens = nltk.word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Re-join tokens into a single string
    return ' '.join(lemmatized_tokens)

# Define the path to your JSON file
intents_file_path = './intents.json'  # Update the path as necessary

# Handling SSL certificate verification for NLTK download
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.data.path.append(os.path.abspath('nltk_data'))

# Load the intents from the JSON file
bot_contents = load_intents(intents_file_path)

# Create the vectorizer and classifier
vectorizer = TfidfVectorizer(ngram_range=(1, 3), stop_words='english')
model = RandomForestClassifier(n_estimators=100)

# Prepare the training data
tags = []
patterns = []
for bot_content in bot_contents:
    for pattern in bot_content['patterns']:
        tags.append(bot_content['tag'])
        patterns.append(preprocess_text(pattern))  # Preprocess each pattern before training

# Training the model
x = vectorizer.fit_transform(patterns)
y = tags
model.fit(x, y)

# A python function to chat with the chatbot
def chatbot(input_text):
    input_text = preprocess_text(input_text)  # Preprocess user input
    input_vec = vectorizer.transform([input_text])
    tag = model.predict(input_vec)[0]
    for bot_content in bot_contents:
        if bot_content['tag'] == tag:
            response = random.choice(bot_content['responses'])
            return response
    return "Sorry, I didn't understand that."

# Deploy the chatbot using Python with streamlit
def main():
    st.title("Chatbot")
    st.write("Welcome to the Chatbot. Please type a message and press Enter to start the conversation.")
    user_input = st.text_input("You:")
    if user_input:
        response = chatbot(user_input)
        st.text_area("Chatbot:", value=response, height=100, max_chars=None)

if __name__ == '__main__':
    main()
