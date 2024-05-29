import os
import json
import nltk
import ssl
import random
import streamlit as st
import re
import time
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
    text = text.lower()  # Convert text to lower case
    text = re.sub(r'[\W_]+', ' ', text)  # Remove non-word characters
    tokens = nltk.word_tokenize(text)  # Tokenize text
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stopwords.words('english')]  # Lemmatize and remove stopwords
    return ' '.join(tokens)

# Define the path to your JSON file
intents_file_path = './intents.json'

# Load the intents from the JSON file
bot_contents = load_intents(intents_file_path)

# Create the vectorizer and classifier
vectorizer = TfidfVectorizer(ngram_range=(1, 3), stop_words='english')
model = RandomForestClassifier(n_estimators=100)
tags = []
patterns = []

for bot_content in bot_contents:
    for pattern in bot_content['patterns']:
        tags.append(bot_content['tag'])
        patterns.append(preprocess_text(pattern))

# Training the model
x = vectorizer.fit_transform(patterns)
y = tags
model.fit(x, y)

def chatbot(input_text):
    input_text = preprocess_text(input_text)
    input_vec = vectorizer.transform([input_text])
    tag = model.predict(input_vec)[0]
    for bot_content in bot_contents:
        if bot_content['tag'] == tag:
            return {'response': random.choice(bot_content['responses']), 'tag': tag}
    return {'response': "Sorry, I didn't understand that.", 'tag': None}

# Testing and metrics
def test_chatbot():
    test_data = [
        ("Hello", "greeting"),
        ("What can you do?", "skill"),
        ("I feel great today.", "happy"),
        ("Goodbye", "goodbye")
    ]
    correct = 0
    response_times = []

    for text, expected in test_data:
        start_time = time.time()
        result = chatbot(text)
        end_time = time.time()

        if result['tag'] == expected:
            correct += 1
        response_times.append(end_time - start_time)

    accuracy = correct / len(test_data)
    average_response_time = sum(response_times) / len(response_times)
    return accuracy, average_response_time

# Streamlit application setup
def main():
    st.title("Chatbot")
    if st.button("Run Test"):
        accuracy, average_response_time = test_chatbot()
        st.write(f"Accuracy: {accuracy:.2f}")
        st.write(f"Average Response Time: {average_response_time:.4f} seconds")

    st.write("Type your message below:")
    user_input = st.text_input("You:")
    if user_input:
        response = chatbot(user_input)
        st.text_area("Chatbot:", value=response['response'], height=100, max_chars=None)

if __name__ == '__main__':
    main()
