import os
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

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

# Define the contents of the chatbot
bot_contents = [
    {"tag": "greeting", "patterns": ["Hi", "Hello", "Hey", "How are you", "What's up"],
     "responses": ["Hi there", "Hello", "Hey", "I'm fine, thank you", "Nothing much"]},
    {"tag": "goodbye", "patterns": ["Bye", "See you later", "Goodbye", "Take care"],
     "responses": ["Goodbye", "See you later", "Take care"]},
    {"tag": "thanks", "patterns": ["Thank you", "Thanks", "Thanks a lot", "I appreciate it"],
     "responses": ["You're welcome", "No problem", "Glad I could help"]},
    {"tag": "about", "patterns": ["What can you do", "Who are you", "What are you", "What is your purpose"],
     "responses": ["I am a chatbot", "My purpose is to assist you", "I can answer questions and provide assistance"]},
    {"tag": "help", "patterns": ["Help", "I need help", "Can you help me", "What should I do"],
     "responses": ["Sure, what do you need help with?", "I'm here to help. What's the problem?", "How can I assist you?"]},
    {"tag": "age", "patterns": ["How old are you", "What's your age"],
     "responses": ["I don't have an age. I'm a chatbot.", "I was just born in the digital world.", "Age is just a number for me."]},
    {"tag": "weather", "patterns": ["What's the weather like", "How's the weather today"],
     "responses": ["I'm sorry, I cannot provide real-time weather information.", "You can check the weather on a weather app or website."]}
]

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
