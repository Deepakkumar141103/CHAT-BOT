import streamlit as st
import random
import nltk
import numpy as np
from keras.models import load_model
import pickle
import json

# Load pre-trained model and data
with open('intent.json') as json_data:
    intents = json.load(json_data)
data = pickle.load(open("training_data", "rb"))
model = load_model('model.h5')  # Replace with the path to your saved model
words = data['words']
classes = data['classes']

stemmer = nltk.stem.LancasterStemmer()

# Function to clean up the user input
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# Function to create a bag of words
def bow(sentence, words, show_details=False):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print(f"found in bag: {w}")
    return np.array(bag)

# Classification function
ERROR_THRESHOLD = 0.25
def classify(sentence):
    bow_input = bow(sentence, words)
    bow_input = np.expand_dims(bow_input, axis=0)
    results = model.predict(bow_input)[0]
    results = [[i, r] for i, r in enumerate(results) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [(classes[r[0]], r[1]) for r in results]

# Generate response function
def response(sentence):
    results = classify(sentence)
    if results:
        for i in intents['intent']:
            if i['tag'] == results[0][0]:
                return random.choice(i['responses'])
    return "Sorry, I didn't understand that."

# Streamlit App
st.title("Chatbot")
st.write("Welcome to the chatbot! Ask me anything.")

# Initialize session state to store chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# User input
user_input = st.text_input("You:", key="input")

# Process user input
if st.button("Send"):
    if user_input:
        st.session_state["messages"].append({"role": "user", "content": user_input})
        bot_response = response(user_input)
        st.session_state["messages"].append({"role": "bot", "content": bot_response})

# Display chat history
for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(f"**Bot:** {msg['content']}")
