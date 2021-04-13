import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import random 
import json
import numpy as np
import pickle
import nltk
from nltk.stem.lancaster import LancasterStemmer
from tensorflow.keras.models import load_model
stemmer = LancasterStemmer()
import speech_recognition as sr
import pyttsx3
from gtts import gTTS
import vlc
import time

with open("chatbot/intents.json") as  file:
    data = json.load(file)

words = pickle.load(open("words.pkl", "rb"))
labels = pickle.load(open("labels.pkl", "rb"))
model = load_model("chatmodel.h5")

def bag_of_words(s, words):
    bag = [0] * len(words)

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return np.array(bag)

def chat():
    print("Start talking with the bot (type quit to stop)!")
    while True:
        print("You: ")
        inp = get_audio()
        if inp.lower() == "quit":
            break
        p = bag_of_words(inp, words)
        res = model.predict(np.array([p]))[0]
        results_index = np.argmax(res)
        tag = labels[results_index]
        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

        response = random.choice(responses)
        print(response)
        speak(response)

def speak(text):
    tts = gTTS(text=text, lang='en', slow = False)
    filename = 'voice.mp3'
    tts.save(filename)
    p = vlc.MediaPlayer(filename)
    p.play()
    time.sleep(1.5)
    duration = p.get_length() / 1000
    time.sleep(duration)
    os.remove(filename)

def get_audio():
	r = sr.Recognizer()
	with sr.Microphone() as source:
		audio = r.listen(source, timeout=3, phrase_time_limit=3)
		said = ""

		try:
		    said = r.recognize_google(audio)
		    print(said)
		except Exception as e:
		    print("Exception: " + str(e))

	return said

chat()