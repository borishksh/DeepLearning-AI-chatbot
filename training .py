import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy as np
import json 
import random
import pickle
import tensorflow as tf

with open("chatbot/intents.json") as  file:
    data = json.load(file)

words = []
labels = []
ignore_letters = ['!', '?', ',', '.']
doc_x = []
doc_y = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        doc_x.append(word_list)
        doc_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

words = [stemmer.stem(w.lower()) for w in words if w not in ignore_letters]
words = sorted(list(set(words)))

labels = sorted(labels)

pickle.dump(words, open("words.pkl", "wb"))
pickle.dump(labels, open("labels.pkl", "wb"))

training = []

output_empty = [0] * len(labels)

for x, doc in enumerate(doc_x):
    bag = []
    wrds = [stemmer.stem(w) for w in doc]

    for w in words:
        bag.append(1) if w in wrds else bag.append(0)
    
    output_row = output_empty[:]
    output_row[labels.index(doc_y[x])] = 1

    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)
train_x = list(training[:, 0])
train_y = list(training[:, 1])

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(len(train_y[0]), activation='softmax'))

sgd = tf.keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

hist = model.fit(train_x, train_y, epochs=200, batch_size=5)
model.save("chatmodel.h5", hist)
