import random
import pickle
import heapq

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.tokenize import RegexpTokenizer,word_tokenize

import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Activation, Dropout
from keras.optimizers import RMSprop

text_df = pd.read_csv("preprocessed_data.csv")
text = list(text_df.text.values)
joined_text = " ".join(text)

with open("joined_text.txt", "w", encoding="utf-8") as f:
    f.write(joined_text)
    
partial_text = joined_text
tokenizer = RegexpTokenizer(r"\w+")
tokens = tokenizer.tokenize(partial_text.lower())
unique_tokens = np.unique(tokens)
unique_token_index = {token: index for index, token in enumerate(unique_tokens)}
n_words = 10
input_words = []
next_word = []
print(unique_token_index)

for i in range(len(tokens) - n_words):
    input_words.append(tokens[i:i + n_words])
    next_word.append(tokens[i + n_words])
X = np.zeros((len(input_words), n_words, len(unique_tokens)), dtype=bool)
y = np.zeros((len(next_word), len(unique_tokens)), dtype=bool)
for i, words in enumerate(input_words):
    for j, word in enumerate(words):
        X[i, j, unique_token_index[word]] = 1
    y[i, unique_token_index[next_word[i]]] = 1

model = Sequential()
model.add(LSTM(128, input_shape=(n_words, len(unique_tokens)), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dense(len(unique_tokens)))
model.add(Activation("softmax"))

optimizer = RMSprop(learning_rate=0.01)
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
history = model.fit(X, y, batch_size=128, epochs=10, shuffle=True).history

model.save("text_gen_model2.h5")
with open("history2.p", "wb") as f:
   pickle.dump(history, f)
model = load_model("text_gen_model2.h5")
history = pickle.load(open("history2.p", "rb"))

def predict_next_word(input_text, n_best):
    input_text = input_text.lower()
    X = np.zeros((1, n_words, len(unique_tokens)))
    for i, word in enumerate(input_text.split()):
        X[0, i, unique_token_index[word]] = 1
    print((unique_tokens))
    predictions = model.predict(X)[0]
    return np.argpartition(predictions, -n_best)[-n_best:]

possible = predict_next_word("seins à trame glandulaire et graisseuse type b de l", 10)

for idx in possible:
    print(unique_tokens[idx])
    
def generate_text(input_text, n_words, creativity):
    word_sequence = input_text.split()
    current = 0
    for _ in range(n_words):
        sub_sequence = " ".join(tokenizer.tokenize(" ".join(word_sequence).lower())[current:current+n_words])
        try:
            choice = unique_tokens[random.choice(predict_next_word(sub_sequence, creativity))]
        except:
            choice = random.choice(unique_tokens)
        word_sequence.append(choice)
        current += 1
    return " ".join(word_sequence)


print(generate_text("seins à trame glandulaire et graisseuse type b de l",30,4))
