from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Activation
from keras.optimizers import RMSprop
from nltk.tokenize import RegexpTokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
import numpy as np
from keras.models import load_model


# Assuming you have joined_text available here
# Load your preprocessed data
text_df = pd.read_csv("preprocessed_data.csv")
text = list(text_df.text.values)
joined_text = " ".join(text)

# Save joined text
with open("joined_text.txt", "w", encoding="utf-8") as f:
    f.write(joined_text)

# Tokenize the text using NLTK's RegexpTokenizer
tokenizer = RegexpTokenizer(r"\w+")
tokens = tokenizer.tokenize(joined_text.lower())

from keras.preprocessing.sequence import pad_sequences
import numpy as np

import numpy as np
import os
from keras.utils import to_categorical

mmapped_filename = 'X_train_mmap.dat'

context_sizes = [10, 15, 20]  # You can adjust these as needed
# X_train = []
# y_train = []
# for n_words in context_sizes:
#     for i in range(len(tokens) - n_words):
#         X_train.append(tokens[i:i + n_words])
#         y_train.append(tokens[i + n_words])

unique_tokens = np.unique(tokens)
unique_token_index = {token: index for index, token in enumerate(unique_tokens)}

# X_train_indices = [[unique_token_index[token] for token in seq] for seq in X_train]
# y_train_indices = [unique_token_index[token] for token in y_train]

# max_sequence_length = max(context_sizes)
# X_train_padded = pad_sequences(X_train_indices, maxlen=max_sequence_length, padding='pre')
# y_train_padded = np.array(y_train_indices)

# n_tokens = len(unique_tokens)
# X_train_shape = (len(X_train_padded), max_sequence_length, n_tokens)

# X_train_mmap = np.memmap(mmapped_filename, dtype=bool, mode='w+', shape=X_train_shape)

# for i, seq in enumerate(X_train_padded):
#     for j, token_index in enumerate(seq):
#         X_train_mmap[i, j, token_index] = 1

# y_train_onehot = to_categorical(y_train_padded, num_classes=n_tokens)

# model = Sequential()
# model.add(LSTM(128, input_shape=(max_sequence_length, n_tokens), return_sequences=True))
# model.add(Dropout(0.2))
# model.add(LSTM(128))
# model.add(Dense(n_tokens))
# model.add(Activation("softmax"))

# optimizer = RMSprop(learning_rate=0.01)
# model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

# history = model.fit(X_train_mmap, y_train_onehot, batch_size=128, epochs=10, shuffle=True).history

# del X_train_mmap
# os.remove(mmapped_filename)


# model.save("text_gen_model_variable2.h5")
model=load_model("text_gen_model_variable2.h5")

def generate_text(model, seed_text, max_sequence_len, unique_tokens, unique_token_index, num_words):
    generated_text = seed_text
    for _ in range(num_words):
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(seed_text)
        
        token_indices = [unique_token_index[token] for token in tokens if token in unique_token_index]
        
        print(token_indices)
        padded_sequence = pad_sequences([token_indices], max_sequence_len, padding='pre')
    
        
        padded_sequence = np.expand_dims(padded_sequence, axis=0)
        
        predicted_probs = model.predict(padded_sequence)[0]
        
        predicted_index = np.argmax(predicted_probs)
        
        predicted_token = [token for token, index in unique_token_index.items() if index == predicted_index][0]
        
        seed_text += " " + predicted_token
        generated_text += " " + predicted_token
    
    return generated_text

seed_text = "echographie echographie echographie"
num_words = 50
predicted_text = generate_text(model, seed_text, 512, unique_tokens, unique_token_index, num_words)
print(predicted_text)
