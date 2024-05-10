import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import RegexpTokenizer
import numpy as np
from nltk.tokenize import RegexpTokenizer
from keras.models import load_model

text_df = pd.read_csv("preprocessed_data.csv")
text = list(text_df.text.values)
joined_text = " ".join(text)

with open("joined_text.txt", "w", encoding="utf-8") as f:
    f.write(joined_text)

partial_text = joined_text

tokenizer = RegexpTokenizer(r"\w+")
tokens = tokenizer.tokenize(partial_text.lower())

max_context_size = 30  # context size

input_sequences = []
output_sequences = []
for i in range(1, len(tokens)):
    context_size = min(max_context_size, i)
    context = tokens[max(0, i - context_size):i]
    input_sequences.append(context)
    output_sequences.append(tokens[i])

unique_tokens = sorted(set(tokens))
token_index = {token: i for i, token in enumerate(unique_tokens)}
input_sequences = [[token_index[token] for token in seq] for seq in input_sequences]
output_sequences = [token_index[token] for token in output_sequences]

# max_sequence_length = max(len(seq) for seq in input_sequences)
# input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre')
# output_sequences = np.array(output_sequences)

# model = Sequential()
# model.add(LSTM(128, input_shape=(max_sequence_length, 1)))
# model.add(Dense(len(unique_tokens)))
# model.add(Activation("softmax"))

# optimizer = RMSprop(learning_rate=0.01)
# model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

# history = model.fit(np.array(input_sequences), output_sequences, batch_size=128, epochs=40, shuffle=True)

# model.save("text_gen_model_variable_context.h5")


def predict_next_word(input_text, tokenizer, model, max_context_size, n=5):
    tokens2 = tokenizer.tokenize(input_text.lower())
    input_sequence = tokens2[-max_context_size:]
    unique_tokens2 = sorted(set(tokens2))
    token_index2 = {token: i for i, token in enumerate(unique_tokens2)}
    input_sequence_indices = [token_index2[token] for token in input_sequence if token in token_index2]
    input_sequence_padded = np.pad(input_sequence_indices, (max_context_size - len(input_sequence_indices), 0), 'constant')
    input_sequence_padded = np.reshape(input_sequence_padded, (1, len(input_sequence_padded), 1))
    predicted_probabilities = model.predict(input_sequence_padded, verbose=0)[0]
    top_n_indices = np.argsort(predicted_probabilities)[-n:][::-1]
    predicted_words = [unique_tokens[i] for i in top_n_indices]
    
    return predicted_words

input_text = "Absence de foyer de micro"
model = load_model("text_gen_model_variable_context.h5")
tokenizer = RegexpTokenizer(r"\w+")
predicted_word = predict_next_word(input_text, tokenizer, model, max_context_size=10)
print("Predicted next word:", predicted_word)
