import pandas as pd
import numpy as np
from keras.layers import Input, Embedding, LSTM, Dense, Bidirectional
from keras.models import Model
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import load_model

df = pd.read_csv('preprocessed_data.csv')
df1, df2 = np.array_split(df, 2)

# Tokenize input sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df1['text'])
input_sequences = tokenizer.texts_to_sequences(df1['text'])
max_sequence_length = max(len(seq) for seq in input_sequences)
input_sequences_padded = pad_sequences(input_sequences, maxlen=max_sequence_length)

# Prepare target sequences
target_sequences = tokenizer.texts_to_sequences(df1['label'])  
target_sequences_padded = pad_sequences(target_sequences, maxlen=max_sequence_length)
vocab_size = len(tokenizer.word_index) + 1  

# Define the model architecture
inputs = Input(shape=(max_sequence_length,))
embedding_dim = 100
embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(inputs)
lstm_layer = Bidirectional(LSTM(units=128, return_sequences=True))(embedding_layer)
outputs = Dense(units=vocab_size, activation='softmax')(lstm_layer)
model = Model(inputs=inputs, outputs=outputs)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(input_sequences_padded, to_categorical(target_sequences_padded, num_classes=vocab_size),
          epochs=1, batch_size=10, validation_split=0.2)

# Save the model
model.save("my_model_bidirectional_lstm.h5")
#model=load_model("my_model.h5")
def generate_hints(input_text, top_n=3):
    input_sequence = tokenizer.texts_to_sequences([input_text])
    input_sequence = [word for word in input_sequence[0] if word != "'"]
    input_sequence_padded = pad_sequences([input_sequence], maxlen=max_sequence_length)
    predicted_probs = model.predict(input_sequence_padded)[0]
    
    top_indices = np.argsort(predicted_probs)[::-1][:top_n]
    
    hints = []
    for index in top_indices:
        if isinstance(index, np.ndarray):
            index = index[0]  
        if index >= len(tokenizer.index_word):
            hint = "Unknown"
        else:
            hint = tokenizer.index_word[index]
        hints.append((hint, predicted_probs[index]))
    
    return hints

input_texts = ["absence", "bi-rads"]
for input_text in input_texts:
    hints = generate_hints(input_text)
    print(f"Hints for '{input_text}':")
    for hint, probability in hints:
        print(f"- {hint}: Probability = {probability}")

