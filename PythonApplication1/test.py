import numpy as np
from nltk.tokenize import RegexpTokenizer
from keras.models import load_model

def predict_next_word(input_text, tokenizer, model, max_context_size):
    # Tokenize input text
    tokens = tokenizer.tokenize(input_text.lower())
    
    # Ensure the input sequence length does not exceed the maximum context size
    input_sequence = tokens[-max_context_size:]
    
    # Convert input sequence to unique tokens
    unique_tokens = sorted(set(tokens))
    
    # Create token index dictionary
    token_index = {token: i for i, token in enumerate(unique_tokens)}
    
    # Convert input sequence to token indices
    input_sequence_indices = [token_index[token] for token in input_sequence if token in token_index]
    
    # Pad input sequence to match the max_sequence_length used during training
    input_sequence_padded = np.pad(input_sequence_indices, (max_context_size - len(input_sequence_indices), 0), 'constant')
    
    # Reshape input sequence for model prediction
    input_sequence_padded = np.reshape(input_sequence_padded, (1, len(input_sequence_padded), 1))
    
    # Predict probabilities for next word
    predicted_probabilities = model.predict(input_sequence_padded, verbose=0)[0]
    
    # Get the index of the word with maximum probability
    predicted_index = np.argmax(predicted_probabilities)
    
    # Convert index back to word
    predicted_word = unique_tokens[predicted_index]
    
    return predicted_word

# Example usage:
input_text = "echographie"
model = load_model("text_gen_model_variable_context.h5")
tokenizer = RegexpTokenizer(r"\w+")
predicted_word = predict_next_word(input_text, tokenizer, model, max_context_size=10)
print("Predicted next word:", predicted_word)
