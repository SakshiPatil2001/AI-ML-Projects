import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the dataset
with open(r"C:\Users\saksh\OneDrive\Desktop\Next word prediction final\Alice's Adventures in Wonderland.txt", encoding='utf-8') as file:
    text = file.read().lower()

# Preprocess text: Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index) + 1

# Generate input sequences
input_sequences = []
for line in text.split("\n"):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# Pad sequences
max_sequence_length = max([len(x) for x in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre')

# Prepare predictors and labels
X = input_sequences[:, :-1]
y = input_sequences[:, -1]
y = to_categorical(y, num_classes=total_words)

# Build the LSTM model
model = Sequential([
    Embedding(total_words, 50, input_length=max_sequence_length-1),
    LSTM(150),
    Dense(total_words, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(X, y, epochs=20, verbose=1)

# Function to generate text based on input word/phrase
def generate_sentence(input_text, max_words, max_sequence_length):
    sentence = input_text
    for _ in range(max_words):
        token_list = tokenizer.texts_to_sequences([sentence])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_length-1, padding='pre')
        predicted = np.argmax(model.predict(token_list, verbose=0), axis=-1)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        if output_word == "":
            break  # Stop generation if no prediction is made
        sentence += " " + output_word
    return sentence

# Test the function
input_word = "wonderland"
generated_sentence = generate_sentence(input_word, max_words=15, max_sequence_length=max_sequence_length)
print(f"Generated sentence: {generated_sentence}")
