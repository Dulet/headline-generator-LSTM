import os
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer


dir = "data/"

headlines = []
for filename in os.listdir(dir):
    if "Articles" in filename:
        headlines_df = pd.read_csv(dir + filename)
        headlines.extend(list(headlines_df.headline.values))
len(headlines)

headlines = [a for a in headlines if a != "Unknown"]
len(headlines)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(headlines)
total_words = len(tokenizer.word_index) + 1
print('Total words: ', total_words)

# sprawdzamy jak są zapisywane słowa
dict = {key: value for key, value in tokenizer.word_index.items()
               if key in ['a','i','a','bike','a','canal','trump']}
print(dict)

sequences = []
for line in headlines:
   # konwersja naszych nagłówków do sekwencji tokenów
    token_list = tokenizer.texts_to_sequences([line])[0]
    
    # tworzenie sekwencji dla każdego nagłówka
    for i in range(1, len(token_list)):
        partial_sequence = token_list[:i+1]
        sequences.append(partial_sequence)

print(tokenizer.sequences_to_texts(sequences[:10]))
sequences[:10]

# dopełniamy sekwencje - tworzymy array z najdłuższą sekwencją słów (najdłuższym artykułem)

from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

max_sequence_len = max([len(x) for x in sequences])

input_sequences = np.array(pad_sequences(sequences, maxlen=max_sequence_len, padding='pre'))
input_sequences[5]

# predyktory to każde słowa oprócz ostatniego
predictors = input_sequences[:,:-1]
# etykiety to ostatnie słowa
labels = input_sequences[:,-1]
labels[:5]

from tensorflow.keras import utils
labels = utils.to_categorical(labels, num_classes=total_words)

from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential

# nasz input to wszystkie słowa oprócz ostatniego
input_len = max_sequence_len - 1 

model = Sequential()

# embedding layer
model.add(Embedding(total_words, 10, input_length=input_len))

# Add LSTM layer with 100 units
model.add(LSTM(100))
model.add(Dropout(0.1))

# Add output layer
model.add(Dense(total_words, activation='softmax'))

model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam')

model.fit(predictors, labels, epochs=30, verbose=1)
def predict(seed_text):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    prediction = np.argmax(model.predict(token_list),axis=1)
    return prediction

prediction = predict("today in new")
print(prediction)

# dekodowanie liczby na słowo
tokenizer.sequences_to_texts([prediction])

def generate_headline(seed_text, next_words=1):
    for _ in range(next_words):
        prediction = predict(seed_text)
        next_word = tokenizer.sequences_to_texts([prediction])[0]
        seed_text += " " + next_word
    return seed_text.title()

seed_texts = [
    'washington',
    'new york',
    'the school',
    'crime has',
    'kraków',
    'Poland']
for seed in seed_texts:
    print(generate_headline(seed, next_words=4))
