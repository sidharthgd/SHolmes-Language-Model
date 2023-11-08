import regex as re
import nltk
import numpy as np
from nltk.tokenize import word_tokenize, sent_tokenize
from tensorflow.keras.utils import pad_sequences, to_categorical


from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, BatchNormalization, SpatialDropout1D, GaussianNoise
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model


def remove_unwanted_characters(txt) -> str:
  newline_str = r"\n+"
  whitespace_str = r"\t|\r"
  quote_str = r"“|”|‘|’"
  roman_numeral_str = r"(XI{0,2}\.)|(VI{0,3}\.)|(IV|IX|I{1,3}\.)"


  txt = re.sub(newline_str, " ", txt)
  txt = re.sub(whitespace_str, "", txt)
  txt = re.sub(quote_str, "", txt)
  txt = re.sub(roman_numeral_str, "", txt)

  return txt

def file_to_sentences(FILE_PATH) -> list:
  with open(FILE_PATH, "r") as file:
    txt = remove_unwanted_characters(file.read())
    # Split into sentences
    sentences = sent_tokenize(txt)

    return sentences

FILE_PATH = "sherlock_holmes_text.txt"

sentences = file_to_sentences(FILE_PATH)

sentences = sentences[4:] # Crops out the preface

sentences = [word_tokenize(sent) for sent in sentences]
sentences[1]

all_words = [word for sentence in sentences for word in sentence]
vocabulary = set(all_words)

list(enumerate(vocabulary, 1))[0:10]

word_to_idx = {word : idx for idx, word in enumerate(vocabulary, 1)}
idx_to_word = {idx : word for word, idx in word_to_idx.items()}
vocab_size = len(vocabulary) + 1

input_sequences = []
for sentence in sentences:
  numerized_sentence = [word_to_idx[word] for word in sentence]
  for i in range(2, len(sentence) + 1):
    ngram = numerized_sentence[:i]
    input_sequences.append(ngram)

input_sequences[5:10]

max_sequence_len = max([len(seq) for seq in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre')

X = [sequence[:-1] for sequence in input_sequences]
y = [sequence[-1] for sequence in input_sequences]
y = to_categorical(y, num_classes=vocab_size)

# Building the RNN model
model = Sequential()

# Embedding layer
model.add(Embedding(vocab_size, 400, input_length=max_sequence_len-1))  # Increased embedding dimensions
model.add(SpatialDropout1D(0.25))
model.add(GaussianNoise(0.1))

# RNN 1
model.add(LSTM(512, dropout=0.25, recurrent_dropout=0.25))  # Increased units, added dropout
model.add(BatchNormalization())
model.add(Dropout(0.3))

# Final Layer
model.add(Dense(vocab_size, activation='softmax'))

model.compile(optimizer=Adam(lr=0.01, clipnorm=1.0), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

early_stop = EarlyStopping(monitor='val_loss', patience=20, verbose=1, restore_best_weights=True)

history = model.fit(X, y, epochs=200, batch_size=32, verbose=1, validation_split=0.2, callbacks=[early_stop])

model.save('model.h5')

model = load_model('model.h5')

model.summary()


def predict_next_word(model, text, max_sequence_len, word_to_index, index_to_word):
  

    # Tokenize the input string
    token_list = [word_to_index[word] for word in word_tokenize(text) if word in word_to_index]

    # Pad the token sequence
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')

    # Predict the token of the next word
    predicted_idx = np.argmax(model.predict(token_list), axis=-1)

    # Convert the token back to a word
    predicted_word = index_to_word.get(predicted_idx[0], '')

    return predicted_word

def predict_next_n_words(model, text, n, max_sequence_len, word_to_index, index_to_word):

    predicted_sequence = []

    for _ in range(n):
        # Tokenize the input string
        token_list = [word_to_index[word] for word in word_tokenize(text) if word in word_to_index]

        # Pad the token sequence
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')

        # Predict the token of the next word
        predicted_idx = np.argmax(model.predict(token_list), axis=-1)

        # Convert the token back to a word
        predicted_word = index_to_word.get(predicted_idx[0], '')

        # Append the predicted word to the sequence and to the text (for the next iteration)
        predicted_sequence.append(predicted_word)
        text += " " + predicted_word

    return ' '.join(predicted_sequence)

input_text = "Sherlock said "
prediction = predict_next_word(model, input_text, max_sequence_len, word_to_idx, idx_to_word)
print(input_text + " " + prediction)
