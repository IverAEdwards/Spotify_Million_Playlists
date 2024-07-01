import json
import pickle
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, concatenate

file_path1 = r"C:\Users\tup30579\Spotify Million\data_store\named_data\all_names.json"
file_path2 = r"C:\Users\tup30579\Spotify Million\data_store\named_data\all_tracks.json"

with open(file_path1, 'r', encoding='utf-8') as file:
    names = json.load(file)
with open(file_path2, 'r', encoding='utf-8') as file:
    tracks = json.load(file)

flat_tracks = [track for sublist in tracks for track in sublist]

# Encode track IDs
track_encoder = LabelEncoder()
track_ids_encoded = track_encoder.fit_transform(flat_tracks)

# Tokenize playlist names
tokenizer = Tokenizer(lower=True)
tokenizer.fit_on_texts(names)

# Convert playlist names to sequences of integers
names_tokenized = tokenizer.texts_to_sequences(names)

# Pad sequences to ensure they have the same length
max_seq_len = max(len(seq) for seq in names_tokenized)
names_padded = pad_sequences(names_tokenized, maxlen=max_seq_len, padding='post')

# Prepare the data for training
X, y = [], []
track_idx = 0
for seq in names_padded:
    for i in range(1, len(seq)):
        X.append(seq[:i])
        y.append(seq[i])
    track_idx += 1

X = pad_sequences(X, maxlen=max_seq_len, padding='post')
y = np.array(y)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)






embedding_dim = 50

# Input for playlist names
input_names = Input(shape=(max_seq_len,))
embedding_names = Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=embedding_dim)(input_names)
lstm_names = LSTM(50)(embedding_names)

# Input for track IDs
input_tracks = Input(shape=(max_seq_len,))
embedding_tracks = Embedding(input_dim=len(track_encoder.classes_), output_dim=embedding_dim)(input_tracks)
lstm_tracks = LSTM(50)(embedding_tracks)

# Concatenate the LSTM outputs
concatenated = concatenate([lstm_names, lstm_tracks])

# Output layer
output = Dense(len(track_encoder.classes_), activation='softmax')(concatenated)

# Create model
model = Model(inputs=[input_names, input_tracks], outputs=output)

# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()

# Train the model
model.fit([X_train, X_train], y_train, epochs=2, batch_size=64, validation_data=([X_test, X_test], y_test))

# Evaluate the model
loss, accuracy = model.evaluate([X_test, X_test], y_test)
print(f"Test Accuracy: {accuracy}")

model.save(r"C:\Users\tup30579\Spotify Million\results\EmbeddedNameModel\name_recommender_model.h5")

# Save the tokenizer
with open(r"C:\Users\tup30579\Spotify Million\results\EmbeddedNameModel\tokenizer.pickle", 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Save the label encoder
with open(r"C:\Users\tup30579\Spotify Million\results\EmbeddedNameModel\track_encoder.pickle", 'wb') as handle:
    pickle.dump(track_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Model, tokenizer, and label encoder saved successfully.")