import tensorflow as tf
import numpy as np
import pandas as pd
# Define the input and output data
data=pd.read_csv("./dataset/tweet_emotions.csv")
# Extract input sentences and corresponding emotions
sentences = [row[2] for row in data]
emotions = [row[1] for row in data]

# Tokenize the input sentences (convert text to numerical representation)
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(sentences)
encoded_sentences = tokenizer.texts_to_sequences(sentences)

# Pad sequences to ensure uniform input size
max_length = max([len(seq) for seq in encoded_sentences])
padded_sentences = tf.keras.preprocessing.sequence.pad_sequences(encoded_sentences, maxlen=max_length, padding='post')

# One-hot encode the output emotions
label_encoder = tf.keras.preprocessing.text.Tokenizer()
label_encoder.fit_on_texts(emotions)
encoded_emotions = np.array(label_encoder.texts_to_matrix(emotions))

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=64, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(label_encoder.word_index)+1, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(padded_sentences, encoded_emotions, epochs=10, batch_size=32)

# Convert the model to TFLite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model to a file
with open('emotion_recognition_model.tflite', 'wb') as f:
    f.write(tflite_model)
