# command to run: python -m flask --app .\flask_app.py run

from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import pickle

flask_app = Flask(__name__)

# Load the pre-trained LSTM model
model = tf.keras.models.load_model('lstm_model.h5')
tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))

ngram_length = 5 

@flask_app.route("/")
def home():
    return render_template("index.html")



# Predicts the next few words based on user input by sending sequnce to the model
def get_next_words(model, tokenizer, user_input, ngram, words_to_predict=3):
    predictions = []
    for _ in range(words_to_predict):
        sequence = tokenizer.texts_to_sequences([user_input])[0]
        padded_sequence = pad_sequences([sequence], maxlen=ngram-1, padding='pre')
        predicted_probs = model.predict(padded_sequence, verbose=0)
        predicted_index = np.argmax(predicted_probs, axis=-1)

        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                next_word = word
                break
        predictions.append(next_word)
        user_input += " " + next_word
    return ' '.join(predictions)




# Predicts the next single word
def get_next_word(model, tokenizer, user_input, ngram):
    sequence = tokenizer.texts_to_sequences([user_input])[0]
    padded_sequence = pad_sequences([sequence], maxlen=ngram-1, padding='pre')
    predicted_probs = model.predict(padded_sequence, verbose=0)
    predicted_index = np.argmax(predicted_probs, axis=-1)
    
    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            return word
    return ""



# Route to handle prediction requests
@flask_app.route('/prediction', methods=['POST'])
def prediction():
    user_input = request.form['input']
    next_word = get_next_word(model, tokenizer, user_input, ngram_length)
    next_words = get_next_words(model, tokenizer, user_input, ngram_length)
    return render_template("index.html", word_predicted=next_word,  words_predicted=next_words, input=user_input)


if __name__ == '__main__':
    flask_app.run(debug=True)
