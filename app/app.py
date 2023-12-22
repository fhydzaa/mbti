import nltk
import tensorflow as tf
import pandas as pd
import re
from flask import Flask, jsonify, request, render_template
from nltk import word_tokenize
from nltk.corpus import stopwords
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from nltk.stem import PorterStemmer, WordNetLemmatizer, wordnet

app = Flask(__name__)
porter_stemmer = PorterStemmer()
wordnet_lemmatizer = WordNetLemmatizer()
max_words = 2500  # Assuming you want to consider the top 10,000 words
max_len = 640  # Assuming you want to limit each comment to 100 words
def convert_mbti(result):
    return pd.Series([
        'I' if result[0] == 1 else 'E',  # I/E
        'N' if result[1] == 1 else 'S',  # N/S
        'T' if result[2] == 1 else 'F',  # T/F
        'J' if result[3] == 1 else 'P',  # J/P
    ])
def lemmatize_with_pos(word, pos):
    # Map POS tags to WordNet POS tags
    pos_mapping = {
        'N': wordnet.NOUN,
        'V': wordnet.VERB,
        'R': wordnet.ADV,
        'J': wordnet.ADJ
    }
    return wordnet_lemmatizer.lemmatize(word, pos=pos_mapping.get(pos, wordnet.NOUN))

# Define a function for lemmatization
def lemmatize_tokens(tokens):
    # Lemmatize each token
    return [lemmatize_with_pos(token, pos[0]) for token, pos in nltk.pos_tag(tokens)]

def clean_text(text):
    # Remove links
    text = re.sub(r'\b(?:https?|ftp):\/\/[^\s/$.?#].[^\s]*\b', ' ', str(text))

    # Convert to lowercase
    text = text.lower()

    # Remove text inside square brackets
    text = re.sub(r'\[.*?\]', ' ', str(text))

    # Remove all punctuation
    text = re.sub(r'[.,|\/#!$%\^&\*;:{}=\-_`~()\[\]"\'<>?]', ' ', str(text))

    # Remove mbti in text
    unique_type_list = ['INFJ', 'ENTP', 'INTP', 'INTJ', 'ENTJ', 'ENFJ', 'INFP', 'ENFP',
       'ISFP', 'ISTP', 'ISFJ', 'ISTJ', 'ESTP', 'ESFP', 'ESTJ', 'ESFJ']
    unique_type_list = [x.lower() for x in unique_type_list]
    for t in unique_type_list:
      text = text.replace(t,"")

    # remove space >1
    text = re.sub(' +', ' ', str(text)).lower()

    # remove multiple character in word
    text = re.sub(r'\b(?:\w*(\w)\1\w*)+\b', '', str(text))

    #remove number
    text = re.sub(r'\d+', '', str(text))

    return text

def remove_stopwords(text, stopwords):
    words = text.split()
    words = [word for word in words if word not in stopwords]
    return ' '.join(words)

def preprocess_text(long_text, tokenizer, max_len):

    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(long_text)

    # Convert text df to numerical sequences
    long_text_sequences = tokenizer.texts_to_sequences(long_text)

    # Pad sequences to ensure uniform length
    # max_sequence_length = 100  # Adjust based on the desired sequence length
    long_text_padded = pad_sequences(long_text_sequences, maxlen=max_len)
    return long_text_padded


@app.route("/")
def Home():
    return {"health_check": "NGENE TO?", "model_version": "OKE?"}

@app.route("/predict", methods=["GET"])
def predict():
    # Get the value of 'parameters' from the query string
    values_input = request.args.get('parameters', '')

    result = []

    # Your long text input
    long_text = values_input

    df_test = pd.DataFrame({'posts': [long_text]})
    df_test['posts_clean'] = df_test['posts'].apply(clean_text)
    stopwords_set = set(stopwords.words('english'))
    df_test['posts_clean'] = df_test['posts_clean'].apply(lambda x: remove_stopwords(x, stopwords_set))
    df_test['posts_tokens'] = df_test['posts_clean'].apply(word_tokenize)
    df_test['posts_lemmatized'] = df_test['posts_tokens'].apply(lemmatize_tokens)
    long_text = df_test['posts_lemmatized']

    models = [load_model(r'Models/best_model_I_E.h5'),
              load_model(r'Models/best_model_J_P.h5'),
              load_model(r'Models/best_model_N_S.h5'),
              load_model(r'Models/best_model_T_F.h5')]

    for model_name, model in zip(['IE', 'JP', 'NS', 'TF'], models):
        tokenizer = Tokenizer(num_words=2500, oov_token="<OOV>")
        input_data = preprocess_text(long_text, tokenizer, max_len)

        # Make predictions
        predictions = model.predict(input_data)
        rounded_predictions = np.round(predictions).astype(int)

        # Print the predictions or use them as needed
        print(f"Predictions for model {model_name}: {rounded_predictions}")
        result.append(rounded_predictions[0])

    mbti_series = np.array(convert_mbti(result))
    mbti_string = ''.join(mbti_series)
    print(mbti_string)
    return jsonify({"mbti": mbti_string})

if __name__ == "__main__":
    app.run(debug=True, port=5002)

