import os

import pandas as pd
import numpy as np
import re

from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet

# Ignore noise warning
import warnings
warnings.filterwarnings("ignore")

import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

max_words = 2500
max_len = 640
embedding_dim = 150
lstm_units_layer1 = 128
lstm_dropout_layer1 = 0.5
lstm_recurrent_dropout_layer1 = 0.2
lstm_units_layer2 = 64
lstm_dropout_layer2 = 0.2
dense_units = 64
porter_stemmer = PorterStemmer()
wordnet_lemmatizer = WordNetLemmatizer()


# Define a function for lemmatization with POS tagging
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
def split_mbti(type):
    return pd.Series([
        1 if type[0] == 'I' else 0,  # I/E
        1 if type[1] == 'N' else 0,  # N/S
        1 if type[2] == 'T' else 0,  # T/F
        1 if type[3] == 'J' else 0   # J/P
    ])

def clean_text(text):
    # Remove links
    text = re.sub(r'\b(?:https?|ftp):\/\/[^\s/$.?#].[^\s]*\b', ' ', text)

    # Convert to lowercase
    text = text.lower()

    # Remove text inside square brackets
    text = re.sub(r'\[.*?\]', ' ', text)

    # Remove all punctuation
    text = re.sub(r'[.,|\/#!$%\^&\*;:{}=\-_`~()\[\]"\'<>?]', ' ', text)

    # Remove mbti in text
    unique_type_list = ['INFJ', 'ENTP', 'INTP', 'INTJ', 'ENTJ', 'ENFJ', 'INFP', 'ENFP',
       'ISFP', 'ISTP', 'ISFJ', 'ISTJ', 'ESTP', 'ESFP', 'ESTJ', 'ESFJ']
    unique_type_list = [x.lower() for x in unique_type_list]
    for t in unique_type_list:
      text = text.replace(t,"")

    # remove space >1
    text = re.sub(' +', ' ', text).lower()

    # remove multiple character in word
    text = re.sub(r'\b(?:\w*(\w)\1\w*)+\b', '', text)

    #remove number
    text = re.sub(r'\d+', '', text)

    return text

def remove_stopwords(text, stopwords):
    words = text.split()
    words = [word for word in words if word not in stopwords]
    return ' '.join(words)

def make_models():
    model = Sequential()
    model.add(Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_len))
    model.add(
        LSTM(units=lstm_units_layer1, dropout=lstm_dropout_layer1, recurrent_dropout=lstm_recurrent_dropout_layer1,
             return_sequences=True))
    model.add(LSTM(units=lstm_units_layer2, dropout=lstm_dropout_layer2))
    model.add(Dense(units=dense_units, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))
    return model

def mbti_model():
    # Load dataset
    df = pd.read_csv('C:\MY FILE\Document\KULIAH\BANGKIT\Tensorflow-Simulation\Capstone Project\mbti\Datasets\mbti_1.csv')

    # Add dataset to train
    X_train = df['posts'].values
    y_train = df['type'].values

    # Add random over sampling to balance data
    ros = RandomOverSampler(sampling_strategy='minority', random_state=42)
    X_train_resampled, y_train_resampled = ros.fit_resample(X_train.reshape(-1, 1), y_train)
    df_resampled = pd.DataFrame({'posts': X_train_resampled.flatten(), 'type': y_train_resampled})
    df = df_resampled

    # Split type column to 4 column and reindex
    df[['I|E', 'N|S', 'T|F', 'J|P']] = df['type'].apply(split_mbti)
    columns = ['type', 'I|E', 'N|S', 'T|F', 'J|P', 'posts']
    df = df.reindex(columns=columns)

    # clean posts text
    df['posts_clean'] = df['posts'].apply(clean_text)

    # Remove stopwords in posts
    stopwords_set = set(stopwords.words('english'))
    df['posts_clean'] = df['posts_clean'].apply(lambda x: remove_stopwords(x, stopwords_set))

    # Tokenize using NLTK's word_tokenize
    df['posts_tokens'] = df['posts_clean'].apply(word_tokenize)

    # Apply lemmatization to the 'tokens' column
    df['posts_lemmatized'] = df['posts_tokens'].apply(lemmatize_tokens)

    # Load model
    model = make_models()

    history_dict = {}
    for dimension in ['I|E', 'N|S', 'T|F', 'J|P']:
        X_train, X_test, y_train, y_test = train_test_split(
            df['posts_lemmatized'].values,
            df[f'{dimension}'].values,
            test_size=0.2,
            random_state=42
        )

        print(f"{dimension} : {df[f'{dimension}'].shape}")
        print(df['posts_lemmatized'].shape)
        # Tokenize the text df
        max_words = 2500  # Assuming you want to consider the top 10,000 words
        max_len = 640  # Assuming you want to limit each comment to 100 words
        tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
        tokenizer.fit_on_texts(X_train)

        # Convert text df to numerical sequences
        X_train_sequences = tokenizer.texts_to_sequences(X_train)
        X_test_sequences = tokenizer.texts_to_sequences(X_test)

        # Pad sequences to ensure uniform length
        # max_sequence_length = 100  # Adjust based on the desired sequence length
        X_train_padded = pad_sequences(X_train_sequences, maxlen=max_len)
        X_test_padded = pad_sequences(X_test_sequences, maxlen=max_len)

        # Build and compile the LSTM model
        model.compile(optimizer=RMSprop(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

        # Train the model
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

        history = model.fit(
            X_train_padded, y_train,
            epochs=1,
            batch_size=128,
            validation_split=0.1,
            callbacks=[early_stopping]
        )
        history_dict[dimension] = history.history
        directory = 'C:\MY FILE\Document\KULIAH\BANGKIT\Tensorflow-Simulation\Capstone Project\mbti\Models'
        filename = f'best_model_{dimension}.h5'

        # Ensure the directory exists
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Save the model to a file
        model.save(os.path.join(directory, filename))
        # Evaluate the model on the test set
        loss, accuracy = model.evaluate(X_test_padded, y_test)
        print(f'Test Accuracy ({dimension}): {accuracy}')
        print(f'Best Test Loss ({dimension}): {loss}')

mbti_model()




