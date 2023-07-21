import re
import streamlit as st
import pandas as pd
import numpy as np
import json
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import load

nltk.download('punkt')
nltk.download('stopwords')

st.write("""
# Prediksi Ulasan Aplikasi Vidio
""")

w2i, i2w = {'Negatif': 0, 'Positif': 1}, {0: 'Negatif', 1: 'Positif'}

input = st.text_input('Teks Ulasan', '')

col1, col2, col3 = st.columns(3)
push = col3.button("Deteksi")

disp1, disp2, disp3 = st.columns(3)

def preprocess_text(text):
    # Menghapus tanda baca
    text = re.sub(r'[^\w\s]', '', text)
    
    # Tokenisasi
    tokens = nltk.word_tokenize(text)
    
    # Normalisasi kata
    tokens = [word.lower() for word in tokens]
    
    # Menghapus stopword
    stop_words = set(stopwords.words('indonesian'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    
    return ' '.join(tokens)

if push:
    preprocessed_input = preprocess_text(input)
    loaded_model = load('decision_tree_model.joblib')
    loaded_vectorizer = load('vectorizer.joblib')
    input_vector = loaded_vectorizer.transform([preprocessed_input])
    label_RF = loaded_model.predict(input_vector)[0]

    if label_RF == 0:
        st.markdown("""
            <style>
            div[data-testid="metric-container"] {
            background-color: rgba(28, 131, 225, 0.1);
            border: 1px solid rgba(28, 131, 225, 0.1);
            padding: 1% 1% 1% 1%;
            border-radius: 2px;
            color: rgb(255, 0, 0);
            overflow-wrap: break-word;
            }

            /* breakline for metric text         */
            div[data-testid="metric-container"] > label[data-testid="stMetricLabel"] > div {
            overflow-wrap: break-word;
            white-space: break-spaces;
            color: red;
            font-size: 200%;
            }
            </style>
            """, unsafe_allow_html=True)

    else:
        st.markdown("""
            <style>
            div[data-testid="metric-container"] {
            background-color: rgba(28, 131, 225, 0.1);
            border: 1px solid rgba(28, 131, 225, 0.1);
            padding: 1% 1% 1% 1%;
            border-radius: 2px;
            color: rgb(13, 252, 13);
            overflow-wrap: break-word;
            }

            /* breakline for metric text         */
            div[data-testid="metric-container"] > label[data-testid="stMetricLabel"] > div {
            overflow-wrap: break-word;
            white-space: break-spaces;
            color: green;
            font-size: 200%;
            }
            </style>
            """, unsafe_allow_html=True)

    score = loaded_model.predict_proba(input_vector)[0, label_RF] * 100
    disp2.metric(i2w[label_RF], f"{score:.2f}%")
