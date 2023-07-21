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
from PIL import Image

nltk.download('punkt')
nltk.download('stopwords')

w2i, i2w = {'Negatif': 0, 'Positif': 1}, {0: 'Negatif', 1: 'Positif'}

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


menu = ["Home","Prediksi Analisis"]
choice = st.sidebar.selectbox("Menu", menu)


if choice == "Home":
    st.title("Selamat datang di Prediksi Ulasan Aplikasi Vidio")
    st.write("Ini adalah aplikasi prediksi ulasan aplikasi Vidio menggunakan model Decision Tree. Aplikasi ini dapat memprediksi apakah ulasan aplikasi Vidio bersifat positif atau negatif berdasarkan teks yang dimasukkan.")
    image = Image.open("logo.png")
    st.image(image, caption='Logo Vidio')
    st.write("Vidio adalah portal online atau situs web streaming video yang didirikan pada tahun 2014. Vidio memungkinkan pengguna untuk menonton dan menikmati berbagai video dan layanan lain seperti live chat dan bermain games melalui jaringan internet dan menyiarkannya secara streaming (live streaming dan Video on Demand). ")
    st.write("Keunggulan Vidio dibandingkan situs video lainnya yaitu Vidio adalah layanan streaming video lokal pertama yang menawarkan 21 saluran gratis (Free to Air/FTA), 32 saluran radio, dan ribuan konten menarik dimulai dengan Vidio Sports, yang menampilkan acara-acara olahraga terbesar dari dalam maupun luar negeri, seperti Liga 1, 2, 3 Indonesia, BWF Series, F1, World Tennis, UCL 2021, UEL 2021, NBA 2021 dan masih banyak lagi. Vidio juga memiliki serial bernama Vidio Originals dengan berbagai cerita pilihan film.")
    st.write("Hingga 4 Februari 2023, Vidio sudah terunduh sebanyak 50 juta dengan rating 3.7 dan 623 ribu ulasan, serta menempati urutan pertama pada kategori top grossing segmentasi entertainment di situs Google Play.")
    st.write("Dengan 1000 ulasan yang dikumpulkan, berikut jumlah data dari masing-masing rating:")
    image = Image.open("jumlah rating.png")
    st.image(image, caption='Diagram Jumlah Data Berdasarkan Rating')
    st.write("Keterangan:")
    st.write("rating 1 = 480")
    st.write("rating 2 = 64")
    st.write("rating 3 = 57")
    st.write("rating 4 = 60") 
    st.write("rating 5 = 339")
    st.write("Dalam penelitian ini, labeling dilakukan dengan metode average labeling dimana rating 1,2 dan 3 adalah negatif, serta 4 dan 5 adalah positif. Maka persentase positif dan negatif didapatkan pada gambar berikut:")
    image = Image.open("persentase data.png")
    st.image(image, caption='Persentase Data Ulasan Positif dan Negatif')
    st.write(" Dari persentase tersebut dapat disimpulkan bahwa pengguna aplikasi Vidio kebanyakan tidak merasa puas dan ini dapat menjadi evaluasi bagi pihak perusahaan untuk memperbaikinya.")
elif choice == "Prediksi Analisis":
    st.write("""
    # Prediksi Ulasan Aplikasi Vidio
    """)
    
    input_text = st.text_input('Teks Ulasan', '')
    label_RF = None

    col1, col2, col3 = st.columns(3)
    push = col3.button("Deteksi")

    disp1, disp2, disp3 = st.columns(3)

    if push:
        preprocessed_input = preprocess_text(input_text)
        loaded_model = load('decision_tree_model3.joblib')
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
