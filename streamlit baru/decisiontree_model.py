import pandas as pd
import numpy as np
import nltk
import pickle
import joblib

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier

nltk.download('punkt')
nltk.download('stopwords')

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

# Load the saved vectorizer
vectorizer = joblib.load('vectorizer.joblib')

# Load the saved model
loaded_model = joblib.load('decision_tree_model.joblib')

def predict_sentiment(text):
    preprocessed_text = preprocess_text(text)
    input_data = [preprocessed_text]

    input_vector = vectorizer.transform(input_data)

    # Prediction
    prediction = loaded_model.predict(input_vector)
    label = prediction.argmax()
    return label


def main():
    st.title('Decision Tree Model Deployment')

    #input variables
    Ulasan = st.text_input('ulasan')
    Cleaning = st.text_input('Cleaning')
    HapusEmoji = st.text_input('HapusEmoji')
    Spasi = st.text_input('Spasi')
    CaseFolding = st.text_input('CaseFolding')
    Tokenizing = st.text_input('Tokenizing')
    Normalized_Words = st.text_input('Normalized_Words')
    Stopword_Removal = st.text_input('Stopword Removal')
    Stemming = st.text_input('Stemming')

     # Combine input variables into a numpy array
    # Combine input variables into a numpy array
    input_data = np.array([[Ulasan, Cleaning, HapusEmoji, Spasi, CaseFolding, Tokenizing, Normalized_Words, Stopword_Removal, Stemming]])

    expected_n_features = loaded_model.n_features_
    n_features = input_data.shape[1]

    if n_features != expected_n_features:
        st.error(f"Jumlah fitur dalam data input ({n_features}) tidak sesuai dengan yang diharapkan oleh model ({expected_n_features}).")
        return
    
    #prediction code
    if st.button('Predict'):
        prediction = loaded_model.predict(input_data)
        st.success('Positif {}'.format(prediction))


    if __name__ =='__main__':
        main()
