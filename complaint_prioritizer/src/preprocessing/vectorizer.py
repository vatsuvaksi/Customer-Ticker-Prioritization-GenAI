# src/preprocessing/vectorizer.py

from sklearn.feature_extraction.text import TfidfVectorizer

def vectorize_corpus(corpus):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    return X, vectorizer
