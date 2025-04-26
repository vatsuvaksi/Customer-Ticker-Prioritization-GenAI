# src/preprocessing/tokenizer.py

import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

def tokenize_text(text):
    tokens = word_tokenize(text)
    return tokens
