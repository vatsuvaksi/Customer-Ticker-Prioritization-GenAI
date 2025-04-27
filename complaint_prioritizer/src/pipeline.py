import os
import pandas as pd
from src.preprocessing.cleaner import clean_text
# from src.preprocessing.tokenizer import tokenize_text
from src.preprocessing.vectorizer import vectorize_corpus
from src.config.config import RAW_DATA_PATH, PROCESSED_DATA_PATH
from src.preprocessing.tokenizer import TextPreprocessor

# Load data
# Clean it
# Tokenize
# Vectorize
# Do SOmthing about it now
# Save processed data
def run_pipeline():
    print("Starting preprocessing pipeline...")

    # Step 1: Load raw data
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"Raw data file not found at {RAW_DATA_PATH}")
    
    print(f"Loading data from {RAW_DATA_PATH}...")
    data = pd.read_csv(RAW_DATA_PATH)


    # Step 2: Clean text
    print("Cleaning text...")
    data['clean_text'] = data['complaint_text'].apply(clean_text)

    # Step 3: Tokenization
    print("Tokenizing text using Lemma and stop words ...")
    textPreProcessor = TextPreprocessor()
    # cleanText , ner = textPreProcessor.preprocess(data['clean_text'])
    data['clean_text_lemma'], data['ner'] = zip(*data['clean_text'].apply(textPreProcessor.preprocess))


    #Step 4: Extracting entitites for future filtering 
    # data['entities'] = ner

    # Step 4: Vectorization (TF-IDF)
    print("Vectorizing text...")
    X_features, vectorizer = vectorize_corpus(data['clean_text_lemma'].apply(lambda x: ' '.join(x)))
    print("tf-id vector " , vectorizer)

    # Step 5: Save processed data
    print(f"Saving processed data to {PROCESSED_DATA_PATH}...")
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    data.to_csv(PROCESSED_DATA_PATH, index=False)

    print("Pipeline completed successfully!")

    return X_features, data

    