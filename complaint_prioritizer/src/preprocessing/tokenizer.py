import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus.reader.wordnet import NOUN, VERB, ADJ, ADV
from src.utils.utils import get_wordnet_pos
from nltk import ne_chunk
from nltk import pos_tag


class TextPreprocessor:
    def __init__(self, remove_stopwords=True):
        self.lemmatizer = WordNetLemmatizer()
        self.remove_stopwords = remove_stopwords
        self.stop_words = set()
        self.__download_nltk_resources()

        if self.remove_stopwords:
            self.stop_words = set(stopwords.words('english'))
    
    def __download_nltk_resources(self):
        def download(resource_path, download_name=None):
            if download_name is None:
                download_name = resource_path.split('/')[-1]
            try:
                nltk.data.find(resource_path)
            except LookupError:
                nltk.download(download_name)
        
        download('tokenizers/punkt')
        download('corpora/wordnet')
        download('corpora/omw-1.4')
        download('corpora/stopwords')
        download('averaged_perceptron_tagger_eng')
        download('maxent_ne_chunker')
        download('maxent_ne_chunker_tab')
        download('words')
        
    def _tokenize(self, text):
        return word_tokenize(text)
    
    def _lemmatize_tokens_and_NER(self, tokens):
        lemmatized = []
        ner = []
        
        # POS tagging for tokens
        tagged_tokens = pos_tag(tokens)
        
        # Perform NER on the entire sentence
        chunked = ne_chunk(tagged_tokens)
        
        # Extract named entities from the chunked tree
        named_entities = []
        for subtree in chunked:
            if isinstance(subtree, nltk.Tree):  # Check if it's a named entity
                entity = " ".join([word for word, tag in subtree.leaves()])
                entity_type = subtree.label()  # 'GPE', 'PERSON', etc.
                named_entities.append((entity, entity_type))

        # Lemmatize tokens
        for token, tag in tagged_tokens:
            wordnet_pos = get_wordnet_pos(tag)  # Convert POS tag to WordNet POS
            lemma = self.lemmatizer.lemmatize(token, pos=wordnet_pos)
            lemmatized.append(lemma)

        # Return both lemmatized tokens and named entities
        return lemmatized, named_entities

    
    def _remove_stop_words(self, tokens):
        return [token for token in tokens if token.lower() not in self.stop_words]
    
    def preprocess(self, text):
        tokens = self._tokenize(text)
        lemmatized_tokens , ner = self._lemmatize_tokens_and_NER(tokens)
        
        if self.remove_stopwords:
            lemmatized_tokens = self._remove_stop_words(lemmatized_tokens)
        
        return lemmatized_tokens , ner
    