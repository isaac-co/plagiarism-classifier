from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams
import string
import re

class TextProcessor:
    
    def __init__(self, language='english'):
        self.language = language
        
        # Instantiate StopWords
        self.stopwords = stopwords.words(self.language)
        
        # Instantiate stemmer or lemmatizer (for english only)
        if self.language == 'english':
            self.lemmatizer = WordNetLemmatizer()        
        else:
            self.stemmer = SnowballStemmer(self.language)
            
    def __str__(self): 
        return f'TextProcessor for {self.language} at 0x{id(self)}'
    
    def __repr__(self): 
        return f'TextProcessor for {self.language} at 0x{id(self)}'
           
    def preprocess(self, text):
        """Parse a text with the following steps:
          - Remove punctuation
          - Remove extra spaces
          - Remove stop words
          - Remove one letter words
          - Lowering the text
          - Stem / Lemmatize the words
          
        Parameters:
            text (str): A string of text
        """
        
        # Remove punctuation
        processed = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra spaces
        processed = re.sub('\s+', ' ', processed)
        
        # Remove stopwords
        processed = [word for word in processed.split() if word not in self.stopwords]
        
        # Remove one letter words
        processed = ' '.join([word for word in processed if len(word) > 1])
        
        # Lowering the text
        processed = processed.lower()
        
        # Stemming / Lemmatazing
        processed = self.stem_lemmatize(processed)
        
        return processed
    
    def stem_lemmatize(self, text):
        """Stem or lemmatize a word depending on the language.
        
        Parameters:
            text (str): A string of text
        """
           
        # Reduce each word
        if self.language == 'english':
            text = [self.lemmatizer.lemmatize(word) for word in text.split()]      
        else:
            text = [self.stemmer.stem(word) for word in text.split()]

        return ' '.join(text)
    
        
    def get_tokens(self, text):
        """Preprocess a text and then return the word tokens
        
        Parameters:
            text (str): A string of text
        """
        proccess = self.preprocess(text)
        
        return word_tokenize(proccess)
    
    def get_ngrams(self, text, n=1):
        """Returns a string with each word transformed into ngrams
        
        Parameters:
            text (str): A string of text
            n (int):    The number of ngrams to use (n=2 : bigrams ; n=3 : trigrams)
        """
        n_grams = []
        for ngram in ngrams(self.get_tokens(text), n):
            n_grams.append(''.join(ngram))

        return ' '.join(n_grams)
    
    def get_ngram_tokens(self, text, n=1):
        """Returns a list with each ngram as a token
        
        Parameters:
            text (str): A string of text
            n (int):    The number of ngrams to use (n=2 : bigrams ; n=3 : trigrams)
        """
        n_grams = []
        for ngram in ngrams(self.get_tokens(text), n):
            #n_grams.append(''.join(ngram))
            n_grams.append(ngram)

        return n_grams