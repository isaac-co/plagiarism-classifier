from .utils import get_language, classify_by_language
from .processing import TextProcessor
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import pairwise
import pandas as pd
import numpy as np
import difflib
import pickle
import os

class TextSimilarity:
    
    _NGRAM_NAMES = ['','unigrams','bigrams','trigrams']
    
    def __init__(self):
        # Instantiate classifier when creating the class
        with open('model/unigrams_model.pkl', 'rb') as f:
            self.__unigram_model = pickle.load(f)
        with open('model/bigrams_model.pkl', 'rb') as f:
            self.__bigram_model = pickle.load(f)
        with open('model/mixed_model.pkl', 'rb') as f:
            self.__mixed_model = pickle.load(f)

    def bow_cosine_similarity(self, t1, t2):
        """Returns the cosine similarity using a BoW vector

        Parameters:
            t1 (str): A string of text
            t2 (str): A string of text
        """
        # Create corpus
        corpus = [t1] + [t2]

        # Create word vector
        vectorizer = CountVectorizer().fit_transform(corpus)
        vectors = vectorizer.toarray()

        # Compute cosine similarity
        similarity_matrix = pairwise.cosine_similarity(vectors)

        # Aggregate the similarity scores
        similarity = np.mean(similarity_matrix[1][0])

        return similarity

    def tfidf_cosine_similarity(self, t1, t2):
        """Returns the cosine similarity using a TF-IDF vector

        Parameters:
            t1 (str): A string of text
            t2 (str): A string of text
        """
        # Create corpus
        corpus = [t1] + [t2]

        # Create word vector
        vectorizer = TfidfVectorizer().fit_transform(corpus)
        vectors = vectorizer.toarray()

        # Compute cosine similarity
        similarity_matrix = pairwise.cosine_similarity(vectors)

        # Aggregate the similarity scores
        similarity = np.mean(similarity_matrix[1][0])

        return similarity

    def get_jaccard_similarity(self, t1, t2):
        """Returns the Jaccard similarity score of two list of word tokens

        Parameters:
            t1 (str): A string of text
            t2 (str): A string of text
        """
        t1_tokens = set(t1.split())
        t2_tokens = set(t2.split())

        intersection = t1_tokens.intersection(t2_tokens)
        union = t1_tokens.union(t2_tokens)

        return float(len(intersection)) / len(union)

    def compare_documents(self, base_document_path, document_to_compare_path, ngrams=[1],
                          txtProcessor=None, predict=True, html=False):
        """Compare two documents with TF-DIF vectorization and returns the Cosine Similarity
        using the specified ngrams. (Unigrams, Bigrams and Trigrams supported)

        Parameters:
            plag_file_paths (list): A list of file paths
            og_file_paths (list): A list of file paths
            ngrams (list): A list of numbers ranging from 1 to 3
                           representing the ngrams it will use to compare
            txtProcessor (class): Existing instance of TextProcessor() to use
        """
        # Read Documents contents
        with open(base_document_path, encoding="utf8") as file:
            base_document = file.read()
        with open(document_to_compare_path, encoding="utf8") as file:
            document_to_compare = file.read()

        if get_language(base_document) != get_language(document_to_compare):
            return 'Documents are not in the same language'

        if txtProcessor is None:
            txtProcessor = TextProcessor(get_language(base_document))

        gram_type = {}

        for n in ngrams:
            # Process both documents
            t1 = txtProcessor.get_ngrams(base_document, n)
            t2 = txtProcessor.get_ngrams(document_to_compare, n)

            # Get match info
            sequence, msize, ma, mb = self.__matches(base_document, document_to_compare)
            gram_type['sequence'] = sequence
            gram_type['length'] = msize
            gram_type['base_pointer'] = ma
            gram_type['compare_pointer'] = mb

            # Calculate cosine similarity
            score = self.tfidf_cosine_similarity(t1, t2)
            gram_type[f'{self._NGRAM_NAMES[n]}'] = round(score, 4)

        df = pd.DataFrame(data=gram_type, index=[0])

        base_name = os.path.basename(base_document_path)
        compare_name = os.path.basename(document_to_compare_path)

        df['base'] = base_name
        df['compare'] = compare_name

        # Get final structure
        cols = ['base','compare'] + [self._NGRAM_NAMES[n] for n in ngrams] + ['sequence','length','base_pointer','compare_pointer']
        df = df[cols]

        if predict:
            if 1 in ngrams and 2 in ngrams:
                model = self.__mixed_model
            elif 1 in ngrams:
                model = self.__unigram_model
            elif 2 in ngrams:
                model = self.__bigram_model

            y_pred = model.predict(df.drop(['base','compare','sequence','length','base_pointer','compare_pointer'], axis=1))
            df['label'] = y_pred

        if html:
            self.__makeHTML(base_document, document_to_compare, base_name, compare_name, txtProcessor)

        return df

    def compare_document_paths(self, base_doc, doc_to_compare, ngrams=[1], predict=True, html=False):
        """Compares each document in plag_file_paths against each document
        in og_file_paths with TF-DIF vectorization and returns the Cosine Similarity
        using the specified ngrams (Unigrams, Bigrams and Trigrams supported).

        Parameters:
            plag_file_paths (list): A list of file paths
            og_file_paths (list): A list of file paths
            ngrams (list): A list of numbers ranging from 1 to 3
                           representing the ngrams it will use to compare
        """
        # Verify paths
        if isinstance(base_doc, str):
            base_path = os.path.dirname(base_doc)
        elif isinstance(base_doc, list) and len(base_doc) > 0:
            base_path = os.path.dirname(base_doc[0])
        else:
            raise ValueError("Invalid input format. Expected file path (string) or list of file paths.")

        if isinstance(doc_to_compare, str):
            compare_path = os.path.dirname(doc_to_compare)
        elif isinstance(doc_to_compare, list) and len(doc_to_compare) > 0:
            compare_path = os.path.dirname(doc_to_compare[0])
        else:
            raise ValueError("Invalid input format. Expected file path (string) or list of file paths.")

        # Classify documents by language
        base_docs = classify_by_language(base_doc)
        docs_to_compare = classify_by_language(doc_to_compare)

        unique_languages = set(list(base_docs.keys())).intersection(set(list(docs_to_compare.keys())))

        # Initialize processors for each language in the documents
        processors = {}
        for key in unique_languages:
            processors[key] = TextProcessor(language=key)

        cols = [self._NGRAM_NAMES[n] for n in ngrams]
        results = pd.DataFrame(columns=['base', 'compare'] + cols)

        for language in unique_languages:
            txtProcessor = processors.get(language)
            scores = {}
            for base_document_path in base_docs[language]:
                for document_to_compare_path in docs_to_compare[language]:
                    # Calculate cosine similarity
                    score = self.compare_documents(base_document_path, document_to_compare_path,
                                                   ngrams=ngrams,
                                                   txtProcessor=txtProcessor,
                                                   predict=False)
                    results = pd.concat([results, score], ignore_index=True)

        # Classify
        labels = self.__classify(results, ngrams, predict=predict)

        # Create HTML documents
        if html:
            # Function to read file contents
            def read_file(file_path):
                with open(file_path, 'r') as f:
                    return f.read()

            # Iterate over the rows of the DataFrame
            for _, row in labels.iterrows():
                base_file = row['base']
                compare_file = row['compare']

                base_file_path = os.path.join(base_path, base_file)
                compare_file_path = os.path.join(compare_path, compare_file)

                base_content = read_file(base_file_path)
                compare_content = read_file(compare_file_path)

                self.__makeHTML(base_content, compare_content, base_file, compare_file, txtProcessor)

        return labels


    def __matches(self, base, compare):
        """Compares each document in plag_file_paths against each document
        in og_file_paths with TF-DIF vectorization and returns the Cosine Similarity
        using the specified ngrams (Unigrams, Bigrams and Trigrams supported).

        Parameters:
            base (str): The reference or target string that you are comparing other strings against.
            compare (str): The string that you are evaluating or measuring against the base string.
        """
        matcher = difflib.SequenceMatcher(a=base, b=compare)
        match = matcher.find_longest_match(0, len(base), 0, len(compare))

        sequence = str(base[match.a:match.a+match.size].strip())

        return sequence, match.size, match.a, match.b

    def __classify(self, df, ngrams, predict=True):
        """Predicts the label of a document given it's cosine similarity scores"""
        # Decide which model to use
        if 1 in ngrams and 2 in ngrams:
            model = self.__mixed_model
        elif 1 in ngrams:
            model = self.__unigram_model
        elif 2 in ngrams:
            model = self.__bigram_model

        # Create df as previously trained
        # Find the rows with maximum unigrams and bigrams values for each base
        cols = [self._NGRAM_NAMES[n] for n in ngrams]
        max_rows = df.groupby('base')[cols].idxmax()
        preds = df.loc[max_rows.values.flatten().tolist()]

        # Filter out duplicate rows
        preds = preds[~preds[cols].duplicated()]

        # Choose the row with the highest bigrams value
        if 2 in ngrams:
            preds = preds.sort_values('bigrams', ascending=False).groupby('base').head(1)

        # Sort by document name
        preds.sort_values(by='base', ascending=True, inplace=True)

        if predict:
            base = preds['base']
            comp = preds['compare']
            seq = preds['sequence']
            leng = preds['length']
            basep = preds['base_pointer']
            compp = preds['compare_pointer']
            preds.drop(['base','compare','sequence','length','base_pointer','compare_pointer'], axis=1, inplace=True)

            # Predict labels
            y_pred = model.predict(preds)

            # Add IDs and predicted label
            preds['base'] = base
            preds['compare'] = comp
            preds['label'] = y_pred
            preds['sequence'] = seq
            preds['length'] = leng
            preds['base_pointer'] = basep
            preds['compare_pointer'] = compp

            # Organize columns
            cols = ['base','compare'] + [self._NGRAM_NAMES[n] for n in ngrams] + ['label','sequence','length','base_pointer','compare_pointer']
            preds = preds[cols]
        
        preds.reset_index(drop=True, inplace=True)
        
        return preds
    
        
    def __makeHTML(self, base, compare, base_name, compare_name, txtProcessor):
        """Generates a HTML with the comparison of the two documents received"""
        
        directory = "output"

        if not os.path.exists(directory):
            os.makedirs(directory)

        html_diff = difflib.HtmlDiff().make_file(
                txtProcessor.get_tokens(base), 
                txtProcessor.get_tokens(compare))

        with open(f'{directory}/{base_name}-{compare_name}.html', 'w', encoding='utf-8') as f:
            f.write(html_diff)
            
        return
