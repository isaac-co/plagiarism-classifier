# File: tests/test_01_processor.py

from unittest import TestCase
from textMinator.processing import TextProcessor

class TestProccesor(TestCase):

    def setUp(self):
        self.p = TextProcessor(language='english')

    def test_preprocessing_special(self):
        self.assertEqual(
            'abc3',
            self.p.preprocess('a!b!!c???3'))
            
    def test_preprocessing_spaces(self):
        self.assertEqual(
            'many space',
            self.p.preprocess('many          spaces'))
        
    def test_preprocessing_stopwords(self):
        self.assertEqual(
            'important word',
            self.p.preprocess('the a an it e important words'))
        
    def test_tokenization(self):
        self.assertEqual(
            ['we', 'love', 'token'],
            self.p.get_tokens('We love tokens!'))
        
    def test_ngrams1(self):
        self.assertEqual(
            'this test ngrams',
            self.p.get_ngrams('This is a test for ngrams', 1))

    def test_ngrams2(self):
        self.assertEqual(
            'thistest testngrams',
            self.p.get_ngrams('This is a test for ngrams', 2))
        
    def test_ngrams_tokens1(self):
        self.assertEqual(
            [('this',), ('test',), ('ngrams',)],
            self.p.get_ngram_tokens('This is a test for ngrams', 1))
        
    def test_ngrams_tokens2(self):
        self.assertEqual(
            [('this', 'test'), ('test', 'ngrams')],
            self.p.get_ngram_tokens('This is a test for ngrams', 2))
        
    def text_multiline(self):
        self.assertEqual(
            'messy text 123 filled random symbol lot space parse',
            self.p.get_ngrams("""Messy text !!# [123[]*[] filled with random symbols

            and      a  lot of spaces 

            to parse"
            """, 1))