# File: tests/test_01_processor.py

from unittest import TestCase
from textMinator.similarity import TextSimilarity
import os
import pandas as pd

class TestSimilarity(TestCase):

    def setUp(self):
        self.s = TextSimilarity()
        self.test_file1 = os.path.join(os.getcwd(), 'tests', 'flat', 'testfile1.txt')
        self.test_file2 = os.path.join(os.getcwd(), 'tests', 'flat', 'testfile2.txt')
        self.test_file3 = os.path.join(os.getcwd(), 'tests', 'flat', 'testfile3.txt')

    def test_bow_same_text(self):
        self.assertEqual(
            1,
            int(self.s.bow_cosine_similarity('two same strings', 'two same strings')))
        
    def test_tdif_same_text(self):
        self.assertEqual(
            1,
            int(self.s.tfidf_cosine_similarity('two same strings', 'two same strings')))
        
    def test_jaccard_same_text(self):
        self.assertEqual(
            1,
            int(self.s.tfidf_cosine_similarity('two same strings', 'two same strings')))
        
    def test_compare_same_documents(self):
        self.assertEqual(
            [['testfile1.txt', 'testfile2.txt', 1.0, 'this is a test file', 19, 0, 0, 'copy']],
            self.s.compare_documents(self.test_file1, self.test_file2).values.tolist())

    def test_compare_different_documents(self):
        self.assertEqual(
            [['testfile1.txt', 'testfile3.txt', 0.0, 'a', 2, 7, 11, 'copy']],
            self.s.compare_documents(self.test_file1, self.test_file3).values.tolist())
        
    def test_compare_same_bigrams(self):
        self.assertEqual(
            [['testfile1.txt', 'testfile2.txt', 1.0, 1.0, 'this is a test file', 19, 0, 0, 'copy']],
            self.s.compare_documents(self.test_file1, self.test_file2, ngrams=[1,2]).values.tolist())

    def test_compare_different_bigrams(self):
        self.assertEqual(
            [['testfile1.txt', 'testfile3.txt', 0.0, 0.0, 'a', 2, 7, 11, 'copy']],
            self.s.compare_documents(self.test_file1, self.test_file3, ngrams=[1,2]).values.tolist())
            