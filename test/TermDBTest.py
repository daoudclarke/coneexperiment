import unittest
import logging
import numpy as np
from numpy import random
from StringIO import StringIO
from utils import randomWord

from sklearn.neighbors import KNeighborsClassifier

from coneexperiment.TermDB import TermDB

class TermDBTestCase(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(filename='log/unittest.log',
                            level=logging.DEBUG,
                            format='%(asctime)s %(process)d %(levelname)s %(message)s')
        logging.info("Starting test: %s", self._testMethodName)
        random.seed(1001)

    def testTermDBTermExists(self):
        term_db = TermDB('test_data/nouns-deps-small-head.mi')
        term_db.nouns.load(['south','termthatdoesntexist'])
        term_vector = term_db.nouns['south']
        self.assertTrue(type(term_vector) == dict)
        self.assertTrue(len(term_vector) > 0)

    def testTermDBTermNotExists(self):
        term_db = TermDB('test_data/nouns-deps-small-head.mi')
        term_db.nouns.load(['south','termthatdoesntexist'])
        term_vector = term_db.nouns['termthatdoesntexist']
        self.assertTrue(type(term_vector) == dict)
        self.assertEqual(0, len(term_vector))

    def testCachedTermsWork(self):
        term_db = TermDB('test_data/nouns-deps-small-head.mi')
        term_db.nouns.load(['south','termthatdoesntexist'])
        term_vector = term_db.nouns['south']
        cached_term_vector = term_db.nouns['south']
        self.assertTrue(type(term_vector) == dict)
        self.assertTrue(len(term_vector) > 0)
        self.assertEqual(term_vector, cached_term_vector)

