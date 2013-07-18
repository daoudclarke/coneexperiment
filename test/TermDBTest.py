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
        logging.info("Starting test: %s", self._testMethodName)
        random.seed(1001)

    def testTermDBTermExists(self):
        term_db = TermDB('test_data/nouns-deps-head.mi.db')
        consideration = term_db['consideration/N']
        self.assertTrue(type(consideration) == dict)

    def testTermDBTermNotExists(self):
        term_db = TermDB('test_data/nouns-deps-head.mi.db')
        consideration = term_db['termthatdoesntexist/N']
        self.assertTrue(type(consideration) == dict)
