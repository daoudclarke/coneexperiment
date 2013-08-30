import unittest
import logging
import numpy as np
from numpy import random
from string import ascii_lowercase
from collections import defaultdict

from sklearn.neighbors import KNeighborsClassifier

from coneexperiment.EntailmentClassifier import EntailmentClassifier        
from utils import testData

class EntailmentClassifierTestCase(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(filename='log/unittest.log',
                            level=logging.DEBUG,
                            format='%(asctime)s %(process)d %(levelname)s %(message)s')
        logging.info("Starting test: %s", self._testMethodName)
        random.seed(1001)

    def testEntailmentClassifier(self):
        for i in range(3):
            # Arrange
            data, test_data, vectors = testData()
            expected = tuple(x[2] for x in test_data)
            test_data = [x[:2] for x in test_data]

            neigh = KNeighborsClassifier(n_neighbors=1)
            classifier = EntailmentClassifier(neigh, vectors)

            # Act
            classifier.fit(data)
            results = classifier.predict(test_data)
            
            # Assert
            self.assertEqual(tuple(results), expected)

    @unittest.skip("This test fails because of a bug in numpy")
    def testEntailmentClassifierEmptyData(self):
        # Arrange
        data, test_data, vectors = testData()
        vectors.nouns = defaultdict(lambda:{})
        neigh = KNeighborsClassifier(n_neighbors=1)
        classifier = EntailmentClassifier(neigh, vectors)

        # Act
        classifier.fit(data)
        classifier.predict(data)

