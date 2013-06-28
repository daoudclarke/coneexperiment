import unittest
import logging
import numpy as np
from numpy import random
from string import ascii_lowercase
from utils import testData

from sklearn.neighbors import KNeighborsClassifier

from coneexperiment.EntailmentClassifier import EntailmentClassifier        

class EntailmentClassifierTestCase(unittest.TestCase):
    def setUp(self):
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

