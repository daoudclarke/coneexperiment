import unittest
import logging
import numpy as np
from numpy import random

from sklearn.neighbors import KNeighborsClassifier

from coneexperiment.EntailmentClassifier import EntailmentClassifier        

class EntailmentClassifierTestCase(unittest.TestCase):
    def setUp(self):
        logging.info("Starting test: %s", self._testMethodName)
        random.seed(1001)

    def testEntailmentClassifier(self):
        for i in range(3):
            # Arrange
            neigh = KNeighborsClassifier(n_neighbors=1)
            unzipped = [('cat','dog', 'banana'),
                        ('animal', 'mosquito', 'fruit'),
                        tuple(random.randint(0,2,3) == 0)]
            data = zip(*unzipped)
            words = unzipped[0] + unzipped[1]
            vectors = {x:random.random(10) for x in words}
            classifier = EntailmentClassifier(neigh, vectors)

            # Act
            classifier.fit(data)
            results = classifier.predict(data)
            
            # Assert
            self.assertEqual(tuple(results), unzipped[2])
