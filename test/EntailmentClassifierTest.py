import unittest
import logging
import numpy as np
from numpy import random
from string import ascii_lowercase
from utils import randomWord

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
            test_words = [randomWord() for j in range(6)]
            words = unzipped[0] + unzipped[1] + tuple(test_words)
            word_vectors = [random.random(10) for j in range(6)]
            vectors = {words[j]:word_vectors[j%6] for j in range(len(words))}
            data = zip(*unzipped)
            test_data = zip(test_words[:3], test_words[3:6], unzipped[2])
            classifier = EntailmentClassifier(neigh, vectors)

            # Act
            classifier.fit(data)
            results = classifier.predict(test_data)
            
            # Assert
            self.assertEqual(tuple(results), unzipped[2])

