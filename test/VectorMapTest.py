import unittest
import logging
import numpy as np
from numpy import random
from StringIO import StringIO
from utils import randomWord

from sklearn.neighbors import KNeighborsClassifier

from coneexperiment.VectorMap import VectorMap

class VectorMapTestCase(unittest.TestCase):
    def setUp(self):
        logging.info("Starting test: %s", self._testMethodName)
        random.seed(1001)

    def testVectorMap(self):
        for i in range(3):
            # Arrange
            data = ''
            all_values = []
            for word in ['consideration', 'banana']:
                values = random.rand(2)
                data += '["%s/N", {"amod-DEP:look": %f, "dobj-HEAD:like": %f}]\n' % (
                    word, values[0], values[1])
                all_values.append(values)

            vectors = VectorMap()
            vectors.load(StringIO(data))
            
            # Act
            consideration = vectors['consideration']
            banana = vectors['banana']
            nonexistent = vectors['nonexistent']

            # Assert
            self.assertTrue(abs(np.dot(all_values[0], all_values[1]) -
                                np.dot(consideration, banana)) <= 1e-5)
            self.assertTrue(abs(np.dot(consideration, nonexistent)) <= 1e-5)
            
