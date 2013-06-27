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
            values = random.rand(2)
            data = '["consideration/N", {"amod-DEP:%s": %f, "dobj-HEAD:%s": %f}]' % (
                randomWord(), values[0], randomWord(), values[1])
            vectors = VectorMap()
            vectors.load(StringIO(data))
            
            # Act
            retrieved = vectors['consideration']
            
            # Assert
            self.assertTrue((abs(values - retrieved) <= 1e-5).all())
            
