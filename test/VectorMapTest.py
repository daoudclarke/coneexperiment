import unittest
import logging
import numpy as np
from numpy import random
from StringIO import StringIO

from sklearn.neighbors import KNeighborsClassifier

from coneexperiment.VectorMap import VectorMap

class VectorMapTestCase(unittest.TestCase):
    def setUp(self):
        logging.info("Starting test: %s", self._testMethodName)
        random.seed(1001)

    def testVectorMap(self):
        # Arrange
        data = StringIO(
            '["consideration/N", {"amod-DEP:await": 5.2, "dobj-HEAD:seek": 0.3}]')
        vectors = VectorMap(data)
        
        # Act
        value = vectors['consideration']
        
        # Assert
        self.assertTrue((value == np.array([0.3, 5.2])).all())
