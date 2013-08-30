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
        logging.basicConfig(filename='log/unittest.log',
                            level=logging.DEBUG,
                            format='%(asctime)s %(process)d %(levelname)s %(message)s')
        logging.info("Starting test: %s", self._testMethodName)
        random.seed(1001)

    def testVectorMap(self):
        vector_map = VectorMap()
        data = '["word/N", {"amod-DEP:look": 2.0, "dobj-HEAD:like": 3.0}]\n'
        vector_map.load(StringIO(data))
        expected = {"amod-DEP:look": 2.0, "dobj-HEAD:like": 3.0}

        self.assertEqual(expected, vector_map['word'])
