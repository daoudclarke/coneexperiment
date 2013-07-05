# Bismillahi-r-Rahmani-r-Rahim
#
# Functional test of the entailment experiment suite

import unittest
import numpy as np
from numpy import random
from StringIO import StringIO
from utils import testData

from coneexperiment.EntailmentSuite import EntailmentSuite

class EntailmentSuiteTestCase(unittest.TestCase):
    def setUp(self):
        random.seed(1001)

    def testEntailmentSuite(self):
        " Functional test of the entailment experiment suite"
        suite = EntailmentSuite(config='test_data/entailment.cfg',
                                ncores=1)
        suite.start()
