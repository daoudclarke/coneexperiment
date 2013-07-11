# Bismillahi-r-Rahmani-r-Rahim
#
# Functional test of the entailment experiment suite

import unittest
import numpy as np
from numpy import random
from StringIO import StringIO
from utils import testData

from coneexperiment.EntailmentSuite import EntailmentSuite
from coneexperiment import evaluate
import os

class EntailmentSuiteTestCase(unittest.TestCase):
    config_path = 'test_data'
    config_name = 'entailment.cfg'
    
    def setUp(self):
        random.seed(1001)

    def testEntailmentSuite(self):
        " Functional test of the entailment experiment suite"
        full_path = os.path.join(self.config_path, self.config_name)
        suite = EntailmentSuite(config=full_path,
                                ncores=1)
        suite.start()

        
        experiments = suite.cfgparser.sections()
        params = suite.get_params(os.path.join(self.config_path, experiments[0]))
        path = os.path.join(params['path'],
                            params['name'])
        rows = evaluate.evaluate_all(path)
        
        self.assertEqual(1, len(rows))
