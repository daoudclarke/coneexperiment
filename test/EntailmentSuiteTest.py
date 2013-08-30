# Bismillahi-r-Rahmani-r-Rahim
#
# Functional test of the entailment experiment suite

import unittest
import numpy as np
from numpy import random
from StringIO import StringIO
from utils import testData
import shutil

from coneexperiment.EntailmentSuite import run_and_evaluate
from coneexperiment import evaluate
import os
import csv
import logging

class EntailmentSuiteTestCase(unittest.TestCase):
    config_path = 'test_data'
    config_name = 'entailment.cfg'
    csv_path = 'test_data/entailment-test/analysis.csv'
    
    def setUp(self):
        logging.basicConfig(filename='log/unittest.log',
                            level=logging.DEBUG,
                            format='%(asctime)s %(process)d %(levelname)s %(message)s')
        logging.info("Starting test: %s", self._testMethodName)
        try:
            shutil.rmtree('test_data/unittest')
        except OSError as e:
            logging.warning("Unable to remove unit test temp data: %s", str(e))
        random.seed(1001)

    def test_run_and_evaluate(self):
        " Functional test of the entailment experiment suite"
        full_path = os.path.join(self.config_path, self.config_name)
        run_and_evaluate(config=full_path,
                         ncores=1)
        csv_reader = csv.DictReader(open(self.csv_path))
        rows = [row for row in csv_reader]
        self.assertEqual(1, len(rows))        

        
        # experiments = suite.cfgparser.sections()
        # params = suite.get_params(os.path.join(self.config_path, experiments[0]))
        # path = os.path.join(params['path'],
        #                     params['name'])
        # rows = evaluate.evaluate_all(path)
        
