# Bismillahi-r-Rahmani-r-Rahim
# Bismilllahi-r-Rahmani-r-Rahim

import unittest
import numpy as np
from numpy import random
from utils import testData

from coneexperiment.ClassifierMaker import ClassifierMaker
from coneexperiment.EntailmentExperiment import EntailmentExperiment

class ClassifierMakerTestCase(unittest.TestCase):
    def setUp(self):
        random.seed(1001)

    def testEntailmentExperiment(self):
        data, test_data, vectors = testData()
        all_data = data + test_data
        #class_values = set(x[2] for x in data)

        maker = ClassifierMaker(vectors)
        classifier = maker.make('knn')

        num_folds = 3
        experiment = EntailmentExperiment(all_data, vectors, classifier, num_folds)
        for fold in range(num_folds):
            results = experiment.runFold(fold)

            
            
