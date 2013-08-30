# Bismillahi-r-Rahmani-r-Rahim
# Bismilllahi-r-Rahmani-r-Rahim

import unittest
import itertools
import numpy as np
from numpy import random
from utils import testData

from coneexperiment.ClassifierMaker import ClassifierMaker
from coneexperiment.EntailmentExperiment import EntailmentExperiment
from coneexperiment.evaluate import accuracy


class EntailmentExperimentTestCase(unittest.TestCase):
    def setUp(self):
        random.seed(1001)

    def testEntailmentExperimentKnn(self):
        results = self.runExperiment('knnP')
        confusion = np.sum([x[0] for x in results], axis=0)
        self.assertEqual(accuracy(confusion), 1.0)

    def testEntailmentExperimentMostFrequent(self):
        results = self.runExperiment('most_frequent')
        confusion = np.sum([x[0] for x in results], axis=0)
        self.assertNotEqual(accuracy(confusion), 1.0)

    def runExperiment(self, classifier_name):
        data, test_data, vectors = testData()
        all_data = (data + test_data)*3
        #print all_data

        maker = ClassifierMaker(vectors, params = {'k':[1]} )
        classifier = maker.make(classifier_name)

        num_folds = 3
        experiment = EntailmentExperiment(all_data, classifier, num_folds)
        results = [experiment.runFold(fold)
                   for fold in range(num_folds)]
        #print results
        return results

            
            
