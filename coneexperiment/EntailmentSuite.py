# Bismillahi-r-Rahmani-r-Rahim

# Experiment Suite for machine learning with cones

from expsuite import PyExperimentSuite

from sklearn.datasets import fetch_mldata, load_svmlight_file
from sklearn import utils
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import confusion_matrix, f1_score
try:
    from sklearn.metrics import accuracy_score
except ImportError:
    from sklearn.metrics import zero_one_score
    accuracy_score = zero_one_score

from EntailmentExperiment import EntailmentExperiment
from ClassifierMaker import ClassifierMaker

# from learncone.ConeEstimatorFactorise import ConeEstimatorFactorise
from learncone.ConeEstimator import ConeEstimator
from learncone.ConeEstimatorGreedy import ConeEstimatorGreedy
from learncone.ConeEstimatorSVM import ConeEstimatorSVM
from learncone.ArtificialData import make_data

import numpy as np
from numpy import random

import hashlib
import os
from datetime import datetime
import logging

class EntailmentSuite(PyExperimentSuite):
    def reset(self, params, rep):
        dataset_path = params['dataset']
        logging.info("Resetting experiment, parameters: %s", str(params))
        random.seed(abs(hash(str(params))))
        with open(dataset_path) as dataset_file:
            dataset = json.load(dataset_file)

        maker = ClassifierMaker()
        classifier = maker.make(params['classifier'])

        num_folds = params['repetitions']
        self.experiment = EntailmentExperiment(dataset, classifier, num_folds)
        self.experiment.setup()
        
    def iterate(self, params, rep, n):
        assert n == 0
        confusion, time, info = self.experiment.run(rep)
        
        return {'rep':rep,
                'iter':n,
                'confusion':confusion,
                'time':time.total_seconds(),
                'classifier':classifier_type,
                'info': info}


if __name__ == '__main__':
    logging.basicConfig(filename='log/experiments.log',
                        level=logging.INFO,
                        format='%(asctime)s %(process)d %(levelname)s %(message)s')
    logging.captureWarnings(True)

    suite = ConeSuite()
    suite.start()
    

