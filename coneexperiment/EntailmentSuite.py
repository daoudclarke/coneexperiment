#!/usr/bin/python
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
from VectorMap import VectorMap

# from learncone.ConeEstimatorFactorise import ConeEstimatorFactorise
from learncone.ConeEstimator import ConeEstimator
from learncone.ConeEstimatorGreedy import ConeEstimatorGreedy
from learncone.ConeEstimatorSVM import ConeEstimatorSVM
from learncone.ArtificialData import make_data

import evaluate

import numpy as np
from numpy import random

import hashlib
import os
import sys
from datetime import datetime
import logging
import json

class EntailmentSuite(PyExperimentSuite):
    def __init__(self, **options):
        self.additional_options = options
        logging.info("Additional options: %s", str(options))
        super(EntailmentSuite, self).__init__()

    def reset(self, params, rep):
        logging.info("Resetting experiment, parameters: %s", str(params))
        datadir = params['datadir']
        dataset_path = os.path.join(datadir, params['dataset'] + '.json')
        random.seed(abs(hash(str(params))))
        with open(dataset_path) as dataset_file:
            dataset = json.load(dataset_file)

        vectors_path = os.path.join(datadir, params['vectors'] + '.json')
        with open(vectors_path) as vectors_file:
            vectors = VectorMap()
            vectors.load(vectors_file)

        maker = ClassifierMaker(vectors, params)
        classifier = maker.make(params['classifier'])

        num_folds = params['repetitions']
        self.experiment = EntailmentExperiment(dataset, classifier, num_folds)
        
    def iterate(self, params, rep, n):
        logging.info("Beggining iteration %d, repetition %d", n, rep)
        assert n == 0
        confusion, time, info = self.experiment.runFold(rep)
        
        return {'rep':rep,
                'iter':n,
                'confusion':confusion,
                'time':time.total_seconds(),
                'classifier':params['classifier'],
                'info': info}

    def parse_opt(self):
        """ parses the command line options for different settings. """
        options, args = super(EntailmentSuite, self).parse_opt()
        self.options.__dict__.update(self.additional_options)

def run_and_evaluate(**suite_params):
    suite = EntailmentSuite(**suite_params)
    suite.start()

    experiments = suite.cfgparser.sections()
    for experiment in experiments:
        logging.info("Running experiment: %s", experiment)
        experiment_path = os.path.join(eval(suite.cfgparser.get('DEFAULT', 'path')),
                                       experiment)
        params = suite.get_params(experiment_path)
        path = os.path.join(params['path'],
                            params['name'])
        rows = evaluate.evaluate_all(path)
        evaluate.write_summary(rows, os.path.join(path, 'analysis.csv'))

if __name__ == '__main__':
    logging.basicConfig(filename='log/experiments.log',
                        level=logging.INFO,
                        format='%(asctime)s %(process)d %(levelname)s %(message)s')
    logging.captureWarnings(True)

    config = sys.argv[1]
    run_and_evaluate(config=config)
    
    
    # import cProfile
    # cProfile.run('suite.start()')
    
    

