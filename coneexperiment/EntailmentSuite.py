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

from EntailmentExperiment import EntailmentExperiment, EntailmentExperimentHeldOut, EntailmentExperimentTrainTest, EntailmentExperimentHeldOutStrict
from ClassifierMaker import ClassifierMaker
#from VectorMap import VectorMap
from TermDB import TermDB

# from learncone.ConeEstimatorFactorise import ConeEstimatorFactorise
from learncone.ConeEstimator import ConeEstimator
from learncone.ConeEstimatorGreedy import ConeEstimatorGreedy
from learncone.ConeEstimatorSVM import ConeEstimatorSVM
from learncone.ArtificialData import make_data
from baseline import tools

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

        vectors_path = os.path.join(datadir, params['vectors'])
        print "DB path: ", vectors_path
        vectors = TermDB(vectors_path)
        terms = set(x[0] for x in dataset) | set(x[1] for x in dataset)
        vectors.nouns.load(terms)

        maker = ClassifierMaker(vectors, params)
        classifier = maker.make(params['classifier'])

        num_folds = params['repetitions']
        self.experiment = EntailmentExperiment(dataset, classifier, num_folds)
        
    def iterate(self, params, rep, n):
        logging.info("Beginning iteration %d, repetition %d", n, rep)
        assert n == 0
        confusion, time, info, classparams, predictions, target = self.experiment.runFold(rep)
        
        return {'rep':rep,
                'iter':n,
                'confusion':confusion,
                'time':time.total_seconds(),
                'classifier':params['classifier'],
                'info': info,
                'params': classparams,
                'predictions': predictions,
                'target': target}

    def parse_opt(self):
        """ parses the command line options for different settings. """
        options, args = super(EntailmentSuite, self).parse_opt()
        self.options.__dict__.update(self.additional_options)

class EntailmentSuiteHeldOut(EntailmentSuite):
    def reset(self,params,rep):
        EntailmentSuite.reset(self,params,rep)#call super method to get vectors path and classifier setup
        blesspath = os.path.join(params['datadir'],params['blesspath'])
        self.experiment = EntailmentExperimentHeldOut(self.experiment.dataset,self.experiment.classifier, self.experiment.num_folds,blesspath)

class EntailmentSuiteHeldOutStrict(EntailmentSuite):
    def reset(self,params,rep):
        EntailmentSuite.reset(self,params,rep)#call super method to get vectors path and classifier setup
        blesspath = os.path.join(params['datadir'],params['blesspath'])
        self.experiment = EntailmentExperimentHeldOutStrict(self.experiment.dataset,self.experiment.classifier, self.experiment.num_folds,blesspath)

class EntailmentSuiteTrainTest(EntailmentSuite):
    def reset(self,params,rep):
        #EntailmentSuite.reset(self,params,rep)
        logging.info("Resetting experiment, parameters: %s", str(params))
        datadir = params['datadir']
        dataset_path = os.path.join(datadir, params['dataset'] + '.json')
        random.seed(abs(hash(str(params))))
        with open(dataset_path) as dataset_file:
            dataset = json.load(dataset_file)

        testset_path = os.path.join(datadir, params['testset'] + '.json')
        with open(testset_path) as dataset_file:
            testset = json.load(dataset_file)

        vectors_path = os.path.join(datadir, params['vectors'])
        print "DB path: ", vectors_path
        vectors = TermDB(vectors_path)
        terms = set(x[0] for x in dataset) | set(x[1] for x in dataset) | set(x[0] for x in testset) | set(x[1] for x in testset)
        vectors.nouns.load(terms)

        maker = ClassifierMaker(vectors, params)
        classifier = maker.make(params['classifier'])

        self.experiment = EntailmentExperimentTrainTest(dataset,classifier, testset)


def run_and_evaluate(**suite_params):


    suite = EntailmentSuite(**suite_params)
    try:
        type = eval(suite.cfgparser.get('DEFAULT', 'type'))
    except:
        logging.info("Warning: type of experiment not specified.  Assuming cross-validation.")
        type="cv"
    if type=="heldout":
        suite = EntailmentSuiteHeldOut(**suite_params)
    elif type=="traintest":
        suite = EntailmentSuiteTrainTest(**suite_params)
    elif type=="heldoutstrict":
        suite = EntailmentSuiteHeldOutStrict(**suite_params)
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
                        level=logging.DEBUG,
                        format='%(asctime)s %(process)d %(levelname)s %(message)s')
    logging.captureWarnings(True)

    config = sys.argv[1]
    run_and_evaluate(config=config)
    
    
    # import cProfile
    # cProfile.run('suite.start()')
    
    

