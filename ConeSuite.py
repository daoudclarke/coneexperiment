# Bismillahi-r-Rahmani-r-Rahim

# Experiment Suite for machine learning with cones

from expsuite import PyExperimentSuite

from sklearn.datasets import fetch_mldata, load_svmlight_file
from sklearn.cross_validation import KFold
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

class SvmlightDataset:
    def __init__(self, t):    
        self.data = np.array(t[0].todense())
        self.target = t[1]

class ConeSuite(PyExperimentSuite):
    def reset(self, params, rep):
        name = params['dataset']
        print params
        random.seed(abs(hash(str(params))))
        if name.startswith('toy'):
            data_dims, cone_dims = [int(x) for x in
                                    name.split('-')[1:]]
            self.dimensions = [cone_dims]
            self.dataset = make_data(data_dims, cone_dims)
        elif name.startswith('wn'):
            self.dimensions = params['dimensions']
            self.dataset = SvmlightDataset(
                load_svmlight_file('../../../Documents/conewordnetdata/data-nouns-deps-mi/' + name + '.mat'))
            print self.dataset.target.shape
            print self.dataset.data.shape
        else:
            self.dimensions = params['dimensions']
            self.dataset = fetch_mldata(name)

        # Ensure that the data is always shuffled the same way:
        # seed RNG on data itself
        seed = int(hashlib.sha1(self.dataset.data).hexdigest()[:7], 16)
        # print len(self.dataset.data), self.dataset.target.shape[0]
        # shuffled_data, shuffled_target = utils.shuffle(
        #     self.dataset.data, self.dataset.target, random_state = seed)
       
        # StratifiedKFold is deterministic
        self.cv = KFold(k = params['repetitions'], n = self.dataset.target.shape[0],
                   shuffle = True, random_state = seed)
                
        train, test = list(self.cv)[rep]
        print len(train), len(test)

        self.X_train = self.dataset.data[train]
        self.X_test = self.dataset.data[test]

        self.y_train = self.dataset.target[train]
        self.y_test = self.dataset.target[test]
        
    def iterate(self, params, rep, n):
        assert n == 0

        classifier_type = params['classifier']
        score = {'f1': f1_score, 'accuracy':accuracy_score}[
            params['score']]
        
        classifier = None
        info_func = lambda x: x.grid_scores_
        if classifier_type == 'svm':
            classifier = GridSearchCV(
                LinearSVC(),
                {'C' : params['costs']})
        elif classifier_type == 'nb':
            classifier = MultinomialNB()
            info_func = lambda x: x.class_log_prior_
        elif classifier_type == 'cone':
            if len(self.dimensions) > 1:
                classifier = GridSearchCV(
                    ConeEstimator(),
                    {'dimensions' : self.dimensions,
                     'noise' : params['noise']},
                    score_func = f1_score)
            else:
                classifier = ConeEstimator(self.dimensions[0])
                info_func = lambda x: x.get_params()
        elif classifier_type == 'conesvm':
            classifier = GridSearchCV(
                ConeEstimatorSVM(),
                {'beta' : params['beta']},
                score_func = f1_score)
        elif classifier_type == 'tree':
            classifier = GridSearchCV(
                DecisionTreeClassifier(random_state=10011),
                {'max_depth' : params['depth']})
        elif classifier_type == 'stratified':
            classifier = DummyClassifier()
            info_func = lambda x: x.get_params()
        else:
            raise Exception(
                "Invalid classifier type: must be 'svm', 'svmcone', 'nb' or 'cone'")

        start = datetime.now()
        classifier.fit(self.X_train, self.y_train)    
        time = datetime.now() - start
        results = classifier.predict(self.X_test)
        logging.info("Different classification values: %s, %s", set(results), set(self.y_test))
        confusion = confusion_matrix(self.y_test, results)

        
        return {'rep':rep,
                'iter':n,
                'confusion':confusion,
                'time':time.total_seconds(),
                'classifier':classifier_type,
                'info': info_func(classifier)}
        
        

if __name__ == '__main__':
    logging.basicConfig(filename='log/experiments.log',
                        level=logging.INFO,
                        format='%(asctime)s %(process)d %(levelname)s %(message)s')

    suite = ConeSuite()
    suite.start()
    

