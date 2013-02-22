# Bismillahi-r-Rahmani-r-Rahim

# Experiment Suite for machine learning with cones

from expsuite import PyExperimentSuite

from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import KFold
from sklearn import utils
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import confusion_matrix, f1_score

# from learncone.ConeEstimatorFactorise import ConeEstimatorFactorise
from learncone.ConeEstimatorGradient import ConeEstimatorGradient

import numpy as np

import hashlib
import os
from datetime import datetime
import logging

class ConeSuite(PyExperimentSuite):
    def reset(self, params, rep):
        self.dataset = fetch_mldata(params['dataset'])

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
        
        classifier = None
        info_func = lambda x: x.grid_scores_
        if classifier_type == 'svm':
            classifier = GridSearchCV(
                LinearSVC(),
                {'C' : [0.0001, 0.001, 0.1, 1, 10, 100, 1000]})
        elif classifier_type == 'nb':
            classifier = MultinomialNB()
            info_func = lambda x: x.class_log_prior_
        elif classifier_type == 'cone':
            classifier = GridSearchCV(
                ConeEstimatorGradient(1),
                {'dimensions' : [2, 3, 4, 5, 10, 15, 20]},
                score_func = f1_score)
        else:
            raise Exception(
                "Invalid classifier type: must be 'svm', 'nb' or 'cone'")

        start = datetime.now()
        classifier.fit(self.X_train, self.y_train)    
        time = datetime.now() - start
        results = classifier.predict(self.X_test)
        logging.info("Different classification values: %s, %s", set(results), set(self.y_test))
        confusion = confusion_matrix(self.y_test, results)

        
        return {'rep':rep,
                'iter':n,
                'confusion':confusion.tolist(),
                'time':time,
                'classifier':classifier_type,
                'info': info_func(classifier)}
        
        

if __name__ == '__main__':
    logging.basicConfig(filename='log/experiments.log',
                        level=logging.INFO,
                        format='%(asctime)s %(thread)d %(levelname)s %(message)s')

    suite = ConeSuite()
    suite.start()
    

