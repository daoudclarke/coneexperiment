# Bismillahi-r-Rahmani-r-Rahim
#
# Class to construct numerous types of classifier

import inspect

from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import confusion_matrix, f1_score

from learncone.ConeEstimatorSVM import ConeEstimatorSVM

from coneexperiment.EntailmentClassifier import EntailmentClassifier
from baseline.baselineClassifier import BaselineEntailmentClassifier
from baseline.baselineClassifier import WidthClassifier, WidthClassifierP,ClassifierP

MAKE_PREFIX = '_make_'

class ClassifierMaker(object):
    def __init__(self, vectors, params = {}):
        self.vectors = vectors
        self.params = params

    def make(self, name):
        method = getattr(self, MAKE_PREFIX + name)
        return method()

    def get_names(self):
        members = inspect.getmembers(self, predicate=inspect.ismethod)
        return [x[0][len(MAKE_PREFIX):]
                for x in members
                if x[0].startswith(MAKE_PREFIX)]
    

    def _make_knn(self):
        neigh = KNeighborsClassifier(n_neighbors=1)
        return EntailmentClassifier(neigh, self.vectors)
        
    def _make_most_frequent(self):
        dummy = DummyClassifier('most_frequent')
        return EntailmentClassifier(dummy, self.vectors)

    def _make_conesvm(self):
        classifier = GridSearchCV(
            ConeEstimatorSVM(),
            {'beta' : self.params['beta'],
             'C' : self.params['costs']},
            score_func = f1_score)
        return EntailmentClassifier(classifier, self.vectors)

    def _make_widthdiff(self):
        classifier = WidthClassifier('widthdiff')
        return BaselineEntailmentClassifier(classifier, self.vectors)

    def _make_widthP(self):
        classifier = WidthClassifierP('widthP')
        return BaselineEntailmentClassifier(classifier,self.vectors)

    def _make_cosine(self):
        classifier = ClassifierP('cosine')
        return BaselineEntailmentClassifier(classifier,self.vectors)

    def _make_lin(self):
        classifier = ClassifierP('lin')
        return BaselineEntailmentClassifier(classifier,self.vectors)
