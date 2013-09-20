# Bismillahi-r-Rahmani-r-Rahim
#
# Class to construct numerous types of classifier

import inspect

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import confusion_matrix, f1_score

from learncone.ConeEstimatorSVM import ConeEstimatorSVM

from coneexperiment.EntailmentClassifier import EntailmentClassifier,AddVectorClassifier,MultVectorClassifier,CatVectorClassifier, SingleVectorClassifier
from baseline.baselineClassifier import BaselineEntailmentClassifier
from baseline.baselineClassifier import WidthClassifierUP, WidthClassifierP,ClassifierP,ClassifierUP

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
    

    def _make_knnP(self):
        neigh = GridSearchCV(KNeighborsClassifier(),{'n_neighbors':self.params['k']},score_func=f1_score)
        return EntailmentClassifier(neigh, self.vectors)

    def _make_knn10(self):
        neigh =KNeighborsClassifier(n_neighbors=10)
        return EntailmentClassifier(neigh,self.vectors)
        
    def _make_most_frequent(self):
        dummy = DummyClassifier('most_frequent')
        return EntailmentClassifier(dummy, self.vectors)

    def _make_linsvmDIFF(self):
        linsvm = LinearSVC()
        return EntailmentClassifier(linsvm,self.vectors)

    def _make_linsvmADD(self):
        linsvm = LinearSVC()
        return AddVectorClassifier(linsvm,self.vectors)

    def _make_linsvmMULT(self):
        linsvm = LinearSVC()
        return MultVectorClassifier(linsvm,self.vectors)

    def _make_linsvmCAT(self):
        linsvm = LinearSVC()
        return CatVectorClassifier(linsvm,self.vectors)

    def _make_linsvmSINGLE(self):
        linsvm = LinearSVC()
        return SingleVectorClassifier(linsvm,self.vectors)

    def _make_conesvm(self):
        classifier = GridSearchCV(
            ConeEstimatorSVM(),
            {'beta' : self.params['beta'],
             'C' : self.params['costs']},
            score_func = f1_score)
        return EntailmentClassifier(classifier, self.vectors)

    def _make_widthdiff(self):
        classifier = WidthClassifierUP('widthdiff')
        return BaselineEntailmentClassifier(classifier, self.vectors)

    def _make_widthdiffP(self):
        classifier = WidthClassifierP('widthdiffP')
        return BaselineEntailmentClassifier(classifier,self.vectors)

    def _make_cosineP(self):
        classifier = ClassifierP('cosine')
        return BaselineEntailmentClassifier(classifier,self.vectors)

    def _make_linP(self):
        classifier = ClassifierP('lin')
        return BaselineEntailmentClassifier(classifier,self.vectors)

    def _make_CRdiffP(self):
        classifier = ClassifierP('CRdiff')
        return BaselineEntailmentClassifier(classifier,self.vectors)

    def _make_clarkediffP(self):
        classifier = ClassifierP('clarkediff')
        return BaselineEntailmentClassifier(classifier,self.vectors)

    def _make_CRdiff(self):
        classifier = ClassifierUP('CRdiff')
        return BaselineEntailmentClassifier(classifier,self.vectors)

    def _make_clarkediff(self):
        classifier = ClassifierUP('clarkediff')
        return BaselineEntailmentClassifier(classifier,self.vectors)

    def _make_invCLP(self):
        classifier = ClassifierP('invCL')
        return BaselineEntailmentClassifier(classifier,self.vectors)