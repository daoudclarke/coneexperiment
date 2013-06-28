# Bismillahi-r-Rahmani-r-Rahim
#
# Class to construct numerous types of classifier

import inspect

from sklearn.neighbors import KNeighborsClassifier

from coneexperiment.EntailmentClassifier import EntailmentClassifier

MAKE_PREFIX = '_make_'

class ClassifierMaker(object):
    def __init__(self, vectors):
        self.vectors = vectors

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
        
