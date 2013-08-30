# Bismillahi-r-Rahmani-r-Rahim
#
# Classifier that operates on pairs of terms

import logging

from sklearn.feature_extraction import DictVectorizer
from scipy import sparse

import numpy as np

#from guppy import hpy

class EntailmentClassifier:
    def __init__(self, classifier, termDb):
        self.termDb = termDb
        self.classifier = classifier
        self.vectorizer = None
        self.output_memory_usage = False

    def memory_usage(self, message):
        if self.output_memory_usage:
            print message
#            h = hpy()
#            print h.heap()

    def fit(self, pairs):
        """Train classifier based on a sequence of (word1, word2, entail) tuples.
        The value of entail is True if word1 entails word2 or False otherwise."""
        data = self.value_map(pairs)
        #print data
        #assert sparse.issparse(data)
        #data = np.atleast2d_or_csr(data, dtype=np.float64, order="C")
        target = np.array([p[2] for p in pairs], dtype=int)
        assert data.shape[0] == target.shape[0]
        logging.info("Number of samples: %d", data.shape[0])
        self.classifier.fit(data, target)

    def predict(self, pairs):
        "Predict whether entailment holds for a sequence of (word1, word2) tuples."        
        data = self.value_map(pairs)
        self.memory_usage("Memory usage before predict():")
        return self.classifier.predict(data)

    def value_map(self, pairs):
        terms = list(set(x[0] for x in pairs) |
                     set(x[1] for x in pairs))
        term_dicts = [self.termDb.nouns[x] for x in terms]
        logging.debug("Term dicts: %s", str(term_dicts)[:1000])

        self.memory_usage("Memory usage before vectorizer():")
        if self.vectorizer:
            term_vectors = self.vectorizer.transform(term_dicts)
        else:
            self.vectorizer = DictVectorizer(sparse=True)
            term_vectors = self.vectorizer.fit_transform(term_dicts)
        term_map = {terms[i]:term_vectors[i] for i in range(len(terms))}
        self.memory_usage("Memory usage after vectorizer():")
        return sparse.vstack(term_map[p[1]] - term_map[p[0]]
                             for p in pairs)
        
