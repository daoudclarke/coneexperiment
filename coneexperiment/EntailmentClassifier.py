# Bismillahi-r-Rahmani-r-Rahim
#
# Classifier that operates on pairs of terms

from sklearn.feature_extraction import DictVectorizer
from scipy import sparse

import numpy as np

class EntailmentClassifier:
    def __init__(self, classifier, termVectors):
        self.termVectors = termVectors
        self.classifier = classifier

    def fit(self, pairs):
        """Train classifier based on a sequence of (word1, word2, entail) tuples.
        The value of entail is True if word1 entails word2 or False otherwise."""
        data = self.value_map(pairs)
        #assert sparse.issparse(data)
        #data = np.atleast2d_or_csr(data, dtype=np.float64, order="C")
        target = np.array([p[2] for p in pairs], dtype=int)
        self.classifier.fit(data, target)

    def predict(self, pairs):
        "Predict whether entailment holds for a sequence of (word1, word2) tuples."
        data = self.value_map(pairs)
        return self.classifier.predict(data)

    def value_map(self, pairs):
        return np.array(
            [self.termVectors[p[1]] - self.termVectors[p[0]]
             for p in pairs])
        
