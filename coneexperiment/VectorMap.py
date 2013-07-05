# Bismillahi-r-Rahmani-r-Rahim
#
# Container for term vectors

import json
import numpy as np
import itertools
from collections import defaultdict

from scipy.sparse import csr_matrix
from sklearn.feature_extraction import DictVectorizer

class VectorMap(object):
    def __init__(self):
        pass

    def load(self, json_file):
        features = [json.loads(x) for x in json_file]
        vectorizer = DictVectorizer(sparse=True)
        vectors = vectorizer.fit_transform(x[1] for x in features)
        words = (x[0].split('/')[0] for x in features)
        zero = csr_matrix((1, vectors.shape[1]))
        self.vector_map = defaultdict(lambda: zero, itertools.izip(words, vectors))

    def __getitem__(self, key):
        return np.array(self.vector_map[key].todense())[0]
