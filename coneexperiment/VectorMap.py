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
    def load(self, json_file):
        features = [json.loads(x) for x in json_file]
        words = [x[0].split('/')[0] for x in features]
        vectors = [f[1] for f in features]
        self.vector_map = defaultdict(lambda: {}, itertools.izip(words, vectors))

    def __getitem__(self, key):
        return self.vector_map[key]
