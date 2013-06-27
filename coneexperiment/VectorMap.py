# Bismillahi-r-Rahmani-r-Rahim
#
# Container for term vectors

import json
import numpy as np
import itertools
from sklearn.feature_extraction import DictVectorizer

class VectorMap(object):
    def __init__(self):
        pass

    def load(self, json_file):
        features = [json.loads(x) for x in json_file]
        #print features
        vectorizer = DictVectorizer(sparse=False)
        vectors = vectorizer.fit_transform(x[1] for x in features)
        
        self.vector_map = {word:vector for word, vector in
                           itertools.izip((x[0].split('/')[0] for x in features),
                                          vectors)}


            

    def __getitem__(self, key):
        return self.vector_map[key]
