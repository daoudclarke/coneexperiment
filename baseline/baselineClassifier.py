__author__ = 'juliewe'

from sklearn.feature_extraction import DictVectorizer
from scipy import sparse

from coneexperiment.EntailmentClassifier import EntailmentClassifier
import logging
import numpy as np


class WidthClassifier:

    def __init__(self,name):
        self.name=name


    def fit(self,data,target):
        print "Baseline fit: Ignoring training data as unsupervised classifier: "+self.name


    def predict(self,data):
        #data is a pair of vstacks (sparse arrays) of term_map for p[0],p[1] for p in pairs
        print "Baseline prediction: "+self.name
        (entailer,entailed)=data
        #print entailer.shape,entailed.shape, entailer.shape[0]
        assert entailer.shape[0] == entailed.shape[0]
        #print "entailer", entailer.getrow(1).getnnz()  #this is non-zero features in sparse array
        #print "entailed", entailed.getrow(1).getnnz()
        tags=[]

        print "Number of samples is : "+str(entailer.shape[0])
        for i in range(entailer.shape[0]):
            wd = entailed.getrow(i).getnnz()-entailer.getrow(i).getnnz()
            if wd>0:
                tags.append(1)
            else:
                tags.append(0)
            if i%10==0: print "Tested: "+str(i)
        return tags

class BaselineEntailmentClassifier(EntailmentClassifier):

    def fit(self, pairs):
        """Train classifier based on a sequence of (word1, word2, entail) tuples.
           The value of entail is True if word1 entails word2 or False otherwise."""
        data = self.value_map(pairs)

        target = np.array([p[2] for p in pairs], dtype=int)
        assert data[0].shape[0] == target.shape[0]
        logging.info("Number of samples: %d", data[0].shape[0])
        self.classifier.fit(data, target)

    def value_map(self, pairs):
        terms = list(set(x[0] for x in pairs) |
                     set(x[1] for x in pairs))
        #print terms
        term_dicts = (self.termDb.nouns[x] for x in terms)
        #print self.termDb.nouns[terms[7]]

        self.memory_usage("Memory usage before vectorizer():")
        if self.vectorizer:
            term_vectors = self.vectorizer.transform(term_dicts)
        else:
            self.vectorizer = DictVectorizer(sparse=True)
            term_vectors = self.vectorizer.fit_transform(term_dicts)
            #print terms[1], term_vectors[1]
        term_map = {terms[i]:term_vectors[i] for i in range(len(terms))}
        self.memory_usage("Memory usage after vectorizer():")

       # p = pairs[1]
       # print p
       # print p[0],term_map[p[0]].getnnz() #this has all features (other than FILTERED) included in count - essentially counts keys
       # print p[1],term_map[p[1]].getnnz()

        return (sparse.vstack(term_map[p[0]] for p in pairs),sparse.vstack(term_map[p[1]] for p in pairs))

