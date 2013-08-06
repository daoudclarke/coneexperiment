__author__ = 'juliewe'

from sklearn.feature_extraction import DictVectorizer
from scipy import sparse

from coneexperiment.EntailmentClassifier import EntailmentClassifier

class WidthClassifier:

    def __init__(self,name):
        self.name=name


    def fit(self,data,target):
        print "Baseline fit"
        print data
        print target


    def predict(self,data):
        #data is a vstack of term_map for p[1]-p[0] for p in pairs
        print "Baseline prediction"
        (entailer,entailed)=data

        print "entailer", entailer.getrow(0).getnnz()
        print "entailed", entailed.getrow(0).getnnz()
        print entailed.getrow(0)

class BaselineEntailmentClassifier(EntailmentClassifier):
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

        p = pairs[0]
        print p
        print p[0],term_map[p[0]].getnnz()
        print p[1],term_map[p[1]].getnnz()
        print term_map[p[1]]
        return (sparse.vstack(term_map[p[0]] for p in pairs),sparse.vstack(term_map[p[1]] for p in pairs))

