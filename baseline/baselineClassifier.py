__author__ = 'juliewe'

from baseline import Separator
from sklearn.feature_extraction import DictVectorizer
from SimScore import SimCalculator

from coneexperiment.EntailmentClassifier import EntailmentClassifier
import logging
import numpy as np


class WidthClassifierUP:

    def __init__(self,name):
        self.name=name
        self.widthparameter=0


    def fit(self,pairs, data,target):
        logging.info("Baseline fit: Ignoring training data as unsupervised classifier: "+self.name)
        #print ("Baseline fit: ignoring training data")

    def predict(self,pairs, term_map):
        #term_map is dictionary from terms (in pairs) to vectors
        #print "Baseline prediction: "+self.name
        #print "Generating width_map from "+str(len(term_map.keys()))+" keys"
        width_map={}
        #done=0
        for term in term_map.keys():
            width_map[term]=term_map[term].getnnz()
        #    done+=1
        #    if done%500==0:print "Processed "+str(done)+" terms"

        #print "Test: grandfather = ",width_map["grandfather"]

        tags=[]
        for p in pairs:
            wd = width_map[p[1]]-width_map[p[0]]
            if wd > self.widthparameter:
                tags.append(1)
            else:
                tags.append(0)

        return np.array(tags,dtype=int)

class WidthClassifierP(WidthClassifierUP):

    def fit(self,pairs,term_map,target):

 #       print "Baseline: setting parameter for "+self.name

        width_map={}

        for term in term_map.keys():
            width_map[term]=term_map[term].getnnz()


        ones=[]
        zeros=[]
        for pair,target in zip(pairs,target):
            wd = width_map[pair[1]]-width_map[pair[0]]
            if target==1:
                ones.append(wd)
            else:
                zeros.append(wd)
#        print len(ones), len(zeros)

        self.widthparameter=float(Separator.separate(ones,zeros))
        logging.info("Baseline: "+self.name+", Parameter set as "+str(self.widthparameter))

class SingleWidthClassifierP:
    def __init__(self,name):
        self.name=name
        self.widthparameter=0

    def fit(self,pairs,term_map,target):

    #       print "Baseline: setting parameter for "+self.name

        width_map={}

        for term in term_map.keys():
            width_map[term]=term_map[term].getnnz()


        ones=[]
        zeros=[]
        for pair,target in zip(pairs,target):
            wd = width_map[pair[1]]
            if target==1:
                ones.append(wd)
            else:
                zeros.append(wd)
            #        print len(ones), len(zeros)

        self.widthparameter=float(Separator.separate(ones,zeros))
        logging.info("Baseline: "+self.name+", Parameter set as "+str(self.widthparameter))



    def predict(self,pairs, term_map):
        #term_map is dictionary from terms (in pairs) to vectors
        #print "Baseline prediction: "+self.name
        #print "Generating width_map from "+str(len(term_map.keys()))+" keys"
        print "Width parameter selected is "+str(self.widthparameter)
        width_map={}
        #done=0
        for term in term_map.keys():
            width_map[term]=term_map[term].getnnz()
            #    done+=1
        #    if done%500==0:print "Processed "+str(done)+" terms"

        #print "Test: event = ",width_map["event"]

        tags=[]
        for p in pairs:
            wd = width_map[p[1]]
            if wd > self.widthparameter:
                tags.append(1)
            else:
                tags.append(0)

        return np.array(tags,dtype=int)

class ClassifierUP():
    def __init__(self,name):
        self.metric=name
        self.make_name()
        #self.param=0
        self.simCalc=SimCalculator()
        #self.reverse=False
        self.param=(0,False) #flag for negating values to swap direction of inequality - false is >, true is <

    def make_name(self):
        self.name=self.metric+"_UP"

    def fit(self,pairs,term_map,target):
        logging.info("Baseline fit: Ignoring training data as unsupervised classifier: "+self.name)

    def predict(self,pairs,term_map):
        #term_map is dictionary from terms (in pairs) to vectors
        #print "Baseline prediction: "+self.name
        #print "Generating width_map from "+str(len(term_map.keys()))+" keys"


        tags=[]
        for pair in pairs:
            wd = self.simCalc.compute_score(pair,term_map,self.metric)
            if self.param[1]:
                wd=-wd
            if wd > self.param[0]:
                tags.append(1)
            else:
                tags.append(0)

        return np.array(tags,dtype=int)

class ClassifierP(ClassifierUP):

    def make_name(self):
        self.name=self.metric+"_P"

    def fit(self,pairs,term_map,target):
        ones=[]
        zeros=[]
        for pair,target in zip(pairs,target):
            score = self.simCalc.compute_score(pair,term_map,self.metric)
            if target==1:
                ones.append(score)
            else:
                zeros.append(score)

        (p1,e1)=Separator.separate(ones,zeros,integer=False)
        (p2,e2)=Separator.separate(zeros,ones,integer=False)
        #e2=-1 # force test of reverse

        if e2<e1:
            self.param=(-float(p2),True)
            print "Reversing ones and zeros"
        else:
            self.param=(float(p1),False)

#        self.param=float(Separator.separate(ones,zeros,integer=False))
        logging.info("Baseline: "+self.name+", Parameter set as "+str(self.param))




class BaselineEntailmentClassifier(EntailmentClassifier):

    def fit(self, pairs):
        """Train classifier based on a sequence of (word1, word2, entail) tuples.
           The value of entail is True if word1 entails word2 or False otherwise."""
        data = self.value_map(pairs)

        target = np.array([p[2] for p in pairs], dtype=int)
        logging.info("Number of samples: %d", target.shape[0])
        self.classifier.fit(pairs,data, target)

    def predict(self, pairs):
        "Predict whether entailment holds for a sequence of (word1, word2) tuples."
        data = self.value_map(pairs)
        self.memory_usage("Memory usage before predict():")
        return self.classifier.predict(pairs,data)

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

        #p = pairs[1]
        #print p
        #print p[0],term_map[p[0]].getnnz() #this has all features (other than FILTERED) included in count - essentially counts keys
        #print p[1],term_map[p[1]].getnnz()

        return term_map

