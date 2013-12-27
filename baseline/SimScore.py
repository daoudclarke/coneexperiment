__author__ = 'juliewe'

import math,os,json,random
from coneexperiment.TermDB import TermDB
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import lil_matrix
import numpy as np
from collections import defaultdict

class SimCalculator(object):

    COMPUTE_PREFIX='_compute_'

    def compute_score(self,pair,term_map,metric):
        method = getattr(self,SimCalculator.COMPUTE_PREFIX+metric)
        avector=term_map[pair[0]]
        bvector=term_map[pair[1]]
        return method(avector, bvector)


    def _compute_cosine(self, avector, bvector):
        dotprod = avector.multiply(bvector).sum()
        denom=avector_length*bvector_length
        if denom==0:
            cosine = 0
        else:
            cosine = dotprod/(avector_length*bvector_length)
        #print pair[0],pair[1], avector_length, bvector_length, dotprod, cosine
        return cosine

    def _compute_lin(self, avector, bvector):
        num=0
        indices = avector.multiply(bvector).nonzero() #only consider elements in intersection
        for i in indices[1]:
            num += avector[0,i]+bvector[0,i]


        den=(avector+bvector).sum()
        if den>0:
            lin=num/den
        else:
            lin=0

        return lin

    def _compute_CRdiff(self, avector, bvector):
        pre = self._compute_pre(avector, bvector, False)
        rec = self._compute_pre(bvector, avector, term_map, False)
        return pre-rec

    def _compute_clarkediff(self, avector, bvector):
        pre = self._compute_pre(avector, bvector, True)
        rec = self._compute_pre(bvector, avector, term_map, True)
        return pre-rec

    def _compute_pre(self, avector, bvector, clarke=False):
        num =0
        indices = avector.multiply(bvector).nonzero()
        for i in indices[1]:
            if clarke:
                num+=min(avector[0,i],bvector[0,i])
            else:
                num+=avector[0,i]
        den = avector.sum()
        if den==0:
            return 0
        else:
            return num/den

    def _compute_invCL(self, avector, bvector):
        pre = self._compute_pre(avector, bvector, True)
        rec = self._compute_pre(bvector, avector, True)
        tmp = pre*(1-rec)
        if tmp>0:
            return math.pow(pre*(1-rec),0.5)
        else:

            return 0

    def _compute_APinc(self, avector, bvector):        
        a_nonzero_sorted = nonzero_sorted_indices(avector)
        b_nonzero_sorted = nonzero_sorted_indices(bvector)
        rank = dict(zip(b_nonzero_sorted,
                        range(1, len(b_nonzero_sorted) + 1)))
        partial_vector = lil_matrix(avector.shape)
        precision_sum = 0.0
        for i in a_nonzero_sorted:
            partial_vector[0,i] = 1
            precision = self._compute_pre(partial_vector, bvector)
            try:
                rel = 1 - rank[i]/float(len(b_nonzero_sorted)+1)
            except KeyError:
                rel = 0.0
            precision_sum += rel*precision
        return precision_sum/len(b_nonzero_sorted)

    def _compute_balAPinc(self, avector, bvector):
        lin = self._compute_lin(avector, bvector)
        ap_inc = self._compute_APinc(avector, bvector)
        return math.sqrt(lin*ap_inc)

def nonzero_sorted_indices(matrix):
    _, a_nonzero = matrix.nonzero()
    a_nonzero = set(a_nonzero)
    a_array = np.asarray(matrix.todense())[0]
    a_sorted_indices = np.argsort(a_array)[::-1]
    a_nonzero_sorted = [x for x in a_sorted_indices
                        if x in a_nonzero]
    return a_nonzero_sorted

if __name__=="__main__":

    print "Testing similarity function"

    params={}
    params['datadir'] = '/Volumes/LocalScratchHD/juliewe/Documents/workspace/coneexperiment/data/'
    params['dataset'] = 'wn-noun-dependencies-original'
    params['vectors'] =  'nouns-deps.mi.db'

    datadir = params['datadir']
    dataset_path = os.path.join(datadir, params['dataset'] + '.json')
    random.seed(abs(hash(str(params))))
    with open(dataset_path) as dataset_file:
        dataset = json.load(dataset_file)

    vectors_path = os.path.join(datadir, params['vectors'])
    print "DB path: ", vectors_path
    vectors = TermDB(vectors_path)

    terms = list(set(x[0] for x in dataset) |
                 set(x[1] for x in dataset))
    term_dicts = (vectors.nouns[x] for x in terms)



    vectorizer = DictVectorizer(sparse=True)
    term_vectors = vectorizer.fit_transform(term_dicts)
        #print terms[1], term_vectors[1]
    term_map = {terms[i]:term_vectors[i] for i in range(len(terms))}

    myCalculator = SimCalculator()

    for pair in dataset:
        print pair[0],pair[1], myCalculator.compute_score(pair,term_map,'invCL')

