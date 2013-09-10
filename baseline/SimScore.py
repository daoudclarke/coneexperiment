__author__ = 'juliewe'

import math,os,json,random
from coneexperiment.TermDB import TermDB
from sklearn.feature_extraction import DictVectorizer

class SimCalculator(object):

    COMPUTE_PREFIX='_compute_'

    def compute_score(self,pair,term_map,metric):

        method = getattr(self,SimCalculator.COMPUTE_PREFIX+metric)
        return method(pair,term_map)


    def _compute_cosine(self,pair,term_map):

        avector=term_map[pair[0]]
        bvector=term_map[pair[1]]
        avector_length = math.pow(avector.multiply(avector).sum(),0.5)
        bvector_length = math.pow(bvector.multiply(bvector).sum(),0.5)
        dotprod = avector.multiply(bvector).sum()
        denom=avector_length*bvector_length
        if denom==0:
            cosine = 0
        else:
            cosine = dotprod/(avector_length*bvector_length)
        #print pair[0],pair[1], avector_length, bvector_length, dotprod, cosine
        return cosine

    def _compute_lin(self,pair,term_map):
        avector=term_map[pair[0]]
        bvector=term_map[pair[1]]
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

    def _compute_CRdiff(self,pair,term_map):

        pre = self._compute_pre(pair,term_map,False)
        rec = self._compute_pre((pair[1],pair[0]),term_map,False)
        return pre-rec

    def _compute_clarkediff(self,pair,term_map):
        pre = self._compute_pre(pair,term_map,True)
        rec = self._compute_pre((pair[1],pair[0]),term_map,True)
        return pre-rec

    def _compute_pre(self,pair,term_map,clarke=False):
        avector=term_map[pair[0]]
        bvector=term_map[pair[1]]
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

    def _compute_invCL(self,pair,term_map):
        pre = self._compute_pre(pair,term_map,True)
        rec = self._compute_pre((pair[1],pair[0]),term_map,True)
        tmp = pre*(1-rec)
        if tmp>0:
            return math.pow(pre*(1-rec),0.5)
        else:

            return 0

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

