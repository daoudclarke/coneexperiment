__author__ = 'juliewe'

import math,os,json,random
from coneexperiment.TermDB import TermDB
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import coo_matrix
import numpy as np

def compute_score(pair,term_map,metric):

    if metric=="cosine":
        return compute_cosine(pair,term_map)
    elif metric =="lin":
        return compute_lin(pair,term_map)
    else:
        print "Unknown sim metric "+metric


def compute_cosine(pair,term_map):

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

def compute_lin(pair,term_map):
    avector=term_map[pair[0]]
    bvector=term_map[pair[1]]
    num=0
    indices = avector.multiply(bvector).nonzero() #only consider elements in intersection
    for i in indices[1]:
        num += min(avector[0,i],bvector[0,i])


    den=(avector+bvector).sum()
    if den>0:
        lin=num/den
    else:
        lin=0

    return lin


if __name__=="__main__":

#    print 0.5, np.ceil(0.5)
#    print "coo_matrix test"
#    row=np.array([0,0,0])
#    col=np.array([0,2,5])
#    data=np.array([0.5,0.6,0.1])
#    m=coo_matrix((data,(row,col)),shape=(1,6))
#    print m.todense()
#    n = np.ceil(m.todense())
#    print n
#
#    exit()
    print "Testing similarity function"

    params={}
    params['datadir'] = '/Volumes/LocalScratchHD/juliewe/Documents/workspace/coneexperiment/data/'
    params['dataset'] = 'wn-noun-dependencies-original'
    params['vectors'] =  'nouns-deps.mi.db'
    #params['classifier'] = 'widthdiff'
    #params['classifier']='cosine'

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

    for pair in dataset:
        print pair[0],pair[1], compute_score(pair,term_map,'lin')

