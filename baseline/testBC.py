__author__ = 'juliewe'

from coneexperiment.ClassifierMaker import ClassifierMaker
from coneexperiment.TermDB import TermDB
import json,os,random
import numpy as np




params={}
params['datadir'] = '/Volumes/LocalScratchHD/juliewe/Documents/workspace/coneexperiment/data/'
params['dataset'] = 'wn-noun-dependencies-original'
params['vectors'] =  'nouns-deps.mi.db'
#params['classifier'] = 'widthdiff'
params['classifier']='lin'

if __name__ == "__main__":
    print "Testing baseline function"

    datadir = params['datadir']
    dataset_path = os.path.join(datadir, params['dataset'] + '.json')
    random.seed(abs(hash(str(params))))
    with open(dataset_path) as dataset_file:
        dataset = json.load(dataset_file)

    vectors_path = os.path.join(datadir, params['vectors'])
    print "DB path: ", vectors_path
    vectors = TermDB(vectors_path)

    maker = ClassifierMaker(vectors, params)
    classifier = maker.make(params['classifier'])


    target = np.array([p[2] for p in dataset], dtype=int)
    classifier.fit(dataset)
    predictions=classifier.predict(dataset)
    print "Predictions:", predictions
    print "Actual:", target



