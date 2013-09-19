__author__ = 'juliewe'

import sys,json,os, random

vectors="data/nouns-deps.mi"


def count(word):

    print "Checking "+vectors+" for "+word
    with open(vectors,'r') as instream:
        for line in instream:
            line=line.rstrip()
            parts=line.split()
            if word==parts[0]:
                print line
                nofeatures = (len(parts)-1)/2
                print word,nofeatures
                features=parts[1:]
                allf=0
                nnz=0
                for i in range(nofeatures):
                    score=float(features[i*2+1])
                    allf+=1
                    if score>0:
                        nnz+=1
                print word, allf, nnz
                break


def readbless(blesspath="../BLESS/data/BLESS.txt"):
    with open(blesspath) as datafile:

        concepts=[]
        for line in datafile:
            fields = line.split('\t')
            concept = fields[0].split('-')[0]
            if concept not in concepts:
                concepts.append(concept)
    return concepts


def split(filename):
    num_folds=5
    bless = readbless()
    foldsize=len(bless)/num_folds

    datapath = os.path.join('data',filename+'.json')
    with open(datapath) as datafile:
        dataset = json.load(datafile)
        #print dataset

    random.seed(abs(hash(str(bless))))
    random.shuffle(bless)

    blessdict={}

    for i in range(len(bless)):
        blessdict[bless[i]]=i/foldsize
    newdataset=[]
    for [w1,w2,sc] in dataset:
        fold=blessdict.get(w1,blessdict.get(w2,-1))
        if fold == -1:
            print "Warning, both concepts not in bless ",w1,w2
        newdataset.append([w1,w2,sc,fold])
    print newdataset

def annotate(num_folds,filename):
    bless=readbless(filename)
    foldsize=len(bless)/num_folds
    random.seed(abs(hash(str(bless))))
    random.shuffle(bless)
    blessdict={}

    for i in range(len(bless)):
        blessdict[bless[i]]=i/foldsize
    return blessdict


if __name__=="__main__":

    if sys.argv[1] == "count":
        word = sys.argv[2]
        count(word)
    elif sys.argv[1] == "split":
        filename = sys.argv[2]
        split(filename)


#abortion 773 761
#event 5075 4765
