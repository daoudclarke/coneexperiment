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

def compare(fp1,fp2):
    with open(fp1) as file1:
        data1=json.load(file1)
    with open(fp2) as file2:
        data2=json.load(file2)
    print "File 1 ",fp1
    print "File 2 ",fp2
    dict1={}
    dict2={}
    for[w1,w2,sc] in data1:
        dict1[w1]=dict1.get(w1,0)+1
        dict1[w2]=dict1.get(w2,0)+1
    for[w1,w2,sc] in data2:
        dict2[w1]=dict2.get(w1,0)+1
        dict2[w2]=dict2.get(w2,0)+1

    inter =0
    for key in dict1.keys():
        if key in dict2.keys():
            inter+=1
    print "% of file 2 concepts by type also in File 1 ",str(inter*100.0/len(dict2.keys()))
    print "% of file 1 concepts by type also in File 2 ",str(inter*100.0/len(dict1.keys()))



if __name__=="__main__":

    datapath="data"
    if sys.argv[1] == "count":
        word = sys.argv[2]
        count(word)
    elif sys.argv[1] == "split":

        filename = sys.argv[2]
        split(filename)

    elif sys.argv[1] == "compare":
        file1 = sys.argv[2]
        file2 = sys.argv[3]
        filepath1=os.path.join(datapath,file1+'.json')
        filepath2=os.path.join(datapath,file2+'.json')
        compare(filepath1,filepath2)

#abortion 773 761
#event 5075 4765
