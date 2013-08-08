__author__ = 'juliewe'

import sys

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



if __name__=="__main__":

    word = sys.argv[1]
    count(word)


#abortion 773 761
#event 5075 4765
