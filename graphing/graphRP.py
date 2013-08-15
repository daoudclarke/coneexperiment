__author__ = 'juliewe'

import scipy as sc
import pylab as pl
import recallPrecision as rp
import sys

display={'*':'','allBLESS-dependencies':'BLESS','nouns-deps.mi.db':'GW','wiki_random.db':'random','wiki_nounsdeps_events.mi.db':'Wiki','wn-noun-dependencies-directional':'directional','wn-noun-dependencies-original':'original','most_frequent':'dummy'}

class Record:

    def __init__(self,line):

        fields=line.split(',')
        if len(fields)==13:
            self.dataset=fields[0]
            self.classifier=fields[1]
            self.vectors=fields[2]
            self.accuracy=float(fields[3])
            self.accerror=float(fields[4])
            self.precision=float(fields[5])
            self.preerror=float(fields[6])
            self.recall=float(fields[7])
            self.recallerror=float(fields[8])
            self.f1=float(fields[9])
            self.f1error=float(fields[10])
            self.time=float(fields[11])
            self.timeerror=float(fields[12])
        else:
            print "No record created for "+line
            print "Number of fields is "+str(len(fields))

    def getlabel(self):
        ds=display.get(self.dataset,self.dataset)
        cl=display.get(self.classifier,self.classifier)
        ve=display.get(self.vectors,self.vectors)
        return ds+'-'+ve+'-'+cl

    def match(self,(dataset,vectorset,classifier)):
        if self.dataset in dataset or '*' in dataset:
            #print "Matched "+dataset
            if self.vectors in vectorset or '*' in vectorset:
                #print "Matched "+vectorset
                if self.classifier in classifier or '*' in classifier:

                    return True
        return False




def random():

    prs = sc.rand(15,2) # precision recall point list
    labels = ["item " + str(i) for i in range(15)] # labels for the points
    rp.plotPrecisionRecallDiagram("footitle", prs, labels)
    pl.show()

def loadfile(filename):
    db={}
    lines=0
    with open(filename,'r') as instream:
        for line in instream:
            if lines>0:
                record=Record(line.rstrip())
                label=record.getlabel()
                db[label]=record
            lines+=1
    print"Read "+str(lines)+" lines from "+filename
    return db

def gencurve(db,selection):
    print "Generating curve for "+str(selection)
    points=[]
    labels=[]
    for item in db.values():
        if item.match(selection):
            points.append((item.precision,item.recall))
            labels.append(item.getlabel())
    rp.plotPrecisionRecallDiagram(maketitle(selection),points,labels)
    pl.show()

def maketitle((dl,vl,cl)):
    ds=''
    cl=''
    ve=''
    for d in dl:
        ds+=display.get(d,d)
    for c in cl:
        cl=display.get(c,c)
    for v in vl:
        ve=display.get(v,v)
    return ds+'-'+ve+'-'+cl

def reverselookup(alist):
    rlist=[]
    for item in alist:
        if isinstance(item,list):
            rlist.append(reverselookup(item))
        else:
            added=False
            for(key,value) in display.items():
                if item==value:
                    rlist.append(key)
                    added=True
            if not added:
                rlist.append(item)
    return rlist

if __name__=="__main__":

    ddatasets=[['BLESS'],['original'],['directional']]
    #dvectors=[['GW','random']]
    dvectors=[['']]
    dclassifiers=[['conesvm','knn','linsvm','invCLP','cosineP','widthdiffP']]
    #dclassifiers =[['']]

    if len(sys.argv)>1:
        filename=sys.argv[1]
    else:
        print "Please enter filename.  Displaying random precision recall curve"
        random()

    db=loadfile(filename)

    datasets=reverselookup(ddatasets)
    vectors=reverselookup(dvectors)
    classifiers=reverselookup(dclassifiers)
    print datasets,vectors,classifiers
    for dataset in datasets:
        for vectorset in vectors:
            for classifier in classifiers:
                gencurve(db,(dataset,vectorset,classifier))

