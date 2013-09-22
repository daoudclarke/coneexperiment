__author__ = 'Julie'

import sys
import scipy as sc
import pylab as pl
import numpy as np
from graphRP import Record,loadfile,display,reverselookup

if __name__=="__main__":

    ddatasets=["BLEent","BLEco"]
    dvectorset=["Wiki"]
    dclassifiers=['linsvmCAT','linsvmADD','linsvmMULT','linsvmDIFF','cosineP','invCLP','knnP','widthdiff']

    datasets=reverselookup(ddatasets)
    vectorset=reverselookup(dvectorset)
    classifiers=reverselookup(dclassifiers)

    filenames=[]
    if len(sys.argv)>2:
        filenames.append(sys.argv[1])
        filenames.append(sys.argv[2])
    else:
        print "scatter takes two filenames as input on command line"


    dbs=[]
    dbs.append(loadfile(filenames[0]))
    dbs.append(loadfile(filenames[1]))



    for vectorset in vectorset:
        classifieracc=[]
        classifiererr=[]
        classifieracc.append({})
        classifiererr.append({})
        classifieracc.append({})
        classifiererr.append({})
        for i,db in enumerate(dbs):
            for record in db.values():
                if record.match((datasets[i],vectorset,classifiers)):
                    #print record.getlabel(),str(record.accuracy),str(record.accerror)
                    classifieracc[i][record.classifier]=record.accuracy
                    classifiererr[i][record.classifier]=record.accerror


        xs=[]
        xerrs=[]
        ys=[]
        yerrs=[]
        labels=[]
        for key in classifieracc[0]:
            xs.append(classifieracc[0][key])
            xerrs.append(classifiererr[0][key])
            ys.append(classifieracc[1][key])
            yerrs.append(classifiererr[1][key])
            labels.append(display.get(key,key))
        print xs, ys
        print xerrs, yerrs
        pl.errorbar(xs,ys,xerr=xerrs,yerr=yerrs,capsize=2,marker='.',linestyle='None',elinewidth=1)
        pl.xlabel("Accuracy on "+ddatasets[0])
        pl.ylabel("Accuracy on "+ddatasets[1])
        yoff={'linsvmMULT':8}
        xoff={}
        for(xpos,ypos,label) in zip (xs,ys,labels):
            pl.annotate(label,(xpos,ypos),xytext=(xoff.get(label,-8),yoff.get(label,-12)),va='bottom',textcoords='offset points')
        #pl.legend()
        pl.yticks(np.arange(0.45,1.03,0.05))
        pl.xticks(np.arange(0.55,1.03,0.05))
        pl.grid(b=True,which='major',axis='both',linestyle='--',linewidth=0.5,color='gray')
        pl.title('Comparison Across Datasets Using '+display.get(vectorset,vectorset)+' Vectors')

        pl.show()

