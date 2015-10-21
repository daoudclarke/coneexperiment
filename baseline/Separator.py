__author__ = 'juliewe'


#find best threshold which optimally "separates" two lists

import bisect

def separate(positives,negatives,trials=1000,integer=True):

    res=0
    positives.sort()
    negatives.sort()
    if(len(positives)>0):
        leastpositive=positives[0]
    else:
        leastpositive=0
    if(len(negatives)>0):
        mostnegative=negatives[len(negatives)-1]
    else:
        mostnegative=0

    #print "Threshold range is "+str(leastpositive)+" to "+str(mostnegative)

    if leastpositive>=mostnegative:
        res= (leastpositive+mostnegative)/2
        bestwrong=0
    else:
        stepsize=float(mostnegative-leastpositive)/float(trials-1)
        if integer:
            if stepsize<1: stepsize=1
        #print stepsize

        thistrial=leastpositive

        bestwrong=len(negatives)+len(positives)+1
        reslist=[]
        while thistrial<=mostnegative:
            #print thistrial
            thiswrong = bisect.bisect_right(positives,thistrial) #returns position of thistrial in list - use bisect_right as >t is classified as positive so we want thistrial placed as far right as possible
            thiswrong+= len(negatives) - bisect.bisect_right(negatives,thistrial)
            #print thistrial, thiswrong, bestwrong
            if thiswrong==bestwrong:
                reslist.append(thistrial)

            if thiswrong<bestwrong:
                bestwrong=thiswrong
                reslist=[thistrial]

            #print reslist

            thistrial+=stepsize
        #print reslist
        res_idx=len(reslist)/2
        res=reslist[res_idx]
    #print res, bestwrong
    return (res,bestwrong)


if __name__=="__main__":

    testpos=[0.5,0.6,0.9,0.7]
    testneg=[0.3,0.2,0.1]

    print separate(testneg,testpos,trials=10,integer=False)
