__author__ = 'juliewe'

import sys,os,ast




def readresults(path):
    #print path
    filelist=os.listdir(path)
    #print filelist
    res=[]
    target=[]
    for filepath in filelist:
        with open(os.path.join(path,filepath),'r') as instream:
            for line in instream:
                try:
                    dict = ast.literal_eval(line.rstrip())
                    #print dict
                    res.append(dict['predictions'])
                    target.append(dict['target'])
                except ValueError:
                    break
    return res,target

if __name__=='__main__':

    if len(sys.argv)<2:
        print "Takes two arguments"
    else:
        classifier1=sys.argv[1]
        classifier2=sys.argv[2]


    resultsdir='/Volumes/LocalScratchHD/juliewe/Documents/workspace/coneexperiment/results/hypernym-test'
    dataset='entpairs_wiki100'
    vectors='wiki_nounsdeps_events_rel100'

    dir1 = 'vectors'+vectors+'dataset'+dataset+'classifier'+classifier1
    dir2 = 'vectors'+vectors+'dataset'+dataset+'classifier'+classifier2
    path1=os.path.join(resultsdir,dir1)
    path2=os.path.join(resultsdir,dir2)


    results1,target1=readresults(path1)
    print results1
    print target1
    results2, target2=readresults(path2)
    print results2
    print target2
    same=0
    diff=0
    samecorrect=0
    for(i,list) in enumerate(results1):
        for(j,item) in enumerate(results1[i]):
            if results1[i][j] == results2[i][j]:
                same+=1
                if target1[i][j] == results1[i][j]:
                    samecorrect+=1
            else:
                diff+=1
    print "Same: "+str(same)
    print "Diff: "+str(diff)
    overlap = (same*100.0)/(same+diff)
    print "Percentage agreement: "+str(overlap)
    print "Same correct: "+str(samecorrect)
    intersectionacc = (samecorrect*100.0)/(same+diff)
    unionacc = ((samecorrect+diff)*100.0)/(same+diff)
    print "Intersection Accuracy "+ str(intersectionacc)
    print "Union Accuracy "+str(unionacc)

