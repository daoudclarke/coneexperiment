__author__ = 'juliewe'

import random,sys

def getword(line):
    line = line.rstrip()
    fields = line.split('\t')
    return(fields[0])

def random_output(word,outstream):
    outstream.write(word)
    for i in range(0,dimensions):
        outstream.write('\t')
        outstream.write('f'+str(i)+'\t')
        outstream.write(str(random.uniform(0,10)))
    outstream.write('\n')

if __name__=="__main__":
    basefile='data/wiki_nounsdeps_events.mi'
    outfile='data/wiki_random'
    dimensions = 10
    if (len(sys.argv)>1):
        dimensions = int (sys.argv[1])

    print "Number of dimensions is "+str(dimensions)
    outfile=outfile+str(dimensions)

    with open(basefile,'r') as instream:
        with open(outfile,'w') as outstream:
            lines=0
            for line in instream:
                word=getword(line)
                random_output(word,outstream)
                lines+=1
                if lines%1000==0:print "Processed "+str(lines)+" lines"