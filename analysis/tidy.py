# Bismillahi-r-Rahmani-r-Rahim
#
# Tidy a csv file so it is ready for presentation

import csv
import sys



def short(f):
    return '%.3f' % float(f)

def tidy(filename):
    f = open(file_name)
    reader = csv.DictReader(f)
    data = [x for x in reader]
    data.sort(key=lambda x: x['dataset'] + x['classifier'])
    fieldnames = ['Dataset', 'SVM', 'Decision Tree', 'Cone']
    output = csv.DictWriter(sys.stdout, fieldnames)
    output.writeheader()
    for i in range(0, len(data), 5):
        cone = data[i]
        cone_greedy = data[i+1]
        stratified = data[i+2]
        svm = data[i+3]
        tree = data[i+4]
        assert cone['classifier'] == 'cone'
        #assert cone_greedy['classifier'] == 'cone-greedy'
        #assert stratified['classifier'] == 'stratified'
        assert svm['classifier'] == 'svm'
        assert tree['classifier'] == 'tree'
        assert cone['dataset'] == svm['dataset']
        assert svm['dataset'] == tree['dataset']
        pm = ' $\pm$ '
        columns = {'Dataset': cone['dataset'],
                   'Stratified': short(stratified['accuracy']) + pm + short(stratified['accuracy error']),
                   'SVM': short(svm['accuracy']) + pm + short(svm['accuracy error']),
                   'Decision Tree': short(tree['accuracy']) + pm + short(tree['accuracy error']),
                   'Cone (Gradient)': short(cone['accuracy']) + pm + short(cone['accuracy error']),
                   'Cone (Greedy)' : short(cone_greedy['accuracy']) + pm + short(cone_greedy['accuracy error'])}

        output.writerow(columns)

if __name__ == "__main__":
    file_name = sys.argv[1]
    tidy(file_name)
