# Bismillahi-r-Rahmani-r-Rahim
#
# Tidy a csv file so it is ready for presentation

import csv
import sys

from itertools import groupby

def short(f):
    return '%.3f' % float(f)

def tidy(filename):
    f = open(file_name)
    reader = csv.DictReader(f)
    data = sorted(reader, key=lambda x: x['dataset'] + x['classifier'])
    
    fieldnames = ['Dataset', 'SVM', 'Decision Tree', 'Cone']
    field_map = {'tree':'Decision Tree', 'svm':'SVM', 'cone':'Cone'}
    output = csv.DictWriter(sys.stdout, fieldnames)
    output.writeheader()
    for k, g in groupby(data, key = lambda x: x['dataset']):
        classifiers = {x['classifier']:x for x in g}
        pm = ' $\pm$ '
        columns = {'Dataset': k}
        for c, row in classifiers.items():
            columns[field_map[c]] = short(row['accuracy']) + pm + short(row['accuracy error'])

        output.writerow(columns)

if __name__ == "__main__":
    file_name = sys.argv[1]
    tidy(file_name)
