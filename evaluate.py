# Bismillahi-r-Rahmani-r-Rahim
#
# Analyse the results of the experiment

from expsuite import PyExperimentSuite

import numpy as np
import sys
import os
import math
import csv

class MissingDataException(Exception):
    pass

def accuracy(confusion):
    return (np.sum(np.diagonal(confusion))/
            float(np.sum(confusion)))

def get_mean_and_error(datasets, function):
    data = [function(x) for x in datasets]
    return np.mean(data), np.std(data)/math.sqrt(len(data))

def evaluate(experiment):
    #print experiment
    suite = PyExperimentSuite()
    params = suite.get_params(experiment)
    reps = params['repetitions']
    datasets = []
    for rep in range(reps):
        results = suite.get_history(experiment, rep, 'all')
        if len(results) == 0:
            raise MissingDataException
        datasets.append(results)
        # print results['info']
        # confusion = results['confusion'][0]
        # print confusion, accuracy(confusion)

    evaluations = [
        ('accuracy', lambda x: accuracy(x['confusion'][0])),
        ('time', lambda x: x['time'][0])]

    # summary = {}
    # for name, eval_func in evaluations:
    #     summary[name] = get_mean_and_error(datasets, eval_func)

    summary = [(name, get_mean_and_error(datasets, eval_func))
               for name, eval_func in evaluations]

    row = [(x,params[x]) for x in ['dataset','classifier']]
    for name, (value, error) in summary:
        row.append( (name,value) )
        row.append( (name + " error", error) )
        
    return row


def write_summary(rows):
    fieldnames = [x[0] for x in rows[0]]
    output = csv.DictWriter(open('output.csv','w'), fieldnames)
    output.writeheader()
    output.writerows([dict(x) for x in rows])

def evaluate_all(path):
    suite = PyExperimentSuite() 
    rows = []
    for experiment in os.listdir(path):
        try:
            joined = os.path.join(path,experiment)
            if os.path.isdir(joined):
                params = suite.get_params(joined)
                row = evaluate(joined)
                rows.append(row)
        except MissingDataException:
            continue

    write_summary(rows)
        

if __name__ == "__main__":
    path = sys.argv[1]
    evaluate_all(path)
