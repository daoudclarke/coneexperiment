# Bismillahi-r-Rahmani-r-Rahim
#
# Analyse the results of the experiment

from expsuite import PyExperimentSuite

import numpy as np
import sys
import os
import math
import csv
from confusionmetrics.metrics import precision, recall, f1_score, accuracy

def get_mean_and_error(datasets, function):
    data = [function(x) for x in datasets]
    return np.mean(data), np.std(data)/math.sqrt(len(data))

    #print experiment
def collect_results(experiment):
    suite = PyExperimentSuite()
    params = suite.get_params(experiment)
    reps = params['repetitions']
    datasets = []
    for rep in range(reps):
        results = suite.get_history(experiment, rep, 'all')
        if len(results) == 0:
            raise MissingDataException
        datasets.append(results)
    return datasets
        # print results['info']
        # confusion = results['confusion'][0]
        # print confusion, accuracy(confusion)

def evaluate(experiment):
    datasets = collect_results(experiment)
    evaluations = [
        ('Accuracy', lambda x: accuracy(x['confusion'][0])),
        ('Precision', lambda x: precision(x['confusion'][0])),
        ('Recall', lambda x: recall(x['confusion'][0])),
        ('F1', lambda x: f1_score(x['confusion'][0])),
        ('Time', lambda x: x['time'][0])]

    # summary = {}
    # for name, eval_func in evaluations:
    #     summary[name] = get_mean_and_error(datasets, eval_func)

    summary = [(name, get_mean_and_error(datasets, eval_func))
               for name, eval_func in evaluations]

    suite = PyExperimentSuite()
    params = suite.get_params(experiment)
    row = [(x,params[x]) for x in ['dataset','classifier']]
    for name, (value, error) in summary:
        row.append( (name,value) )
        row.append( (name + " error", error) )
        
    return row


def write_summary(rows, name):
    fieldnames = [x[0] for x in rows[0]]
    output = csv.DictWriter(open(name,'w'), fieldnames)
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
    return rows

def evaluate_dimensions(path):
    suite = PyExperimentSuite() 
    rows = []
    for experiment in os.listdir(path):
        try:
            joined = os.path.join(path,experiment)
            if os.path.isdir(joined):
                params = suite.get_params(joined)
                if params['classifier'] != 'cone':
                    continue
                datasets = collect_results(joined)
                dimensions = params['dimensions']
                values = {d:[] for d in dimensions}
                for data in datasets:
                    for dimension_row in data['info'][0]:
                        d = dimension_row[0]['dimensions']
                        values[d] += dimension_row[2]
                print values
                means = [(d, get_mean_and_error(values[d],
                                                lambda x:x))
                         for d in dimensions]
                rows = [[('dimensions', x[0]),
                         ('accuracy', x[1][0]),
                         ('error', x[1][1])] for x in means]
                write_summary(rows, 'analysis/' + params['dataset'] + '_dims.csv')

        except MissingDataException:
            continue


def _test():
    import doctest
    doctest.testmod(verbose=True)
        

if __name__ == "__main__":
    if len(sys.argv) <= 1:
        _test()
    else:
        path = sys.argv[1]
        rows = evaluate_all(path)
        write_summary(rows, 'analysis/output.csv')
        evaluate_dimensions(path)


