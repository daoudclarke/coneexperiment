# Bismillahi-r-Rahmani-r-Rahim
#
# Analyse the results of the experiment

from expsuite import PyExperimentSuite

import numpy as np
import sys
import os
import math

class MissingDataException(Exception):
    pass

def accuracy(confusion):
    return (np.sum(np.diagonal(confusion))/
            float(np.sum(confusion)))

def get_mean_and_error(datasets, function):
    data = [function(x) for x in datasets]
    return np.mean(data), np.std(data)/math.sqrt(len(data))

def evaluate(experiment):
    print experiment
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

    evaluations = {
        'accuracy' : lambda x: accuracy(x['confusion'][0]),
        'time' : lambda x: x['time'][0]
        }

    for name, eval_func in evaluations.items():
        print name, get_mean_and_error(datasets, eval_func)

def evaluate_all(path):
    for experiment in os.listdir(path):
        try:
            joined = os.path.join(path,experiment)
            if os.path.isdir(joined):
                evaluate(joined)
        except MissingDataException:
            continue

if __name__ == "__main__":
    path = sys.argv[1]
    evaluate_all(path)
