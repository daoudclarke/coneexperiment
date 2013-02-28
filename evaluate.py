# Bismillahi-r-Rahmani-r-Rahim
#
# Analyse the results of the experiment

from expsuite import PyExperimentSuite

import numpy as np
import sys

def accuracy(confusion):
    return ((confusion[0][0] + confusion[1][1])/
            float(np.sum(confusion)))

def evaluate(experiment):
    suite = PyExperimentSuite()
    params = suite.get_params(experiment)
    reps = params['repetitions']
    for rep in range(reps):
        results = suite.get_history(experiment, rep, 'all')
        if len(results) == 0:
            continue
        print results['info']
        confusion = results['confusion'][0]
        print confusion, accuracy(confusion)

if __name__ == "__main__":
    path = sys.argv[1]
    evaluate(path)
