import unittest
from scipy.sparse import lil_matrix
from numpy import random

from baseline.SimScore import SimCalculator

import cProfile
import pstats
    

term_map = {}
def setup():
    dims = 300000
    for name in ['animal', 'cat']:
        matrix = lil_matrix((1, dims))
        indices = random.randint(0, dims, 1000)
        for i in indices:
            matrix[0,i] = random.random()
        term_map[name] = matrix
        #print name, matrix

def run():
    sim = SimCalculator()
    pair = ('cat', 'animal')

    score = sim.compute_score(
        pair, term_map, 'balAPinc')
    print "Score: ", score

if __name__ == "__main__":
    random.seed(101)
    setup()
    stats_name = 'balapincstats'
    cProfile.run('run()', stats_name)
    p = pstats.Stats(stats_name)
    p.strip_dirs().sort_stats('cumulative').print_stats(30)
