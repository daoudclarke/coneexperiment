import unittest
from scipy.sparse import csr_matrix

from baseline.SimScore import SimCalculator

class SimScoreTestCase(unittest.TestCase):
    def testBalAPinc(self):
        term_map = {
            'animal': csr_matrix([0.1, 0.0, 0.3, 0.4]),
            'cat': csr_matrix([0.0, 0.5, 0.0, 1.0])
            }
        sim = SimCalculator()
        pair = ('cat', 'animal')
        score = sim.compute_score(pair, term_map, 'balAPinc')
        self.assertEqual(score, 0.25)
