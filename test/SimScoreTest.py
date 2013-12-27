import unittest
from scipy.sparse import csr_matrix

from baseline.SimScore import SimCalculator

    

class SimScoreTestCase(unittest.TestCase):
    def setUp(self):
        self.term_map = {
            'animal': csr_matrix([0.1, 0.0, 0.3, 0.4]),
            'cat': csr_matrix([0.0, 0.5, 0.0, 1.0])
            }
        self.sim = SimCalculator()
        self.pair = ('cat', 'animal')

    def test_APinc(self):
        score = self.sim.compute_score(
            self.pair, self.term_map, 'APinc')
        self.assertEqual(score, 0.25)

    def test_balAPinc(self):
        score = self.sim.compute_score(
            self.pair, self.term_map, 'balAPinc')
        self.assertGreater(score, 0.0)
