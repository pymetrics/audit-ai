import unittest
import numpy as np

from auditai.misc import bias_test_check, compare_groups


class TestMisc(unittest.TestCase):
    def setUp(self):
        self.labels = [0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0]
        self.results = [0.16540467, 0.09153486, 0.70881489, 0.82012178, 0.54649991,
                        0.16622152, 0.69941364, 0.31745991, 0.72577634, 0.23298507,
                        0.71889626, 0.30512787, 0.87571838, 0.01221256, 0.76493222,
                        0.7943308 , 0.12108529, 0.41547848, 0.25132058, 0.33009967]

    def test_bias_test_check(self):
        category = ['Gender']
        test_thresh = 0.50
        bias_test_check(self.labels, self.results, category=category,
                        test_thresh=test_thresh)
        bias_test_check(self.labels, self.results, category=None,
                        test_thresh=None)

    def test_compare_groups(self):
        for i in range(1,10):
            thresh = i/10
            statistics = [j.get(thresh) for j in
                          compare_groups(self.labels, self.results, low=thresh,
                                         num=1)]
            self.assertTrue(len(statistics) == 5)
