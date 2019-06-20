import unittest

from auditai.misc import compare_groups


class TestMisc(unittest.TestCase):
    def setUp(self):
        self.labels = [0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1]
        self.results = [0.16540, 0.09153, 0.70881, 0.82012,
                        0.16622, 0.69941, 0.31745, 0.72577,
                        0.71889, 0.30512, 0.87571, 0.01221,
                        0.79433, 0.12108, 0.41547, 0.25132]

    def test_compare_groups(self):
        for i in range(1, 10):
            thresh = i/10
            statistics = [j.get(thresh) for j in
                          compare_groups(self.labels, self.results, low=thresh,
                                         num=1)]
            self.assertTrue(len(statistics) == 5)
