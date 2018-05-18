import unittest

from numpy.testing import assert_almost_equal

from auditai.stats import ztest


class TestZtest(unittest.TestCase):

    def test_ztest(self):
        T10 = [True] * 10
        F10 = [False] * 10
        labels = ['F'] * 20 + ['M'] * 20
        decisions = T10 + F10 + T10 + F10
        z, p = ztest(labels, decisions)
        self.assertEqual(z, 0.0)
        decisions = T10 + F10 + T10 + T10
        z, p = ztest(labels, decisions)
        assert_almost_equal(z, -3.6514837167011072)
