import unittest

from numpy.testing import assert_almost_equal

from auditai.utils.crosstabs import top_bottom_crosstab, crosstab_ztest


class TestTBCrosstab(unittest.TestCase):

    def test_tbcross_shape(self):
        T10 = [True] * 10
        F10 = [False] * 10
        labels = ['F'] * 20 + ['M'] * 20
        decisions = T10 + F10 + T10 + T10
        ctab = top_bottom_crosstab(labels, decisions)
        self.assertEqual(len(ctab), 2)
        self.assertEqual(len(ctab[0]), 2)
        self.assertEqual(ctab[0][1], 10)
        self.assertEqual(ctab[1][1], 20)


class TestCtabZtest(unittest.TestCase):

    def test_crosstab_ztest(self):
        z, p = crosstab_ztest([[10, 10], [10, 10]])
        self.assertEqual(z, 0.0)
        z, p = crosstab_ztest([[10, 10], [0, 20]])
        assert_almost_equal(z, -3.6514837167011072)
        z, p = crosstab_ztest([[78, 5], [87, 12]])
        assert_almost_equal(z, -1.4078304151258787)
