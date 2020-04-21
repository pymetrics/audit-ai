import unittest

from numpy.testing import assert_almost_equal

from auditai.stats import ztest, fisher_exact_test, chi2_test, bayes_factor


class TestZtest(unittest.TestCase):

    def setUp(self):
        T10 = [True] * 10
        F10 = [False] * 10
        self.labels = ['F'] * 20 + ['M'] * 20
        self.results = T10 + F10 + T10 + F10
        self.results_2 = T10 + F10 + T10 + T10

    def test_ztest(self):
        z, p = ztest(self.labels, self.results)
        self.assertEqual(z, 0.0)
        z, p = ztest(self.labels, self.results_2)
        assert_almost_equal(z, -3.6514837167011072)

    def test_fisher_exact_test(self):
        oddsratio, pvalue = fisher_exact_test(self.labels, self.results)
        self.assertEqual(oddsratio, 1.0)
        self.assertEqual(pvalue, 1.0)

    def test_chi2_test(self):
        chi2_stat, pvalue = chi2_test(self.labels, self.results)
        self.assertEqual(chi2_stat, 0.0)
        self.assertEqual(pvalue, 1.0)

    def test_bayes_factor(self):
        oddsratio, bf = bayes_factor(self.labels, self.results)
        self.assertEqual(oddsratio, 1.0)
        self.assertAlmostEqual(bf, 0.5500676358744966)
        with self.assertRaises(ValueError):
            oddsratio, bf = bayes_factor(self.labels, self.results,
                                         priors=self.results)
