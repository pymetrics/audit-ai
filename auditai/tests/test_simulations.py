import unittest

from sklearn.ensemble import RandomForestClassifier
import pandas as pd

import auditai.simulations as sim


class TestGenerateBayesFactors(unittest.TestCase):
    def setUp(self):
        self.data = pd.read_csv('../data/GermanCreditData.csv')
        self.features = ['age',
                         'duration',
                         'amount',
                         'dependents',
                         'inst_rate',
                         'num_credits'
                         ]
        X = self.data[self.features]
        y = self.data['status']

        self.clf = RandomForestClassifier()
        self.clf.fit(X, y)

    def test_use_default_values(self):
        thresholds, ratios, N = sim.generate_bayesfactors(self.clf, self.data,
                                                          self.features,
                                                          ['under_30',
                                                           'is_female'])
        # make sure all thresholds have ratio values
        for key in ratios.keys():
            self.assertEqual(len(ratios[key]), len(thresholds))

    def test_add_low_high(self):
        thresholds, ratios, N = sim.generate_bayesfactors(self.clf, self.data,
                                                          self.features,
                                                          ['under_30',
                                                           'is_female'],
                                                          low=.01,
                                                          high=.99,
                                                          num=99)
        # make sure all thresholds have ratio values
        for key in ratios.keys():
            self.assertEqual(len(ratios[key]), len(thresholds))

    def test_bad_case_all_users_pass(self):
        with self.assertRaises(ValueError):
            thresholds, ratios, N = sim.generate_bayesfactors(self.clf,
                                                              self.data,
                                                              self.features,
                                                              ['under_30',
                                                               'is_female'],
                                                              low=.00,
                                                              high=.99,
                                                              num=100)

    def test_bad_case_all_users_fail(self):
        with self.assertRaises(ValueError):
            thresholds, ratios, N = sim.generate_bayesfactors(self.clf,
                                                              self.data,
                                                              self.features,
                                                              ['under_30',
                                                               'is_female'],
                                                              low=.01,
                                                              high=1.,
                                                              num=100)

    def test_single_threshold(self):
        thresholds, ratios, N = sim.generate_bayesfactors(self.clf, self.data,
                                                          self.features,
                                                          ['under_30',
                                                           'is_female'],
                                                          threshold=.7)
        # make sure all thresholds have ratio values
        for key in ratios.keys():
            self.assertEqual(len(ratios[key]), len(thresholds))
            self.assertEqual(len(ratios[key][0.7]), 11)
