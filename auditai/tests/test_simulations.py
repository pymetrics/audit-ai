import unittest
import os

from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np

import auditai.simulations as sim


class TestGenerateBayesFactors(unittest.TestCase):
    def setUp(self):
        data_path = os.path.join(os.path.dirname(__file__), '..', '..',
                                 'data', 'GermanCreditData.csv')
        self.data = pd.read_csv(data_path)
        self.features = ['age',
                         'duration',
                         'amount',
                         'dependents',
                         'inst_rate',
                         'num_credits'
                         ]
        X = self.data[self.features]
        y = self.data['status']

        self.clf = LogisticRegression()
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

    def test_expected_values_0_01(self):
        np.random.seed(42)
        expected_is_female = [0.9964569398234155,
                              1.0035990084294402,
                              0.006555291338954991,
                              0.006589967705087398,
                              0.9844427430302514,
                              0.9895322416249943,
                              1.0105784931410273,
                              1.0158031109908876,
                              (683, 9.0),
                              (309, 3.0),
                              np.inf]

        expected_under_30 = [1.0156517754102912,
                             0.9846637559926464,
                             0.008848752073678487,
                             0.008532082550902103,
                             1.0017184895413358,
                             0.9656369218482502,
                             1.0355859215014185,
                             0.9982844586036186,
                             (627, 4.0),
                             (365, 8.0),
                             np.inf]

        _, ratios, _ = sim.generate_bayesfactors(self.clf, self.data,
                                                 self.features,
                                                 ['under_30',
                                                  'is_female'],
                                                 threshold=.01)
        self.assertTrue(np.all(np.isclose(
                               ratios['is_female: 0 over 1'][0.01][:8],
                               expected_is_female[:8])))
        self.assertTrue(np.all(np.isclose(
                               ratios['under_30: 0 over 1'][0.01][:8],
                               expected_under_30[:8])))
