import sys
import unittest
import mock
import pytest
import os

from sklearn.ensemble import RandomForestClassifier
import pandas as pd

import auditai.viz as viz


@pytest.mark.skipif(sys.version_info < (3, 6), reason="requires python3.6+")
class TestBiasBarPlot(unittest.TestCase):
    def setUp(self):
        self.bias_report = {'ethnicity': {'averages': [0.291, 0.303, 0.317,
                                                       0.338, 0.301],
                                          'categories': ['asian',
                                                         'black',
                                                         'hispanic-latino',
                                                         'two-or-more-groups',
                                                         'white'],
                                          'errors': [[0.278, 0.304],
                                                     [0.273, 0.334],
                                                     [0.288, 0.347],
                                                     [0.297, 0.38],
                                                     [0.29, 0.313]]},
                            'gender': {'averages': [0.293, 0.308],
                                       'categories': ['F', 'M'],
                                       'errors': [[0.282, 0.304],
                                                  [0.297, 0.319]]}}

    @mock.patch('matplotlib.pyplot.show')
    @mock.patch('matplotlib.pyplot.savefig')
    def test_have_bias_report_case(self, mock_plt_savefig, mock_plt_show):
        viz.bias_bar_plot(bias_report=self.bias_report,
                          ref_threshold=0.7)


@pytest.mark.skipif(sys.version_info < (3, 6), reason="requires python3.6+")
class TestGetBiasPlots(unittest.TestCase):
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

        self.clf = RandomForestClassifier()
        self.clf.fit(X, y)

    @mock.patch('matplotlib.pyplot.show')
    @mock.patch('matplotlib.pyplot.savefig')
    def test_use_default_values(self, mock_plt_savefig, mock_plt_show):
        viz.get_bias_plots(self.clf, self.data,
                           self.features, ['under_30', 'is_female'])
