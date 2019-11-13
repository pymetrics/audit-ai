import unittest
import io
import sys
import re
import pandas as pd

from auditai import misc


def pass_fail_count(output):
    pass_cnt = len(re.findall('pass', output))
    fail_cnt = len(re.findall('fail', output))
    return pass_cnt, fail_cnt


class TestMisc(unittest.TestCase):
    def setUp(self):
        self.labels = [0, 0, 0, 0,
                       1, 1, 1, 1, 1, 1]
        self.results = [0.25, 0.25, 0.75, 0.75,
                        0.25, 0.25, 0.25, 0.75, 0.75, 0.75]

    def test_bias_test_check_all_pass(self):
        """
        Testing unbias results.
        Default test_thresh = 0.50 and all tests pass.
        """
        capturedOutput = io.StringIO()
        sys.stdout = capturedOutput

        misc.bias_test_check(self.labels, self.results, category='test_group')

        sys.stdout = sys.__stdout__

        pass_cnt, fail_cnt = pass_fail_count(capturedOutput.getvalue())
        self.assertEqual(pass_cnt, 5)
        self.assertEqual(fail_cnt, 0)

    def test_bias_test_check_below_min_thresh(self):
        """
        Testing unbias results at a test_threshold below min(results).
        Unable to run all tests all labels are classified into one group.
        """
        capturedOutput = io.StringIO()
        sys.stdout = capturedOutput

        misc.bias_test_check(self.labels, self.results, category='test_group',
                             test_thresh=0.20)

        sys.stdout = sys.__stdout__

        pass_cnt, fail_cnt = pass_fail_count(capturedOutput.getvalue())
        self.assertEqual(pass_cnt, 0)
        self.assertEqual(fail_cnt, 0)

    def test_bias_test_completely_bias(self):
        """
        Testing bias results at a test_threshold of 0.50. All tests will fail.
        """
        labels = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
        results = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

        capturedOutput = io.StringIO()
        sys.stdout = capturedOutput

        misc.bias_test_check(labels, results, category='test_group')

        sys.stdout = sys.__stdout__

        pass_cnt, fail_cnt = pass_fail_count(capturedOutput.getvalue())
        self.assertEqual(pass_cnt, 0)
        self.assertEqual(fail_cnt, 5)

    def test_one_way_mi(self):
        df = pd.DataFrame({'feat1': [1, 2, 3, 2, 1],
                           'feat2': [3, 2, 1, 2, 3],
                           'feat3': [1, 2, 3, 2, 1],
                           'feat4': [1, 2, 3, 2, 1],
                           'group': [1, 1, 3, 4, 5],
                           'y':     [1, 2, 1, 1, 2]})
        expected_output = {
            'feature': {0: 'feat1', 1: 'feat2', 2: 'feat3', 3: 'feat4'},
            'group': {0: 0.12, 1: 0.12, 2: 0.12, 3: 0.12},
            'group_scaled': {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0},
            'y': {0: 0.12, 1: 0.12, 2: 0.12, 3: 0.12},
            'y_scaled': {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0}}

        features = ['feat1', 'feat2', 'feat3', 'feat4']
        group_column = 'group'
        y_var = 'y'

        output = misc.one_way_mi(df, features, group_column, y_var, (4, 2))
        for col in [group_column, y_var,
                    group_column+'_scaled', y_var+'_scaled']:
            output[col] = output[col].round(2)
        self.assertEqual(output.to_dict(), expected_output)
