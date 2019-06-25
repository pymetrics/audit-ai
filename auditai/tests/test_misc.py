import unittest
import io
import sys
import re

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
