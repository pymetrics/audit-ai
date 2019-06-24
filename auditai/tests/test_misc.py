import unittest
import io
import sys

from auditai import misc


class TestMisc(unittest.TestCase):
    def setUp(self):
        self.labels = [0, 0, 0, 0,
                       1, 1, 1, 1, 1, 1]
        self.results = [0.25, 0.25, 0.75, 0.75,
                        0.25, 0.25, 0.25, 0.75, 0.75, 0.75]

    def test_bias_test_check_all_pass(self):

        capturedOutput = io.StringIO()
        sys.stdout = capturedOutput

        misc.bias_test_check(self.labels, self.results, category='test_group')

        sys.stdout = sys.__stdout__

        expected_output = """
            *test_group passes 4/5 test at 0.50*
            *test_group passes Fisher exact test at 0.50*
            *test_group passes Chi squared test at 0.50*
            *test_group passes z test at 0.50*
            *test_group passes Bayes Factor test at 0.50*
            """
        self.assertEqual(' '.join(expected_output.split()),
                         ' '.join(capturedOutput.getvalue().split()))

    def test_bias_test_check_below_min_thresh(self):

        capturedOutput = io.StringIO()
        sys.stdout = capturedOutput

        misc.bias_test_check(self.labels, self.results, category='test_group',
                             test_thresh=0.20)

        sys.stdout = sys.__stdout__

        expected_output = """
            Unable to run 4/5 test
            Unable to run Fisher exact test
            Unable to run Chi squared test
            Unable to run z test
            Unable to run Bayes Factor test
            """
        self.assertEqual(' '.join(expected_output.split()),
                         ' '.join(capturedOutput.getvalue().split()))

    def test_bias_test_completely_bias(self):
        labels = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
        results = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

        capturedOutput = io.StringIO()
        sys.stdout = capturedOutput

        misc.bias_test_check(labels, results, category='test_group',
                             test_thresh=0.50)

        sys.stdout = sys.__stdout__

        expected_output = """
            Unable to run 4/5 test
            *test_group fails Fisher exact test at 0.50*
             - test_group minimum proportion at 0.50: 0.005
            *test_group fails Chi squared test at 0.50*
             - test_group minimum proportion at 0.50: 0.012
            *test_group fails z test at 0.50*
             - test_group minimum proportion at 0.50: 0.002
            *test_group fails Bayes Factor test at 0.50*
             - test_group minimum proportion at 0.50: 88.846
            """
        self.assertEqual(' '.join(expected_output.split()),
                         ' '.join(capturedOutput.getvalue().split()))
