import unittest

from auditai.utils.functions import get_unique_name, two_tailed_ztest, dirichln


class TestUtils(unittest.TestCase):
    def test_get_unique_name(self):
        new_name = 'matched'
        name_list = ['feat1', 'feat2', 'matched']

        output = get_unique_name(new_name, name_list)
        self.assertEqual(output, 'matched_new')

        name_list.append(output)
        output = get_unique_name(new_name, name_list)
        self.assertEqual(output, 'matched_new_new')

    def test_two_tailed_ztest(self):
        success1, success2, total1, total2 = 5, 10, 10, 10
        zstat, p_value = two_tailed_ztest(success1, success2, total1, total2)
        output = tuple(round(i, 2) for i in (zstat, p_value))
        self.assertEqual(output, (-2.58, 0.01))

    def dirichln(self):
        output = round(dirichln([1, 2]), 2)
        self.assertEqual(output, -0.69)

        output = round(dirichln([1]), 2)
        self.assertEqual(output, 0.0)
