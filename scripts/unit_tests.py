import unittest
import config
import utils

import torch


class PairCountTestCase(unittest.TestCase):

    def setUp(self):
        self.filename = config.MSA_FILESd[1]

    def test_main_counts(self):
        interaction = False
        print("Runing c++ version")
        counts_main_cpp, counts_int_cpp = utils.calc_mutant_counts(
            self.filename, cpp=True, interaction=interaction)
        print("Runing torch version")
        counts_main, counts_int = utils.calc_mutant_counts(
            self.filename, cpp=False, interaction=interaction)
        print("Done calculating")
        self.assertIsNone(counts_int_cpp)
        self.assertIsNone(counts_int)
        self.assertFalse(torch.all(counts_main == 0)) # it is not zeros
        self.assertTrue(torch.all(counts_main == counts_main_cpp).item())

    def test_interaction_counts(self):
        interaction = True
        counts_main_cpp, counts_int_cpp = utils.calc_mutant_counts(
            self.filename, cpp=True, interaction=interaction)
        counts_main, counts_int = utils.calc_mutant_counts(
            self.filename, cpp=False, interaction=interaction)
        self.assertFalse(torch.all(counts_int == 0)) # it is not zeros
        self.assertTrue(torch.all(counts_int == counts_int_cpp).item())




if __name__ == '__main__':
    unittest.main()


