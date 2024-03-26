import unittest

from amlro.amlro import main


class TestOptimizerAmlro(unittest.TestCase):
    def test_amlro(self):
        result = main()

        self.assertEqual(5, result)
