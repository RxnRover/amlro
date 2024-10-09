import unittest

import numpy as np

from amlro import pareto


class TestIsParetoDominant(unittest.TestCase):

    def test_is_pareto_dominant_min(self):

        # Test case where both objectives are minimized
        directions = ["min", "min"]

        # point 1 dominate point 2
        point1 = [1, 2]
        point2 = [2, 3]
        result = pareto.is_pareto_dominant(point1, point2, directions)
        self.assertTrue(result, "Expected point1 to dominate point2")

        # point1 does not dominate point2
        point1 = [2, 4]
        point2 = [1, 2]
        result = pareto.is_pareto_dominant(point1, point2, directions)
        self.assertFalse(result, "Expected point1 to not dominate point2")

    def test_is_pareto_dominant_max(self):
        # Test case where both objectives are maximized
        directions = ["max", "max"]

        # point 1 dominate point 2
        point1 = [5, 7]
        point2 = [3, 6]
        result = pareto.is_pareto_dominant(point1, point2, directions)
        self.assertTrue(result, "Expected point1 to dominate point2")

        # point1 does not dominate point2
        point1 = [3, 4]
        point2 = [5, 6]
        result = pareto.is_pareto_dominant(point1, point2, directions)
        self.assertFalse(result, "Expected point1 to not dominate point2")

    def test_is_pareto_dominant_mixed(self):
        # Test case where one objective is minimized and the other is maximized

        directions = [
            "min",
            "max",
        ]  # Minimize first objective, maximize second objective

        # point 1 dominate point 2
        point1 = [2, 4]
        point2 = [3, 3]
        result = pareto.is_pareto_dominant(point1, point2, directions)
        self.assertTrue(result, "Expected point1 to dominate point2")

        #  point1 does not dominate point2
        point1 = [4, 2]
        point2 = [2, 4]
        result = pareto.is_pareto_dominant(point1, point2, directions)
        self.assertFalse(result, "Expected point1 to not dominate point2")

    def test_is_pareto_dominant_identical(self):
        # Test case where point1 and point2 are identical
        point1 = [1, 1]
        point2 = [1, 1]
        directions = ["min", "min"]
        result = pareto.is_pareto_dominant(point1, point2, directions)
        self.assertFalse(
            result, "Expected identical points to not dominate each other"
        )


class TestIdentifyParetoFront(unittest.TestCase):

    def test_identify_pareto_front_min(self):

        # Test case where both objectives are minimized
        points = [[1, 2], [2, 1], [2, 3], [3, 3]]
        directions = ["min", "min"]
        nfeatures = 0  # assuming points only have objective values

        pareto_front = pareto.identify_pareto_front(
            points, directions, nfeatures
        )
        expected_front = [[1, 2], [2, 1]]
        np.testing.assert_array_equal(pareto_front, expected_front)

    def test_identify_pareto_front_max(self):

        # Test case where both objectives are maximized
        points = [[2, 3], [3, 2], [2, 1], [1, 1]]

        directions = ["max", "max"]
        nfeatures = 0  # assuming points only have objective values

        pareto_front = pareto.identify_pareto_front(
            points, directions, nfeatures
        )
        expected_front = [
            [2, 3],
            [3, 2],
        ]
        np.testing.assert_array_equal(pareto_front, expected_front)

    def test_identify_pareto_front_mixed(self):

        # Test case where one objective is minimized and the other is maximized
        points = [[1, 2], [3, 1], [2, 4], [4, 3]]
        directions = ["min", "max"]
        nfeatures = 0  # assuming points only have objective values

        pareto_front = pareto.identify_pareto_front(
            points, directions, nfeatures
        )
        expected_front = [[1, 2], [2, 4]]
        np.testing.assert_array_equal(pareto_front, expected_front)
