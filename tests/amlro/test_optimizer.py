import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from amlro import optimizer


class TestLoadData(unittest.TestCase):

    def setUp(self):
        # Common data for all tests
        self.reactions_data_path = "tmp/reactions_data.csv"
        self.full_combo_path = "tmp/full_combo.csv"
        self.config = {"objectives": ["Yield"]}

        # Mock data for reactions_data.csv
        self.mock_reactions_data = pd.DataFrame(
            {
                "f1": [100.0, 150.0, 140.0],
                "f2": [1, 2, 3],
                "f3": ["A", "B", "C"],
                "Yield": [80, 85, 90],
            }
        )

        # Mock data for full_combo.csv which have full reaction space
        self.mock_combination_file = pd.DataFrame(
            {
                "f1": [100.0, 120.0, 130.0, 150.0],
                "f2": [1, 2, 3, 2],
                "f3": ["A", "B", "C", "B"],
            }
        )

    @patch("amlro.optimizer.pd.read_csv")  # Patching pd.read_csv
    def test_load_data(self, mock_read_csv):

        # Define what the mocked pd.read_csv will return for each call
        mock_read_csv.side_effect = [
            self.mock_reactions_data,
            self.mock_combination_file,
        ]

        # Call the function under test
        x_train, y_train, test_data = optimizer.load_data(
            self.reactions_data_path, self.full_combo_path, self.config
        )

        # Assertions
        expected_x_train = pd.DataFrame(
            {
                "f1": [100.0, 150.0, 140.0],
                "f2": [1, 2, 3],
                "f3": ["A", "B", "C"],
            }
        )
        expected_y_train = pd.DataFrame({"Yield": [80, 85, 90]})
        expected_test_data = pd.DataFrame(
            {"f1": [120.0, 130.0], "f2": [2, 3], "f3": ["B", "C"]}
        )

        pd.testing.assert_frame_equal(x_train, expected_x_train)
        pd.testing.assert_frame_equal(y_train, expected_y_train)
        pd.testing.assert_frame_equal(test_data, expected_test_data)

        # Verifying that pd.read_csv was called twice
        self.assertEqual(mock_read_csv.call_count, 2)
        mock_read_csv.assert_any_call(self.reactions_data_path)
        mock_read_csv.assert_any_call(self.full_combo_path)


class TestPredictNextParameters(unittest.TestCase):

    def setUp(self):
        # Setup common mock config
        self.data = pd.DataFrame({"f1": [1, 2, 3, 4], "f2": [10, 20, 30, 40]})
        self.config_single_obj = {
            "objectives": ["Yield"],
            "directions": ["max"],
        }
        self.config_multi_obj = {
            "objectives": ["Yield", "side_products"],
            "directions": ["max", "min"],
        }
        self.regr = MagicMock()
        self.batch_size = 2

    def test_single_objective(self):
        # Mock the return value of the `predict` method for single objective
        self.regr.predict.return_value = np.array([80, 90, 70, 85])
        # Call the function
        result = optimizer.predict_next_parameters(
            self.regr, self.data, self.config_single_obj, self.batch_size
        )

        # Assert that the mock was called with the data
        # mock_predict.assert_called_once_with(self.data)

        expected_best_parameters = pd.DataFrame({"f1": [2, 4], "f2": [20, 40]})

        self.regr.predict.assert_called_once_with(self.data)
        # Check that the result is as expected
        self.assertEqual(len(result), self.batch_size)
        pd.testing.assert_frame_equal(
            result.reset_index(drop=True),
            expected_best_parameters.reset_index(drop=True),
        )

    def test_multiple_objectives(self):
        # Mock the return value of the `predict` method for multiple objectives
        self.regr.predict.return_value = np.array(
            [[80, 0.8], [90, 0.75], [70, 0.9], [85, 0.7]]
        )

        # Call the function
        result = optimizer.predict_next_parameters(
            self.regr, self.data, self.config_multi_obj, self.batch_size
        )
        print(result)
        expected_best_parameters = pd.DataFrame(
            {"f1": [2.0, 4.0], "f2": [20.0, 40.0]}
        )

        # Assert that the mock was called with the data
        self.regr.predict.assert_called_once_with(self.data)

        # Check that the result is as expected for multi-objective
        self.assertEqual(len(result), self.batch_size)
        pd.testing.assert_frame_equal(
            result.reset_index(drop=True),
            expected_best_parameters.reset_index(drop=True),
        )


class TestCategoricalFeatureEncodingDecoding(unittest.TestCase):

    def setUp(self):

        self.config = {
            "continuous": {"feature_names": ["f1", "f2"]},
            "categorical": {
                "feature_names": ["f3", "f4"],
                "values": [["A", "B"], ["X", "Y"]],
            },
        }

    def test_categorical_feature_encoding(self):

        prev_parameters = [0.5, 10, "B", "Y"]
        expected_encoded = [0.5, 10, 1, 1]  # "B" -> 1, "Y" -> 1

        encoded = optimizer.categorical_feature_encoding(
            self.config, prev_parameters
        )

        # self.assertEqual(encoded, np.array(expected_encoded))
        # ValueError: The truth value of an array with more than one element is
        # ambiguous. Use a.any() or a.all()
        np.testing.assert_array_equal(encoded, np.array(expected_encoded))

    def test_categorical_feature_decoding(self):
        best_combo = [0.5, 10, 1, 1]  # 1 -> "B", 1 -> "Y"
        expected_decoded = [0.5, 10, "B", "Y"]

        decoded = optimizer.categorical_feature_decoding(
            self.config, best_combo
        )

        self.assertEqual(decoded, expected_decoded)
