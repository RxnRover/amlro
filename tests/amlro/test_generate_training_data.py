import os
import unittest
from unittest.mock import patch

from amlro.generate_training_data import generate_training_data


class TestGenerateTrainingData(unittest.TestCase):

    def setUp(self):

        self.exp_dir = "test_dir"
        self.config = {
            "continuous": {"feature_names": ["f1", "f2"]},
            "categorical": {"feature_names": ["f3"]},
            "objectives": ["z1"],
        }
        self.reaction_combos = [[0.4, 0.5, "A"], [0.5, 0.5, "B"]]
        self.training_dataset_path = os.path.join(
            self.exp_dir, "reactions_data.csv"
        )
        self.training_dataset_decoded_path = os.path.join(
            self.exp_dir, "reactions_decoded.csv"
        )

    @patch("amlro.generate_training_data.write_data_to_training")
    @patch("amlro.generate_training_data.load_training_combo_file")
    @patch("amlro.optimizer.optimizer.categorical_feature_encoding")
    def test_generate_training_data_itr_0(
        self, mock_encoding, mock_load_combo, mock_write
    ):

        # Mock the return value of categorical_feature_encoding
        mock_encoding.return_value = []

        # Mock the return value of load_training_combo_file
        mock_load_combo.return_value = self.reaction_combos

        # function inputs
        parameters = []
        obj_values = []
        itr = 0
        termination = False

        result = generate_training_data(
            self.exp_dir, self.config, parameters, obj_values, itr, termination
        )

        # Check that categorical_feature_encoding is called with appropriate arguments
        mock_encoding.assert_called_once_with(self.config, parameters)

        # Check that write_data_to_training is called with appropriate arguments
        # If the mock was called multiple times, assert_any_call will pass
        # if any of the calls match the specified arguments.
        mock_write.assert_any_call(self.training_dataset_path, "f1,f2,f3,z1")
        mock_write.assert_any_call(
            self.training_dataset_decoded_path, "f1,f2,f3,z1"
        )

        # Check the return value
        # Since itr = 0, it returns the first combination
        self.assertEqual(result, [0.4, 0.5, "A"])

    @patch("amlro.generate_training_data.write_data_to_training")
    @patch("amlro.generate_training_data.load_training_combo_file")
    @patch("amlro.optimizer.optimizer.categorical_feature_encoding")
    def test_generate_training_data_itr_1(
        self, mock_encoding, mock_load_combo, mock_write
    ):

        # Mock the return value of categorical_feature_encoding
        mock_encoding.return_value = [0.4, 0.5, 1]

        # Mock the return value of load_training_combo_file
        mock_load_combo.return_value = self.reaction_combos

        # Define function inputs
        parameters = [0.4, 0.5, "A"]
        obj_values = [0.9]
        itr = 1
        termination = False

        result = generate_training_data(
            self.exp_dir, self.config, parameters, obj_values, itr, termination
        )

        # Check that categorical_feature_encoding is called with appropriate arguments
        mock_encoding.assert_called_once_with(self.config, parameters)

        # Check that write_data_to_training is called with appropriate arguments
        mock_write.assert_any_call(self.training_dataset_path, "0.4,0.5,1,0.9")
        mock_write.assert_any_call(
            self.training_dataset_decoded_path, "0.4,0.5,A,0.9"
        )

        # Check the return value
        # Since itr = 1, it returns the second combination
        self.assertEqual(result, [0.5, 0.5, "B"])

    @patch("amlro.generate_training_data.write_data_to_training")
    @patch("amlro.generate_training_data.load_training_combo_file")
    @patch("amlro.optimizer.optimizer.categorical_feature_encoding")
    def test_generate_training_data_termination(
        self, mock_encoding, mock_load_combo, mock_write
    ):
        # Mock the return value of categorical_feature_encoding
        mock_encoding.return_value = [0.5, 0.5, 2]

        # Mock the return value of load_training_combo_file
        mock_load_combo.return_value = self.reaction_combos

        # Define function inputs
        parameters = [0.5, 0.5, "B"]
        obj_values = [0.9]
        itr = 2
        termination = True

        result = generate_training_data(
            self.exp_dir, self.config, parameters, obj_values, itr, termination
        )

        # Check that categorical_feature_encoding is called with appropriate arguments
        mock_encoding.assert_called_once_with(self.config, parameters)

        # Check that write_data_to_training is called with appropriate arguments
        mock_write.assert_any_call(self.training_dataset_path, "0.5,0.5,2,0.9")
        mock_write.assert_any_call(
            self.training_dataset_decoded_path, "0.5,0.5,B,0.9"
        )

        # Check that the function returns None due to termination
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
