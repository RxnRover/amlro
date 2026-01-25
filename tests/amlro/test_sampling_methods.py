import unittest

import numpy as np
import pandas as pd

from amlro.sampling_methods import (
    feature_scaling,
    latin_hypercube_sampling,
    random_sampling,
    sobol_sequnce_sampling,
)


class TestSamplingMethods(unittest.TestCase):

    def setUp(self):

        self.df = pd.DataFrame(
            {
                "f1": np.linspace(0, 1, 100),
                "f2": np.linspace(10, 20, 100),
            }
        )
        self.config = {
            "continuous": {
                "feature_names": ["f1", "f2"],
                "bounds": [(0, 1), (10, 20)],
                "resolutions": [0.1, 1],
            },
            "categorical": {
                "feature_names": ["f3"],
                "values": [["A", "B", "C"]],
            },
        }

    def test_random_sampling(self):

        # Test generating subsample from given reaction condition dataframe using
        # random sampling method.

        sample_size = 10
        sampled_df = random_sampling(self.df, sample_size=sample_size)

        # Check that the sample size is correct
        self.assertEqual(len(sampled_df), sample_size)

        # Ensure that the sampled DataFrame has the same columns
        self.assertEqual(
            sampled_df.columns.to_list(), self.df.columns.to_list()
        )

    def test_latin_hypercube_sampling(self):

        # Test generating subsample from given feature config dictionary using
        # latin hypercube sampling method.

        sample_size = 10
        sampled_df = latin_hypercube_sampling(
            self.config, sample_size=sample_size
        )

        # Check that the sample size is correct
        self.assertEqual(len(sampled_df), sample_size)

        # Ensure that the sampled DataFrame has the correct columns
        expected_columns = (
            self.config["continuous"]["feature_names"]
            + self.config["categorical"]["feature_names"]
        )

        self.assertEqual(sampled_df.columns.to_list(), expected_columns)

    def test_sobol_sequnce_sampling(self):

        # Test generating subsample from given feature config dictionary using
        # Sobol sequnce sampling method.

        sample_size = 10
        sampled_df = sobol_sequnce_sampling(
            self.config, sample_size=sample_size
        )

        # Check that the sample size is correct
        self.assertEqual(len(sampled_df), sample_size)

        # Ensure that the sampled DataFrame has the correct columns
        expected_columns = (
            self.config["continuous"]["feature_names"]
            + self.config["categorical"]["feature_names"]
        )

        self.assertEqual(sampled_df.columns.to_list(), expected_columns)


class TestFeatureScaling(unittest.TestCase):

    def test_continuous_only(self):

        # Test feature scaling of given reaction condition sample from [0,1] to scale of
        # given feature bounds using continous features only.

        samples = np.array([[0.1, 1.0], [0.5, 0.1], [0.14, 1.0]])
        config = {
            "continuous": {
                "feature_names": ["feature1", "feature2"],
                "bounds": [(0, 10), (20, 30)],
                "resolutions": [1.0, 0.5],
            },
            "categorical": {"feature_names": [], "values": []},
        }
        expected_output = pd.DataFrame(
            {"feature1": [1.0, 5.0, 1.4], "feature2": [30.0, 21.0, 30.0]}
        )

        result = feature_scaling(samples, config)
        pd.testing.assert_frame_equal(result, expected_output)

    def test_categorical_only(self):

        # Test feature scaling of given reaction condition sample from [0,1] to scale of
        # given feature bounds using categorical features only.

        samples = np.array([[0.0], [0.25], [0.50], [0.75], [1.0]])

        config = {
            "continuous": {
                "feature_names": [],
                "bounds": [],
                "resolutions": [],
            },
            "categorical": {
                "feature_names": ["feature1"],
                "values": [["A", "B", "C", "D"]],
            },
        }

        expected_output = pd.DataFrame({"feature1": ["A", "B", "C", "D", "D"]})

        result = feature_scaling(samples, config)
        pd.testing.assert_frame_equal(result, expected_output)

    def test_mixed_data(self):

        # Test feature scaling of given reaction condition sample from [0,1] to scale of
        # given feature bounds using both continous and categorical  features.

        samples = np.array([[0.1, 0.25], [0.5, 0.75]])
        config = {
            "continuous": {
                "feature_names": ["feature1"],
                "bounds": [(0, 10)],
                "resolutions": [1.0],
            },
            "categorical": {
                "feature_names": ["feature2"],
                "values": [["A", "B", "C", "D"]],
            },
        }
        expected_output = pd.DataFrame(
            {"feature1": [1.0, 5.0], "feature2": ["B", "D"]}
        )

        result = feature_scaling(samples, config)
        pd.testing.assert_frame_equal(result, expected_output)
