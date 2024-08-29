import unittest

import pandas as pd

from amlro import generate_reaction_conditions


class TestGetReactionScope(unittest.TestCase):

    def setUp(self):

        self.config = {
            "continuous": {
                "bounds": [[0, 1]],
                "resolutions": [1],
                "feature_names": ["f1"],
            },
            "categorical": {
                "feature_names": ["f2"],
                "values": [["A", "B"]],
            },
        }

        self.expected_full_reaction_scope = pd.DataFrame(
            [[0, "A"], [0, "B"], [1, "A"], [1, "B"]], columns=["f1", "f2"]
        )

        self.expected_full_reaction_scope_encoded = pd.DataFrame(
            [[0, 0], [0, 1], [1, 0], [1, 1]], columns=["f1", "f2"]
        )

    def test_get_reaction_scope_random(self):

        # Test generating the full reaction scope and generating training reaction
        # conditions with random sampling. Here we are testing length of the
        # dataframes and column names.
        (
            reaction_conditions_df,
            reaction_conditions_encoded_df,
            training_conditions_df,
        ) = generate_reaction_conditions.get_reaction_scope(
            self.config, sampling="random", training_size=2
        )

        # Check that the generated reaction space is the correct full reaction space
        pd.testing.assert_frame_equal(
            reaction_conditions_df,
            self.expected_full_reaction_scope,
            check_dtype=False,
        )
        pd.testing.assert_frame_equal(
            reaction_conditions_encoded_df,
            self.expected_full_reaction_scope_encoded,
            check_dtype=False,
        )

        # Check the length of the training conditions
        self.assertEqual(len(training_conditions_df), 2)

    def test_get_reaction_scope_lhs(self):

        # Test generating the full reaction scope and generating training reaction
        # conditions with laten hypercube sampling. Here we are testing length of
        # the dataframes and column names.
        (
            reaction_conditions_df,
            reaction_conditions_encoded_df,
            training_conditions_df,
        ) = generate_reaction_conditions.get_reaction_scope(
            self.config, sampling="lhs", training_size=2
        )

        # Check that the generated reaction space is the correct full reaction space
        pd.testing.assert_frame_equal(
            reaction_conditions_df,
            self.expected_full_reaction_scope,
            check_dtype=False,
        )
        pd.testing.assert_frame_equal(
            reaction_conditions_encoded_df,
            self.expected_full_reaction_scope_encoded,
            check_dtype=False,
        )

        # Check the length of the training conditions
        self.assertEqual(len(training_conditions_df), 2)

    def test_get_reaction_scope_sobol(self):

        # Test generating the full reaction scope and generating training reaction
        # conditions with Sobol sequnce sampling. Here we are testing length of
        # the dataframes and column names.
        (
            reaction_conditions_df,
            reaction_conditions_encoded_df,
            training_conditions_df,
        ) = generate_reaction_conditions.get_reaction_scope(
            self.config, sampling="sobol", training_size=2
        )

        # Check that the generated reaction space is the correct full reaction space
        pd.testing.assert_frame_equal(
            reaction_conditions_df,
            self.expected_full_reaction_scope,
            check_dtype=False,
        )
        pd.testing.assert_frame_equal(
            reaction_conditions_encoded_df,
            self.expected_full_reaction_scope_encoded,
            check_dtype=False,
        )

        # Check the length of the training conditions
        self.assertEqual(len(training_conditions_df), 2)

    def test_get_reaction_scope_invalid_sampling(self):

        # Test invalid sampling method
        with self.assertRaises(ValueError) as context:
            generate_reaction_conditions.get_reaction_scope(
                self.config, sampling="invalid", training_size=2
            )
        expected_msg = "Incorrect sampling method"
        self.assertIn(expected_msg, str(context.exception))


class TestGenerateReactionGrid(unittest.TestCase):

    def test_continous_features(self):

        # Test generating reaction grid using continous features only.
        # Both generated encoded and decoded reaction grid dataframes were tested here.
        config = {
            "continuous": {
                "bounds": [[0, 1], [0, 1]],
                "resolutions": [1, 1],
                "feature_names": ["f1", "f2"],
            },
            "categorical": {"feature_names": [], "values": []},
        }

        corr = [[0, 0], [0, 1], [1, 0], [1, 1]]

        df, df_encoded = generate_reaction_conditions.generate_reaction_grid(
            config
        )
        combos = pd.DataFrame(df).values.tolist()

        # Check that the generated reaction space is the correct full reaction space
        self.assertEqual(combos, corr)
        pd.testing.assert_frame_equal(df, df_encoded)

    def test_categorical_features(self):

        # Test generating reaction grid using categorical features only.
        # Both generated encoded and decoded reaction grid dataframes were tested here.
        config = {
            "continuous": {
                "bounds": [],
                "resolutions": [],
                "feature_names": [],
            },
            "categorical": {
                "feature_names": ["Animal", "Colour"],
                "values": [["cat", "dog"], ["black", "grey"]],
            },
        }

        corr = [
            ["cat", "black"],
            ["cat", "grey"],
            ["dog", "black"],
            ["dog", "grey"],
        ]

        corr_encoded = [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ]

        df, df_encoded = generate_reaction_conditions.generate_reaction_grid(
            config
        )
        combos = pd.DataFrame(df).values.tolist()
        combos_encoded = pd.DataFrame(df_encoded).values.tolist()

        # Check that the generated reaction space is the correct full reaction space
        self.assertEqual(combos, corr)
        self.assertEqual(combos_encoded, corr_encoded)

    def test_mixed_features(self):

        # Test generating reaction grid using both continous and categoriucal features.
        # Both generated encoded and decoded reaction grid dataframes were tested here.
        config = {
            "continuous": {
                "bounds": [[0, 1]],
                "resolutions": [1],
                "feature_names": ["f1"],
            },
            "categorical": {
                "feature_names": ["Animal", "Colour"],
                "values": [["cat", "dog"], ["black", "grey"]],
            },
        }

        corr = [
            [0, "cat", "black"],
            [0, "cat", "grey"],
            [0, "dog", "black"],
            [0, "dog", "grey"],
            [1, "cat", "black"],
            [1, "cat", "grey"],
            [1, "dog", "black"],
            [1, "dog", "grey"],
        ]

        corr_encoded = [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ]

        df, df_encoded = generate_reaction_conditions.generate_reaction_grid(
            config
        )
        combos = pd.DataFrame(df).values.tolist()
        combos_encoded = pd.DataFrame(df_encoded).values.tolist()

        # Check that the generated reaction space is the correct full reaction space
        self.assertEqual(combos, corr)
        self.assertEqual(combos_encoded, corr_encoded)
