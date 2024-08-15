import unittest

import pandas as pd

from amlro import generate_reaction_conditions


class TestGetCombos(unittest.TestCase):

    def test_continous_features(self):
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

        self.assertEqual(combos, corr)
        pd.testing.assert_frame_equal(df, df_encoded)

    def test_categorical_features(self):

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

        self.assertEqual(combos, corr)
        self.assertEqual(combos_encoded, corr_encoded)

    def test_mixed_features(self):
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

        self.assertEqual(combos, corr)
        self.assertEqual(combos_encoded, corr_encoded)
