import unittest

from amlro import validations


class TestValidateReactionScopeConfig(unittest.TestCase):

    def test_valid_config(self):

        # Testing valid config dictionary and it shouldnt raise any exceptions.
        # Valid config should have correct lengths of bounds, resolutions,
        # feature names and values, min bounds should lower than max bounds ,
        # and resolution cannot be negative.

        config = {
            "continuous": {
                "bounds": [(0, 10), (5, 15)],
                "resolutions": [0.1, 1.0],
                "feature_names": ["f1", "f2"],
            },
            "categorical": {"feature_names": ["f3"], "values": [["A", "B"]]},
        }
        # Should not raise any exception
        validations.validate_reaction_scope_config(config)

    def test_invalid_bounds(self):

        # Test case with invalid bounds where min > max.
        config = {
            "continuous": {
                "bounds": [(10, 5), (5, 15)],
                "resolutions": [0.1, 1.0],
                "feature_names": ["f1", "f2"],
            },
            "categorical": {"feature_names": [], "values": []},
        }
        with self.assertRaises(ValueError) as context:
            validations.validate_reaction_scope_config(config)

        expected_msg = (
            "Max bound must be greater than or equal to the min bound"
        )

        self.assertIn(expected_msg, str(context.exception))

    def test_invalid_resolutions(self):

        # Test case with invalid resolution (non-positive values).

        config = {
            "continuous": {
                "bounds": [(0, 10), (5, 15)],
                "resolutions": [-0.1, 0],  # Invalid resolutions
                "feature_names": ["f1", "f2"],
            },
            "categorical": {"feature_names": [], "values": []},
        }

        with self.assertRaises(ValueError) as context:
            validations.validate_reaction_scope_config(config)

        expected_msg = "Resolutions must all be positive, nonzero values"
        self.assertIn(expected_msg, str(context.exception))

    def test_inconsistent_lengths_resoultions(self):

        # Test with inconsistent lengths of resolutions.
        # one bound and feature names, and two resolutions, inconsistent length
        config = {
            "continuous": {
                "bounds": [(0, 10)],
                "resolutions": [0.1, 1.0],
                "feature_names": ["f1"],
            },
            "categorical": {"feature_names": [], "values": []},
        }
        with self.assertRaises(ValueError) as context:
            validations.validate_reaction_scope_config(config)

        expected_msg = (
            "Lengths of continuous bounds, resolutions, and "
            "feature names must match."
        )

        self.assertIn(expected_msg, str(context.exception))

    def test_inconsistent_lengths_bounds(self):

        # Test with inconsistent lengths of continous bounds.
        # one bound, and two resolutions and feature names, inconsistent length
        config = {
            "continuous": {
                "bounds": [(0, 10)],
                "resolutions": [0.1, 1.0],
                "feature_names": ["f1", "f2"],
            },
            "categorical": {"feature_names": [], "values": []},
        }
        with self.assertRaises(ValueError) as context:
            validations.validate_reaction_scope_config(config)

        expected_msg = (
            "Lengths of continuous bounds, resolutions, and "
            "feature names must match."
        )

        self.assertIn(expected_msg, str(context.exception))

    def test_inconsistent_lengths_continous_feature_names(self):

        # Test with inconsistent lengths of continous feature names.
        # one bound and resoultion, and two feature names, inconsistent length
        config = {
            "continuous": {
                "bounds": [(0, 10)],
                "resolutions": [0.1],
                "feature_names": ["f1", "f2"],
            },
            "categorical": {"feature_names": [], "values": []},
        }
        with self.assertRaises(ValueError) as context:
            validations.validate_reaction_scope_config(config)

        expected_msg = (
            "Lengths of continuous bounds, resolutions, and "
            "feature names must match."
        )

        self.assertIn(expected_msg, str(context.exception))

    def test_inconsistent_lengths_categorical_feature_names(self):

        # Test with inconsistent lengths of categorical feature names.
        # one categorical value list, and two categorical feature names,
        # inconsistent length
        config = {
            "continuous": {
                "bounds": [(0, 10)],
                "resolutions": [0.1],
                "feature_names": ["f1"],
            },
            "categorical": {
                "feature_names": ["f2", "f3"],
                "values": [["A", "B"]],
            },
        }
        with self.assertRaises(ValueError) as context:
            validations.validate_reaction_scope_config(config)

        expected_msg = (
            "Lengths of categorical feature names, and values must match. "
        )

        self.assertIn(expected_msg, str(context.exception))

    def test_type_categorical_values(self):

        # Test case for  categorical values key has values of  list of lists.
        # values list with strings not list of strings
        config = {
            "continuous": {
                "bounds": [(0, 10)],
                "resolutions": [0.1],
                "feature_names": ["f1"],
            },
            "categorical": {"feature_names": ["f2"], "values": ["A"]},
        }
        with self.assertRaises(ValueError) as context:
            validations.validate_reaction_scope_config(config)

        expected_msg = "Categorical values must be lists of lists. "

        self.assertIn(expected_msg, str(context.exception))
