from typing import Dict


def validate_reaction_scope_config(config: Dict) -> None:
    """Validates the part of the configuration dictionary for generating grids.

    :param config: Configuration to be checked
    :type config: Dict

    :raises ValueError: At least one given bound is invalid.
    :raises ValueError: At least one given resolution is invalid.
    """

    # Check for inconsistent lengths for continous keys
    if len(config["continuous"]["bounds"]) != len(
        config["continuous"]["resolutions"]
    ) or len(config["continuous"]["bounds"]) != len(
        config["continuous"]["feature_names"]
    ):
        msg = "Lengths of continuous bounds, resolutions, and feature names must match."
        msg += "Given bounds: {}, Given resolutions: {}, Given feture names: {}".format(
            len(config["continuous"]["bounds"]),
            len(config["continuous"]["resolutions"]),
            len(config["continuous"]["feature_names"]),
        )
        raise ValueError(msg)

    # Check for invalid bounds
    for bound in config["continuous"]["bounds"]:
        if bound[0] > bound[1]:
            msg = "Max bound must be greater than or equal to the min "
            msg += "bound. Given bounds: Min = {}, Max = {}".format(
                bound[0], bound[1]
            )
            raise (ValueError(msg))

    # Check for invalid resolutions
    for resolution in config["continuous"]["resolutions"]:
        if resolution <= 0:
            msg = "Resolutions must all be positive, nonzero values. "
            msg += "Given resolutions: {}".format(
                config["continuous"]["resolutions"]
            )
            raise (ValueError(msg))

    # Check if categorical values are lists of lists
    if not all(
        isinstance(values, list) for values in config["categorical"]["values"]
    ):
        msg = "Categorical values must be lists of lists. "
        msg += "Given values: {}".format(len(config["categorical"]["values"]))
        raise ValueError(msg)

    # Check for inconsistent lengths for categorical keys
    if len(config["categorical"]["feature_names"]) != len(
        config["categorical"]["values"]
    ):
        msg = "Lengths of categorical feature names, and values must match. "
        msg += "Given feature names: {}, Given values: {}".format(
            len(config["categorical"]["feature_names"]),
            len(config["categorical"]["values"]),
        )
        raise ValueError(msg)


def validate_optimizer_config(config: Dict) -> None:
    """Validates the configdict for the optimizer and generate training set.

    :param config: Configuration to be checked
    :type config: Dict

    :raises ValueError: If lengths of directions and objectives do not match.
    :raises ValueError: If any direction is not 'min' or 'max'.
    """

    # Check feature related config
    validate_reaction_scope_config(config)

    # Check if directions and objectives have the same length
    if len(config["directions"]) != len(config["objectives"]):
        msg = "Lengths of directions and objectives must match. "
        msg += "Given directions: {}, Given objectives: {}".format(
            len(config["directions"]), len(config["objectives"])
        )
        raise ValueError(msg)

    # Check if directions only contain 'min' or 'max'
    valid_directions = {"min", "max"}
    for direction in config["directions"]:
        if direction not in valid_directions:
            msg = f"Invalid direction '{direction}' found. "
            msg += "Directions must only contain 'min' or 'max'. "
            msg += "Given directions: {}".format(config["directions"])
            raise ValueError(msg)
