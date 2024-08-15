import itertools
import os
from typing import Dict

import numpy as np
import pandas as pd

from amlro.optimizer import optimizer
from amlro.sampling_methods import (
    latin_hypercube_sampling,
    random_sampling,
    sobol_sequnce_sampling,
)


def validate_config(config: Dict) -> None:
    """Validates the configuration dictionary for generating grids.

    :param config: Configuration to be checked
    :type config: Dict

    :raises ValueError: At least one given bound is invalid.
    :raises ValueError: At least one given resolution is invalid.
    """

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


def get_reaction_scope(
    config: Dict,
    sampling: str = "random",
    training_size: int = 20,
    write_files: bool = False,
    exp_dir: str = None,
) -> pd.DataFrame:
    """Generate the full reaction space and training
    reaction conditions and if required it can called to write function
    to generate full combo file and traning combo files.
    User need to define parameter configurations, training sampling techniques
    and traning set size.

    :param config: Dictionary of parameters, their bounds and resolution.
    :type config: Dict
    :param sampling: Sampling methods for generating traning reaction conditions,
                    defaults to 'random'
    :type sampling: str, optional
    :param training_size: Training set size required for initial experiments,
                         defaults to 20
    :type training_size: int, optional
    :param write_files: Option to enable writing files,
                        defaults to False
    :type write_files: bool, optional
    :param exp_dir: experimental directory for saving data files,
                    defaults to None
    :type exp_dir: _type_, optional
    :return: returning two dataframes with full reaction space
             and training conditions
    :rtype: pd.DataFrame
    :raises ValueError: incorrect sampling method.
    """

    reaction_conditions_df, reaction_conditions_encoded_df = (
        generate_reaction_grid(config)
    )

    if sampling == "random":
        training_conditions_df = random_sampling(
            reaction_conditions_df, training_size
        )
    elif sampling == "lhs":
        training_conditions_df = latin_hypercube_sampling(config, training_size)
    elif sampling == "sobol":
        training_conditions_df = sobol_sequnce_sampling(config, training_size)
    else:
        error = "chosse correct sampling method from [random,lhs,sobol]"
        raise (ValueError(error))

    if write_files:
        writing_reaction_scope(
            reaction_conditions_df,
            reaction_conditions_encoded_df,
            training_conditions_df,
            exp_dir,
        )

    return (
        reaction_conditions_df,
        reaction_conditions_encoded_df,
        training_conditions_df,
    )


def generate_reaction_grid(config: Dict) -> pd.DataFrame:
    """
    Generates all reaction conditions in the requested reaction space.

    Given all continuous and categorical input parameters defining a reaction
    space, every reaction parameter combination is generated, forming a
    uniform grid of points over the reaction space. From the config dictionary,
    the ``config["continuous"]["feature_names"]``, ``config["continuous"]
    ["feature_bounds"]``, and ``config["continuous"]["resolutions"]`` are used to
    define the reaction space. If there are categorical parameters, then
    ``config["categorical"]["feature_names"]`` and
    ``config["categorical"]["values"]`` are also required. It is allowed for there to
    be only continuous, only categorical, or a mixture of both types of
    parameters to be present.

    An encoded variant is also returned with categorical parameter values
    represented as numbers for ease-of-use by other algorithms and optimizers.

    :param config: Dictionary of parameters, their bounds and resolution.
    :type config: Dict
    :return: reaction space and encoded reaction space.
    :rtype: pd.DataFrame
    """

    validate_config(config)
    feature_names = (
        config["continuous"]["feature_names"]
        + config["categorical"]["feature_names"]
    )
    feature_arrays = []

    for i in range(len(config["continuous"]["feature_names"])):
        min_val = config["continuous"]["bounds"][i][0]
        max_val = config["continuous"]["bounds"][i][1]
        resolution = config["continuous"]["resolutions"][i]
        feature_arrays.append(
            np.arange(min_val, max_val + resolution, resolution)
        )

    for i in range(len(config["categorical"]["feature_names"])):
        feature_arrays.append(config["categorical"]["values"][i])

    # Generate all combinations
    all_combinations = itertools.product(*feature_arrays)

    df = pd.DataFrame(all_combinations, columns=feature_names)

    # encoding the reaction combinations
    encoded_data = df.apply(
        lambda row: optimizer.categorical_feature_encoding(
            config, row.tolist()
        ),
        axis=1,
    )
    encoded_df = pd.DataFrame(encoded_data.tolist(), columns=df.columns)

    return df, encoded_df


def writing_reaction_scope(
    full_combo_df: pd.DataFrame,
    full_combo_encoded_df: pd.DataFrame,
    training_combo_df: pd.DataFrame,
    exp_dir: str,
):
    """writing reaction scopes into files,
    full combo file (full reaction space) and
    training combo file (sub sample of reaction conditions
    to generate training  data file).

    :param full_combo_df: full reaction space dataframe
    :type full_combo_df: pd.DataFrame
    :param full_combo_encoded_df: full reaction space dataframe that
    catogerical varible encoded into numerical
    :type full_combo_encoded_df: pd.DataFrame
    :param training_combo_df: training reaction conditions dataframe
    :type training_combo_df: pd.DataFrame
    :param exp_dir: experimental directory for saving data files,
    :type exp_dir: str
    """
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    full_combo_path = os.path.join(exp_dir, "full_combo_file.txt")
    full_combo_decoded_path = os.path.join(
        exp_dir, "full_combo_decoded_file.txt"
    )
    training_combo_path = os.path.join(exp_dir, "training_combo_file.txt")

    full_combo_encoded_df.to_csv(full_combo_path, index=False)
    # Decoded full combo file is writing if its
    # smaller reaction space (less than 20000)
    if full_combo_df.shape[0] <= 20000:
        full_combo_df.to_csv(full_combo_decoded_path, index=False)

    training_combo_df.to_csv(training_combo_path, index=False)
