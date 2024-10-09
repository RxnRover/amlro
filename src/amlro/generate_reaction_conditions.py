import itertools
import os
from typing import Dict

import numpy as np
import pandas as pd

from amlro.const import (
    FULL_COMBO_DECODED_FILENAME,
    FULL_COMBO_FILENAME,
    TRAINING_COMBO_FILENAME,
)
from amlro.optimizer import categorical_feature_encoding
from amlro.sampling_methods import (
    latin_hypercube_sampling,
    random_sampling,
    sobol_sequnce_sampling,
)
from amlro.validations import validate_reaction_scope_config


def get_reaction_scope(
    config: Dict,
    sampling: str = "random",
    training_size: int = 20,
    write_files: bool = False,
    exp_dir: str = None,
) -> pd.DataFrame:
    """Generate the full reaction space and training reaction conditions.

    If required  write function can be enabled to generate full combo file and
    traning combo files.

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
    :type exp_dir: str, optional
    :return: returning two dataframes with full reaction space
             and training conditions
    :rtype: pd.DataFrame
    :raises ValueError: incorrect sampling method.
    """
    validate_reaction_scope_config(config)

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
        error = "Incorrect sampling method" + str(sampling) + " provided."
        " Valid methods [random, lhs, sobol]"
        raise (ValueError(error))

    if write_files:
        write_reaction_scope(
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
        lambda row: categorical_feature_encoding(config, row.tolist()),
        axis=1,
    )
    encoded_df = pd.DataFrame(encoded_data.tolist(), columns=df.columns)

    return df, encoded_df


def write_reaction_scope(
    full_combo_df: pd.DataFrame,
    full_combo_encoded_df: pd.DataFrame,
    training_combo_df: pd.DataFrame,
    exp_dir: str,
):
    """write reaction scopes into files

    This code will generate following two files, full combo file (full reaction space)
    and training combo file (sub sample of reaction conditions
    to generate training  data file) in given experimental directory.

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

    full_combo_path = os.path.join(exp_dir, FULL_COMBO_FILENAME)
    full_combo_decoded_path = os.path.join(exp_dir, FULL_COMBO_DECODED_FILENAME)
    training_combo_path = os.path.join(exp_dir, TRAINING_COMBO_FILENAME)

    full_combo_encoded_df.to_csv(full_combo_path, index=False)
    # Decoded full combo file is writing if its
    # smaller reaction space (less than 20000)
    if full_combo_df.shape[0] <= 20000:
        full_combo_df.to_csv(full_combo_decoded_path, index=False)

    training_combo_df.to_csv(training_combo_path, index=False)
