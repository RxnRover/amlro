import itertools
import os
from typing import Dict, List

import numpy as np
import pandas as pd
from pyDOE2 import lhs
from scipy.stats.qmc import Sobol
from sklearn.utils import resample

from amlro.optimizer import optimizer


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
    write_files=False,
    exp_dir=None,
) -> pd.DataFrame:
    """This function will generate the full reaction space and training
    reaction conditions and if required it can called to write function
    to generate full combo file and traning combo files.
    User need to define parameter configurations, traning sampling techniques
    and traning set size.

    :param config: Dictionary of parameters, their bounds and resolution.
    :type config: Dict
    :param sampling: Sampling methods for generating traning reaction conditions,
                    defaults to 'random'
    :type sampling: str, optional
    :param training_size: Training set size required for initial experiments,
                         defaults to 20
    :type training_size: int, optional
    :param write_files: Option to enable writting files,
                        defaults to False
    :type write_files: bool, optional
    :param exp_dir: experimental directory for saving data files,
                    defaults to None
    :type exp_dir: _type_, optional
    :return: returning two dataframes with full reaction space
             and training conditions
    :rtype: pd.DataFrame
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
        print("chosse correct sampling method from [random,lhs,sobol]")

    if write_files:
        writting_reaction_scope(
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
    """This function will read the config dictionary and
    generate the full reaction connditions dataframe.

    :param config: Dictionary of parameters, their bounds and resolution.
    :type config: Dict
    :return: all the parameter combinations within bounds.
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

    encoded_data = df.apply(
        lambda row: optimizer.categorical_feature_encoding(
            config, row.tolist()
        ),
        axis=1,
    )
    encoded_df = pd.DataFrame(encoded_data.tolist(), columns=df.columns)

    return df, encoded_df


def random_sampling(df: pd.DataFrame, training_size=20) -> pd.DataFrame:
    """This function will generate subsample from full reaction space
    using random sampling.

    :param df: Full reaction space dataframe
    :type df: pd.DataFrame
    :param training_size: traning size/sub sample size, defaults to 20
    :type training_size: int, optional
    :return: training reaction conditions dataframe
    :rtype: pd.DataFrame
    """

    training_df = resample(df, n_samples=training_size)

    return training_df


def latin_hypercube_sampling(config, training_size=20) -> pd.DataFrame:
    """This function will generate subsample from full reaction space
    using latent hypercube sampling.

    :param config: Dictionary of parameters, their bounds and resolution.
    :type config: Dict
    :param training_size: traning size/sub sample size, defaults to 20
    :type training_size: int, optional
    :return: training reaction conditions dataframe
    :rtype: pd.DataFrame
    """

    feature_names = (
        config["continuous"]["feature_names"]
        + config["categorical"]["feature_names"]
    )

    n_features = len(feature_names)

    # Generate LHS samples
    lhs_samples = lhs(n_features, samples=training_size)

    training_df = feature_scaling(lhs_samples, config)

    return training_df


def sobol_sequnce_sampling(config, training_size=20) -> pd.DataFrame:
    """This function will generate subsample from full reaction space
    using sobol sequnce sampling.

    :param config: Dictionary of parameters, their bounds and resolution.
    :type config: Dict
    :param training_size: traning size/sub sample size, defaults to 20
    :type training_size: int, optional
    :return: training reaction conditions dataframe
    :rtype: pd.DataFrame
    """

    feature_names = (
        config["continuous"]["feature_names"]
        + config["categorical"]["feature_names"]
    )
    n_features = len(feature_names)

    # Generate Sobol samples
    sobol_engine = Sobol(
        d=n_features, scramble=False
    )  # make scramble True to make randomized
    sobol_samples = sobol_engine.random(n=training_size)

    training_df = feature_scaling(sobol_samples, config)

    return training_df


def feature_scaling(samples: List[List], config: Dict) -> pd.DataFrame:
    """This function will scale and map the continous and
    categorical features from latin hypercube and sobol sampling space.
    These methods will generate coordinates
    between 0,1 for each dimention.

    :param samples: sub sample generated from sampling method
    :type samples: List[List]
    :param config: Dictionary of parameters, their bounds and resolution.
    :type config: Dict
    :return: scaled traning reaction conditions dataframe
    :rtype: pd.DataFrame
    """

    feature_names = (
        config["continuous"]["feature_names"]
        + config["categorical"]["feature_names"]
    )
    n_continuous = len(config["continuous"]["feature_names"])

    scaled_df = pd.DataFrame(columns=feature_names)

    # Scaling continuous features
    for i, (bounds, feature_name) in enumerate(
        zip(
            config["continuous"]["bounds"],
            config["continuous"]["feature_names"],
        )
    ):

        min_val, max_val = bounds
        resolution = config["continuous"]["resolutions"][i]
        sample_scaled = samples[:, i] * (max_val - min_val) + min_val
        scaled_df[feature_name] = (
            np.round(sample_scaled / resolution) * resolution
        )

    # Map and scale categorical features
    for i, (values, feature_name) in enumerate(
        zip(
            config["categorical"]["values"],
            config["categorical"]["feature_names"],
        )
    ):

        num_categories = len(values)
        cat_samples = samples[:, n_continuous + i] * num_categories
        rounded_samples = np.floor(cat_samples).astype(int)
        scaled_df[feature_name] = [
            values[index] for index in rounded_samples
        ]  # if out of boubd values[min(index, num_categories - 1)]

    return scaled_df


def writting_reaction_scope(
    full_combo_df: pd.DataFrame,
    full_combo_encoded_df: pd.DataFrame,
    training_combo_df: pd.DataFrame,
    exp_dir: str,
):
    """writting reaction scopes into files,
    full combo file - full reaction space and
    training combo file - sub sample of reaction conditions
    to generate training  data file.

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

    full_combo_path = os.path.join(
        exp_dir, "full_combo_file.txt"
    )  # writting encoded version

    full_combo_decoded_path = os.path.join(
        exp_dir, "full_combo_decoded_file.txt"
    )

    training_combo_path = os.path.join(exp_dir, "training_combo_file.txt")

    full_combo_encoded_df.to_csv(full_combo_path, index=False)
    # Decoded full combo file is writting if its smaller reaction space
    if full_combo_df.shape[0] <= 20000:
        full_combo_df.to_csv(full_combo_decoded_path, index=False)

    training_combo_df.to_csv(training_combo_path, index=False)
