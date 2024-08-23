from typing import Dict, List

import numpy as np
import pandas as pd
from pyDOE2 import lhs
from scipy.stats.qmc import Sobol
from sklearn.utils import resample


def random_sampling(df: pd.DataFrame, sample_size: int = 20) -> pd.DataFrame:
    """Generate subsample from full reaction space using random sampling.

    :param df: Dataframe with full reaction space
    :type df: pd.DataFrame
    :param sample_size: sub sample size, defaults to 20
    :type sample_size: int, optional
    :return: sub sample of reaction space needed for training set generation
    :rtype: pd.DataFrame
    """

    sample_df = resample(df, n_samples=sample_size)

    return sample_df


def latin_hypercube_sampling(
    config: Dict, sample_size: int = 20
) -> pd.DataFrame:
    """Generate subsample from full reaction space using  latent hypercube sampling.

    :param config: Dictionary of parameters, their bounds and resolution.
    :type config: Dict
    :param sample_size: sub sample size, defaults to 20
    :type sample_size: int, optional
    :return: sub sample of reaction space needed for training set generation
    :rtype: pd.DataFrame
    """

    feature_names = (
        config["continuous"]["feature_names"]
        + config["categorical"]["feature_names"]
    )

    n_features = len(feature_names)

    # Generate LHS samples
    lhs_samples = lhs(n_features, samples=sample_size)

    training_df = feature_scaling(lhs_samples, config)

    return training_df


def sobol_sequnce_sampling(config: Dict, sample_size: int = 20) -> pd.DataFrame:
    """Generate subsample from full reaction space using sobol sequnce sampling.

    :param config: Dictionary of parameters, their bounds and resolution.
    :type config: Dict
    :param sample_size: sub sample size, defaults to 20
    :type sample_size: int, optional
    :return: sub sample of reaction space needed for training set generation
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
    sobol_samples = sobol_engine.random(n=sample_size)

    training_df = feature_scaling(sobol_samples, config)

    return training_df


def feature_scaling(samples: List[List], config: Dict) -> pd.DataFrame:
    """This function will scale and map the continous and
    categorical features from latin hypercube and sobol sampling space.
    These methods will generate coordinates
    between [0,1] for each dimention.

    From the config dictionary,
    the ``config["continuous"]["feature_names"]``, ``config["continuous"]
    ["feature_bounds"]``, and ``config["continuous"]["resolutions"]`` are
    used rescale the continous features and ``config["continuous"]["bounds"]``
    and ``config["continuous"]["feature_names"]`` are used to
    rescale categorical features.

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
        print(min_val, max_val)
        resolution = config["continuous"]["resolutions"][i]
        sample_scaled = samples[:, i] * (max_val - min_val) + min_val
        print(samples[:, i], sample_scaled)
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
        # need to check min between index and categories-1 to handle
        # index out of bounds when sample value is equal to 1.0
        scaled_df[feature_name] = [
            values[min(index, num_categories - 1)] for index in rounded_samples
        ]

    return scaled_df
