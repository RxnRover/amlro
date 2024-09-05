import os
from typing import Dict, List

import pandas as pd

from amlro.optimizer import optimizer


def generate_training_data(
    exp_dir: str,
    config: Dict,
    parameters: List = [],
    obj_values: List = [],
    itr: int = 0,
    termination: bool = False,
) -> List[float]:
    """Generates a training dataset for the ML model.

    This function handles the generation of training data for the optimization
    process involving experimental reactions. The function is designed to work
    iteratively,where each iteration represents a new training reaction.
    Depending on the iteration number (`itr`) and the termination flag,
    it writes the experimental data to files and provides the
    next set of reaction conditions.

    :param exp_dir: experimental directory for saving data files,
                    defaults to None
    :type exp_dir: str, optional
    :param config: Dictionary of parameters
    :type config: Dict
    :param parameters: parameter set from previous experiment or initial parameters,
                       defaults to [].
    :type parameters: List, optional
    :param obj_values: experimental yield from previous experiment.
    :type obj_values: List, optional
    :param itr: Experiment iteration, starting from 0, defaults to 0.
    :type itr: int, optional
    :param termination: termination of the training function after last iteration
                        without returning next reaction conditions, defaults to False
    :type termination: bool, optional
    :return: parameter set for next experiment.
    :rtype: List
    """

    training_dataset_path = os.path.join(exp_dir, "reactions_data.csv")
    training_dataset_decoded_path = os.path.join(
        exp_dir, "reactions_decoded.csv"
    )
    training_combo_path = os.path.join(exp_dir, "training_combo.csv")

    parameters_encoded = optimizer.categorical_feature_encoding(
        config, parameters
    )
    prev_parameters = (
        ",".join([str(elem) for elem in parameters])
        + ","
        + ",".join([str(obj_val) for obj_val in obj_values])
    )
    prev_parameters_encoded = (
        ",".join([str(elem) for elem in parameters_encoded])
        + ","
        + ",".join([str(obj_val) for obj_val in obj_values])
    )

    if itr == 0:
        continous_names = config["continuous"]["feature_names"]
        cat_names = config["categorical"]["feature_names"]
        obj_names = config["objectives"]

        col_names = continous_names + cat_names + obj_names
        col_names = ",".join([str(elem) for elem in col_names])

        write_data_to_training(training_dataset_path, col_names)
        write_data_to_training(training_dataset_decoded_path, col_names)

    if itr != 0:
        write_data_to_training(training_dataset_path, prev_parameters_encoded)
        write_data_to_training(training_dataset_decoded_path, prev_parameters)
        print("writing data to training dataset files...")

    if termination:
        # termination of function without returning reaction conditions
        # and complete writing last results to the file.
        # should activated in last iteration+1
        print("Training set generation completed..")
        return

    data = load_training_combo_file(training_combo_path)

    # data_decoded = optimizer.categorical_feature_decoding(config, data[itr])
    print("training  parameter combination...", data)

    return data[itr]


def load_training_combo_file(training_combo: str) -> List[float]:
    """Loads the training data.

    Reads the combination file as pandas data frame and return the
    reaction combination data as list.

    :param training_combo: training parameter combination file path
    :type training_combo: str
    :return: training parameter combination list
    :rtype: List[float]
    """
    data = pd.read_csv(training_combo, skiprows=0)

    return data.values.tolist()


def writing_training_file(exp_dir: str, objectives: List[List], config: Dict):
    """writes the experiment data.

    writing results when batch of training experiments results/objectives collected.

    :param exp_dir: experimental directory for saving data files
    :type exp_dir: str
    :param objectives: objective values for training reaction conditions
    :type objectives: List[List]
    """

    training_combo_path = os.path.join(exp_dir, "training_combo.csv")

    reaction_conditions_df = pd.read_csv(training_combo_path, skiprows=0)

    encoded_data = reaction_conditions_df.apply(
        lambda row: optimizer.categorical_feature_encoding(
            config, row.tolist()
        ),
        axis=1,
    )
    encoded_df = pd.DataFrame(
        encoded_data.tolist(), columns=reaction_conditions_df.columns
    )

    objective_values_df = pd.DataFrame(objectives, columns=config["objectives"])
    training_encoded_df = pd.concat([encoded_df, objective_values_df], axis=1)
    training_df = pd.concat(
        [reaction_conditions_df, objective_values_df], axis=1
    )

    training_dataset_path = os.path.join(exp_dir, "reactions_data.csv")
    training_dataset_decoded_path = os.path.join(
        exp_dir, "reactions_decoded.csv"
    )
    # file writing
    training_encoded_df.to_csv(training_dataset_path, index=False)
    training_df.to_csv(training_dataset_decoded_path, index=False)


def write_data_to_training(training_file: str, prev_parameters: str) -> None:
    """writes the reaction data

    Writes the prev reaction conditions and experimental objective values
    at the end of the training data file.

    :param training_file: traning set file path
    :type training_file: str
    :param prev_parameters: previous reaction conditions and objective values
    :type prev_parameters: str
    """
    # Open the file in append & read mode ('a+')
    with open(training_file, "a+") as file_object:
        file_object.write(prev_parameters + "\n")
