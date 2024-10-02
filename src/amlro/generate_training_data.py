import os
from typing import Any, Dict, List

import pandas as pd

from amlro.const import REACTION_DATA_FILENAME, TRAINING_COMBO_FILENAME
from amlro.optimizer import optimizer


def generate_training_data(
    exp_dir: str,
    config: Dict,
    parameters: List = [],
    obj_values: List = [],
    filename: str = REACTION_DATA_FILENAME,
    termination: bool = False,
) -> List[Any]:
    """Generates a training dataset for the ML model.

    This function handles the generation of training data point for the optimization
    process involving experimental reactions. The function is designed to work
    iteratively,where each iteration represents a generate new training reaction.
    Depending on the iteration number (`itr`) and the termination flag,
    it writes the experimental data to files and provides the
    next set of reaction conditions.

    :param exp_dir: experimental directory for saving data files
    :type exp_dir: str
    :param config: Dictionary of parameters
    :type config: Dict
    :param parameters: Previous experiment parameters or initial parameters,
                       defaults to [].
    :type parameters: List, optional
    :param obj_values: experimental yield from previous experiment.
    :type obj_values: List, optional
    :param filename: filename for reaction data file, filename for reaction data file,
                     defaults to the value of `amlro.const.REACTION_DATA_FILENAME`.
    :type filename: str, optional
    :param termination: termination of the training function after last iteration
                        without returning next reaction conditions, defaults to False
    :type termination: bool, optional
    :return: parameter set for next experiment.
    :rtype: List
    """

    write_data_to_training_files(
        exp_dir=exp_dir,
        config=config,
        parameters=parameters,
        obj_values=obj_values,
        filename=filename,
    )
    if termination:
        # termination of function without returning reaction conditions
        # and complete writing last results to the file.
        # should activated in last iteration+1
        print("Training set generation completed..")
        return None

    # Reads the training combo file and returns the next reaction conditions.
    next_conditions = get_next_training_conditions(
        exp_dir=exp_dir, config=config, filename=filename
    )

    if next_conditions:
        return next_conditions
    else:
        print("Training set generation completed.")
        return None


def get_next_training_conditions(
    exp_dir: str, config: Dict, filename: str = REACTION_DATA_FILENAME
) -> List[Any]:
    """Returns next reaction conditions from the training reaction space.

    :param exp_dir: experimental directory for saving data files
    :type exp_dir: str
    :param config: Dictionary of parameters
    :type config: Dict
    :param filename: filename for reaction data file, filename for reaction data file,
                     defaults to the value of `amlro.const.REACTION_DATA_FILENAME`.
    :type filename: str, optional
    :return: parameter set for next experiment.
    :rtype: List
    """

    training_combo_path = os.path.join(exp_dir, TRAINING_COMBO_FILENAME)
    name, extension = os.path.splitext(filename)
    decoded_filename = f"{name}_decoded{extension}"

    reaction_data_decoded_path = os.path.join(exp_dir, decoded_filename)

    training_df = pd.read_csv(training_combo_path, header=0)
    reaction_data_df = pd.read_csv(reaction_data_decoded_path, header=0)

    target_columns = config["objectives"]
    reaction_data_df = reaction_data_df.drop(target_columns, axis=1)

    if not reaction_data_df.empty:

        reactions_to_perform_df = pd.merge(
            training_df, reaction_data_df, how="left", indicator=True
        )

        reactions_to_perform_df = reactions_to_perform_df[
            reactions_to_perform_df["_merge"] == "left_only"
        ].drop(columns="_merge")
    else:
        reactions_to_perform_df = training_df

    if not reactions_to_perform_df.empty:
        return reactions_to_perform_df.iloc[0].to_list()
    else:
        return None


def write_data_to_training_files(
    exp_dir: str,
    config: Dict,
    parameters: List = [],
    obj_values: List = [],
    filename: str = REACTION_DATA_FILENAME,
):
    """Writes experimental reaction data into the reaction data files.

    :param exp_dir: experimental directory for saving data files
    :type exp_dir: str
    :param config: Dictionary of parameters
    :type config: Dict
    :param filename: filename for reaction data file, filename for reaction data file,
                     defaults to the value of `amlro.const.REACTION_DATA_FILENAME`.
    :type filename: str, optional
    :param obj_values: experimental yield from previous experiment.
    :type obj_values: List, optional
    :param filename: filename for reaction data file, defaults to reaction_data.csv.
    :type filename: str, optional
    """

    reaction_data_path = os.path.join(exp_dir, filename)

    name, extension = os.path.splitext(filename)
    decoded_filename = f"{name}_decoded{extension}"

    reaction_data_decoded_path = os.path.join(exp_dir, decoded_filename)

    if not os.path.exists(reaction_data_path):
        continous_names = config["continuous"]["feature_names"]
        cat_names = config["categorical"]["feature_names"]
        obj_names = config["objectives"]

        col_names = continous_names + cat_names + obj_names
        col_names = ",".join([str(elem) for elem in col_names])

        write_line_to_a_file(reaction_data_path, col_names)
        write_line_to_a_file(reaction_data_decoded_path, col_names)

    # Each iteration (excepts initial itr), previous experimental results and
    # conditions are write to the reaction data files.
    else:

        if not parameters:
            # stop writing if previous reaction data file exists.
            return

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
        write_line_to_a_file(reaction_data_path, prev_parameters_encoded)
        write_line_to_a_file(reaction_data_decoded_path, prev_parameters)
        print("writing data to training dataset files...")


def write_line_to_a_file(training_file: str, reaction_results: str) -> None:
    """writes a data line to a file

    Writes the prev reaction conditions and experimental objective values
    at the end of the training data file.

    :param training_file: traning set file path
    :type training_file: str
    :param reaction_results: previous reaction conditions and objective values
    :type reaction_results: str
    """

    # Open the file in append & read mode ('a+') for writes results each iteration
    with open(training_file, "a+") as file_object:
        file_object.write(reaction_results + "\n")


def load_training_conditions(
    training_combo: str = TRAINING_COMBO_FILENAME,
) -> List[Any]:  # pragma: no cov
    """Loads the training data.

    Reads the combination file as pandas data frame and return the
    reaction combination data as list.

    :param training_combo: training parameter combination file path, defaults
         to the value of `amlro.const.TRAINING_COMBO_FILE`.
    :type training_combo: str
    :return: training parameter combination list
    :rtype: List
    """

    data = pd.read_csv(training_combo, skiprows=0)

    return data.values.tolist()


def write_training_data(
    exp_dir: str,
    objectives: List[List],
    config: Dict,
    filename: str = REACTION_DATA_FILENAME,
):  # pragma: no cov
    """writes the experiment data.

    This function can be used if we needs to perform all the training conditions
    together and write the experimental results together. Batch of training
    experiments results/objectives are neded to collected.

    :param exp_dir: experimental directory for saving data files
    :type exp_dir: str
    :param objectives: objective values for training reaction conditions
    :type objectives: List[List]
    :param filename: filename for reaction data file, filename for reaction data file,
                     defaults to the value of `amlro.const.REACTION_DATA_FILENAME`.
    :type filename: str, optional

    :raises ValueError: if lengths of training conditions and objectives dont match.
    """

    training_combo_path = os.path.join(exp_dir, TRAINING_COMBO_FILENAME)

    reaction_conditions_df = pd.read_csv(training_combo_path, skiprows=0)

    if len(reaction_conditions_df) != len(objectives):
        msg = "Lengths of training conditions and experimental results must match."
        msg += "Given reaction conditions: {}, Given objectives: {} ".format(
            len(reaction_conditions_df),
            len(objectives),
        )
        raise ValueError(msg)

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

    training_dataset_path = os.path.join(exp_dir, filename)
    name, extension = os.path.splitext(filename)
    decoded_filename = f"{name}_decoded{extension}"
    training_dataset_decoded_path = os.path.join(exp_dir, decoded_filename)
    # file writing
    training_encoded_df.to_csv(training_dataset_path, index=False)
    training_df.to_csv(training_dataset_decoded_path, index=False)
