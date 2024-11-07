import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.multioutput import MultiOutputRegressor

from amlro.const import FULL_COMBO_FILENAME, REACTION_DATA_FILENAME
from amlro.ml_models import get_regressor_model
from amlro.pareto import identify_pareto_front
from amlro.validations import validate_optimizer_config


def load_data(
    reactions_data: str, combination_file: str, config: Dict
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loading dataset files

    Reads the training set file and all combination file as pandas data
    frames and split into x train , y train and test datasets. When loading
    the combination file, data rows will be deleted if they are included
    in training file.

    :param training_file: path to the training set file.
    :type training_file: str
    :param combination_file: path to the combination file.
    :type combination_file: str
    :param config: Dictionary of optimizer parameters.
    :type config: Dict
    :return: x and y training datasets and test dataset.
    :rtype: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
    """

    train = pd.read_csv(reactions_data)

    target_columns = config["objectives"]
    y_train = train[target_columns]

    x_train = train.drop(target_columns, axis=1)

    data = pd.read_csv(combination_file)

    data = data.drop_duplicates()

    data = data.merge(
        x_train, on=x_train.columns.to_list(), how="left", indicator=True
    )
    data = data[data["_merge"] == "left_only"].drop(columns="_merge")
    data = data.reset_index(drop=True)

    return x_train, y_train, data


def model_training(
    x_train: pd.DataFrame, y_train: pd.DataFrame, model: str = "gb"
) -> object:
    """Train the regressor model and return the best model.

    :param x_train: training dataset that contains feature values.
    :type x_train: pd.Dataframe
    :param y_train: target dataframe that contains objective values.
    :type y_train: pd.Dataframe
    :param model: Regressor model name.
        Valid options: 'ela_net', 'dtree', 'rf', 'gb', 'xgb', 'aboost', 'svr', 'knn',
        'bayesian_ridge'. Default is 'gb' (Gradient Boosting).
    :type model: str
    :return: trained regressor model
    :rtype: model object with a `predict` method (e.g., sklearn regressor)
    """

    # instead shufflesplit we can use  KFold(n_splits=5, shuffle=True)
    # or RepeatedKFold(n_splits=5, n_repeats=3).
    y_train = y_train.values.ravel()
    SEED = 42
    kfold = ShuffleSplit(n_splits=10, test_size=0.2, random_state=SEED)

    regr, param_grid = get_regressor_model(model)

    if model != "svr" and model != "knn" and model != "bayesian_ridge":
        regr.set_params(random_state=SEED)

    grid = GridSearchCV(
        estimator=regr, param_grid=param_grid, cv=kfold, n_jobs=6
    )

    grid.fit(x_train, y_train)
    pd.DataFrame([grid.best_params_], columns=grid.best_params_.keys())

    regr = grid.best_estimator_

    return regr


def mo_model_training(
    x_train: pd.DataFrame, y_train: pd.DataFrame, model: str = "gb"
) -> object:
    """Train the multi output regressor model and return the best model.

    :param x_train: training dataset that contains feature values.
    :type x_train: pd.Dataframe
    :param y_train: target dataframe that contains objective values.
    :type y_train: pd.Dataframe
    :param model: Regressor model name.
        Valid options: 'ela_net', 'dtree', 'rf', 'gb', 'xgb', 'aboost', 'svr', 'knn',
        'bayesian_ridge'. Default is 'gb' (Gradient Boosting).
    :type model: str
    :return: trained regressor model
    :rtype:  model object with a `predict` method (e.g., sklearn regressor)
    """

    kfold = ShuffleSplit(n_splits=10, test_size=0.2)

    # base regressor model
    regr, param_grid = get_regressor_model(model)

    # multi objective regressor model
    mo_regr = MultiOutputRegressor(regr)

    param_grid = {"n_jobs": [2, 4, 10]}

    grid = GridSearchCV(
        estimator=mo_regr, param_grid=param_grid, cv=kfold, n_jobs=6
    )

    grid.fit(x_train, y_train)
    pd.DataFrame([grid.best_params_], columns=grid.best_params_.keys())

    mo_regr = grid.best_estimator_

    return mo_regr


def predict_next_parameters(
    regr, data: pd.DataFrame, config: Dict, batch_size: int = 1
) -> pd.DataFrame:
    """Predicts the yield from all the combination data using a trained
    regressor model and return the best combinations.

    The function handles both single-objective and multi-objective optimization.
    In the single-objective case, it sorts the predicted results based on the
    specified direction (maximization or minimization).
    In the multi-objective case, it identifies the Pareto front and ranks solutions
    by computing weighted sums of normalized objectives. Weights are defined as
    -1 for min and +1 for max.

    :param regr: trained regressor model
    :type regr: model object with a `predict` method (e.g., sklearn regressor)
    :param data: test dataset that contains full reaction space
    :type data: pd.Dataframe
    :param config: Dictionary of optimizer parameters
    :type config: Dict
    :param batch_size: Number of reactions conditions need as predictions
        defaults to 1.
    :type batch_size: int, optional
    :return: batch of best predicted parameter
    :rtype: pd.Dataframe
    """

    pred = pd.DataFrame(regr.predict(data), columns=config["objectives"])

    prediction_df = pd.concat([data, pred], axis=1)

    if len(config["directions"]) == 1:
        if config["directions"][0] == "max":
            best_parameters = prediction_df.sort_values(
                by=[config["objectives"][0]], ascending=False
            ).iloc[:batch_size]
        elif config["directions"][0] == "min":
            best_parameters = prediction_df.sort_values(
                by=[config["objectives"][0]], ascending=True
            ).iloc[:batch_size]

        best_parameters = best_parameters.drop(config["objectives"][0], axis=1)

    elif len(config["directions"]) > 1:
        nfeatures = len(data.columns)

        weights = np.array(
            [-1 if dir == "min" else 1 for dir in config["directions"]]
        )

        # Identify Pareto front
        pareto_front = identify_pareto_front(
            prediction_df.values, config["directions"], nfeatures
        )

        # finding the weighted sum of objective values for pareto solutions
        normalized_front = (pareto_front - pareto_front.min()) / (
            pareto_front.max() - pareto_front.min()
        )
        weighted_sums = np.sum(
            normalized_front[:, nfeatures:] * weights, axis=1
        )

        num_solutions = min(len(pareto_front), batch_size)

        # Get the indices of the top `num_solutions` weighted sums
        best_indexs = np.argsort(weighted_sums)[-num_solutions:][::-1]
        best_solutions = pareto_front[best_indexs]
        best_parameters = best_solutions[:, :nfeatures]

        best_parameters = pd.DataFrame(best_parameters, columns=data.columns)

    return best_parameters


def get_optimized_parameters(
    exp_dir: str,
    config: Dict,
    parameters_list: List[List] = [[]],
    objectives_list: List[List] = [[]],
    model: str = "gb",
    filename: str = REACTION_DATA_FILENAME,
    batch_size: int = 1,
    termination: bool = False,
) -> List[List]:
    """
    Trains a machine learning model using the provided training data and
    configuration, then predicts and retrieves the next best reaction
    parameters based on the optimization objectives.
    Then, writes the training data and predictions to files.

    :param exp_dir: experimental directory for saving data files
    :type exp_dir: str
    :param config: Configuration dictionary specifying features, objectives,
        directions, and other settings for optimizer.
    :type config: Dict
    :param parameters_list: List of reaction conditions for training,
        defaults to [[]].
    :type parameters_list: List[List], optional
    :param objectives_list: List of objective values corresponding to the reaction
        conditions, defaults to [[]].
    :type objectives_list: List[List], optional
    :param model: The machine learning model type to use, e.g., 'gb' for Gradient
        Boosting, defaults to 'gb'.
    :type model: str, optional
    :param filename: The name of the file where reaction data is stored,
        defaults to REACTION_DATA_FILENAME.
    :type filename: str, optional
    :param batch_size: The number of best reaction conditions to return,
        defaults to 1.
    :type batch_size: int, optional
    :param termination: If True, the function will terminate early after saving
        data, defaults to False.
    :type termination: bool, optional
    :return: A list of lists containing the next predicted reaction conditions.
    :rtype: List[List]
    """

    validate_optimizer_config(config)
    # Encode and decode the provided parameters and objectives and
    # make a str for writes.
    parameters_encoded, parameters_decoded = stringify_parameters_objectives(
        parameters_list, objectives_list, config
    )

    # File paths for saving training data and decoded data
    reaction_data_path = os.path.join(exp_dir, filename)
    name, extension = os.path.splitext(filename)
    decoded_filename = f"{name}_decoded{extension}"
    reaction_data_decoded_path = os.path.join(exp_dir, decoded_filename)
    full_combo_path = os.path.join(exp_dir, FULL_COMBO_FILENAME)

    # Write encoded and decoded parameter-objective combinations to file
    if len(parameters_list) != 0:
        write_data_to_training(reaction_data_path, parameters_encoded)
        write_data_to_training(reaction_data_decoded_path, parameters_decoded)
        print("writing data to training dataset files...")

    # Exit early if termination flag is set
    if termination:
        return None

    # Load training data and full reaction space
    x_train, y_train, data = load_data(
        reaction_data_path, full_combo_path, config
    )

    print("Data Loading for Machine Learning Model...")
    print("Training ML model " + model + " ...")

    # Train the correct ml model for single or multi objective optimization.
    if len(config["objectives"]) == 1:
        regr = model_training(x_train, y_train, model)
    else:
        regr = mo_model_training(x_train, y_train, model)

    # Predict the next best reaction conditions.
    best_combo = predict_next_parameters(regr, data, config, batch_size)

    # Decode the best reaction conditions.
    next_best_conditions = []

    for _, conditions in best_combo.iterrows():
        conditions_list = conditions.tolist()
        next_best_conditions.append(
            categorical_feature_decoding(config, conditions_list)
        )

    print("Best parameter combination...", next_best_conditions)

    return next_best_conditions


def stringify_parameters_objectives(
    parameters_list: List[List], objectives_list: List[List], config: Dict
) -> Tuple[List[str], List[str]]:
    """Combine reaction parameters and respective objectives values
    as a comma sepearated string for writing into reaction data file.

    Convert parameters and their respective objective values into
    comma-separated strings, both encoded and decoded, for writing into a
    reaction data file. The function encodes categorical features based on
    the provided configuration.

    :param parameters_list: A list of reaction conditions, where each condition
        is a list of parameter values.
    :type parameters_list: List[List]
    :param objectives_list: A list of objective values where list of corrosponding
        objective values for each reaction condition.
    :type objectives_list: List[List]
    :param config: Dictionary of optimizer parameters
    :type config: Dict
    :return: Two lists of strings: (1) the parameters with encoded categorical
        features combined with objective values, and (2) the original
        (decoded) parameter values combined with objective values.
    :rtype: Tuple[List[str], List[str]]
    """

    parameters_encoded = []
    parameters_decoded = []

    for parameters, objectives in zip(parameters_list, objectives_list):
        encoded = categorical_feature_encoding(config, parameters)
        parameters_encoded.append(
            ",".join([str(elem) for elem in encoded])
            + ","
            + ",".join([str(obj_val) for obj_val in objectives])
        )
        parameters_decoded.append(
            ",".join([str(elem) for elem in parameters])
            + ","
            + ",".join([str(obj_val) for obj_val in objectives])
        )

    return parameters_encoded, parameters_decoded


def categorical_feature_decoding(
    config: Dict, best_combo: List[Any]
) -> List[Any]:
    """This method converts encoded parameter list into decoded list.
    Convert categorical feature values back into its names.

    :param config: Initial reaction feature configurations
    :type config: Dict
    :param best_combo: parameter list required for decoding
    :type best_combo: List[Any]
    :return: Decoded parameter list
    :rtype: List[Any]
    """

    numerical_feature_count = len(config["continuous"]["feature_names"])
    numerical_combo = best_combo[0:numerical_feature_count]
    cat_combo = best_combo[numerical_feature_count:]

    for i in range(len(cat_combo)):
        x = config["categorical"]["values"][i]
        print(cat_combo[i])
        cat_combo[i] = x[int(cat_combo[i])]

    best_combo_with_names = []
    [best_combo_with_names.append(elem) for elem in numerical_combo]
    [best_combo_with_names.append(elem) for elem in cat_combo]

    return best_combo_with_names


def categorical_feature_encoding(
    config: Dict, prev_parameters: List[Any]
) -> List[Any]:
    """This method converts decoded parameter list into encoded list.
    Convert categorical feature values into its numerical values.

    :param config: Initial reaction feature configurations
    :type config: Dict
    :param prev_parameters: parameter list required for encoding
    :type prev_parameters: List[Any]
    :return: encoded parameter list
    :rtype: List[Any]
    """

    numerical_feature_count = len(config["continuous"]["feature_names"])
    numerical_combo = prev_parameters[0:numerical_feature_count]
    cat_combo = prev_parameters[numerical_feature_count:]

    for i in range(len(cat_combo)):
        cat_list = config["categorical"]["values"][i]

        for x in range(len(cat_list)):
            if cat_list[x] == cat_combo[i]:
                cat_combo[i] = int(x)
    prev_parameters_encode = []
    [prev_parameters_encode.append(elem) for elem in numerical_combo]
    [prev_parameters_encode.append(elem) for elem in cat_combo]

    return np.array(prev_parameters_encode)


def write_data_to_training(
    training_file: str, parameter_list: List[str]
) -> None:
    """writing the prev best predicted combination and
    experimental yield at the end of the training set file.

    :param training_file: traning set file path
    :type training_file: str
    :param prev_parameters: previous best combo and yield
    :type prev_parameters: List[str]
    """

    # Open the file in append & read mode ('a+')
    for parameters in parameter_list:
        with open(training_file, "a+") as file_object:
            file_object.write(parameters + "\n")
