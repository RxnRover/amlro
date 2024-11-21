""" This module provides a framework for retrieving various regression models
    along with their corresponding hyperparameter grids for hyper parameter tuning.
    Regression Models:
    - ElasticNet
    - Decision Tree
    - Random Forest
    - Gradient Boosting
    - XGBoost
    - AdaBoost
    - Support Vector Regressor
    - K-Nearest Neighbors (KNN)
    - Bayesian Ridge

    Each function returns a regressor model object and a hyperparameter grid for use
    in hyperparameter optimization (e.g., using grid search
    in `optimizer.model_training`).

    The main function `get_regressor_model(model: str)` serves as the entry point
    to select a model based on user input, returning the chosen model and grid
    for training and tuning.

    Usage:
        model, param_grid = get_regressor_model('xgb')
"""

from typing import Dict, Tuple

import xgboost
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import BayesianRidge, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


def elastic_net_regressor() -> Tuple[ElasticNet, Dict]:
    """
    Creates an ElasticNet regressor with a grid of hyperparameters.

    :return: The ElasticNet model and hyperparameter grid.
    :rtype: Tuple[ElasticNet,Dict]
    """

    regr = ElasticNet()

    param_grid = {
        "alpha": [0.1, 0.5, 1.0],
        "l1_ratio": [0.2, 0.5, 0.8],
    }

    return regr, param_grid


def decision_tree_regressor() -> Tuple[DecisionTreeRegressor, Dict]:
    """
    Creates a DecisionTree regressor with a grid of hyperparameters.

    :return: The DecisionTreeRegressor model and hyperparameter grid.
    :rtype: Tuple[DecisionTreeRegressor, Dict]
    """

    regr = DecisionTreeRegressor()

    param_grid = {
        "max_depth": [None, 2, 4],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": [None, "sqrt", "log2"],
    }

    return regr, param_grid


def random_forest_regressor() -> Tuple[RandomForestRegressor, Dict]:
    """
    Creates a RandomForest regressor with a grid of hyperparameters.

    :return: The RandomForestRegressor model and hyperparameter grid.
    :rtype: Tuple[RandomForestRegressor, Dict]
    """

    regr = RandomForestRegressor()

    estimators_int = list(range(100, 500, 50))
    param_grid = {
        "n_estimators": estimators_int,
        "max_depth": [None, 2, 4],
        "max_features": ["auto", "sqrt", "log2"],
    }

    return regr, param_grid


def gradient_boost_regressor() -> Tuple[GradientBoostingRegressor, Dict]:
    """
    Creates a GradientBoosting regressor with a grid of hyperparameters.

    :return: The GradientBoostingRegressor model and hyperparameter grid.
    :rtype: Tuple[GradientBoostingRegressor, Dict]
    """

    regr = GradientBoostingRegressor()

    estimators_int = list(range(100, 500, 50))
    param_grid = {
        "n_estimators": estimators_int,
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [None, 2, 4],
    }

    return regr, param_grid


def xgboost_regressor() -> Tuple[xgboost.XGBRegressor, Dict]:
    """
    Creates an XGBoost regressor with a grid of hyperparameters.

    :return: The XGBoost model and hyperparameter grid.
    :rtype: Tuple[xgboost.XGBRegressor, Dict]
    """

    regr = xgboost.XGBRegressor()

    estimators_int = list(range(100, 500, 50))
    param_grid = {
        "n_estimators": estimators_int,
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [2, 3],
        "subsample": [0.8, 1.0],
    }

    return regr, param_grid


def adaboost_regressor() -> Tuple[AdaBoostRegressor, Dict]:
    """
    Creates an AdaBoost regressor with a grid of hyperparameters.

    :return: The AdaBoostRegressor model and hyperparameter grid.
    :rtype: Tuple[AdaBoostRegressor, Dict]
    """

    regr = AdaBoostRegressor()

    estimators_int = list(range(100, 500, 50))
    param_grid = {
        "n_estimators": estimators_int,
        "learning_rate": [0.01, 0.1, 0.2],
        "loss": ["linear", "square", "exponential"],
    }

    return regr, param_grid


def support_vector_regressor() -> Tuple[SVR, Dict]:
    """
    Creates a Support Vector Regressor (SVR) with a grid of hyperparameters.

    :return: The SVR model and hyperparameter grid.
    :rtype: Tuple[SVR, Dict]
    """

    regr = SVR()

    param_grid = {
        "kernel": ["linear", "poly", "rbf", "sigmoid"],
        "epsilon": [0.1, 0.01, 0.05],
    }

    return regr, param_grid


def knn_regressor() -> Tuple[KNeighborsRegressor, Dict]:
    """
    Creates a K-Nearest Neighbors (KNN) regressor with a grid of hyperparameters.

    :return: The KNeighborsRegressor model and hyperparameter grid.
    :rtype: Tuple[KNeighborsRegressor, Dict]
    """

    regr = KNeighborsRegressor()

    param_grid = {
        "n_neighbors": [3, 5, 7, 10],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan", "minkowski"],
    }

    return regr, param_grid


def bayesian_ridge_regressor() -> Tuple[BayesianRidge, Dict]:
    """
    Creates a BayesianRidge regressor with a grid of hyperparameters.

    :return: The BayesianRidge model and hyperparameter grid.
    :rtype: Tuple[BayesianRidge, Dict]
    """

    regr = BayesianRidge()

    param_grid = {
        "alpha_1": [1e-6, 1e-5, 1e-4],
        "alpha_2": [1e-6, 1e-5, 1e-4],
        "lambda_1": [1e-6, 1e-5, 1e-4],
        "lambda_2": [1e-6, 1e-5, 1e-4],
        "tol": [1e-4, 1e-3, 1e-2],
    }

    return regr, param_grid


def get_regressor_model(model: str, seed=None) -> Tuple[object, Dict]:
    """
    Retrieves the specified regressor model and its hyperparameter grid.

    :param model: The name of the model to retrieve. Supported values are:
        'ela_net', 'dtree', 'rf', 'gb', 'xgb', 'aboost', 'svr', 'knn',
        'bayesian_ridge'.
    :type model: str
    :return: The selected model and its hyperparameter grid.
    :rtype: Tuple[model object, Dict]
    """

    param_grid = {}
    model = model.lower()

    if model == "ela_net":
        regr, param_grid = elastic_net_regressor()
    elif model == "dtree":
        regr, param_grid = decision_tree_regressor()
    elif model == "rf":
        regr, param_grid = random_forest_regressor()
    elif model == "gb":
        regr, param_grid = gradient_boost_regressor()
    elif model == "xgb":
        regr, param_grid = xgboost_regressor()
    elif model == "aboost":
        regr, param_grid = adaboost_regressor()
    elif model == "svr":
        regr, param_grid = support_vector_regressor()
    elif model == "knn":
        regr, param_grid = knn_regressor()
    elif model == "bayesian_ridge":
        regr, param_grid = bayesian_ridge_regressor()
    else:
        raise ValueError("Invalid regressor name given: {}".format(model))

    if model != "svr" and model != "knn" and model != "bayesian_ridge":
        print("Setting random state " + str(seed) + " for ML model")
        regr.set_params(random_state=seed)

    return regr, param_grid
