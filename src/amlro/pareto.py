from typing import List

import numpy as np


def is_pareto_dominant(point1: List, point2: List, directions: List) -> bool:
    """Determines whether one point dominates another in a multi-objective space
    according to Pareto dominance.

    Pareto dominance is a concept used in multi-objective optimization to compare
    two points (or solutions) based on multiple objectives. A point `A` is said to
    Pareto-dominate another point `B` if `A` is no worse than `B` in all objectives
    and better than `B` in at least one objective.

    This function checks whether `point1` Pareto-dominates `point2` considering
    the specified optimization directions for each objective.

    :param point1: List of objective values for the first point
    :type point1: List
    :param point2: List of objective values for the second point
    :type point2: List
    :param directions: Optimization direction for each objective. Each entry should be
                     "min" for minimization or "max" for maximization.
    :type directions: List
    :return: True if `point1` Pareto-dominates `point2`, otherwise False.
    :rtype: bool
    """

    all_dominate = True
    for p1, p2, direction in zip(point1, point2, directions):
        if direction == "max" and p1 < p2:
            all_dominate = False
            break
        elif direction == "min" and p1 > p2:
            all_dominate = False
            break
    # check points are not identical in all the objectives
    identical = any(p1 != p2 for p1, p2 in zip(point1, point2))
    return all_dominate and identical


def identify_pareto_front(
    prediction_data: List, directions: List, nfeatures: int
) -> np.array:
    """Identifies the Pareto front from a list of points in a multi-objective space.

    The Pareto front is a set of non-dominated points in a multi-objective optimization
    problem. A point is considered to be on the Pareto front if no other point in
    the set dominates it. This function examines each point and determines whether it
    should be included in the Pareto front.

    :param prediction_data: A list  of points/predictions, where each point contains
        both feature values and objective values. The objective values should follow
        the features in each point.
    :type prediction_data: List
    :param directions: Optimization direction for each objective. Each entry should be
             "min" for minimization or "max" for maximization.
    :type directions: List
    :param nfeatures: Length of the feature space
    :type nfeatures: int
    :return: List of pareto solutions
    :rtype: np.array
    """

    pareto_front = []

    for i, first_point in enumerate(prediction_data):
        # checking each point dominated by other points in the prediction dataset
        # only non doiminated solutions by other points become pareto solution
        if not any(
            is_pareto_dominant(
                other_point[nfeatures:], first_point[nfeatures:], directions
            )
            for j, other_point in enumerate(prediction_data)
            if i != j
        ):
            pareto_front.append(first_point)
    return np.array(pareto_front)
