.. _configurations:

Configuration
=============

This section describes the **AMLRO configuration dictionary**, which defines the
reaction search space, objectives.
All AMLRO workflows rely on this configuration as the **single source of truth**
for reaction space generation, training data handling, and active learning.

AMLRO is designed so that users **do not need to modify optimization logic,
models, or acquisition strategies**. Instead, all behavior is controlled through
this configuration dictionary and three high-level entry-point functions.

Overview
--------

The configuration dictionary specifies:

- Continuous reaction parameters (bounds, resolution, names)
- Categorical reaction parameters (explicit value lists)
- Optimization objectives and directions

A minimal configuration dictionary **must contain all required keys**, even if
some sections (e.g., categorical variables) are empty.

Minimal Configuration Template
------------------------------

The following example shows the **required structure only**.
Values are placeholders and should be replaced by user-defined settings.

.. code-block:: python

   config = {
       "continuous": {
           "bounds": [
               [min_value, max_value],
               ...
           ],
           "resolutions": [
               step_size,
               ...
           ],
           "feature_names": [
               "feature_1",
               "feature_2",
               ...
           ],
       },
       "categorical": {
           "feature_names": [],
           "values": []
       },
       "objectives": [
           "objective_1",
           "objective_2",
           ...
       ],
       "directions": [
           "max",
           "min",
           ...
       ],
   }

.. important::

   Even if no categorical variables are used, the ``categorical`` block
   **must still be present** with empty lists.

Configuration Sections
----------------------

Continuous Parameters
~~~~~~~~~~~~~~~~~~~~~

The ``continuous`` block defines numerical reaction parameters such as
temperature, time, concentration, voltage, or flow rate.

Required keys:

- ``bounds``
  A list of ``[min, max]`` pairs defining the allowed range for each parameter.

- ``resolutions``
  Step sizes used to discretize each continuous parameter.

- ``feature_names``
  Human-readable names for each parameter. These names become column headers
  in all generated CSV files.

Discretization controls the **granularity of the reaction grid**, balancing
combinatorial explosion against practical experimental resolution.

Categorical Parameters
~~~~~~~~~~~~~~~~~~~~~~

The ``categorical`` block defines discrete reaction parameters such as solvent,
catalyst, base, or ligand identity.

Required keys:

- ``feature_names``
  Names of categorical parameters.

- ``values``
  A list of value lists corresponding to each categorical parameter.

Example structure (format only):

.. code-block:: python

   "categorical": {
       "feature_names": ["category_1", "category_2"],
       "values": [
           ["value_a", "value_b"],
           ["value_x", "value_y"]
       ]
   }

If categorical parameters are used, AMLRO internally encodes them as integers
(``0, 1, 2, ...``) according to their order in ``values``.

Objectives and Directions
~~~~~~~~~~~~~~~~~~~~~~~~~

The ``objectives`` list defines the names of experimental or simulated outcomes
to be optimized.

The ``directions`` list specifies whether each objective should be minimized or
maximized.

.. important::

   - ``objectives`` and ``directions`` **must have the same length**
   - Allowed direction values are ``"min"`` or ``"max"``

Objective names become column headers in all training and reaction data files.

Relationship to the AMLRO Workflow
----------------------------------

This configuration dictionary is used by all three AMLRO entry-point functions:

1. ``get_reaction_scope``
   Generates the full reaction space and initial training conditions.

2. ``generate_training_data``
   Manages experimental or simulated feedback and training dataset construction.

3. ``get_optimized_parameters``
   Performs active learning–based prediction and batch selection.

Users interact **only** with these functions and the configuration dictionary;
all optimization logic, multi-objective handling, and model training are managed
internally by AMLRO.
