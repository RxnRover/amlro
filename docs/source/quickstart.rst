.. _quickstart:
Quickstart
==========

This guide demonstrates a minimal AMLRO workflow for reaction optimization.
It mirrors the three core steps of AMLRO and is intended to introduce the
framework with the least possible setup.
Each step has a dedicated function that can be called by the user:

1. **Reaction Space Generation** → ``get_reaction_scope()``
2. **Training Set Generation** → ``generate_training_data()``
3. **Active Learning Prediction** → ``get_optimized_parameters()``

AMLRO is designed for *iterative* optimization workflows. Between optimization
steps, experimental or computational feedback must be provided by the user.

Step 0: Import Packages
----------------------

.. code-block:: python

   from amlro.generate_reaction_conditions import get_reaction_scope
   from amlro.generate_training_data import generate_training_data
   from amlro.optimizer import get_optimized_parameters
   import pandas as pd

Step 1: Define the Configuration Dictionary
-------------------------------------------

AMLRO is configured using a single dictionary that defines reaction parameters,
objectives, sampling strategy, and model settings.

.. code-block:: python

   config = {
       "continuous": {
           "feature_names": ["temperature", "time"],
           "bounds": [[20, 80], [1, 10]],
           "step_sizes": [5, 1]
       },
       "categorical": {
           "solvent": ["THF", "MeCN", "DMF"]
       },
       "objectives": {
           "yield": "max"
       },
       "model": "random_forest"
   }

An experiment directory is used to store all intermediate files:

.. code-block:: python

   exp_dir = "exp_data"

Step 2: Generate Reaction Space and Initial Training Conditions
---------------------------------------------------------------

.. code-block:: python

   get_reaction_scope(
       config=config,
       sampling="lhs",
       training_size=10,
       write_files=True,
       exp_dir=exp_dir
   )

This step generates the following files:

- ``full_combo.csv``: encoded full reaction space
- ``full_combo_decoded.csv``: human-readable reaction space
- ``training_combo.csv``: initial reaction conditions to evaluate

To inspect the initial training conditions:

.. code-block:: python

   df = pd.read_csv(f"{exp_dir}/training_combo.csv")
   print(df)

Step 3: Generate the Training Dataset
------------------------------------

After performing experiments or simulations for the reaction conditions listed
in ``training_combo.csv``, objective values must be provided before proceeding.

Once objective values are available, generate or update the training dataset:

.. code-block:: python

   generate_training_data(
       exp_dir=exp_dir,
       config=config,
       filename="reactions_data.csv"
   )

This step creates empty files (if not already present) or updates the following
datasets:

- ``reactions_data.csv``: encoded dataset used for machine learning model training
- ``reactions_data_decoded.csv``: human-readable version of the training dataset

This function is designed to be called **iteratively**, where reaction conditions
are evaluated round-by-round and experimental feedback is incorporated after
each iteration.

**Providing Experimental or Computational Feedback:

AMLRO supports both manual and programmatic ways of supplying objective values.

**Option A: Manual update (recommended for open loop experimental workflows)**

If experimental results are obtained outside of AMLRO, users may manually
create or update ``reactions_data.csv`` before running the active learning step.

To do so:

- Open ``reactions_data.csv``
- Copy initial reaction conditions from ``training_combo.csv``
- Add columns corresponding to the objective names defined in ``config["objectives"]``
- Ensure no additional columns are present
- For categorical features, use encoded values (``0, 1, 2, ...``) following the
  order specified in the configuration dictionary.
- Also create a ``reactions_data_decoded.csv`` file with actual categorical values.

This option is also suitable when users already possess prior experimental data.

.. important::

   When updating training or reaction data files manually, they must contain
   **only** the following columns:

   - Feature columns defined in ``config``
   - Objective columns defined in ``config["objectives"]``

**Option B: Programmatic update (simulations or benchmarks)**

When objective values can be computed programmatically (e.g., benchmark
functions or simulations), users may iteratively update the dataset using a
``for`` loop by calling ``generate_training_data()`` function.

In this case, objective values are computed and appended automatically,
provided that column names exactly match those defined in the configuration.

This approach is demonstrated in the Branin tutorial and Google Colab example.

Step 4: Predict Next Optimal Reaction Conditions
------------------------------------------------

.. code-block:: python

   next_conditions = get_optimized_parameters(
       exp_dir=exp_dir,
       config=config,
       batch_size=5
       filename='reactions_data.csv'
   )

This completes one AMLRO iteration. The workflow can be repeated by adding new
experimental feedback and re-running Steps 4 and 5.

Next Steps
----------

AMLRO also supports fully automated closed-loop optimization when objective
values can be computed programmatically.

For an interactive example using the Branin benchmark function, see:

- :doc:`tutorials/branin_example`
- Google Colab notebook (linked in the tutorial)

A web-based interface for interactive use is under development.
