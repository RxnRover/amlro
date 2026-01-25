.. _training_data:

Training Data Generation
=======================

This section describes how AMLRO incorporates **experimental or computational
feedback** to build the training dataset used for active learning.

This step bridges **reaction space definition** and **model-driven optimization**.

Overview
--------

After generating the initial reaction conditions (``training_combo.csv``),
objective values must be provided before AMLRO can train a model.

The training data generation step:

- Collects reaction conditions and objective values
- Builds a cumulative training dataset
- Supports both **open-loop** (manual/inetractive notebook) and **closed-loop** (automated) workflows

This functionality is accessed through the ``generate_training_data`` entry-point
function.

Entry-Point Function
--------------------

.. code-block:: python

   generate_training_data(
       exp_dir=exp_dir,
       config=config,
       parameters=parameters,
       obj_values=objectives,
       filename='reactions_data.csv',
       termination=False
   )

This function is designed to be called **iteratively**, once objective values
become available.

Generated Files
---------------

During training data generation, AMLRO creates or updates the following files
inside ``exp_dir``, filename can be defined by user and default is ``reactions_data.csv``:

- ``reactions_data.csv``
  Encoded dataset used for machine learning model training

- ``reactions_data_decoded.csv``
  Human-readable version of the training dataset

These files grow incrementally as new experimental or computational results
are added.

Open-Loop Workflow (Manual Update)
----------------------------------

This option is recommended for **experimental workflows** or expensive
simulations where objective values are not available programmatically.

Workflow
~~~~~~~~

1. Perform experiments or simulations for the conditions listed in
   ``training_combo.csv``
2. Manually create or update ``reactions_data.csv``
3. Proceed to active learning once sufficient data is available

File Format Requirements
~~~~~~~~~~~~~~~~~~~~~~~~

When creating or editing ``reactions_data.csv`` manually:

- Include **only**:
  - Feature columns defined in ``config["continuous"]["feature_names"]``
  - Feature columns defined in ``config["categorical"]["feature_names"]``
  - Objective columns defined in ``config["objectives"]``
- Do **not** include additional columns
- Categorical variables must be encoded as integer indices
  corresponding to their order in ``config["categorical"]["values"]``
- Also create a ``reactions_data_decoded.csv`` file including reaction data with actual categorical values.

.. important::

   The column names and ordering must match the configuration exactly.
   AMLRO does not perform automatic column reconciliation.

Open-loop workflows allow AMLRO to be used with laboratory notebooks,
external data acquisition systems, or third-party simulation pipelines.

Interactive Open-Loop Workflows
-------------------------------

AMLRO is designed to support **interactive optimization workflows**
without requiring manual editing of CSV files.

This is achieved by separating:

- The **AMLRO backend** (reaction space, training data, optimization)
- Uxsing **interactive frontends** - Interactive Google Colab notebook.

*Local web-based interface will be released near future.

Closed-Loop Workflow (Automated / Benchmarking)
-----------------------------------------------

For simulations, benchmarks, or algorithm development, AMLRO supports
a **fully automated closed-loop setup**.

In this mode, objective values are computed programmatically and fed
back into AMLRO in each iteration.

Minimal Closed-Loop Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   parameters = []
   objectives = []

   for i in range(training_size):

       parameters = generate_training_data(
           exp_dir=exp_dir,
           config=config,
           parameters=parameters,
           obj_values=objectives
       )

       objectives = objective_function(parameters)

   generate_training_data(
       exp_dir=exp_dir,
       config=config,
       parameters=parameters,
       obj_values=objectives,
       termination=True
   )

Explanation
~~~~~~~~~~~

- Each iteration retrieves the **next reaction condition**
- The user-defined ``objective_function`` evaluates the objectives
- Results are appended to ``reactions_data.csv``
- The final call with ``termination=True`` ensures that all remaining
  reaction conditions are written without requesting further feedback

This workflow enables:

- End-to-end autonomous optimization
- Synthetic benchmarks (e.g., Branin, analytical test functions)
- Integration with external automation systems

Relationship to the AMLRO Workflow
----------------------------------

Training data generation:

- Converts raw experimental results into a structured dataset
- Maintains full compatibility with manual workflows
- Acts as the **only required input** for active learning optimization

Once training data is available, users may proceed to:

- Batch selection and model training
- Prediction of optimal reaction conditions
