.. _reaction_space:

Reaction Space Generation
=========================

This section describes how AMLRO constructs the **reaction search space** and
selects the **initial training conditions** using the
``get_reaction_scope`` entry-point function.

This is the **first step** in any AMLRO workflow.

Overview
--------

Reaction space generation serves two purposes:

1. Construct the **full combinatorial reaction space** based on user-defined
   continuous and categorical parameters.
2. Select an **initial subset of reactions** for training using a chosen
   sampling strategy.

The reaction scope is generated using:

.. code-block:: python

   get_reaction_scope(
       config=config,
       sampling="sobol",
       training_size=10,
       write_files=True,
       exp_dir=exp_dir
   )

This function generates all reaction combinations and writes them to ``full_combo.csv``
for  use as reaction grid for active learning optimization.

Function Arguments
------------------

Configuration Dictionary
~~~~~~~~~~~~~~~~~~~~~~~~

``config`` defines the **reaction parameters and objectives** and is described
in detail in :doc:`configuration`.

Only reaction variables and vales are read from ``config`` at this stage.

Sampling Strategy
~~~~~~~~~~~~~~~~~

The ``sampling`` argument controls how the **initial training reactions**
are selected from the full reaction space.

Supported options include:

- ``"lhs"`` – Latin Hypercube Sampling
- ``"sobol"`` – Sobol low-discrepancy sequence
- ``"random"`` – Random sampling

The sampling method affects **only the initial training set** and does not
alter the full reaction space.

Implementation details:

- Latin Hypercube Sampling (LHS) is implemented using **PyDOE2**
  with a min–max criterion, 1000 iterations, and a fixed random seed
  for reproducibility.
- Sobol sampling is implemented using ``scipy.qmc``.
- Random sampling uses uniform random selection.

Training Size
~~~~~~~~~~~~~

``training_size`` specifies the number of reaction conditions selected
for the **initial training set**.

This value should reflect experimental or computational budget constraints.
Typical values range from 5 to 50 reactions, depending on problem complexity.
Donot go lesser than 5.

Experiment Directory
~~~~~~~~~~~~~~~~~~~~

``exp_dir`` defines the directory where all generated reaction space
and training files are written.

If the directory does not exist, it will be created automatically.
This directory should use with full optimization cycle. Specially if you manually
adding ``reaction_data.csv`` includes here.

Example:

.. code-block:: python

   exp_dir = "exp_data"

File Generation
---------------

When ``write_files=True``, the reaction space generation step produces
the following files inside ``exp_dir``:

- ``full_combo.csv``
  Encoded representation of the complete reaction space.
  Categorical variables are stored as integer indices.

- ``full_combo_decoded.csv``
  Human-readable version of the full reaction space with original
  categorical values restored. (This file is generated if only reaction space <= 20000)

- ``training_combo.csv``
  Initial subset of reaction conditions selected using the specified
  sampling strategy.
  These reactions are intended to be performed experimentally or
  evaluated via simulation.

These files allow users to **inspect, visualize, or modify**
reaction conditions prior to experimentation.

Inspecting the Reaction Space
-----------------------------

Users may inspect the generated reaction combinations using standard
data analysis tools.

Example:

.. code-block:: python

   import pandas as pd

   df = pd.read_csv("exp_data/training_combo.csv")
   print(df.head())

This is particularly useful for verifying parameter ranges,
sampling behavior, and categorical encoding.

Relationship to the AMLRO Workflow
----------------------------------

Reaction space generation is a **one-time initialization step**.
Once completed, users proceed to:

1. Perform experiments or simulations for conditions listed in
   ``training_combo.csv``
2. Invoke active learning to propose new reaction conditions

For details on how experimental feedback is incorporated, see
:doc:`training_data.rst`.

For details on batch selection and optimization, see
:doc:`active_learning.rst`.
