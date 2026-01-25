.. _active_learning:

Active Learning Optimization
============================

Once an initial training dataset is available, AMLRO can be used to
**predict optimal reaction conditions** and iteratively refine them
through active learning.

This step closes the optimization loop by combining:
- Machine learning model training
- Prediction over the full reaction space
- Selection of the next batch of experiments

Overview
--------

The active learning stage assumes that:

- ``reactions_data.csv`` already exists
- filename can be defined by user and default is ``reactions_data.csv``
- Feature and objective columns match the configuration
- At least one round of experimental or computational feedback
  has been collected

From this point onward, AMLRO repeatedly:
1. Trains an ML model on available data
2. Predicts objective values for all candidate reactions
3. Selects the best next batch of conditions
4. Waits for user feedback (open-loop) or computes objectives (closed-loop)

Entry-Point Function
--------------------

The active learning process is accessed through a single entry-point:

.. code-block:: python

   get_optimized_parameters(
       exp_dir=exp_dir,
       config=config,
       parameters_list=parameters,
       objectives_list=objectives,
       batch_size=5,
       filename='reactions_data.csv',
       termination=False
   )

This function can be called **iteratively** to perform multi-round
optimization.

Generated Output
----------------

Each call to ``get_optimized_parameters``:

- Trains an ML model using ``reactions_data.csv``
- Predicts objective values for the full reaction space
- Selects the next batch of reaction conditions
- Appends results from the previous round (if provided)

Open-Loop Optimization
----------------------

In open-loop scenarios, experiments or simulations must be performed
externally between optimization cycles.

Workflow
~~~~~~~~

1. AMLRO suggests a batch of reaction conditions
2. User performs experiments or simulations
3. Objective values are collected
4. Results are written back to ``reactions_data.csv``
5. AMLRO is called again to suggest the next batch

This process continues until satisfactory performance is achieved.

Interactive Interfaces
~~~~~~~~~~~~~~~~~~~~~~

To avoid manual file editing, users are encouraged to use:

- intercative Google Colab notebook
- Programmatic CSV updates
- Future web interfaces or lab automation tools

AMLRO does not require a specific interface, only correctly formatted data.

Closed-Loop Optimization (Autonomous)
-------------------------------------

For benchmarks, simulations, or automated pipelines, AMLRO supports
fully closed-loop optimization with a fixed evaluation budget.

In this mode:
- Objective values are computed computationaly or feedback autonomus experiments setup
- Optimization proceeds without user intervention
- A predefined budget controls termination

Minimal Closed-Loop Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   parameters = []
   objectives = []

   budget = 10

   for i in range(budget):

       parameters = get_optimized_parameters(
           exp_dir=exp_dir,
           config=config,
           parameters_list=parameters,
           objectives_list=objectives,
           batch_size=2
       )

       #should be handle correctly with batch size and MO criteria [[],[]]
       objectives = objective_function(parameters)


   get_optimized_parameters(
       exp_dir=exp_dir,
       config=config,
       parameters_list=parameters,
       objectives_list=objectives,
       batch_size=1,
       termination=True
   )

Explanation
~~~~~~~~~~~

- Each iteration selects a new batch of reaction conditions
- Objective values are collected automatically
- Results are appended to the training dataset
- The final call with ``termination=True`` ensures clean shutdown
  and final data persistence

This setup is suitable for:
- Analytical test functions (e.g., Branin)
- High-throughput experiments
- Algorithm benchmarking

Batch Size and Budget
---------------------

- ``batch_size`` controls how many reaction conditions are suggested per iteration
- The optimization budget is controlled externally (e.g., loop length)
- AMLRO does not enforce a stopping criterion by default

Users may stop optimization based on:
- Objective convergence
- Experimental constraints
- Resource limits

Relationship to the AMLRO Workflow
----------------------------------

Active learning is the final stage of AMLRO and depends on:

- Reaction space generation
- Training data generation
- User-provided objective feedback

Together, these three stages form a complete
**iterative reaction optimization framework**.
