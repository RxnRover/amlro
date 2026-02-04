..
    These are examples of badges you might want to add to your README:
    please update the URLs accordingly

     .. image:: https://api.cirrus-ci.com/github/<USER>/amlro.svg?branch=main
         :alt: Built Status
         :target: https://cirrus-ci.com/github/<USER>/amlro
     .. image:: https://readthedocs.org/projects/amlro/badge/?version=latest
         :alt: ReadTheDocs
         :target: https://amlro.readthedocs.io/en/stable/
     .. image:: https://img.shields.io/coveralls/github/<USER>/amlro/main.svg
         :alt: Coveralls
         :target: https://coveralls.io/r/<USER>/amlro
     .. image:: https://img.shields.io/pypi/v/amlro.svg
         :alt: PyPI-Server
         :target: https://pypi.org/project/amlro/
     .. image:: https://img.shields.io/conda/vn/conda-forge/amlro.svg
         :alt: Conda-Forge
         :target: https://anaconda.org/conda-forge/amlro
     .. image:: https://pepy.tech/badge/amlro/month
         :alt: Monthly Downloads
         :target: https://pepy.tech/project/amlro
     .. image:: https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter
         :alt: Twitter
         :target: https://twitter.com/amlro

.. image:: https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold
    :alt: Project generated with PyScaffold
    :target: https://pyscaffold.org/

|

AMLRO
=====

This is the documentation of **amlro**.

Active Machine Learning Reaction Optimizer (AMLRO) for data-efficient reaction
process condition discovery and optimization.

.. image:: /docs/source/_static/AMLRO.jpg
    :width: 60%
    :align: center
    :alt: AMLRO overview

AMLRO is an open-source framework designed to accelerate chemical reaction
optimization using active learning with classical machine learning regression
models. AMLRO integrates space-filling sampling strategies (e.g., Sobol and
Latin Hypercube sampling) with iterative model training, prediction, and
experiment selection to efficiently navigate complex reaction spaces. The
platform supports multiple regression models, flexible multi-objective
definitions, and user-defined parameter bounds, enabling data-efficient
optimization from small initial datasets. AMLRO is designed for ease of use by
experimentalists and can operate as a standalone decision-support tool or be
integrated into closed-loop automated experimentation workflows.

AMLRO follows a three-step workflow:

1. Reaction space generation
2. Training set generation with experimental feedback
3. Active learning-loop -> prediction of optimal reaction conditions.

Click the badge below to open AMLRO Interactive notebook in **Google Colab**:

.. image:: https://colab.research.google.com/assets/colab-badge.svg
    :alt: open in colab
    :target: https://colab.research.google.com/github/RxnRover/amlro/blob/main/notebooks/AMLRO_interactive_colab.ipynb

.. _pyscaffold-notes:

Making Changes & Contributing
-----------------------------

This project uses pre-commit_, please make sure to install it before making any
changes:

::

    pip install pre-commit
    cd amlro
    pre-commit install

It is a good idea to update the hooks to the latest version:

::

    pre-commit autoupdate

Don't forget to tell your contributors to also install and use pre-commit.

.. _pre-commit: https://pre-commit.com/

Note
----

This project has been set up using PyScaffold 4.5. For details and usage
information on PyScaffold see https://pyscaffold.org/.
