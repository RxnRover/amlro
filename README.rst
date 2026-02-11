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

.. image:: https://img.shields.io/badge/DOI-10.11578/dc.20260205.1-blue
   :target: https://doi.org/10.11578/dc.20260205.1

.. image:: https://readthedocs.org/projects/amlro/badge/?version=latest
   :target: https://rxnrover.github.io/amlro/
   :alt: Documentation Status

AMLRO
=====

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

For tutorials and interactive notebooks, see the documentation.
📘 `Documentation <https://rxnrover.github.io/amlro/>`_

Click the badge below to open AMLRO Interactive notebook in **Google Colab**:

.. image:: https://colab.research.google.com/assets/colab-badge.svg
    :alt: open in colab
    :target: https://colab.research.google.com/github/RxnRover/amlro/blob/main/notebooks/AMLRO_interactive_colab.ipynb


Citation
--------

If you use AMLRO in your research, please cite:

Kulathunga, D. P. et al.
*RxnRover/amlro*.
Computer Software.
USDOE Office of Energy Efficiency and Renewable Energy (EERE),
Advanced Materials & Manufacturing Technologies Office (AMMTO), 2026.
DOI: https://doi.org/10.11578/dc.20260205.1

BibTeX:

.. code-block:: bibtex

   @misc{doecode_174798,
     title        = {RxnRover/amlro},
     author       = {Kulathunga, Dulitha Prasanna and Crandall, Zachery},
     abstractNote = {AMLRO (Active Machine Learning Reaction Optimizer)...},
     doi          = {10.11578/dc.20260205.1},
     url          = {https://doi.org/10.11578/dc.20260205.1},
     howpublished = {[Computer Software] \url{https://doi.org/10.11578/dc.20260205.1}},
     year         = {2026},
     month        = {feb}
   }


Quick Installation
------------------

Create a virtual environment (recommended):

.. code-block:: bash

    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

Clone and install:

.. code-block:: bash

    git clone https://github.com/RxnRover/amlro.git
    cd amlro
    pip install -e .
