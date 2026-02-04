AMLRO
=====

This is the documentation of **amlro**.

Active Machine Learning Reaction Optimizer (AMLRO) for data-efficient reaction
process condition discovery and optimization.

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

.. image:: _static/AMLRO.jpg
    :width: 80%
    :align: center
    :alt: AMLRO workflow overview

Key Features
------------

- Active learning for reaction optimization
- Continuous and categorical reaction parameters
- Modular regression and acquisition strategies
- Designed for experimental feedback loops

Get Started
-----------

- :doc:`installation`
- :doc:`quickstart`
- :doc:`tutorials/index`
- :doc:`api/modules`

Contents
~~~~~~~~

.. toctree::
    :maxdepth: 2

    installation
    quickstart
    user_guide/index
    tutorials/index
    License <license>
    Authors <authors>
    Module Reference <api/modules>

Indices and tables
~~~~~~~~~~~~~~~~~~

- :ref:`genindex`
- :ref:`modindex`
- :ref:`search`

.. _autodoc: https://www.sphinx-doc.org/en/master/ext/autodoc.html

.. _classical style: https://www.sphinx-doc.org/en/master/domains.html#info-field-lists

.. _google style: https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings

.. _matplotlib: https://matplotlib.org/contents.html#

.. _numpy: https://numpy.org/doc/stable

.. _numpy style: https://numpydoc.readthedocs.io/en/latest/format.html

.. _pandas: https://pandas.pydata.org/pandas-docs/stable

.. _python: https://docs.python.org/

.. _python domain syntax: https://www.sphinx-doc.org/en/master/usage/restructuredtext/domains.html#the-python-domain

.. _references: https://www.sphinx-doc.org/en/stable/markup/inline.html

.. _restructuredtext: https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html

.. _scikit-learn: https://scikit-learn.org/stable

.. _scipy: https://docs.scipy.org/doc/scipy/reference/

.. _sphinx: https://www.sphinx-doc.org/

.. _toctree: https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html
