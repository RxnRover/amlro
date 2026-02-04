.. _installation:

Installation
============

AMLRO is a Python package and can be installed using ``pip`` or directly
from the source code.

Requirements
------------

* Python >= 3.9
* scikit-learn pandas numpy matplotlib seaborn flask xgboost Joblib pyDOE3

Optional dependencies may be required for specific workflows (e.g.
RDKit for cheminformatics or PyTorch for neural network models).

Virtual environments
--------------------

We strongly recommend installing AMLRO inside a virtual environment
(virtualenv or conda) to avoid dependency conflicts.

Example using ``venv``:

.. code-block:: bash

   python -m venv venv
   source venv/bin/activate   # Linux / macOS
   .\venv\Scripts\activate    # Windows


Install from source (recommended)
---------------------------------

AMLRO is currently under active development and is not yet available on PyPI.
To install AMLRO, clone the repository and install it locally.

.. code-block:: bash

   git clone https://github.com/<your-org>/AMLRO.git
   cd AMLRO
   pip install -e .

For development (documentation, testing, formatting):

.. code-block:: bash

   pip install -e .[dev]

Install via pip (not yet available on PyPI)
-----------------------------

The easiest way to install AMLRO is from PyPI:

.. code-block:: bash

   pip install amlro


Verify installation
-------------------

To verify that AMLRO is installed correctly:

.. code-block:: python

   import amlro
   print(amlro.__version__)
