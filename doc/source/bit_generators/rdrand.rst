Hardware-based Random Number Generator (RDRAND)
-----------------------------------------------

.. module:: randomgen.rdrand

.. currentmodule:: randomgen.rdrand

.. autoclass:: RDRAND

Seeding and State
=================

.. autosummary::
   :toctree: generated/

   ~RDRAND.seed
   ~RDRAND.state
   ~RDRAND.from_seed_seq

Parallel generation
===================
.. autosummary::
   :toctree: generated/

   ~RDRAND.jumped

Extending
=========
.. autosummary::
   :toctree: generated/

   ~RDRAND.cffi
   ~RDRAND.ctypes

Testing
=======
.. autosummary::
   :toctree: generated/

   ~RDRAND.random_raw
