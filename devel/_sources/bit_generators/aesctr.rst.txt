AES Counter-based RNG
---------------------

.. module:: randomgen.aes

.. currentmodule:: randomgen.aes

.. autoclass:: AESCounter

Seeding and State
=================

.. autosummary::
   :toctree: generated/

   ~AESCounter.seed
   ~AESCounter.state

Parallel generation
===================
.. autosummary::
   :toctree: generated/

   ~AESCounter.advance
   ~AESCounter.jump
   ~AESCounter.jumped

Extending
=========
.. autosummary::
   :toctree: generated/

   ~AESCounter.cffi
   ~AESCounter.ctypes

Testing
=======
.. autosummary::
   :toctree: generated/

   ~AESCounter.random_raw
