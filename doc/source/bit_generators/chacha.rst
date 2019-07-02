ChaCha Cipher-based RNG
-----------------------

.. module:: randomgen.chacha

.. currentmodule:: randomgen.chacha

.. autoclass:: ChaCha

Seeding and State
=================

.. autosummary::
   :toctree: generated/

   ~ChaCha.seed
   ~ChaCha.state
   ~ChaCha.from_seed_seq

Parallel generation
===================
.. autosummary::
   :toctree: generated/

   ~ChaCha.advance
   ~ChaCha.jump
   ~ChaCha.jumped

Extending
=========
.. autosummary::
   :toctree: generated/

   ~ChaCha.cffi
   ~ChaCha.ctypes

Testing
=======
.. autosummary::
   :toctree: generated/

   ~ChaCha.random_raw
