Middle Square with Weyl increment (64-bit, Squares)
---------------------------------------------------

.. module:: randomgen.squares

.. currentmodule:: randomgen.squares

.. autoclass:: Squares

Seeding and State
=================

.. autosummary::
   :toctree: generated/

   ~Squares.seed
   ~Squares.state

Parallel generation
===================
.. autosummary::
   :toctree: generated/

   ~Squares.advance
   ~Squares.jumped

Extending
=========
.. autosummary::
   :toctree: generated/

   ~Squares.cffi
   ~Squares.ctypes

Testing
=======
.. autosummary::
   :toctree: generated/

   ~Squares.random_raw

Key Generation
==============

A convenience function for pre-generating keys that can be
used with the ``key`` argument of Squares is available. This
function transforms the entropy in a :class:`~numpy.random.SeedSequence`
into a 64-bit unsigned integer key that can be used with Squares.

.. autosummary::
   :toctree: generated/

   generate_keys

